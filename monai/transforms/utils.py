# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import warnings

from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import paddle

from monai.config import DtypeLike, IndexSelection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utils_pytorch_numpy_unification import (
    any_np_pt,
    nonzero,
    ravel,
    unravel_index,
    where,
)
from monai.utils import (
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    min_version,
    optional_import,
)
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)
ndimage, _ = optional_import("scipy.ndimage")
cp, has_cp = optional_import("cupy")
cp_ndarray, _ = optional_import("cupy", name="ndarray")
exposure, has_skimage = optional_import("skimage.exposure")

__all__ = [
    "compute_divisible_spatial_size",
    "generate_pos_neg_label_crop_centers",
    "generate_spatial_bounding_box",
    "is_positive",
    "map_binary_to_indices",
    "map_spatial_axes",
    "rescale_array",
    "convert_pad_mode",
]


def is_positive(img):
    """
    Returns a boolean version of `img` where the positive values are converted into True, the other values are False.
    """
    return img > 0


def rescale_array(
    arr: NdarrayOrTensor,
    minv: Optional[float] = 0.0,
    maxv: Optional[float] = 1.0,
    dtype: Optional[Union[DtypeLike, paddle.dtype]] = np.float32,
) -> NdarrayOrTensor:
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    If either `minv` or `maxv` is None, it returns `(a - min_a) / (max_a - min_a)`.

    Args:
        arr: input array to rescale.
        minv: minimum value of target rescaled array.
        maxv: maxmum value of target rescaled array.
        dtype: if not None, convert input array to dtype before computation.

    """
    if dtype is not None:
        arr, *_ = convert_data_type(arr, dtype=dtype)
    mina = arr.min()
    maxa = arr.max()

    if mina == maxa:
        return arr * minv if minv is not None else arr

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    if (minv is None) or (maxv is None):
        return norm
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default


def map_binary_to_indices(
    label: NdarrayOrTensor, image: Optional[NdarrayOrTensor] = None, image_threshold: float = 0.0
) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
    """
    Compute the foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])``
    ``foreground indices = np.array([1, 2, 3, 5, 6, 7])`` and ``background indices = np.array([0, 4, 8])``

    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.
    """

    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    label_flat = ravel(any_np_pt(label, 0))  # in case label has multiple dimensions
    fg_indices = nonzero(label_flat)
    if image is not None:
        img_flat = ravel(any_np_pt(image > image_threshold, 0))
        img_flat, *_ = convert_to_dst_type(img_flat, label, dtype=img_flat.dtype)
        bg_indices = nonzero(img_flat & ~label_flat)
    else:
        bg_indices = nonzero(~label_flat)

    # no need to save the indices in GPU, otherwise, still need to move to CPU at runtime when crop by indices
    fg_indices, *_ = convert_data_type(fg_indices)
    bg_indices, *_ = convert_data_type(bg_indices)
    return fg_indices, bg_indices


def correct_crop_centers(
    centers: List[Union[int, paddle.Tensor]],
    spatial_size: Union[Sequence[int], int],
    label_spatial_shape: Sequence[int],
    allow_smaller: bool = False,
):
    """
    Utility to correct the crop center if the crop size and centers are not compatible with the image size.

    Args:
        centers: pre-computed crop centers of every dim, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if any(np.subtract(label_spatial_shape, spatial_size) < 0):
        if not allow_smaller:
            raise ValueError("The size of the proposed random crop ROI is larger than the image size.")
        spatial_size = tuple(min(l, s) for l, s in zip(label_spatial_shape, spatial_size))

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1
    valid_centers = []
    for c, v_s, v_e in zip(centers, valid_start, valid_end):
        _c = int(convert_data_type(c, np.ndarray)[0])  # type: ignore
        center_i = min(max(_c, v_s), v_e - 1)
        valid_centers.append(int(center_i))
    return valid_centers


def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: NdarrayOrTensor,
    bg_indices: NdarrayOrTensor,
    rand_state: Optional[np.random.RandomState] = None,
    allow_smaller: bool = False,
) -> List[List[int]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape)
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return centers


def generate_spatial_bounding_box(
    img: NdarrayOrTensor,
    select_fn: Callable = is_positive,
    channel_indices: Optional[IndexSelection] = None,
    margin: Union[Sequence[int], int] = 0,
) -> Tuple[List[int], List[int]]:
    """
    generate the spatial bounding box of foreground in the image with start-end positions (inclusive).
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:

        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]

    The bounding boxes edges are aligned with the input image edges.
    This function returns [-1, -1, ...], [-1, -1, ...] if there's no positive intensity.

    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """
    data = img[list(ensure_tuple(channel_indices))] if channel_indices is not None else img
    data = select_fn(data).any(0)
    ndim = len(data.shape)
    margin = ensure_tuple_rep(margin, ndim)
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(itertools.combinations(reversed(range(ndim)), ndim - 1)):
        dt = data
        if len(ax) != 0:
            dt = any_np_pt(dt, ax)

        if not dt.any():
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        arg_max = where(dt == dt.max())[0]
        min_d = max(arg_max[0] - margin[di], 0)
        max_d = arg_max[-1] + margin[di] + 1

        box_start[di] = min_d.detach().cpu().item() if isinstance(min_d, paddle.Tensor) else min_d  # type: ignore
        box_end[di] = max_d.detach().cpu().item() if isinstance(max_d, paddle.Tensor) else max_d  # type: ignore

    return box_start, box_end


def map_spatial_axes(
    img_ndim: int, spatial_axes: Optional[Union[Sequence[int], int]] = None, channel_first: bool = True
) -> List[int]:
    """
    Utility to map the spatial axes to real axes in channel first/last shape.
    For example:
    If `channel_first` is True, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [1, 2, 3]
    [0, 1] -> [1, 2]
    [0, -1] -> [1, -1]
    If `channel_first` is False, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [0, 1, 2]
    [0, 1] -> [0, 1]
    [0, -1] -> [0, -2]

    Args:
        img_ndim: dimension number of the target image.
        spatial_axes: spatial axes to be converted, default is None.
            The default `None` will convert to all the spatial axes of the image.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints.
        channel_first: the image data is channel first or channel last, default to channel first.

    """
    if spatial_axes is None:
        spatial_axes_ = list(range(1, img_ndim) if channel_first else range(img_ndim - 1))

    else:
        spatial_axes_ = []
        for a in ensure_tuple(spatial_axes):
            if channel_first:
                spatial_axes_.append(a if a < 0 else a + 1)
            else:
                spatial_axes_.append(a - 1 if a < 0 else a)

    return spatial_axes_


def compute_divisible_spatial_size(spatial_shape: Sequence[int], k: Union[Sequence[int], int]):
    """
    Compute the target spatial size which should be divisible by `k`.

    Args:
        spatial_shape: original spatial shape.
        k: the target k for each spatial dimension.
            if `k` is negative or 0, the original size is preserved.
            if `k` is an int, the same `k` be applied to all the input spatial dimensions.

    """
    k = fall_back_tuple(k, (1,) * len(spatial_shape))
    new_size = []
    for k_d, dim in zip(k, spatial_shape):
        new_dim = int(np.ceil(dim / k_d) * k_d) if k_d > 0 else dim
        new_size.append(new_dim)

    return new_size


def convert_pad_mode(dst: NdarrayOrTensor, mode: Union[NumpyPadMode, PytorchPadMode, str]):
    """
    Utility to convert padding mode between numpy array and PyTorch Tensor.

    Args:
        dst: target data to convert padding mode for, should be numpy array or PyTorch Tensor.
        mode: current padding mode.

    """
    mode = mode.value if isinstance(mode, (NumpyPadMode, PytorchPadMode)) else mode
    if isinstance(dst, paddle.Tensor):
        if mode == "wrap":
            mode = "circular"
        if mode == "edge":
            mode = "replicate"
        return look_up_option(mode, PytorchPadMode)
    if isinstance(dst, np.ndarray):
        if mode == "circular":
            mode = "wrap"
        if mode == "replicate":
            mode = "edge"
        return look_up_option(mode, NumpyPadMode)
    raise ValueError(f"unsupported data type: {type(dst)}.")

