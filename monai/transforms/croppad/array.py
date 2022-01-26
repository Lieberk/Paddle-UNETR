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
"""
A collection of "vanilla" transforms for crop and pad operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from itertools import chain
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import paddle
from paddle.nn.functional import pad as pad_pt

from monai.config import IndexSelection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Randomizable, Transform
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    convert_pad_mode,
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    is_positive,
    map_binary_to_indices,
)
from monai.transforms.utils_pytorch_numpy_unification import floor_divide, maximum
from monai.utils import (
    Method,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
)
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

__all__ = [
    "Pad",
    "SpatialPad",
    "BorderPad",
    "SpatialCrop",
    "CropForeground",
    "RandCropByPosNegLabel",
]


class Pad(Transform):
    """
    Args:
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            to_pad: List[Tuple[int, int]],
            mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.CONSTANT,
            **kwargs,
    ) -> None:
        self.to_pad = to_pad
        self.mode = mode
        self.kwargs = kwargs

    @staticmethod
    def _np_pad(img: np.ndarray, all_pad_width, mode, **kwargs) -> np.ndarray:
        return np.pad(img, all_pad_width, mode=mode, **kwargs)  # type: ignore

    @staticmethod
    def _pt_pad(img: paddle.Tensor, all_pad_width, mode, **kwargs) -> paddle.Tensor:
        pt_pad_width = [val for sublist in all_pad_width[1:] for val in sublist[::-1]][::-1]
        # torch.pad expects `[B, C, H, W, [D]]` shape
        return pad_pt(img.unsqueeze(0), pt_pad_width, mode=mode, **kwargs).squeeze(0)

    def __call__(
            self, img: NdarrayOrTensor, mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"`` or ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to `self.mode`.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

        """
        if not np.asarray(self.to_pad).any():
            # all zeros, skip padding
            return img
        mode = convert_pad_mode(dst=img, mode=mode or self.mode).value
        pad = self._pt_pad if isinstance(img, paddle.Tensor) else self._np_pad
        return pad(img, self.to_pad, mode, **self.kwargs)  # type: ignore


class SpatialPad(Transform):
    """
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.

    Args:
        spatial_size: the spatial size of output data after padding, if a dimension of the input
            data size is bigger than the pad size, will not pad that dimension.
            If its components have non-positive values, the corresponding size of input image will be used
            (no padding). for example: if the spatial size of input data is [30, 30, 30] and
            `spatial_size=[32, 25, -1]`, the spatial size of output data will be [32, 30, 30].
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Pad.backend

    def __init__(
            self,
            spatial_size: Union[Sequence[int], int],
            method: Union[Method, str] = Method.SYMMETRIC,
            mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.CONSTANT,
            **kwargs,
    ) -> None:
        self.spatial_size = spatial_size
        self.method: Method = look_up_option(method, Method)
        self.mode = mode
        self.kwargs = kwargs

    def _determine_data_pad_width(self, data_shape: Sequence[int]) -> List[Tuple[int, int]]:
        spatial_size = fall_back_tuple(self.spatial_size, data_shape)
        if self.method == Method.SYMMETRIC:
            pad_width = []
            for i, sp_i in enumerate(spatial_size):
                width = max(sp_i - data_shape[i], 0)
                pad_width.append((width // 2, width - (width // 2)))
            return pad_width
        return [(0, max(sp_i - data_shape[i], 0)) for i, sp_i in enumerate(spatial_size)]

    def __call__(
            self, img: NdarrayOrTensor, mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to `self.mode`.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

        """
        data_pad_width = self._determine_data_pad_width(img.shape[1:])
        all_pad_width = [(0, 0)] + data_pad_width
        if not np.asarray(all_pad_width).any():
            # all zeros, skip padding
            return img

        padder = Pad(all_pad_width, mode or self.mode, **self.kwargs)
        return padder(img)


class BorderPad(Transform):
    """
    Pad the input data by adding specified borders to every dimension.

    Args:
        spatial_border: specified size for every spatial border. Any -ve values will be set to 0. It can be 3 shapes:

            - single int number, pad all the borders with the same size.
            - length equals the length of image shape, pad every spatial dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [2, 1],
              pad every border of H dim with 2, pad every border of W dim with 1, result shape is [1, 8, 6].
            - length equals 2 x (length of image shape), pad every border of every dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [1, 2, 3, 4], pad top of H dim with 1,
              pad bottom of H dim with 2, pad left of W dim with 3, pad right of W dim with 4.
              the result shape is [1, 7, 11].
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Pad.backend

    def __init__(
            self,
            spatial_border: Union[Sequence[int], int],
            mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.CONSTANT,
            **kwargs,
    ) -> None:
        self.spatial_border = spatial_border
        self.mode = mode
        self.kwargs = kwargs

    def __call__(
            self, img: NdarrayOrTensor, mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to `self.mode`.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

        Raises:
            ValueError: When ``self.spatial_border`` does not contain ints.
            ValueError: When ``self.spatial_border`` length is not one of
                [1, len(spatial_shape), 2*len(spatial_shape)].

        """
        spatial_shape = img.shape[1:]
        spatial_border = ensure_tuple(self.spatial_border)
        if not all(isinstance(b, int) for b in spatial_border):
            raise ValueError(f"self.spatial_border must contain only ints, got {spatial_border}.")
        spatial_border = tuple(max(0, b) for b in spatial_border)

        if len(spatial_border) == 1:
            data_pad_width = [(spatial_border[0], spatial_border[0]) for _ in spatial_shape]
        elif len(spatial_border) == len(spatial_shape):
            data_pad_width = [(sp, sp) for sp in spatial_border[: len(spatial_shape)]]
        elif len(spatial_border) == len(spatial_shape) * 2:
            data_pad_width = [(spatial_border[2 * i], spatial_border[2 * i + 1]) for i in range(len(spatial_shape))]
        else:
            raise ValueError(
                f"Unsupported spatial_border length: {len(spatial_border)}, available options are "
                f"[1, len(spatial_shape)={len(spatial_shape)}, 2*len(spatial_shape)={2 * len(spatial_shape)}]."
            )

        all_pad_width = [(0, 0)] + data_pad_width
        padder = Pad(all_pad_width, mode or self.mode, **self.kwargs)
        return padder(img)


class SpatialCrop(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            roi_center: Union[Sequence[int], NdarrayOrTensor, None] = None,
            roi_size: Union[Sequence[int], NdarrayOrTensor, None] = None,
            roi_start: Union[Sequence[int], NdarrayOrTensor, None] = None,
            roi_end: Union[Sequence[int], NdarrayOrTensor, None] = None,
            roi_slices: Optional[Sequence[slice]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
        roi_start_torch: paddle.Tensor

        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError("Only slice steps of 1/None are currently supported")
            self.slices = list(roi_slices)
        else:
            if roi_center is not None and roi_size is not None:
                roi_center, *_ = convert_data_type(
                    data=roi_center, output_type=paddle.Tensor, dtype=paddle.int16, wrap_sequence=True
                )
                roi_size, *_ = convert_to_dst_type(src=roi_size, dst=roi_center, wrap_sequence=True)
                roi_center = roi_center.cast('float32')
                _zeros = paddle.zeros_like(roi_center)
                roi_start_torch = maximum(roi_center - floor_divide(roi_size.cast('float32'), paddle.to_tensor(2, dtype='float32')), _zeros)
                roi_end_torch = maximum(roi_start_torch + roi_size, roi_start_torch)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start_torch, *_ = convert_data_type(  # type: ignore
                    data=roi_start, output_type=paddle.Tensor, dtype=paddle.int16, wrap_sequence=True
                )
                roi_start_torch = maximum(roi_start_torch, paddle.zeros_like(roi_start_torch.cast('float32')))  # type: ignore
                roi_end_torch, *_ = convert_to_dst_type(src=roi_end, dst=roi_start_torch, wrap_sequence=True)
                roi_end_torch = maximum(roi_end_torch, roi_start_torch)
            # convert to slices (accounting for 1d)
            if roi_start_torch.numel() == 1:
                self.slices = [slice(int(roi_start_torch.item()), int(roi_end_torch.item()))]
            else:
                self.slices = [slice(int(s), int(e)) for s, e in zip(roi_start_torch, roi_end_torch)]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.slices), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + self.slices[:sd]
        return img[tuple(slices)]


class CropForeground(Transform):
    """
    Crop an image using a bounding box. The bounding box is generated by selecting foreground using select_fn
    at channels channel_indices. margin is added in each spatial dimension of the bounding box.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box of foreground object.
    For example:

    .. code-block:: python

        image = np.array(
            [[[0, 0, 0, 0, 0],
              [0, 1, 2, 1, 0],
              [0, 1, 3, 2, 0],
              [0, 1, 2, 1, 0],
              [0, 0, 0, 0, 0]]])  # 1x5x5, single channel 5x5 image


        def threshold_at_one(x):
            # threshold at 1
            return x > 1


        cropper = CropForeground(select_fn=threshold_at_one, margin=0)
        print(cropper(image))
        [[[2, 1],
          [3, 2],
          [2, 1]]]

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            select_fn: Callable = is_positive,
            channel_indices: Optional[IndexSelection] = None,
            margin: Union[Sequence[int], int] = 0,
            return_coords: bool = False,
            k_divisible: Union[Sequence[int], int] = 1,
            mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
            **np_kwargs,
    ) -> None:
        """
        Args:
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
                more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        """
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.mode: NumpyPadMode = look_up_option(mode, NumpyPadMode)
        self.np_kwargs = np_kwargs

    def compute_bounding_box(self, img: NdarrayOrTensor):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(img, self.select_fn, self.channel_indices, self.margin)
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        return box_start_, box_end_

    def crop_pad(
            self,
            img: NdarrayOrTensor,
            box_start: np.ndarray,
            box_end: np.ndarray,
            mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
    ):
        """
        Crop and pad based on the bounding box.

        """
        cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        return BorderPad(spatial_border=pad, mode=mode or self.mode, **self.np_kwargs)(cropped)

    def __call__(self, img: NdarrayOrTensor, mode: Optional[Union[NumpyPadMode, str]] = None):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img)
        cropped = self.crop_pad(img, box_start, box_end, mode)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped


class RandCropByPosNegLabel(Randomizable, Transform):
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            spatial_size: Union[Sequence[int], int],
            label: Optional[NdarrayOrTensor] = None,
            pos: float = 1.0,
            neg: float = 1.0,
            num_samples: int = 1,
            image: Optional[NdarrayOrTensor] = None,
            image_threshold: float = 0.0,
            fg_indices: Optional[NdarrayOrTensor] = None,
            bg_indices: Optional[NdarrayOrTensor] = None,
            allow_smaller: bool = False,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[int]]] = None
        self.fg_indices = fg_indices
        self.bg_indices = bg_indices
        self.allow_smaller = allow_smaller

    def randomize(
            self,
            label: NdarrayOrTensor,
            fg_indices: Optional[NdarrayOrTensor] = None,
            bg_indices: Optional[NdarrayOrTensor] = None,
            image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            if self.fg_indices is not None and self.bg_indices is not None:
                fg_indices_ = self.fg_indices
                bg_indices_ = self.bg_indices
            else:
                fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    def __call__(
            self,
            img: NdarrayOrTensor,
            label: Optional[NdarrayOrTensor] = None,
            image: Optional[NdarrayOrTensor] = None,
            fg_indices: Optional[NdarrayOrTensor] = None,
            bg_indices: Optional[NdarrayOrTensor] = None,
    ) -> List[NdarrayOrTensor]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image

        self.randomize(label, fg_indices, bg_indices, image)
        results: List[NdarrayOrTensor] = []
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
                results.append(cropper(img))

        return results
