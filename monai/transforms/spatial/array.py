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
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""
import warnings
from typing import Optional, Sequence, Tuple, Union

import utils.utils as utils
import numpy as np
import paddle

from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import compute_shape_offset, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform
from monai.transforms.transform import RandomizableTransform, Transform
from monai.transforms.utils import (
    map_spatial_axes,
)
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    ensure_tuple,
    optional_import,
)
from monai.utils.enums import TransformBackends
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

nib, _ = optional_import("nibabel")

__all__ = [
    "Spacing",
    "Orientation",
    "Flip",
    "Rotate90",
    "RandFlip",
]

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]


class Spacing(Transform):
    """
    Resample input image into the specified `pixdim`.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        pixdim: Union[Sequence[float], float],
        diagonal: bool = False,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        image_only: bool = False,
    ) -> None:
        """
        Args:
            pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
                items of the pixdim sequence map to the spatial dimensions of input image, if length
                of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
                if shorter, will pad with `1.0`.
                if the components of the `pixdim` are non-positive values, the transform will use the
                corresponding components of the original pixdim, which is computed from the `affine`
                matrix of input image.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
            image_only: return just the image or the image, the old affine and new affine. Default is `False`.

        """
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype
        self.image_only = image_only

    def __call__(
        self,
        data_array: NdarrayOrTensor,
        affine: Optional[NdarrayOrTensor] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: DtypeLike = None,
        output_spatial_shape: Optional[np.ndarray] = None,
    ) -> Union[NdarrayOrTensor, Tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]]:
        """
        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
            output_spatial_shape: specify the shape of the output data_array. This is typically useful for
                the inverse of `Spacingd` where sometimes we could not compute the exact shape due to the quantization
                error with the affine.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``pixdim`` is nonpositive.

        Returns:
            data_array (resampled into `self.pixdim`), original affine, current affine.

        """
        _dtype = dtype or self.dtype or data_array.dtype
        sr = int(data_array.ndim - 1)
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            # default to identity
            affine_np = affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_np, *_ = convert_data_type(affine, np.ndarray)  # type: ignore
            affine_ = to_affine_nd(sr, affine_np)

        out_d = self.pixdim[:sr]
        if out_d.size < sr:
            out_d = np.append(out_d, [1.0] * (sr - out_d.size))

        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        output_shape, offset = compute_shape_offset(data_array.shape[1:], affine_, new_affine)
        new_affine[:sr, -1] = offset[:sr]
        transform = np.linalg.inv(affine_) @ new_affine
        # adapt to the actual rank
        transform = to_affine_nd(sr, transform)

        # no resampling if it's identity transform
        if np.allclose(transform, np.diag(np.ones(len(transform))), atol=1e-3):
            output_data = data_array
        else:
            # resample
            affine_xform = AffineTransform(
                normalized=False,
                mode=look_up_option(mode or self.mode, GridSampleMode),
                padding_mode=look_up_option(padding_mode or self.padding_mode, GridSamplePadMode),
                align_corners=self.align_corners if align_corners is None else align_corners,
                reverse_indexing=True,
            )
            data_array_t: paddle.Tensor
            data_array_t, *_ = convert_data_type(data_array, paddle.Tensor, dtype=_dtype)  # type: ignore
            output_data = affine_xform(
                # AffineTransform requires a batch dim
                data_array_t.unsqueeze(0),
                convert_data_type(transform, paddle.Tensor, dtype=_dtype)[0],
                spatial_size=output_shape if output_spatial_shape is None else output_spatial_shape,
            ).squeeze(0)

        output_data, *_ = convert_to_dst_type(output_data, data_array, dtype=paddle.float32)
        new_affine = to_affine_nd(affine_np, new_affine)  # type: ignore
        new_affine, *_ = convert_to_dst_type(src=new_affine, dst=affine, dtype=paddle.float32)

        if self.image_only:
            return output_data
        return output_data, affine, new_affine


class Orientation(Transform):
    """
    Change the input image's orientation into the specified based on `axcodes`.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        axcodes: Optional[str] = None,
        as_closest_canonical: bool = False,
        labels: Optional[Sequence[Tuple[str, str]]] = tuple(zip("LPI", "RAS")),
        image_only: bool = False,
    ) -> None:
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.
            image_only: if True return only the image volume, otherwise return (image, affine, new_affine).

        Raises:
            ValueError: When ``axcodes=None`` and ``as_closest_canonical=True``. Incompatible values.

        See Also: `nibabel.orientations.ornt2axcodes`.

        """
        if axcodes is None and not as_closest_canonical:
            raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
        if axcodes is not None and as_closest_canonical:
            warnings.warn("using as_closest_canonical=True, axcodes ignored.")
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels
        self.image_only = image_only

    def __call__(
        self, data_array: NdarrayOrTensor, affine: Optional[NdarrayOrTensor] = None
    ) -> Union[NdarrayOrTensor, Tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]]:
        """
        original orientation of `data_array` is defined by `affine`.

        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``axcodes`` spatiality differs from ``data_array``.

        Returns:
            data_array [reoriented in `self.axcodes`] if `self.image_only`, else
            (data_array [reoriented in `self.axcodes`], original axcodes, current axcodes).

        """
        data_array_np, *_ = convert_data_type(data_array, np.ndarray)  # type: ignore
        sr = data_array_np.ndim - 1
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            # default to identity
            affine_np = affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_np, *_ = convert_data_type(affine, np.ndarray)  # type: ignore
            affine_ = to_affine_nd(sr, affine_np)

        src = nib.io_orientation(affine_)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            if self.axcodes is None:
                raise AssertionError
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
            if len(dst) < sr:
                raise ValueError(
                    f"axcodes must match data_array spatially, got axcodes={len(self.axcodes)}D data_array={sr}D"
                )
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        ornt = spatial_ornt.copy()
        ornt[:, 0] += 1  # skip channel dim
        ornt = np.concatenate([np.array([[0, 1]]), ornt])
        shape = data_array_np.shape[1:]
        data_array_np = np.ascontiguousarray(nib.orientations.apply_orientation(data_array_np, ornt))
        new_affine = affine_ @ nib.orientations.inv_ornt_aff(spatial_ornt, shape)
        new_affine = to_affine_nd(affine_np, new_affine)
        out, *_ = convert_to_dst_type(src=data_array_np, dst=data_array)
        new_affine, *_ = convert_to_dst_type(src=new_affine, dst=affine, dtype=paddle.float32)

        if self.image_only:
            return out
        return out, affine, new_affine


class Rotate90(Transform):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    See np.rot90 for additional details:
    https://numpy.org/doc/stable/reference/generated/numpy.rot90.html.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                If axis is negative it counts from the last to the first axis.
        """
        self.k = k
        spatial_axes_: Tuple[int, int] = ensure_tuple(spatial_axes)  # type: ignore
        if len(spatial_axes_) != 2:
            raise ValueError("spatial_axes must be 2 int numbers to indicate the axes to rotate 90 degrees.")
        self.spatial_axes = spatial_axes_

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        rot90 = utils.rot90 if isinstance(img, paddle.Tensor) else np.rot90
        out: NdarrayOrTensor = rot90(img, self.k, map_spatial_axes(img.ndim, self.spatial_axes))
        out, *_ = convert_data_type(out, dtype=img.dtype)
        return out


class Flip(Transform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        self.spatial_axis = spatial_axis

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        if isinstance(img, np.ndarray):
            return np.ascontiguousarray(np.flip(img, map_spatial_axes(img.ndim, self.spatial_axis)))
        return paddle.flip(img, map_spatial_axes(img.ndim, self.spatial_axis))


class RandFlip(RandomizableTransform):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        RandomizableTransform.__init__(self, prob)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize(None)

        if not self._do_transform:
            return img

        return self.flipper(img)
