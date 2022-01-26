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
A collection of dictionary-based wrappers around the "vanilla" transforms for spatial operations
defined in :py:class:`monai.transforms.spatial.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import (
    Orientation,
    RandFlip,
    Rotate90,
    Spacing,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    ensure_tuple_rep,
)
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import

nib, _ = optional_import("nibabel")

__all__ = [
    "Spacingd",
    "Orientationd",
    "RandRotate90d",
    "RandFlipd",
    "SpacingD",
    "SpacingDict",
    "OrientationD",
    "OrientationDict",
    "RandRotate90D",
    "RandFlipD",
    "RandFlipDict",
]

GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]], NumpyPadMode, PytorchPadMode, str]


class Spacingd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Spacing`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    see also:
        :py:class:`monai.transforms.Spacing`
    """

    backend = Spacing.backend

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Union[Sequence[float], float],
        diagonal: bool = False,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Optional[Union[Sequence[DtypeLike], DtypeLike]] = np.float64,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
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

                    np.diag((pixdim_0, pixdim_1, pixdim_2, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, the axes orientation, orthogonal rotation and
                translations components from the original affine will be
                preserved in the target affine. This option will not flip/swap
                axes against the original ones.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, affine, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys=None, use `key_{postfix}` to to fetch the meta data according
                to the key data, default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        """
        super().__init__(keys, allow_missing_keys)
        self.spacing_transform = Spacing(pixdim, diagonal=diagonal)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, NdarrayOrTensor]]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: Dict = dict(data)
        for key, mode, padding_mode, align_corners, dtype, meta_key, meta_key_postfix in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            original_spatial_shape = d[key].shape[1:]
            d[key], old_affine, new_affine = self.spacing_transform(
                data_array=d[key],
                affine=meta_data["affine"],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
            )
            self.push_transform(
                d,
                key,
                extra_info={
                    "meta_key": meta_key,
                    "old_affine": old_affine,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                },
                orig_size=original_spatial_shape,
            )
            # set the 'affine' key
            meta_data["affine"] = new_affine
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key, dtype in self.key_iterator(d, self.dtype):
            transform = self.get_most_recent_transform(d, key)
            if self.spacing_transform.diagonal:
                raise RuntimeError(
                    "Spacingd:inverse not yet implemented for diagonal=True. "
                    + "Please raise a github issue if you need this feature"
                )
            # Create inverse transform
            meta_data = d[transform[TraceKeys.EXTRA_INFO]["meta_key"]]
            old_affine = np.array(transform[TraceKeys.EXTRA_INFO]["old_affine"])
            mode = transform[TraceKeys.EXTRA_INFO]["mode"]
            padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
            align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
            orig_size = transform[TraceKeys.ORIG_SIZE]
            orig_pixdim = np.sqrt(np.sum(np.square(old_affine), 0))[:-1]
            inverse_transform = Spacing(orig_pixdim, diagonal=self.spacing_transform.diagonal)
            # Apply inverse
            d[key], _, new_affine = inverse_transform(
                data_array=d[key],
                affine=meta_data["affine"],  # type: ignore
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False if align_corners == TraceKeys.NONE else align_corners,
                dtype=dtype,
                output_spatial_shape=orig_size,
            )
            meta_data["affine"] = new_affine  # type: ignore
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class Orientationd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Orientation`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After reorienting the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.
    """

    backend = Orientation.backend

    def __init__(
        self,
        keys: KeysCollection,
        axcodes: Optional[str] = None,
        as_closest_canonical: bool = False,
        labels: Optional[Sequence[Tuple[str, str]]] = tuple(zip("LPI", "RAS")),
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
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
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, affine, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to to fetch the meta data according
                to the key data, default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        See Also:
            `nibabel.orientations.ornt2axcodes`.

        """
        super().__init__(keys, allow_missing_keys)
        self.ornt_transform = Orientation(axcodes=axcodes, as_closest_canonical=as_closest_canonical, labels=labels)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, NdarrayOrTensor]]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: Dict = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]
            d[key], old_affine, new_affine = self.ornt_transform(d[key], affine=meta_data["affine"])
            self.push_transform(d, key, extra_info={"meta_key": meta_key, "old_affine": old_affine})
            d[meta_key]["affine"] = new_affine
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            meta_data: Dict = d[transform[TraceKeys.EXTRA_INFO]["meta_key"]]  # type: ignore
            orig_affine = transform[TraceKeys.EXTRA_INFO]["old_affine"]
            orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
            inverse_transform = Orientation(
                axcodes=orig_axcodes, as_closest_canonical=False, labels=self.ornt_transform.labels
            )
            # Apply inverse
            d[key], _, new_affine = inverse_transform(d[key], affine=meta_data["affine"])
            meta_data["affine"] = new_affine
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandRotate90d(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate90`.
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    backend = Rotate90.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: Tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        super().randomize(None)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        self.randomize()
        d = dict(data)

        # FIXME: here we didn't use array version `RandRotate90` transform as others, because we need
        # to be compatible with the random status of some previous integration tests
        rotator = Rotate90(self._rand_k, self.spatial_axes)
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = rotator(d[key])
            self.push_transform(d, key, extra_info={"rand_k": self._rand_k})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Create inverse transform
                num_times_rotated = transform[TraceKeys.EXTRA_INFO]["rand_k"]
                num_times_to_rotate = 4 - num_times_rotated
                inverse_transform = Rotate90(num_times_to_rotate, self.spatial_axes)
                # Apply inverse
                d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandFlipd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandFlip.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.flipper = RandFlip(prob=1.0, spatial_axis=spatial_axis)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandFlipd":
        super().set_random_state(seed, state)
        self.flipper.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key], randomize=False)
            self.push_transform(d, key)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Inverse is same as forward
                d[key] = self.flipper(d[key], randomize=False)
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


SpacingD = SpacingDict = Spacingd
OrientationD = OrientationDict = Orientationd
RandRotate90D = RandRotate90Dict = RandRotate90d
RandFlipD = RandFlipDict = RandFlipd
