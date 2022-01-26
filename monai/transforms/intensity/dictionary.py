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
A collection of dictionary-based wrappers around the "vanilla" transforms for intensity adjustment
defined in :py:class:`monai.transforms.intensity.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Dict, Hashable, Mapping, Optional, Tuple, Union

import numpy as np

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.intensity.array import (
    RandScaleIntensity,
    RandShiftIntensity,
    ScaleIntensityRange,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import ensure_tuple, ensure_tuple_rep

__all__ = [
    "RandShiftIntensityD",
    "RandShiftIntensityDict",
    "RandScaleIntensityD",
    "RandScaleIntensityDict",
    "ScaleIntensityRangeD",
    "ScaleIntensityRangeDict",
]


class RandShiftIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandShiftIntensity`.
    """

    backend = RandShiftIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        offsets: Union[Tuple[float, float], float],
        factor_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offsets: offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            factor_key: if not None, use it as the key to extract a value from the corresponding
                meta data dictionary of `key` at runtime, and multiply the random `offset` to shift intensity.
                Usually, `IntensityStatsd` transform can pre-compute statistics of intensity values
                and store in the meta data.
                it also can be a sequence of strings, map to `keys`.
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                used to extract the factor value is `factor_key` is not None.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to to fetch the meta data according
                to the key data, default is `meta_dict`, the meta data is a dictionary object.
                used to extract the factor value is `factor_key` is not None.
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.factor_key = ensure_tuple_rep(factor_key, len(self.keys))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.shifter = RandShiftIntensity(offsets=offsets, prob=1.0)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandShiftIntensityd":
        super().set_random_state(seed, state)
        self.shifter.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        # all the keys share the same random shift factor
        self.shifter.randomize(None)
        for key, factor_key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.factor_key, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            factor: Optional[float] = d[meta_key].get(factor_key) if meta_key in d else None
            d[key] = self.shifter(d[key], factor=factor, randomize=False)
        return d


class RandScaleIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    """

    backend = RandScaleIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        factors: Union[Tuple[float, float], float],
        prob: float = 0.1,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            dtype: output data type, if None, same as input image. defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.scaler = RandScaleIntensity(factors=factors, dtype=dtype, prob=1.0)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandScaleIntensityd":
        super().set_random_state(seed, state)
        self.scaler.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        # all the keys share the same random scale factor
        self.scaler.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key], randomize=False)
        return d


class ScaleIntensityRanged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


RandShiftIntensityD = RandShiftIntensityDict = RandShiftIntensityd
RandScaleIntensityD = RandScaleIntensityDict = RandScaleIntensityd
ScaleIntensityRangeD = ScaleIntensityRangeDict = ScaleIntensityRanged

