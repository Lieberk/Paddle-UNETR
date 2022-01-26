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
A collection of "vanilla" transforms for intensity adjustment
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Any, Optional, Tuple, Union
from warnings import warn

import numpy as np
import paddle

from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import RandomizableTransform, Transform
from monai.transforms.utils import rescale_array
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils import (
    convert_data_type,
)
from monai.utils.enums import TransformBackends

__all__ = [
    "ShiftIntensity",
    "RandShiftIntensity",
    "ScaleIntensity",
    "RandScaleIntensity",
    "ScaleIntensityRange",
]


class ShiftIntensity(Transform):
    """
    Shift intensity uniformly for the entire image with specified `offset`.

    Args:
        offset: offset value to shift the intensity of image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, img: NdarrayOrTensor, offset: Optional[float] = None) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """

        offset = self.offset if offset is None else offset
        out = img + offset
        out, *_ = convert_data_type(data=out, dtype=img.dtype)

        return out


class RandShiftIntensity(RandomizableTransform):
    """
    Randomly shift intensity with randomly picked offset.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, offsets: Union[Tuple[float, float], float], prob: float = 0.1) -> None:
        """
        Args:
            offsets: offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            prob: probability of shift.
        """
        RandomizableTransform.__init__(self, prob)
        if isinstance(offsets, (int, float)):
            self.offsets = (min(-offsets, offsets), max(-offsets, offsets))
        elif len(offsets) != 2:
            raise ValueError("offsets should be a number or pair of numbers.")
        else:
            self.offsets = (min(offsets), max(offsets))
        self._offset = self.offsets[0]
        self._shfiter = ShiftIntensity(self._offset)

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._offset = self.R.uniform(low=self.offsets[0], high=self.offsets[1])

    def __call__(self, img: NdarrayOrTensor, factor: Optional[float] = None, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.

        Args:
            img: input image to shift intensity.
            factor: a factor to multiply the random offset, then shift.
                can be some image specific value at runtime, like: max(img), etc.

        """
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        return self._shfiter(img, self._offset if factor is None else self._offset * factor)


class ScaleIntensity(Transform):
    """
    Scale the intensity of input image to the given value range (minv, maxv).
    If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        minv: Optional[float] = 0.0,
        maxv: Optional[float] = 1.0,
        factor: Optional[float] = None,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        """
        Args:
            minv: minimum value of output data.
            maxv: maximum value of output data.
            factor: factor scale by ``v = v * (1 + factor)``. In order to use
                this parameter, please set both `minv` and `maxv` into None.
            channel_wise: if True, scale on each channel separately. Please ensure
                that the first dimension represents the channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
        """
        self.minv = minv
        self.maxv = maxv
        self.factor = factor
        self.channel_wise = channel_wise
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.

        Raises:
            ValueError: When ``self.minv=None`` or ``self.maxv=None`` and ``self.factor=None``. Incompatible values.

        """
        if self.minv is not None or self.maxv is not None:
            if self.channel_wise:
                out = [rescale_array(d, self.minv, self.maxv, dtype=self.dtype) for d in img]
                ret = paddle.stack(out) if isinstance(img, paddle.Tensor) else np.stack(out)  # type: ignore
            else:
                ret = rescale_array(img, self.minv, self.maxv, dtype=self.dtype)
        else:
            ret = (img * (1 + self.factor)) if self.factor is not None else img

        ret, *_ = convert_data_type(ret, dtype=self.dtype or img.dtype)
        return ret


class RandScaleIntensity(RandomizableTransform):
    """
    Randomly scale the intensity of input image by ``v = v * (1 + factor)`` where the `factor`
    is randomly picked.
    """

    backend = ScaleIntensity.backend

    def __init__(
        self, factors: Union[Tuple[float, float], float], prob: float = 0.1, dtype: DtypeLike = np.float32
    ) -> None:
        """
        Args:
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of scale.
            dtype: output data type, if None, same as input image. defaults to float32.

        """
        RandomizableTransform.__init__(self, prob)
        if isinstance(factors, (int, float)):
            self.factors = (min(-factors, factors), max(-factors, factors))
        elif len(factors) != 2:
            raise ValueError("factors should be a number or pair of numbers.")
        else:
            self.factors = (min(factors), max(factors))
        self.factor = self.factors[0]
        self.dtype = dtype

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        return ScaleIntensity(minv=None, maxv=None, factor=self.factor, dtype=self.dtype)(img)


class ScaleIntensityRange(Transform):
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    When `b_min` or `b_max` are `None`, `scacled_array * (b_max - b_min) + b_min` will be skipped.
    If `clip=True`, when `b_min`/`b_max` is None, the clipping is not performed on the corresponding edge.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        dtype = self.dtype or img.dtype
        if self.a_max - self.a_min == 0.0:
            warn("Divide by zero (a_min == a_max)", Warning)
            if self.b_min is None:
                return img - self.a_min
            return img - self.a_min + self.b_min

        img = (img - self.a_min) / (self.a_max - self.a_min)
        if (self.b_min is not None) and (self.b_max is not None):
            img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = clip(img, self.b_min, self.b_max)
        ret, *_ = convert_data_type(img, dtype=dtype)

        return ret




