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
A collection of "vanilla" transforms for utility functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Optional

import paddle

from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils import (
    convert_to_numpy,
    convert_to_tensor,
    optional_import,
)
from monai.utils.enums import TransformBackends

PILImageImage, has_pil = optional_import("PIL.Image", name="Image")
pil_image_fromarray, _ = optional_import("PIL.Image", name="fromarray")
cp, has_cp = optional_import("cupy")


__all__ = [
    "AddChannel",
    "ToTensor",
    "ToNumpy",
]


class AddChannel(Transform):
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return img[None]


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    Input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
    Will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
    For dictionary, list or tuple, convert every item to a Tensor if applicable and `wrap_sequence=False`.

    Args:
        dtype: target data type to when converting to Tensor.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, dtype: Optional[paddle.dtype] = None, wrap_sequence: bool = True
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.wrap_sequence = wrap_sequence

    def __call__(self, img: NdarrayOrTensor):
        """
        Apply the transform to `img` and make it contiguous.
        """
        return convert_to_tensor(img, dtype=self.dtype, wrap_sequence=self.wrap_sequence)


class ToNumpy(Transform):
    """
    Converts the input data to numpy array, can support list or tuple of numbers and PyTorch Tensor.

    Args:
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[array(1), array(2)]`, if `True`, then `[1, 2]` -> `array([1, 2])`.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dtype: DtypeLike = None, wrap_sequence: bool = True) -> None:
        super().__init__()
        self.dtype = dtype
        self.wrap_sequence = wrap_sequence

    def __call__(self, img: NdarrayOrTensor):
        """
        Apply the transform to `img` and make it contiguous.
        """
        return convert_to_numpy(img, dtype=self.dtype, wrap_sequence=self.wrap_sequence)