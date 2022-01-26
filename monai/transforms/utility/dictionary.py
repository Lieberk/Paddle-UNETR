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
A collection of dictionary-based wrappers around the "vanilla" transforms for utility functions
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from typing import Dict, Hashable, Mapping, Optional

import paddle

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.transforms.utility.array import (
    AddChannel,
    ToNumpy,
    ToTensor,
)

__all__ = [
    "AddChannelD",
    "AddChannelDict",
    "AddChanneld",
]


class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = AddChannel.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.adder = AddChannel()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d


class ToTensord(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    backend = ToTensor.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Optional[paddle.dtype] = None,
        wrap_sequence: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: target data content type to convert, for example: torch.float, etc.
            wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
                E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToTensor(dtype=dtype, wrap_sequence=wrap_sequence)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # Create inverse transform
            inverse_transform = ToNumpy()
            # Apply inverse
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


AddChannelD = AddChannelDict = AddChanneld
ToTensorD = ToTensorDict = ToTensord

