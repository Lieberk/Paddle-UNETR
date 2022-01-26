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

import os
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union

import numpy as np
import paddle

# Commonly used concepts
# This module provides naming and type specifications for commonly used concepts
# within the MONAI package. The intent is to explicitly identify information
# that should be used consistently throughout the entire MONAI package.
#
# A type would be named as type_definitions.KeysCollection
# which includes a meaningful name for the consent in the name itself. The
# definitions in this file map context meaningful names to the underlying
# object properties that define the expected API.
#
# A conceptual type is represented by a new type name but is also one which
# can be different depending on an environment (i.e. differences for python 3.6 vs 3.9
# may be implemented). Consistent use of the concept and recorded documentation of
# the rationale and convention behind it lowers the learning curve for new
# developers. For readability, short names are preferred.
__all__ = [
    "KeysCollection",
    "IndexSelection",
    "DtypeLike",
    "NdarrayTensor",
    "NdarrayOrTensor",
    "TensorOrList",
    "PathLike",
]


#: KeysCollection
#
# The KeyCollection type is used to for defining variables
# that store a subset of keys to select items from a dictionary.
# The container of keys must contain hashable elements.
# NOTE:  `Hashable` is not a collection, but is provided as a
#        convenience to end-users.  All supplied values will be
#        internally converted to a tuple of `Hashable`'s before
#        use
KeysCollection = Union[Collection[Hashable], Hashable]

#: IndexSelection
#
# The IndexSelection type is used to for defining variables
# that store a subset of indices to select items from a List or Array like objects.
# The indices must be integers, and if a container of indices is specified, the
# container must be iterable.
IndexSelection = Union[Iterable[int], int]

#: Type of datatypes: Adapted from https://github.com/numpy/numpy/blob/v1.21.4/numpy/typing/_dtype_like.py#L121
DtypeLike = Union[np.dtype, type, str, None]

#: NdarrayTensor
#
# Generic type which can represent either a numpy.ndarray or a torch.Tensor
# Unlike Union can create a dependence between parameter(s) / return(s)
NdarrayTensor = TypeVar("NdarrayTensor", np.ndarray, paddle.Tensor)


#: NdarrayOrTensor: Union of numpy.ndarray and torch.Tensor to be used for typing
NdarrayOrTensor = Union[np.ndarray, paddle.Tensor]

#: TensorOrList: The TensorOrList type is used for defining `batch-first Tensor` or `list of channel-first Tensor`.
TensorOrList = Union[paddle.Tensor, Sequence[paddle.Tensor]]

#: PathLike: The PathLike type is used for defining a file path.
PathLike = Union[str, os.PathLike]
