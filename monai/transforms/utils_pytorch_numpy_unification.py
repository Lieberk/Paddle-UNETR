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

from typing import Sequence, Union

import numpy as np
import paddle

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.misc import is_module_ver_at_least

__all__ = [
    "clip",
    "where",
    "nonzero",
    "floor_divide",
    "unravel_index",
    "ravel",
    "any_np_pt",
    "maximum",
]


def clip(a: NdarrayOrTensor, a_min, a_max) -> NdarrayOrTensor:
    """`np.clip` with equivalent implementation for torch."""
    result: NdarrayOrTensor
    if isinstance(a, np.ndarray):
        result = np.clip(a, a_min, a_max)
    else:
        result = paddle.clip(a, a_min, a_max)
    return result


def where(condition: NdarrayOrTensor, x=None, y=None) -> NdarrayOrTensor:
    """
    Note that `torch.where` may convert y.dtype to x.dtype.
    """
    result: NdarrayOrTensor
    if isinstance(condition, np.ndarray):
        if x is not None:
            result = np.where(condition, x, y)
        else:
            result = np.where(condition)
    else:
        if x is not None:
            x = paddle.Tensor(x, device=condition.device)
            y = paddle.Tensor(y, device=condition.device, dtype=x.dtype)
            result = paddle.where(condition, x, y)
        else:
            result = paddle.where(condition)  # type: ignore
    return result


def nonzero(x: NdarrayOrTensor):
    """`np.nonzero` with equivalent implementation for torch.

    Args:
        x: array/tensor

    Returns:
        Index unravelled for given shape
    """
    if isinstance(x, np.ndarray):
        return np.nonzero(x)[0]
    return paddle.nonzero(x).flatten()


def floor_divide(a: NdarrayOrTensor, b) -> NdarrayOrTensor:
    """`np.floor_divide` with equivalent implementation for torch.

    As of pt1.8, use `torch.div(..., rounding_mode="floor")`, and
    before that, use `torch.floor_divide`.

    Args:
        a: first array/tensor
        b: scalar to divide by

    Returns:
        Element-wise floor division between two arrays/tensors.
    """
    if isinstance(a, paddle.Tensor):
        if is_module_ver_at_least(paddle, (1, 8, 0)):
            ipt = paddle.divide(a.cast('float32'), paddle.to_tensor(b, dtype='float32'))
            abs_ipt = paddle.abs(ipt)
            abs_ipt = paddle.floor(abs_ipt)
            return abs_ipt
        return paddle.floor_divide(a, b)
    return np.floor_divide(a, b)


def unravel_index(idx, shape):
    """`np.unravel_index` with equivalent implementation for torch.

    Args:
        idx: index to unravel
        shape: shape of array/tensor

    Returns:
        Index unravelled for given shape
    """
    if isinstance(idx, paddle.Tensor):
        coord = []
        for dim in reversed(shape):
            coord.append(idx % dim)
            idx = floor_divide(idx, dim)
        return paddle.stack(coord[::-1])
    return np.asarray(np.unravel_index(idx, shape))


def ravel(x: NdarrayOrTensor):
    """`np.ravel` with equivalent implementation for torch.

    Args:
        x: array/tensor to ravel

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, paddle.Tensor):
        if hasattr(paddle, "ravel"):  # `ravel` is new in torch 1.8.0
            return x.ravel()
        return x.flatten()
    return np.ravel(x)


def any_np_pt(x: NdarrayOrTensor, axis: Union[int, Sequence[int]]):
    """`np.any` with equivalent implementation for torch.

    For pytorch, convert to boolean for compatibility with older versions.

    Args:
        x: input array/tensor
        axis: axis to perform `any` over

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, np.ndarray):
        return np.any(x, axis)

    # pytorch can't handle multiple dimensions to `any` so loop across them
    axis = [axis] if not isinstance(axis, Sequence) else axis
    for ax in axis:
        try:
            x = paddle.any(x, ax)
        except RuntimeError:
            # older versions of pytorch require the input to be cast to boolean
            x = paddle.any(x.cast('bool'), ax)
    return x


def maximum(a: NdarrayOrTensor, b: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.maximum` with equivalent implementation for torch.

    `torch.maximum` only available from pt>1.6, else use `torch.stack` and `torch.max`.

    Args:
        a: first array/tensor
        b: second array/tensor

    Returns:
        Element-wise maximum between two arrays/tensors.
    """
    if isinstance(a, paddle.Tensor) and isinstance(b, paddle.Tensor):
        # is torch and has torch.maximum (pt>1.6)
        if hasattr(paddle, "maximum"):  # `maximum` is new in torch 1.7.0
            return paddle.maximum(a.cast('float32'), b.cast('float32'))
        return paddle.stack((a, b)).max(dim=0)[0]
    return np.maximum(a, b)
