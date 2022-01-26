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

import collections.abc
import inspect
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import paddle

from monai.utils.module import version_leq

__all__ = [
    "first",
    "issequenceiterable",
    "ensure_tuple",
    "ensure_tuple_size",
    "ensure_tuple_rep",
    "fall_back_tuple",
    "get_seed",
    "MAX_SEED",
    "ImageMetaKey",
    "is_module_ver_at_least",
    "has_option",
]

_seed = None
MAX_SEED = np.iinfo(np.uint32).max + 1  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default


def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if isinstance(obj, paddle.Tensor):
        return int(obj.dim()) > 0  # a 0-d tensor is not iterable
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes))


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        return (vals,)

    return tuple(vals)


def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    new_tup = ensure_tuple(tup) + (pad_val,) * dim
    return new_tup[:dim]


def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, paddle.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def fall_back_tuple(
    user_provided: Any, default: Union[Sequence, np.ndarray], func: Callable = lambda x: x and x > 0
) -> Tuple[Any, ...]:
    """
    Refine `user_provided` according to the `default`, and returns as a validated tuple.

    The validation is done for each element in `user_provided` using `func`.
    If `func(user_provided[idx])` returns False, the corresponding `default[idx]` will be used
    as the fallback.

    Typically used when `user_provided` is a tuple of window size provided by the user,
    `default` is defined by data, this function returns an updated `user_provided` with its non-positive
    components replaced by the corresponding components from `default`.

    Args:
        user_provided: item to be validated.
        default: a sequence used to provided the fallbacks.
        func: a Callable to validate every components of `user_provided`.

    Examples::

        >>> fall_back_tuple((1, 2), (32, 32))
        (1, 2)
        >>> fall_back_tuple(None, (32, 32))
        (32, 32)
        >>> fall_back_tuple((-1, 10), (32, 32))
        (32, 10)
        >>> fall_back_tuple((-1, None), (32, 32))
        (32, 32)
        >>> fall_back_tuple((1, None), (32, 32))
        (1, 32)
        >>> fall_back_tuple(0, (32, 32))
        (32, 32)
        >>> fall_back_tuple(range(3), (32, 64, 48))
        (32, 1, 2)
        >>> fall_back_tuple([0], (32, 32))
        ValueError: Sequence must have length 2, got length 1.

    """
    ndim = len(default)
    user = ensure_tuple_rep(user_provided, ndim)
    return tuple(  # use the default values if user provided is not valid
        user_c if func(user_c) else default_c for default_c, user_c in zip(default, user)
    )


def get_seed() -> Optional[int]:
    return _seed


class ImageMetaKey:
    """
    Common key names in the meta data header of images
    """

    FILENAME_OR_OBJ = "filename_or_obj"
    PATCH_INDEX = "patch_index"


def has_option(obj, keywords: Union[str, Sequence[str]]) -> bool:
    """
    Return a boolean indicating whether the given callable `obj` has the `keywords` in its signature.
    """
    if not callable(obj):
        return False
    sig = inspect.signature(obj)
    return all(key in sig.parameters for key in ensure_tuple(keywords))


def is_module_ver_at_least(module, version):
    """Determine if a module's version is at least equal to the given value.

    Args:
        module: imported module's name, e.g., `np` or `torch`.
        version: required version, given as a tuple, e.g., `(1, 8, 0)`.
    Returns:
        `True` if module is the given version or newer.
    """
    test_ver = ".".join(map(str, version))
    return module.__version__ != test_ver and version_leq(test_ver, module.__version__)
