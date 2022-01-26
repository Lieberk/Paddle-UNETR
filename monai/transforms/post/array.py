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
A collection of "vanilla" transforms for the model output tensors
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import warnings
from typing import Optional

import numpy as np
import paddle

from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils import TransformBackends, convert_data_type, look_up_option
from monai.utils.type_conversion import convert_to_dst_type

import utils.utils as utils

__all__ = [
    "AsDiscrete",
]


class AsDiscrete(Transform):
    """
    Execute after model forward to transform model output to discrete values.
    It can complete below operations:

        -  execute `argmax` for input logits values.
        -  threshold input value to 0.0 or 1.0.
        -  convert input value to One-Hot format.
        -  round the value to the closest integer.

    Args:
        argmax: whether to execute argmax function on input data before transform.
            Defaults to ``False``.
        to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
            Defaults to ``None``.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
            Defaults to ``None``.
        rounding: if not None, round the data according to the specified option,
            available options: ["torchrounding"].

    Example:

        >>> transform = AsDiscrete(argmax=True)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[1.0, 1.0]]]

        >>> transform = AsDiscrete(threshold=0.6)
        >>> print(transform(np.array([[[0.0, 0.5], [0.8, 3.0]]])))
        # [[[0.0, 0.0], [1.0, 1.0]]]

        >>> transform = AsDiscrete(argmax=True, to_onehot=2, threshold=0.5)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[0.0, 0.0]], [[1.0, 1.0]]]

    .. deprecated:: 0.6.0
        ``n_classes`` is deprecated, use ``to_onehot`` instead.

    .. deprecated:: 0.7.0
        ``num_classes`` is deprecated, use ``to_onehot`` instead.
        ``logit_thresh`` is deprecated, use ``threshold`` instead.
        ``threshold_values`` is deprecated, use ``threshold`` instead.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        argmax: bool = False,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,  # deprecated
        num_classes: Optional[int] = None,  # deprecated
        logit_thresh: float = 0.5,  # deprecated
        threshold_values: Optional[bool] = False,  # deprecated
    ) -> None:
        self.argmax = argmax
        if isinstance(to_onehot, bool):  # for backward compatibility
            warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
            to_onehot = num_classes if to_onehot else None
        self.to_onehot = to_onehot

        if isinstance(threshold, bool):  # for backward compatibility
            warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
            threshold = logit_thresh if threshold else None
        self.threshold = threshold

        self.rounding = rounding

    def __call__(
        self,
        img: NdarrayOrTensor,
        argmax: Optional[bool] = None,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,  # deprecated
        num_classes: Optional[int] = None,  # deprecated
        logit_thresh: Optional[float] = None,  # deprecated
        threshold_values: Optional[bool] = None,  # deprecated
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the input tensor data to convert, if no channel dimension when converting to `One-Hot`,
                will automatically add it.
            argmax: whether to execute argmax function on input data before transform.
                Defaults to ``self.argmax``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                Defaults to ``self.to_onehot``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                Defaults to ``self.threshold``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"].

        .. deprecated:: 0.6.0
            ``n_classes`` is deprecated, use ``to_onehot`` instead.

        .. deprecated:: 0.7.0
            ``num_classes`` is deprecated, use ``to_onehot`` instead.
            ``logit_thresh`` is deprecated, use ``threshold`` instead.
            ``threshold_values`` is deprecated, use ``threshold`` instead.

        """
        if isinstance(to_onehot, bool):
            warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
            to_onehot = num_classes if to_onehot else None
        if isinstance(threshold, bool):
            warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
            threshold = logit_thresh if threshold else None

        img_t: paddle.Tensor
        img_t, *_ = convert_data_type(img, paddle.Tensor)  # type: ignore
        if argmax or self.argmax:
            img_t = paddle.argmax(img_t, axis=0, keepdim=True)

        to_onehot = self.to_onehot if to_onehot is None else to_onehot
        if to_onehot is not None:
            if not isinstance(to_onehot, int):
                raise AssertionError("the number of classes for One-Hot must be an integer.")
            img_t = utils.one_hot(img_t, num_classes=to_onehot, dim=0)

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img_t = img_t >= threshold

        rounding = self.rounding if rounding is None else rounding
        if rounding is not None:
            look_up_option(rounding, ["torchrounding"])
            img_t = paddle.round(img_t)

        img, *_ = convert_to_dst_type(img_t, img, dtype=paddle.float32)
        return img