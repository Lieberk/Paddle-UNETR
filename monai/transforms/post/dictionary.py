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
A collection of dictionary-based wrappers around the "vanilla" transforms for model output tensors
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import warnings
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union


from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.post.array import (
    AsDiscrete,
)
from monai.transforms.transform import MapTransform
from monai.utils import ensure_tuple_rep

__all__ = [
    "AsDiscreteD",
    "AsDiscreteDict",
    "AsDiscreted",
]


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    backend = AsDiscrete.backend

    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[Optional[int]], Optional[int]] = None,
        threshold: Union[Sequence[Optional[float]], Optional[float]] = None,
        rounding: Union[Sequence[Optional[str]], Optional[str]] = None,
        allow_missing_keys: bool = False,
        n_classes: Optional[Union[Sequence[int], int]] = None,  # deprecated
        num_classes: Optional[Union[Sequence[int], int]] = None,  # deprecated
        logit_thresh: Union[Sequence[float], float] = 0.5,  # deprecated
        threshold_values: Union[Sequence[bool], bool] = False,  # deprecated
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold value.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"]. it also can be a sequence of str or None,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        .. deprecated:: 0.6.0
            ``n_classes`` is deprecated, use ``to_onehot`` instead.

        .. deprecated:: 0.7.0
            ``num_classes`` is deprecated, use ``to_onehot`` instead.
            ``logit_thresh`` is deprecated, use ``threshold`` instead.
            ``threshold_values`` is deprecated, use ``threshold`` instead.

        """
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        to_onehot_ = ensure_tuple_rep(to_onehot, len(self.keys))
        num_classes = ensure_tuple_rep(num_classes, len(self.keys))
        self.to_onehot = []
        for flag, val in zip(to_onehot_, num_classes):
            if isinstance(flag, bool):
                warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
                self.to_onehot.append(val if flag else None)
            else:
                self.to_onehot.append(flag)

        threshold_ = ensure_tuple_rep(threshold, len(self.keys))
        logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.threshold = []
        for flag, val in zip(threshold_, logit_thresh):
            if isinstance(flag, bool):
                warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
                self.threshold.append(val if flag else None)
            else:
                self.threshold.append(flag)

        self.rounding = ensure_tuple_rep(rounding, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, argmax, to_onehot, threshold, rounding in self.key_iterator(
            d, self.argmax, self.to_onehot, self.threshold, self.rounding
        ):
            d[key] = self.converter(d[key], argmax, to_onehot, threshold, rounding)
        return d


AsDiscreteD = AsDiscreteDict = AsDiscreted

