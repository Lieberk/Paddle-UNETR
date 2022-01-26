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

from typing import Union

import numpy as np
import paddle
from monai.utils import MetricReduction, look_up_option, optional_import

binary_erosion, _ = optional_import("scipy.ndimage.morphology", name="binary_erosion")
distance_transform_edt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_edt")
distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")

__all__ = ["ignore_background", "do_metric_reduction"]


def ignore_background(y_pred: Union[np.ndarray, paddle.Tensor], y: Union[np.ndarray, paddle.Tensor]):
    """
    This function is used to remove background (the first channel) for `y_pred` and `y`.

    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        y: ground truth, the first dim is batch.

    """

    y = y[:, 1:] if y.shape[1] > 1 else y
    y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred
    return y_pred, y


def do_metric_reduction(f: paddle.Tensor, reduction: Union[MetricReduction, str] = MetricReduction.MEAN):
    """
    This function is to do the metric reduction for calculated `not-nan` metrics of each sample's each class.
    The function also returns `not_nans`, which counts the number of not nans for the metric.

    Args:
        f: a tensor that contains the calculated metric scores per batch and
            per class. The first two dims should be batch and class.
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``.
            if "none", return the input f tensor and not_nans.
        Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.

    Raises:
        ValueError: When ``reduction`` is not one of
            ["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].
    """

    # some elements might be Nan (if ground truth y was missing (zeros))
    # we need to account for it
    nans = paddle.isnan(f)
    not_nans = (~nans).float()

    t_zero = paddle.zeros(1, dtype=f.dtype)
    reduction = look_up_option(reduction, MetricReduction)
    if reduction == MetricReduction.NONE:
        return f, not_nans

    f[nans] = 0
    if reduction == MetricReduction.MEAN:
        # 2 steps, first, mean by channel (accounting for nans), then by batch
        not_nans = not_nans.sum(dim=1)
        f = paddle.where(not_nans > 0, f.sum(axis=1) / not_nans, t_zero)  # channel average

        not_nans = (not_nans > 0).float().sum(dim=0)
        f = paddle.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average

    elif reduction == MetricReduction.SUM:
        not_nans = not_nans.sum(dim=[0, 1])
        f = paddle.sum(f, axis=[0, 1])  # sum over the batch and channel dims
    elif reduction == MetricReduction.MEAN_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = paddle.where(not_nans > 0, f.sum(axis=0) / not_nans, t_zero)  # batch average
    elif reduction == MetricReduction.SUM_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = f.sum(axis=0)  # the batch sum
    elif reduction == MetricReduction.MEAN_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = paddle.where(not_nans > 0, f.sum(axis=1) / not_nans, t_zero)  # channel average
    elif reduction == MetricReduction.SUM_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = f.sum(axis=1)  # the channel sum
    elif reduction != MetricReduction.NONE:
        raise ValueError(
            f"Unsupported reduction: {reduction}, available options are "
            '["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].'
        )
    return f, not_nans
