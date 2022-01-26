# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
import paddle.nn.functional as F


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def pd_index_slice(x, index, axes):
    y = paddle.to_tensor([0])
    for i, k in enumerate(index):
        xs = x.slice(axes=[axes], starts=[k], ends=[k + 1])
        if i == 0:
            y = xs
        else:
            y = paddle.concat([y, xs], axes)
    return y


def rot90(inp, k, dims):
    le = len(inp.shape)
    new_dims = list(range(le))
    new_dims[dims[0]] = dims[1]
    new_dims[dims[1]] = dims[0]
    flip_dim = min(dims)
    for _ in range(k):
        inp = paddle.transpose(inp, new_dims)
        inp = paddle.flip(inp, [flip_dim])
    return inp


def one_hot(labels: paddle.Tensor, num_classes: int, dim: int = 1) -> paddle.Tensor:
    # scatter_()
    out = F.one_hot(labels.squeeze(dim).cast('int64'), num_classes)
    if dim == 1:
        out = out.transpose([0, 4, 1, 2, 3])
    else:
        out = out.transpose([3, 0, 1, 2])
    return out
