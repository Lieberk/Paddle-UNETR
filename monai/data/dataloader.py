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

import paddle
from paddle.io import DataLoader as _PaddleDataLoader
from paddle.io import Dataset

from monai.data.utils import list_data_collate, set_rnd

__all__ = ["DataLoader"]


class DataLoader(_PaddleDataLoader):

    def __init__(self, dataset: Dataset, num_workers: int = 0, **kwargs) -> None:
        if num_workers == 0:
            # this is to make the behavior consistent when num_workers == 0
            # paddle.int64 doesn't work well on some versions of windows
            _seed = paddle.rand(shape=[1]).multiply(paddle.to_tensor(1e+9))
            set_rnd(dataset, int(_seed))
        if "collate_fn" not in kwargs:
            kwargs.update({"collate_fn": list_data_collate})

        super().__init__(dataset=dataset, num_workers=num_workers, **kwargs)
