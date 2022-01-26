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
import sys
import warnings
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

from paddle.io import Dataset
from paddle.io import Subset

from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform
from monai.utils import  min_version, optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")


class Dataset(Dataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class CacheDataset(Dataset):
    """
    Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.

    By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
    If the requested data is not in the cache, all transforms will run normally
    (see also :py:class:`monai.data.dataset.Dataset`).

    Users can set the cache rate or number of items to cache.
    It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

    The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
    interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
    `Randomizable` `Transform` within a `Compose` instance.
    So to improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomized ones when composing the chain of transforms.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadImaged(),
            AddChanneld(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadImaged`, `AddChanneld`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.

    During training call `set_data()` to update input data and recompute cache content, note that it requires
    `persistent_workers=False` in the PyTorch DataLoader.

    Note:
        `CacheDataset` executes non-random transforms and prepares cache content in the main process before
        the first epoch, then all the subprocesses of DataLoader will read the same cache content in the main process
        during training. it may take a long time to prepare cache content according to the size of expected cache data.
        So to debug or verify the program before real training, users can set `cache_rate=0.0` or `cache_num=0` to
        temporarily skip caching.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: Optional[int] = None,
        progress: bool = True,
        copy_cache: bool = True,
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker processes to use.
                If num_workers is None then the number returned by os.cpu_count() is used.
            progress: whether to display a progress bar.
            copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
                default to `True`. if the random transforms don't modify the cached content
                (for example, randomly crop from the cached image and deepcopy the crop region)
                or if every cache item is only used once in a `multi-processing` environment,
                may set `copy=False` for better performance.
        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.progress = progress
        self.copy_cache = copy_cache
        self.cache_num = min(int(cache_num), int(len(data) * cache_rate), len(data))
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self._cache: List = self._fill_cache()

    def set_data(self, data: Sequence):
        """
        Set the input data and run deterministic transforms to generate cache content.

        Note: should call this func after an entire epoch and must set `persistent_workers=False`
        in PyTorch DataLoader, because it needs to create new worker processes based on new
        generated cache content.

        """
        self.data = data
        self._cache = self._fill_cache()

    def _fill_cache(self) -> List:
        if self.cache_num <= 0:
            return []
        if self.progress and not has_tqdm:
            warnings.warn("tqdm is not installed, will not show the caching progress bar.")
        with ThreadPool(self.num_workers) as p:
            if self.progress and has_tqdm:
                return list(
                    tqdm(
                        p.imap(self._load_cache_item, range(self.cache_num)),
                        total=self.cache_num,
                        desc="Loading dataset",
                    )
                )
            return list(p.imap(self._load_cache_item, range(self.cache_num)))

    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]
        for _transform in self.transform.transforms:  # type:ignore
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item = apply_transform(_xform, item)
        return item

    def _transform(self, index: int):
        if index % len(self) >= self.cache_num:  # support negative index
            # no cache for this index, execute all the transforms directly
            return super()._transform(index)
        # load data from cache and execute from the first random transform
        start_run = False
        if self._cache is None:
            self._cache = self._fill_cache()
        data = self._cache[index]
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.transform.transforms:
            if start_run or isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                # only need to deep copy data on first non-deterministic transform
                if not start_run:
                    start_run = True
                    if self.copy_cache:
                        data = deepcopy(data)
                data = apply_transform(_transform, data)
        return data


