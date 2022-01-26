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
A collection of dictionary-based wrappers around the "vanilla" transforms for IO functions
defined in :py:class:`monai.transforms.io.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Optional, Union

import numpy as np

from monai.config import DtypeLike, KeysCollection
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage
from monai.transforms.transform import MapTransform
from monai.utils import ensure_tuple, ensure_tuple_rep

__all__ = ["LoadImaged", "LoadImageD", "LoadImageDict"]


class LoadImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadImage`,
    It can load both image data and metadata. When loading a list of files in one key,
    the arrays will be stacked and a new dimension will be added as the first dimension
    In this case, the meta data of the first image will be used to represent the stacked result.
    The affine transform of all the stacked images should be same.
    The output metadata field will be created as ``meta_keys`` or ``key_{meta_key_postfix}``.

    If reader is not specified, this class automatically chooses readers
    based on the supported suffixes and in the following order:

        - User-specified reader at runtime when calling this loader.
        - User-specified reader in the constructor of `LoadImage`.
        - Readers from the last to the first in the registered list.
        - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
          (npz, npy -> NumpyReader), (others -> ITKReader).

    Note:

        - If `reader` is specified, the loader will attempt to use the specified readers and the default supported
          readers. This might introduce overheads when handling the exceptions of trying the incompatible loaders.
          In this case, it is therefore recommended to set the most appropriate reader as
          the last item of the `reader` parameter.

    See also:

        - tutorial: https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb

    """

    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
        image_only: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            reader: register reader to load image file and meta data, if None, still can register readers
                at runtime or use the default readers. If a string of reader name provided, will construct
                a reader object with the `*args` and `**kwargs` parameters, supported reader name: "NibabelReader",
                "PILReader", "ITKReader", "NumpyReader".
            dtype: if not None convert the loaded image data to this data type.
            meta_keys: explicitly indicate the key to store the corresponding meta data dictionary.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.
            image_only: if True return dictionary containing just only the image volumes, otherwise return
                dictionary containing image data array and header dict per input key.
            allow_missing_keys: don't raise exception if key is missing.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.
        """
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                if not isinstance(data, np.ndarray):
                    raise ValueError("loader must return a numpy array (because image_only=True was used).")
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Meta data with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        return d


LoadImageD = LoadImageDict = LoadImaged
