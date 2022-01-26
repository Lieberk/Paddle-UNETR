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
A collection of "vanilla" transforms for IO functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import inspect
import logging
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from monai.config import DtypeLike, PathLike
from monai.data.image_reader import ImageReader, ITKReader, NibabelReader, NumpyReader, PILReader
from monai.transforms.transform import Transform
from monai.utils import ImageMetaKey as Key
from monai.utils import OptionalImportError, ensure_tuple, look_up_option, optional_import

nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")

__all__ = ["LoadImage", "SUPPORTED_READERS"]

SUPPORTED_READERS = {
    "itkreader": ITKReader,
    "numpyreader": NumpyReader,
    "pilreader": PILReader,
    "nibabelreader": NibabelReader,
}


def switch_endianness(data, new="<"):
    """
    Convert the input `data` endianness to `new`.

    Args:
        data: input to be converted.
        new: the target endianness, currently support "<" or ">".
    """
    if isinstance(data, np.ndarray):
        # default to system endian
        sys_native = "<" if (sys.byteorder == "little") else ">"
        current_ = sys_native if data.dtype.byteorder not in ("<", ">") else data.dtype.byteorder
        if new not in ("<", ">"):
            raise NotImplementedError(f"Not implemented option new={new}.")
        if current_ != new:
            data = data.byteswap().newbyteorder(new)
    elif isinstance(data, tuple):
        data = tuple(switch_endianness(x, new) for x in data)
    elif isinstance(data, list):
        data = [switch_endianness(x, new) for x in data]
    elif isinstance(data, dict):
        data = {k: switch_endianness(v, new) for k, v in data.items()}
    elif not isinstance(data, (bool, str, float, int, type(None))):
        raise RuntimeError(f"Unknown type: {type(data).__name__}")
    return data


class LoadImage(Transform):
    """
    Load image file or files from provided path based on reader.
    If reader is not specified, this class automatically chooses readers
    based on the supported suffixes and in the following order:

        - User-specified reader at runtime when calling this loader.
        - User-specified reader in the constructor of `LoadImage`.
        - Readers from the last to the first in the registered list.
        - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
          (npz, npy -> NumpyReader), (others -> ITKReader).

    See also:

        - tutorial: https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb

    """

    def __init__(self, reader=None, image_only: bool = False, dtype: DtypeLike = np.float32, *args, **kwargs) -> None:
        """
        Args:
            reader: reader to load image file and meta data

                - if `reader` is None, a default set of `SUPPORTED_READERS` will be used.
                - if `reader` is a string, the corresponding item in `SUPPORTED_READERS` will be used,
                  and a reader instance will be constructed with the `*args` and `**kwargs` parameters.
                  the supported reader names are: "nibabelreader", "pilreader", "itkreader", "numpyreader".
                - if `reader` is a reader class/instance, it will be registered to this loader accordingly.

            image_only: if True return only the image volume, otherwise return image data array and header dict.
            dtype: if not None convert the loaded image to this data type.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.

        Note:

            - The transform returns an image data array if `image_only` is True,
              or a tuple of two elements containing the data array, and the meta data in a dictionary format otherwise.
            - If `reader` is specified, the loader will attempt to use the specified readers and the default supported
              readers. This might introduce overheads when handling the exceptions of trying the incompatible loaders.
              In this case, it is therefore recommended to set the most appropriate reader as
              the last item of the `reader` parameter.

        """

        self.auto_select = reader is None
        self.image_only = image_only
        self.dtype = dtype

        self.readers: List[ImageReader] = []
        for r in SUPPORTED_READERS:  # set predefined readers as default
            try:
                self.register(SUPPORTED_READERS[r](*args, **kwargs))
            except OptionalImportError:
                logging.getLogger(self.__class__.__name__).debug(
                    f"required package for reader {r} is not installed, or the version doesn't match requirement."
                )
            except TypeError:  # the reader doesn't have the corresponding args/kwargs
                logging.getLogger(self.__class__.__name__).debug(
                    f"{r} is not supported with the given parameters {args} {kwargs}."
                )
                self.register(SUPPORTED_READERS[r]())
        if reader is None:
            return  # no user-specified reader, no need to register

        for _r in ensure_tuple(reader):
            if isinstance(_r, str):
                the_reader = look_up_option(_r.lower(), SUPPORTED_READERS)
                try:
                    self.register(the_reader(*args, **kwargs))
                except OptionalImportError:
                    warnings.warn(
                        f"required package for reader {r} is not installed, or the version doesn't match requirement."
                    )
                except TypeError:  # the reader doesn't have the corresponding args/kwargs
                    warnings.warn(f"{r} is not supported with the given parameters {args} {kwargs}.")
                    self.register(the_reader())
            elif inspect.isclass(_r):
                self.register(_r(*args, **kwargs))
            else:
                self.register(_r)  # reader instance, ignoring the constructor args/kwargs
        return

    def register(self, reader: ImageReader):
        """
        Register image reader to load image file and meta data.

        Args:
            reader: reader instance to be registered with this loader.

        """
        if not isinstance(reader, ImageReader):
            warnings.warn(f"Preferably the reader should inherit ImageReader, but got {type(reader)}.")
        self.readers.append(reader)

    def __call__(self, filename: Union[Sequence[PathLike], PathLike], reader: Optional[ImageReader] = None):
        """
        Load image file and meta data from the given filename(s).
        If `reader` is not specified, this class automatically chooses readers based on the
        reversed order of registered readers `self.readers`.

        Args:
            filename: path file or file-like object or a list of files.
                will save the filename to meta_data with key `filename_or_obj`.
                if provided a list of files, use the filename of first file to save,
                and will stack them together as multi-channels data.
                if provided directory path instead of file path, will treat it as
                DICOM images series and read.
            reader: runtime reader to load image file and meta data.

        """
        filename = tuple(f"{Path(s).expanduser()}" for s in ensure_tuple(filename))  # allow Path objects
        img = None
        if reader is not None:
            img = reader.read(filename)  # runtime specified reader
        else:
            for reader in self.readers[::-1]:
                if self.auto_select:  # rely on the filename extension to choose the reader
                    if reader.verify_suffix(filename):
                        img = reader.read(filename)
                        break
                else:  # try the user designated readers
                    try:
                        img = reader.read(filename)
                    except Exception as e:
                        logging.getLogger(self.__class__.__name__).debug(
                            f"{reader.__class__.__name__}: unable to load {filename}.\n" f"Error: {e}"
                        )
                    else:
                        break

        if img is None or reader is None:
            if isinstance(filename, tuple) and len(filename) == 1:
                filename = filename[0]
            raise RuntimeError(
                f"cannot find a suitable reader for file: {filename}.\n"
                "    Please install the reader libraries, see also the installation instructions:\n"
                "    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies.\n"
                f"   The current registered: {self.readers}.\n"
            )

        img_array, meta_data = reader.get_data(img)
        img_array = img_array.astype(self.dtype, copy=False)

        if self.image_only:
            return img_array
        meta_data[Key.FILENAME_OR_OBJ] = f"{ensure_tuple(filename)[0]}"  # Path obj should be strings for data loader
        # make sure all elements in metadata are little endian
        meta_data = switch_endianness(meta_data, "<")

        return img_array, meta_data
