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

import os
from monai.utils.module import OptionalImportError, optional_import

try:
    _, HAS_EXT = optional_import("monai._C")
    USE_COMPILED = HAS_EXT and os.getenv("BUILD_MONAI", "0") == "1"
except (OptionalImportError, ImportError, AttributeError):
    HAS_EXT = USE_COMPILED = False

psutil, has_psutil = optional_import("psutil")
psutil_version = psutil.__version__ if has_psutil else "NOT INSTALLED or UNKNOWN VERSION."

__all__ = [
    "USE_COMPILED",
    "IgniteInfo",
]


class IgniteInfo:
    """
    Config information of the PyTorch ignite package.

    """

    OPT_IMPORT_VERSION = "0.4.4"

