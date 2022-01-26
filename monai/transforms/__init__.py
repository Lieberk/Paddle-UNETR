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


from .compose import Compose, OneOf
from .inverse import InvertibleTransform, TraceableTransform
from .io.array import SUPPORTED_READERS, LoadImage
from .io.dictionary import LoadImaged, LoadImageD, LoadImageDict
from .transform import MapTransform, Randomizable, RandomizableTransform, ThreadUnsafe, Transform, apply_transform

from .compose import Compose, OneOf
from .croppad.array import (
    BorderPad,
    CropForeground,
    Pad,
    RandCropByPosNegLabel,
    SpatialCrop,
    SpatialPad,
)
from .croppad.dictionary import (
    CropForegroundd,
    CropForegroundD,
    CropForegroundDict,
    PadModeSequence,
    RandCropByPosNegLabeld,
    RandCropByPosNegLabelD,
    RandCropByPosNegLabelDict,
)
from .intensity.array import (
    RandScaleIntensity,
    RandShiftIntensity,
    ScaleIntensity,
    ScaleIntensityRange,
    ShiftIntensity,
)
from .intensity.dictionary import (
    RandScaleIntensityd,
    RandScaleIntensityD,
    RandScaleIntensityDict,
    RandShiftIntensityd,
    RandShiftIntensityD,
    RandShiftIntensityDict,
    ScaleIntensityRanged,
    ScaleIntensityRangeD,
    ScaleIntensityRangeDict,
)
from .inverse import InvertibleTransform, TraceableTransform
from .io.array import SUPPORTED_READERS, LoadImage
from .io.dictionary import LoadImaged, LoadImageD, LoadImageDict
from .post.array import (
    AsDiscrete,
)
from .post.dictionary import (
    AsDiscreteD,
    AsDiscreted,
    AsDiscreteDict,
)
from .spatial.array import (
    Flip,
    Orientation,
    RandFlip,
    Rotate90,
    Spacing,
)
from .spatial.dictionary import (
    Orientationd,
    OrientationD,
    OrientationDict,
    RandFlipd,
    RandFlipD,
    RandFlipDict,
    RandRotate90d,
    RandRotate90D,
    RandRotate90Dict,
    Spacingd,
    SpacingD,
    SpacingDict,
)
from .utility.array import (
    AddChannel,
    ToNumpy,
    ToTensor,
)
from .utility.dictionary import (
    AddChanneld,
    AddChannelD,
    AddChannelDict,
    ToTensord,
)
from .utils import (
    compute_divisible_spatial_size,
    convert_pad_mode,
    generate_pos_neg_label_crop_centers,
    is_positive,
    map_binary_to_indices,
    rescale_array,
)
from .utils_pytorch_numpy_unification import (
    any_np_pt,
    clip,
    floor_divide,
    maximum,
    nonzero,
    ravel,
    unravel_index,
    where,
)
