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

import math

import pickle
import warnings
from copy import deepcopy
from itertools import zip_longest
from pathlib import PurePath
from typing import Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import paddle
from paddle.fluid.dataloader.dataloader_iter import default_collate_fn


from monai.config.type_definitions import PathLike
from monai.utils import (
    MAX_SEED,
    BlendMode,
    ensure_tuple,
    ensure_tuple_size,
    fall_back_tuple,
    first,
    look_up_option,
    optional_import,
)

pd, _ = optional_import("pandas")
DataFrame, _ = optional_import("pandas", name="DataFrame")
nib, _ = optional_import("nibabel")


__all__ = [
    "dense_patch_slices",
    "get_valid_patch_size",
    "list_data_collate",
    "set_rnd",
    "correct_nifti_header_if_necessary",
    "rectify_header_sform_qform",
    "zoom_affine",
    "compute_shape_offset",
    "to_affine_nd",
    "compute_importance_map",
    "is_supported_format",
    "decollate_batch",
    "SUPPORTED_PICKLE_MOD",
]

# module to be used by `torch.save`
SUPPORTED_PICKLE_MOD = {"pickle": pickle}


def dense_patch_slices(
    image_size: Sequence[int], patch_size: Sequence[int], scan_interval: Sequence[int]
) -> List[Tuple[slice, ...]]:

    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    return [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]


def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


def list_data_collate(batch: Sequence):

    elem = batch[0]
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    key = None
    try:
        return default_collate_fn(data)
    except RuntimeError as re:
        re_str = str(re)
        if "equal size" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                "\n\nMONAI hint: if your transforms intentionally create images of different shapes, creating your "
                + "`DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem (check its "
                + "documentation)."
            )
        raise RuntimeError(re_str) from re
    except TypeError as re:
        re_str = str(re)
        if "numpy" in re_str and "Tensor" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                "\n\nMONAI hint: if your transforms intentionally create mixtures of torch Tensor and numpy ndarray, "
                + "creating your `DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem "
                + "(check its documentation)."
            )
        raise TypeError(re_str) from re


def _non_zipping_check(batch_data, detach, pad, fill_value):

    if isinstance(batch_data, Mapping):
        _deco = {key: decollate_batch(batch_data[key], detach, pad=pad, fill_value=fill_value) for key in batch_data}
    elif isinstance(batch_data, Iterable):
        _deco = [decollate_batch(b, detach, pad=pad, fill_value=fill_value) for b in batch_data]
    else:
        raise NotImplementedError(f"Unable to de-collate: {batch_data}, type: {type(batch_data)}.")
    batch_size, non_iterable = 0, []
    for k, v in _deco.items() if isinstance(_deco, Mapping) else enumerate(_deco):
        if not isinstance(v, Iterable) or isinstance(v, (str, bytes)) or (isinstance(v, paddle.Tensor) and v.ndim == 0):
            # Not running the usual list decollate here:
            # don't decollate ['test', 'test'] into [['t', 't'], ['e', 'e'], ['s', 's'], ['t', 't']]
            # torch.tensor(0) is iterable but iter(torch.tensor(0)) raises TypeError: iteration over a 0-d tensor
            non_iterable.append(k)
        elif hasattr(v, "__len__"):
            batch_size = max(batch_size, len(v))
    return batch_size, non_iterable, _deco


def decollate_batch(batch, detach: bool = True, pad=True, fill_value=None):

    if batch is None:
        return batch
    if isinstance(batch, (float, int, str, bytes)):
        return batch
    if isinstance(batch, paddle.Tensor):
        if detach:
            batch = batch.detach()
        if batch.ndim == 0:
            return batch.item() if detach else batch
        out_list = paddle.unbind(batch, axis=0)
        if out_list[0].ndim == 0 and detach:
            return [t.item() for t in out_list]
        return list(out_list)

    b, non_iterable, deco = _non_zipping_check(batch, detach, pad, fill_value)
    if b <= 0:  # all non-iterable, single item "batch"? {"image": 1, "label": 1}
        return deco
    if pad:  # duplicate non-iterable items to the longest batch
        for k in non_iterable:
            deco[k] = [deepcopy(deco[k]) for _ in range(b)]
    if isinstance(deco, Mapping):
        _gen = zip_longest(*deco.values(), fillvalue=fill_value) if pad else zip(*deco.values())
        return [dict(zip(deco, item)) for item in _gen]
    if isinstance(deco, Iterable):
        _gen = zip_longest(*deco, fillvalue=fill_value) if pad else zip(*deco)
        return [list(item) for item in _gen]
    raise NotImplementedError(f"Unable to de-collate: {batch}, type: {type(batch)}.")


def set_rnd(obj, seed: int) -> int:

    if not hasattr(obj, "__dict__"):
        return seed  # no attribute
    if hasattr(obj, "set_random_state"):
        obj.set_random_state(seed=seed % MAX_SEED)
        return seed + 1  # a different seed for the next component
    for key in obj.__dict__:
        if key.startswith("__"):  # skip the private methods
            continue
        seed = set_rnd(obj.__dict__[key], seed=seed)
    return seed


def correct_nifti_header_if_necessary(img_nii):

    if img_nii.header.get("dim") is None:
        return img_nii  # not nifti?
    dim = img_nii.header["dim"][0]
    if dim >= 5:
        return img_nii  # do nothing for high-dimensional array
    # check that affine matches zooms
    pixdim = np.asarray(img_nii.header.get_zooms())[:dim]
    norm_affine = np.sqrt(np.sum(np.square(img_nii.affine[:dim, :dim]), 0))
    if np.allclose(pixdim, norm_affine):
        return img_nii
    if hasattr(img_nii, "get_sform"):
        return rectify_header_sform_qform(img_nii)
    return img_nii


def rectify_header_sform_qform(img_nii):

    d = img_nii.header["dim"][0]
    pixdim = np.asarray(img_nii.header.get_zooms())[:d]
    sform, qform = img_nii.get_sform(), img_nii.get_qform()
    norm_sform = np.sqrt(np.sum(np.square(sform[:d, :d]), 0))
    norm_qform = np.sqrt(np.sum(np.square(qform[:d, :d]), 0))
    sform_mismatch = not np.allclose(norm_sform, pixdim)
    qform_mismatch = not np.allclose(norm_qform, pixdim)

    if img_nii.header["sform_code"] != 0:
        if not sform_mismatch:
            return img_nii
        if not qform_mismatch:
            img_nii.set_sform(img_nii.get_qform())
            return img_nii
    if img_nii.header["qform_code"] != 0:
        if not qform_mismatch:
            return img_nii
        if not sform_mismatch:
            img_nii.set_qform(img_nii.get_sform())
            return img_nii

    norm = np.sqrt(np.sum(np.square(img_nii.affine[:d, :d]), 0))
    warnings.warn(f"Modifying image pixdim from {pixdim} to {norm}")

    img_nii.header.set_zooms(norm)
    return img_nii


def zoom_affine(affine: np.ndarray, scale: Sequence[float], diagonal: bool = True):

    affine = np.array(affine, dtype=float, copy=True)
    if len(affine) != len(affine[0]):
        raise ValueError(f"affine must be n x n, got {len(affine)} x {len(affine[0])}.")
    scale_np = np.array(scale, dtype=float, copy=True)

    d = len(affine) - 1
    # compute original pixdim
    norm = np.sqrt(np.sum(np.square(affine), 0))[:-1]
    if len(scale_np) < d:  # defaults based on affine
        scale_np = np.append(scale_np, norm[len(scale_np) :])
    scale_np = scale_np[:d]
    scale_np = np.asarray(fall_back_tuple(scale_np, norm))

    scale_np[scale_np == 0] = 1.0
    if diagonal:
        return np.diag(np.append(scale_np, [1.0]))
    rzs = affine[:-1, :-1]  # rotation zoom scale
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    s = np.sign(np.diag(zs)) * np.abs(scale_np)
    # construct new affine with rotation and zoom
    new_affine = np.eye(len(affine))
    new_affine[:-1, :-1] = rotation @ np.diag(s)
    return new_affine


def compute_shape_offset(
    spatial_shape: Union[np.ndarray, Sequence[int]], in_affine: np.ndarray, out_affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    shape = np.array(spatial_shape, copy=True, dtype=float)
    sr = len(shape)
    in_affine = to_affine_nd(sr, in_affine)
    out_affine = to_affine_nd(sr, out_affine)
    in_coords = [(0.0, dim - 1.0) for dim in shape]
    corners = np.asarray(np.meshgrid(*in_coords, indexing="ij")).reshape((len(shape), -1))
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    corners = in_affine @ corners
    corners_out = np.linalg.inv(out_affine) @ corners
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(corners_out.ptp(axis=1) + 1.0)
    if np.allclose(nib.io_orientation(in_affine), nib.io_orientation(out_affine)):
        # same orientation, get translate from the origin
        offset = in_affine @ ([0] * sr + [1])
        offset = offset[:-1] / offset[-1]
    else:
        # different orientation, the min is the origin
        corners = corners[:-1] / corners[-1]
        offset = np.min(corners, 1)
    return out_shape.astype(int, copy=False), offset


def to_affine_nd(r: Union[np.ndarray, int], affine: np.ndarray) -> np.ndarray:

    affine_np = np.array(affine, dtype=np.float64)
    if affine_np.ndim != 2:
        raise ValueError(f"affine must have 2 dimensions, got {affine_np.ndim}.")
    new_affine = np.array(r, dtype=np.float64, copy=True)
    if new_affine.ndim == 0:
        sr: int = int(new_affine.astype(np.uint))
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=np.float64)
    d = max(min(len(new_affine) - 1, len(affine_np) - 1), 1)
    new_affine[:d, :d] = affine_np[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_np[:d, -1]
    return new_affine


def compute_importance_map(
    patch_size: Tuple[int, ...],
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
) -> paddle.Tensor:

    mode = look_up_option(mode, BlendMode)
    if mode == BlendMode.CONSTANT:
        importance_map = paddle.ones(patch_size).cast('float32')
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are [{BlendMode.CONSTANT}, {BlendMode.CONSTANT}]."
        )

    return importance_map


def is_supported_format(filename: Union[Sequence[PathLike], PathLike], suffixes: Sequence[str]) -> bool:

    filenames: Sequence[PathLike] = ensure_tuple(filename)
    for name in filenames:
        tokens: Sequence[str] = PurePath(name).suffixes
        if len(tokens) == 0 or all("." + s.lower() not in "".join(tokens) for s in suffixes):
            return False

    return True

