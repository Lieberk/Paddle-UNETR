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
Utilities and types for defining networks, these depend on Pypaddle.
"""

from typing import Optional, Sequence
import paddle

__all__ = [
    "normalize_transform",
    "to_norm_affine",
]


def normalize_transform(
    shape: Sequence[int],
    dtype: Optional[paddle.dtype] = None,
    align_corners: bool = False,
) -> paddle.Tensor:
    """
    Compute an affine matrix according to the input shape.
    The transform normalizes the homogeneous image coordinates to the
    range of `[-1, 1]`.

    Args:
        shape: input spatial shape
        dtype: data type of the returned affine
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pypaddle.org/docs/stable/nn.functional.html#paddle.nn.functional.grid_sample
    """
    norm = paddle.to_tensor(shape, dtype=paddle.float64)  # no in-place change
    if align_corners:
        norm[norm <= 1.0] = 2.0
        norm = 2.0 / (norm - 1.0)
        norm = paddle.diag(paddle.concat((norm, paddle.ones((1,), dtype=paddle.float64))))
        norm[:-1, -1] = -1.0
    else:
        norm[norm <= 0.0] = 2.0
        norm = 2.0 / norm
        norm = paddle.diag(paddle.concat((norm, paddle.ones((1,), dtype=paddle.float64))))
        norm[:-1, -1] = 1.0 / paddle.to_tensor(shape, dtype=paddle.float64) - 1.0
    norm = norm.unsqueeze(0).cast(dtype=dtype)
    norm.stop_gradient = True
    return norm


def to_norm_affine(
    affine: paddle.Tensor, src_size: Sequence[int], dst_size: Sequence[int], align_corners: bool = False
) -> paddle.Tensor:
    """
    Given ``affine`` defined for coordinates in the pixel space, compute the corresponding affine
    for the normalized coordinates.

    Args:
        affine: Nxdxd batched square matrix
        src_size: source image spatial shape
        dst_size: target image spatial shape
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pypaddle.org/docs/stable/nn.functional.html#paddle.nn.functional.grid_sample

    Raises:
        TypeError: When ``affine`` is not a ``paddle.Tensor``.
        ValueError: When ``affine`` is not Nxdxd.
        ValueError: When ``src_size`` or ``dst_size`` dimensions differ from ``affine``.

    """
    if not isinstance(affine, paddle.Tensor):
        raise TypeError(f"affine must be a paddle.Tensor but is {type(affine).__name__}.")
    if affine.ndimension() != 3 or affine.shape[1] != affine.shape[2]:
        raise ValueError(f"affine must be Nxdxd, got {tuple(affine.shape)}.")
    sr = affine.shape[1] - 1
    if sr != len(src_size) or sr != len(dst_size):
        raise ValueError(f"affine suggests {sr}D, got src={len(src_size)}D, dst={len(dst_size)}D.")

    src_xform = normalize_transform(src_size, affine.dtype, align_corners)
    dst_xform = normalize_transform(dst_size, affine.dtype, align_corners)
    return src_xform @ affine @ paddle.inverse(dst_xform)

