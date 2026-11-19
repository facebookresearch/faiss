# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA per-owner statistics for the IVF-sharded centroids."""

import torch
import triton
import triton.language as tl


@triton.jit
def _owned_statistics(rows, winners, centroids, axes, stats, removal, margins, objectives,
                       n, d, k, centroid_begin, row_stride, stats_stride,
                       COST: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    winner = tl.load(winners + row, mask=row < n, other=-1)
    owned = (row < n) & (winner >= centroid_begin) & (winner < centroid_begin + k)
    label = (winner - centroid_begin).to(tl.int64)
    col = tl.arange(0, BLOCK_D)
    live = owned[:, None] & (col < d)[None, :]
    value = tl.load(rows + row[:, None].to(tl.int64) * row_stride + col[None, :], mask=live, other=0.0).to(tl.float32)
    center = tl.load(centroids + label[:, None] * d + col[None, :], mask=live, other=0.0)
    y = value - center
    tl.atomic_add(stats + label[:, None] * stats_stride + col[None, :], value, mask=live, sem="relaxed")
    tl.atomic_add(stats + label * stats_stride + d, 1.0, mask=owned, sem="relaxed")
    # Square in FP32, then sum in FP64, as the ordinary owner SSE.
    squared = (y * y).to(tl.float64)
    tl.store(objectives + tl.program_id(0), tl.sum(tl.sum(squared, axis=1), axis=0))
    if COST:
        axis = tl.load(axes + label[:, None] * d + col[None, :], mask=live, other=0.0)
        p = tl.sum(y * axis, axis=1)
        tl.atomic_add(removal + label[:, None] * stats_stride + col[None, :], p[:, None] * y, mask=live, sem="relaxed")
        margin = tl.load(margins + row, mask=owned, other=0.0)
        margin = tl.where(margin == float("-inf"), -3.4028234663852886e38, margin)
        margin = tl.where((margin == margin) & (margin < float("inf")), margin, 0.0)
        tl.atomic_add(removal + label * stats_stride + d, margin, mask=owned, sem="relaxed")


def accumulate_owned(rows, winners, centroids, axes, stats, removal, objective, margins, centroid_begin):
    """Accumulate owner sums/counts, SSE and Oja/removal sums in one row pass."""
    if (stats.dtype != torch.float32 or rows.dtype not in (torch.float32, torch.bfloat16)
            or rows.shape[1] > 4096 or rows.stride(1) != 1 or stats.stride(1) != 1
            or torch.are_deterministic_algorithms_enabled()):
        return False
    block_d = triton.next_power_of_2(rows.shape[1])
    block_r = 4 if block_d <= 1024 else 1
    blocks = triton.cdiv(len(rows), block_r)
    if blocks:
        partial = objective.new_empty(blocks)
        with torch.cuda.device(rows.device):
            _owned_statistics[(blocks,)](
                rows, winners, centroids, axes, stats, removal, margins, partial,
                len(rows), rows.shape[1], len(centroids), centroid_begin, rows.stride(0), stats.stride(0),
                COST=margins is not None, BLOCK_R=block_r, BLOCK_D=block_d,
                num_warps=4 if block_d <= 2048 else 8, enable_fp_fusion=False)
        objective.add_(partial.sum())
    return True
