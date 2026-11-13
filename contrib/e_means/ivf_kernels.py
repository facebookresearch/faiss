# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA candidate reduction for the bounded IVF BMM search."""

import torch
import triton
import triton.language as tl


@triton.jit
def _update_probes(distances, candidate_ids, probes, groups, scores, ids, norms,
                   nrows, width, queries_per_group, nprobes,
                   BLOCK_B: tl.constexpr, BLOCK_R: tl.constexpr,
                   TOP2: tl.constexpr, GROUPED: tl.constexpr, DOTS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    probe = tl.load(probes + row, mask=row < nrows, other=nprobes).to(tl.int64)
    live = (row < nrows) & (probe < nprobes)
    group = tl.load(groups + row, mask=live, other=0) if GROUPED else row // queries_per_group
    col = tl.arange(0, BLOCK_B)
    value = tl.load(distances + row[:, None].to(tl.int64) * width + col[None, :],
                    mask=live[:, None] & (col < width)[None, :], other=float("inf"))
    if DOTS:
        norm = tl.load(norms + group[:, None].to(tl.int64) * width + col[None, :],
                       mask=live[:, None] & (col < width)[None, :], other=float("inf"))
        value = tl.where(live[:, None] & (col < width)[None, :], norm - 2.0 * value, float("inf"))
    candidate = tl.load(candidate_ids + group[:, None].to(tl.int64) * width + col[None, :],
                        mask=live[:, None] & (col < width)[None, :], other=0x7fffffffffffffff)
    first = tl.min(value, axis=1)
    first_id = tl.min(tl.where((value == first[:, None]) & (tl.abs(value) < float("inf")),
                              candidate, 0x7fffffffffffffff), axis=1)
    stride: tl.constexpr = 2 if TOP2 else 1
    old = tl.load(scores + probe * stride, mask=live, other=float("inf"))
    old_id = tl.load(ids + probe * stride, mask=live, other=0x7fffffffffffffff)
    if TOP2:
        second_values = tl.where(candidate == first_id[:, None], float("inf"), value)
        second = tl.min(second_values, axis=1)
        second_id = tl.min(tl.where((second_values == second[:, None]) & (tl.abs(second_values) < float("inf")),
                                   candidate, 0x7fffffffffffffff), axis=1)
        old_second = tl.load(scores + probe * 2 + 1, mask=live, other=float("inf"))
        old_second_id = tl.load(ids + probe * 2 + 1, mask=live, other=0x7fffffffffffffff)
        lane = tl.arange(0, 4)
        merged = tl.where(lane[None, :] == 0, first[:, None],
                          tl.where(lane[None, :] == 1, second[:, None],
                                   tl.where(lane[None, :] == 2, old[:, None], old_second[:, None])))
        merged_ids = tl.where(lane[None, :] == 0, first_id[:, None],
                              tl.where(lane[None, :] == 1, second_id[:, None],
                                       tl.where(lane[None, :] == 2, old_id[:, None], old_second_id[:, None])))
        best = tl.min(merged, axis=1)
        best_id = tl.min(tl.where(merged == best[:, None], merged_ids, 0x7fffffffffffffff), axis=1)
        other = tl.min(tl.where(merged_ids == best_id[:, None], float("inf"), merged), axis=1)
        other_id = tl.min(tl.where((merged == other[:, None]) & (merged_ids != best_id[:, None]),
                                  merged_ids, 0x7fffffffffffffff), axis=1)
        tl.store(scores + probe * 2 + 1, other, mask=live)
        tl.store(ids + probe * 2 + 1, other_id, mask=live)
    else:
        better = (first < old) | ((first == old) & (first_id < old_id))
        best, best_id = tl.where(better, first, old), tl.where(better, first_id, old_id)
    tl.store(scores + probe * stride, best, mask=live)
    tl.store(ids + probe * stride, best_id, mask=live)


@triton.jit
def _merge_probes(scores, ids, output_scores, output_ids, n, nprobe,
                   TOP2: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_P: tl.constexpr):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    stride: tl.constexpr = 2 if TOP2 else 1
    lane = tl.arange(0, BLOCK_P)
    offset = row[:, None].to(tl.int64) * nprobe * stride + lane[None, :]
    live = (row < n)[:, None] & (lane < nprobe * stride)[None, :]
    value = tl.load(scores + offset, mask=live, other=float("inf"))
    candidate = tl.load(ids + offset, mask=live, other=0x7fffffffffffffff)
    first = tl.min(value, axis=1)
    first_id = tl.min(tl.where(value == first[:, None], candidate, 0x7fffffffffffffff), axis=1)
    tl.store(output_scores + row * stride, first, mask=row < n)
    tl.store(output_ids + row * stride, tl.where(first_id == 0x7fffffffffffffff, -1, first_id), mask=row < n)
    if TOP2:
        second = tl.min(tl.where(candidate == first_id[:, None], float("inf"), value), axis=1)
        second_id = tl.min(tl.where((value == second[:, None]) & (candidate != first_id[:, None]),
                                   candidate, 0x7fffffffffffffff), axis=1)
        tl.store(output_scores + row * 2 + 1, second, mask=row < n)
        tl.store(output_ids + row * 2 + 1, tl.where(second_id == 0x7fffffffffffffff, -1, second_id), mask=row < n)


def update_probes(distances, candidate_ids, probes, scores, ids, *, groups=None, norms=None):
    """Reduce a BMM chunk and merge it into disjoint query/probe entries."""
    width = distances.shape[-1]
    rows = distances.numel() // width
    block_b = triton.next_power_of_2(width)
    block_r = 4 if block_b <= 1024 else 1
    with torch.cuda.device(distances.device):
        _update_probes[(triton.cdiv(rows, block_r),)](
            distances, candidate_ids, probes, groups, scores, ids, norms,
            rows, width, distances.shape[-2], len(scores),
            BLOCK_B=block_b, BLOCK_R=block_r, TOP2=scores.shape[1] == 2,
            GROUPED=groups is not None, DOTS=norms is not None,
            num_warps=4 if block_b <= 2048 else 8, enable_fp_fusion=False)


def merge_probes(scores, ids, n, nprobe):
    width = scores.shape[1]
    output_scores = scores.new_empty((n, width))
    output_ids = ids.new_empty((n, width), dtype=torch.int32)
    if n:
        block_p = triton.next_power_of_2(nprobe * width)
        block_r = max(1, min(32, 1024 // block_p))
        with torch.cuda.device(scores.device):
            _merge_probes[(triton.cdiv(n, block_r),)](
                scores, ids, output_scores, output_ids, n, nprobe,
                TOP2=width == 2, BLOCK_R=block_r, BLOCK_P=block_p, num_warps=4)
    return (output_scores, output_ids) if width == 2 else (output_scores[:, 0], output_ids[:, 0])


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
