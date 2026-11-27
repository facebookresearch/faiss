# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels for the exact balancing step (``e_means_balanced_exact``).

``top_m`` returns, for every row, the m centroids of smallest biased score
||c_j||^2 - b_j - 2 x.c_j, streaming the centroids in tiles and keeping a
running list of m candidates per row in registers, so the (rows, k) score
matrix is never written. A tile is merged into the list only as many times as
the most demanding row of the block needs (at most m), so once the lists are
tight most tiles cost one comparison per score, as in the plain assignment.

``exact_costs`` recomputes the true squared distance from each row to each of
its m candidates by direct difference in float32.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:
    raise ImportError("the exact balancing kernels require the triton package") from error


_BLOCK_B = 64
_BLOCK_K = 128
_NUM_WARPS = 4
_NUM_STAGES = 3
# Up to this dimension the row block stays in registers for the whole scan.
_FULL_D_MAX = 128
_BLOCK_D = 64


@triton.jit(do_not_specialize=["B", "K"])
def _top_m_kernel(
    x_ptr, c_ptr, norm_ptr, label_ptr, score_ptr,
    B, K, D, stride_x, stride_c,
    M: tl.constexpr, MW: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    FULL_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_B + tl.arange(0, BLOCK_B)
    row_mask = row < B
    slot = tl.arange(0, MW)
    # Padding slots hold -inf: never the list's maximum, so never replaced.
    best = tl.where(slot[None, :] < M, float("inf"), float("-inf")) + tl.zeros(
        (BLOCK_B, MW), tl.float32)
    label = tl.full((BLOCK_B, MW), -1, tl.int32)
    worst, worst_slot = tl.max(best, axis=1, return_indices=True)
    if FULL_D:
        dcol = tl.arange(0, BLOCK_D)
        xr = tl.load(x_ptr + row[:, None] * stride_x + dcol[None, :],
                     mask=row_mask[:, None] & (dcol[None, :] < D), other=0.0)
    for k0 in range(0, K, BLOCK_K):
        centroid = k0 + tl.arange(0, BLOCK_K)
        cmask = centroid < K
        if FULL_D:
            c = tl.load(c_ptr + centroid[:, None].to(tl.int64) * stride_c + dcol[None, :],
                        mask=cmask[:, None] & (dcol[None, :] < D), other=0.0)
            dot = tl.dot(xr, tl.trans(c), out_dtype=tl.float32)
        else:
            dot = tl.zeros((BLOCK_B, BLOCK_K), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                dcol = d0 + tl.arange(0, BLOCK_D)
                dmask = dcol < D
                xr = tl.load(x_ptr + row[:, None] * stride_x + dcol[None, :],
                             mask=row_mask[:, None] & dmask[None, :], other=0.0)
                c = tl.load(c_ptr + centroid[:, None].to(tl.int64) * stride_c + dcol[None, :],
                            mask=cmask[:, None] & dmask[None, :], other=0.0)
                dot = tl.dot(xr, tl.trans(c), acc=dot, out_dtype=tl.float32)
        norm = tl.load(norm_ptr + centroid, mask=cmask, other=float("inf"))
        score = norm[None, :] - 2.0 * dot                                   # (BLOCK_B, BLOCK_K)
        if M == 1:
            tile_best, tile_arg = tl.min(score, axis=1, return_indices=True,
                                         return_indices_tie_break_left=True)
            take = tile_best < worst
            hit = take[:, None] & (slot[None, :] == 0)
            best = tl.where(hit, tile_best[:, None], best)
            label = tl.where(hit, (tile_arg + k0)[:, None].to(tl.int32), label)
            worst = tl.where(take, tile_best, worst)
            rounds = 0
        else:
            below = tl.sum((score < worst[:, None]).to(tl.int32), axis=1)
            rounds = tl.minimum(tl.max(below, axis=0), M)
        kcol = tl.arange(0, BLOCK_K)
        for _ in range(rounds):
            tile_best, tile_arg = tl.min(score, axis=1, return_indices=True)
            take = tile_best < worst
            hit = take[:, None] & (slot[None, :] == worst_slot[:, None])
            best = tl.where(hit, tile_best[:, None], best)
            label = tl.where(hit, (tile_arg + k0)[:, None].to(tl.int32), label)
            score = tl.where(take[:, None] & (kcol[None, :] == tile_arg[:, None]),
                             float("inf"), score)
            worst, worst_slot = tl.max(best, axis=1, return_indices=True)
    out = row[:, None] * M + slot[None, :]
    omask = row_mask[:, None] & (slot[None, :] < M)
    tl.store(label_ptr + out, label, mask=omask)
    tl.store(score_ptr + out, best, mask=omask)


def top_m(x, centroids16, norms, m, out_labels=None, out_scores=None):
    """Per row, the m smallest ``norms[j] - 2 x.c_j`` (unordered).

    x (n, d) and centroids16 (k, d) are 16-bit CUDA tensors of the same dtype
    (the product accumulates in float32); norms (k,) float32. Returns
    (labels (n, m) int32, scores (n, m) float32).
    """
    n, d = x.shape
    k = centroids16.shape[0]
    assert x.dtype == centroids16.dtype and x.dtype in (torch.float16, torch.bfloat16)
    assert 1 <= m <= k
    x = x if x.stride(1) == 1 else x.contiguous()
    centroids16 = centroids16.contiguous()
    if out_labels is None:
        out_labels = torch.empty((n, m), dtype=torch.int32, device=x.device)
    if out_scores is None:
        out_scores = torch.empty((n, m), dtype=torch.float32, device=x.device)
    if n == 0:
        return out_labels, out_scores
    full_d = d <= _FULL_D_MAX
    block_d = max(16, triton.next_power_of_2(d)) if full_d else _BLOCK_D
    grid = (triton.cdiv(n, _BLOCK_B),)
    _top_m_kernel[grid](
        x, centroids16, norms, out_labels, out_scores,
        n, k, d, x.stride(0), centroids16.stride(0),
        M=m, MW=triton.next_power_of_2(m),
        BLOCK_B=_BLOCK_B, BLOCK_K=_BLOCK_K, BLOCK_D=block_d, FULL_D=full_d,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return out_labels, out_scores


@triton.jit(do_not_specialize=["B"])
def _exact_costs_kernel(x_ptr, c_ptr, label_ptr, out_ptr, B, D, stride_x, stride_c,
                        M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    pair = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = pair < B * M
    row = pair // M
    centroid = tl.load(label_ptr + pair, mask=mask, other=0).to(tl.int64)
    total = tl.zeros((BLOCK_R,), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        dcol = d0 + tl.arange(0, BLOCK_D)
        dmask = mask[:, None] & (dcol[None, :] < D)
        xv = tl.load(x_ptr + row[:, None] * stride_x + dcol[None, :], mask=dmask, other=0.0)
        cv = tl.load(c_ptr + centroid[:, None] * stride_c + dcol[None, :], mask=dmask, other=0.0)
        diff = xv.to(tl.float32) - cv
        total += tl.sum(diff * diff, axis=1)
    tl.store(out_ptr + pair, total, mask=mask)


def exact_costs(x, centroids, labels, out=None):
    """||x_i - c_{labels[i, l]}||^2 by direct difference in float32.

    x (n, d) any float dtype on CUDA, centroids (k, d) float32, labels (n, m)
    int32. Returns (n, m) float32.
    """
    n, m = labels.shape
    d = x.shape[1]
    x = x if x.stride(1) == 1 else x.contiguous()
    centroids = centroids.contiguous()
    labels = labels.contiguous()
    if out is None:
        out = torch.empty((n, m), dtype=torch.float32, device=x.device)
    if n == 0:
        return out
    block_r = 64
    block_d = min(128, max(16, triton.next_power_of_2(d)))
    grid = (triton.cdiv(n * m, block_r),)
    _exact_costs_kernel[grid](x, centroids, labels, out, n, d, x.stride(0), centroids.stride(0),
                              M=m, BLOCK_R=block_r, BLOCK_D=block_d, num_warps=4)
    return out


@triton.jit
def _relax_kernel(src_ptr, dst_ptr, w_ptr, key_ptr, flag_ptr, E, BLOCK: tl.constexpr):
    """One in-place Bellman-Ford sweep over the move list.

    key[j] packs (distance bits << 32 | hops) of cluster j; the distances
    are >= 0, so their float bits order like the floats and an int64
    atomic_min keeps the lexicographic minimum (distance, then hops).
    """
    edge = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = edge < E
    src = tl.load(src_ptr + edge, mask=mask, other=0).to(tl.int64)
    dst = tl.load(dst_ptr + edge, mask=mask, other=0).to(tl.int64)
    w = tl.load(w_ptr + edge, mask=mask, other=0.0)
    key = tl.load(key_ptr + dst, mask=mask, other=0)
    dist = (key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    hops = key & 0xFFFFFFFF
    ok = mask & (dist < float("inf"))
    through = w + dist
    new = (through.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (hops + 1)
    old = tl.atomic_min(key_ptr + src, new, mask=ok)
    improved = ok & (old > new)
    if tl.max(improved.to(tl.int32), axis=0) > 0:
        tl.atomic_add(flag_ptr, 1)


def relax(src, dst, w, key, flag):
    """One sweep; ``flag`` (1,) int32 is incremented if any distance improved."""
    e = src.shape[0]
    if e == 0:
        return
    block = 1024
    _relax_kernel[(triton.cdiv(e, block),)](src, dst, w, key, flag, e, BLOCK=block, num_warps=4)


@triton.jit
def _relax_csr_kernel(offset_ptr, dst_ptr, w_ptr, key_ptr, flag_ptr, BLOCK: tl.constexpr):
    """One in-place sweep, one program per cluster over its outgoing moves.

    The moves are grouped by source cluster (offsets, CSR). The program keeps
    the smallest packed (distance, hops) over its moves and writes it if it
    beats the cluster's key: no atomics, and reads of keys already updated in
    this sweep only speed the convergence.
    """
    cluster = tl.program_id(0).to(tl.int64)
    begin = tl.load(offset_ptr + cluster)
    end = tl.load(offset_ptr + cluster + 1)
    if end > begin:
        best = tl.full((BLOCK,), 0x7FFFFFFFFFFFFFFF, tl.int64)
        for e0 in range(begin, end, BLOCK):
            edge = e0 + tl.arange(0, BLOCK)
            mask = edge < end
            dst = tl.load(dst_ptr + edge, mask=mask, other=0).to(tl.int64)
            w = tl.load(w_ptr + edge, mask=mask, other=0.0)
            key = tl.load(key_ptr + dst, mask=mask, other=0)
            dist = (key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            ok = mask & (dist < float("inf"))
            through = w + dist
            new = (through.to(tl.int32, bitcast=True).to(tl.int64) << 32) | ((key & 0xFFFFFFFF) + 1)
            best = tl.minimum(best, tl.where(ok, new, 0x7FFFFFFFFFFFFFFF))
        best_all = tl.min(best, axis=0)
        old = tl.load(key_ptr + cluster)
        if best_all < old:
            tl.store(key_ptr + cluster, best_all)
            tl.atomic_add(flag_ptr, 1)


def relax_csr(offsets, dst, w, key, flag):
    """One sweep over moves grouped by source (offsets (k + 1,) int64)."""
    k = offsets.shape[0] - 1
    _relax_csr_kernel[(k,)](offsets, dst, w, key, flag, BLOCK=64, num_warps=2)


@triton.jit(do_not_specialize=["n_levels"])
def _push_kernel(offset_ptr, n_levels, src_ptr, dst_ptr, units_ptr, cap_ptr, moved_ptr,
                 BLOCK: tl.constexpr):
    """The capacity-aware push of one batch, in a single program.

    The free moves are sorted by hop level of their source (level offsets in
    ``offset_ptr``). First, by increasing level, a cluster's capacity grows by
    one for each of its moves whose target has capacity (cap holds the room of
    the clusters with room on entry). Then, by decreasing level, each move
    whose target still has capacity claims one of its source's units with an
    atomic decrement; a claimed unit moves to the target, whose capacity drops
    by one. Moves of a level only read levels already settled, and a barrier
    separates the levels.
    """
    for lv in range(0, n_levels):
        begin = tl.load(offset_ptr + lv)
        end = tl.load(offset_ptr + lv + 1)
        for e0 in range(begin, end, BLOCK):
            edge = e0 + tl.arange(0, BLOCK)
            mask = edge < end
            src = tl.load(src_ptr + edge, mask=mask, other=0)
            dst = tl.load(dst_ptr + edge, mask=mask, other=0)
            ok = mask & (tl.load(cap_ptr + dst, mask=mask, other=0) > 0)
            tl.atomic_add(cap_ptr + src, ok.to(tl.int32), mask=ok)
        tl.debug_barrier()
    for i in range(0, n_levels):
        lv = n_levels - 1 - i
        begin = tl.load(offset_ptr + lv)
        end = tl.load(offset_ptr + lv + 1)
        for e0 in range(begin, end, BLOCK):
            edge = e0 + tl.arange(0, BLOCK)
            mask = edge < end
            src = tl.load(src_ptr + edge, mask=mask, other=0)
            dst = tl.load(dst_ptr + edge, mask=mask, other=0)
            ok = mask & (tl.load(cap_ptr + dst, mask=mask, other=0) > 0)
            ok = ok & (tl.load(units_ptr + src, mask=mask, other=0) > 0)
            old = tl.atomic_add(units_ptr + src, -1, mask=ok)
            won = ok & (old > 0)
            tl.atomic_add(units_ptr + src, 1, mask=ok & ~won)
            tl.atomic_add(units_ptr + dst, 1, mask=won)
            tl.atomic_add(cap_ptr + dst, -1, mask=won)
            tl.store(moved_ptr + edge, won.to(tl.int8), mask=mask)
        tl.debug_barrier()


def push(offsets, src, dst, units, cap):
    """Run ``_push_kernel``; units and cap (k,) int32 are updated in place.

    Returns the (e,) int8 flags of the moves taken."""
    moved = torch.zeros(src.shape[0], dtype=torch.int8, device=src.device)
    if src.shape[0]:
        _push_kernel[(1,)](offsets, offsets.shape[0] - 1, src, dst, units, cap, moved,
                           BLOCK=1024, num_warps=8)
    return moved
