# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernel for the balanced E-means capacity cap's repair rounds.

A repair round assigns again the rows evicted from over-full clusters, with
every full cluster masked out (its norm set to +inf). How many rows are
evicted is only known on the GPU. ``masked_assign`` therefore reads the row
count from device memory: a persistent grid walks (row tile, centroid slice)
work items up to that count, so the host never waits for it and the call can
be recorded in a CUDA graph. Each item reduces its slice to one candidate per
row and merges it with a 64-bit atomic minimum of (ordered score, index), so
the result is the exact argmin with ties to the lower index, whatever the
order the items ran in. Rows are read through a list of positions into the
batch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_assign_kernel(
    rows, row_stride, positions, count_ptr, centroids, centroid_stride, norms, full, cap_ptr,
    exclude, packed, k, d, n_slices,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    SLICE_TILES: tl.constexpr, ONE_D_TILE: tl.constexpr, HAS_FULL: tl.constexpr,
    FULL_FROM_COUNTS: tl.constexpr, EXCLUDE: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    count = tl.load(count_ptr)
    n_items = tl.cdiv(count, BLOCK_M) * n_slices
    offs_d0 = tl.arange(0, BLOCK_D)
    for item in range(pid, n_items, nprog):
        tile = item // n_slices
        k_begin = (item % n_slices) * (SLICE_TILES * BLOCK_K)
        offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < count
        pos = tl.load(positions + offs_m, mask=mask_m, other=0).to(tl.int64)
        row_base = rows + pos[:, None] * row_stride
        if EXCLUDE:
            # The row's own nearest centroid: the search finds its runner-up.
            own = tl.load(exclude + pos, mask=mask_m, other=-1)
        if ONE_D_TILE:
            x = tl.load(row_base + offs_d0[None, :],
                        mask=mask_m[:, None] & (offs_d0[None, :] < d), other=0.0)
        best = tl.full((BLOCK_M,), float("inf"), tl.float32)
        best_idx = tl.zeros((BLOCK_M,), tl.int32)
        for t in range(SLICE_TILES):
            offs_k = k_begin + t * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k
            c_base = centroids + offs_k.to(tl.int64)[None, :] * centroid_stride
            if ONE_D_TILE:
                c = tl.load(c_base + offs_d0[:, None],
                            mask=mask_k[None, :] & (offs_d0[:, None] < d), other=0.0)
                acc = tl.dot(x, c)
            else:
                acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
                for d0 in range(0, d, BLOCK_D):
                    offs_d = d0 + offs_d0
                    xd = tl.load(row_base + offs_d[None, :],
                                 mask=mask_m[:, None] & (offs_d[None, :] < d), other=0.0)
                    c = tl.load(c_base + offs_d[:, None],
                                mask=mask_k[None, :] & (offs_d[:, None] < d), other=0.0)
                    acc += tl.dot(xd, c)
            norm = tl.load(norms + offs_k, mask=mask_k, other=float("inf"))
            if HAS_FULL:
                masked = tl.load(full + offs_k, mask=mask_k, other=1)
                norm = tl.where(masked != 0, float("inf"), norm)
            if FULL_FROM_COUNTS:
                held = tl.load(full + offs_k, mask=mask_k, other=0)
                norm = tl.where(held >= tl.load(cap_ptr), float("inf"), norm)
            score = norm[None, :] - 2.0 * acc
            score = tl.where(mask_k[None, :], score, float("inf"))
            if EXCLUDE:
                score = tl.where(offs_k[None, :] == own[:, None], float("inf"), score)
            block_best = tl.min(score, axis=1)
            block_idx = (tl.argmin(score, axis=1) + k_begin + t * BLOCK_K).to(tl.int32)
            better = block_best < best
            best = tl.where(better, block_best, best)
            best_idx = tl.where(better, block_idx, best_idx)
        # Order-preserving int32 view of the score, then (score, index) in 64 bits.
        bits = best.to(tl.int32, bitcast=True)
        key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        word = (key.to(tl.int64) << 32) | best_idx.to(tl.int64)
        tl.atomic_min(packed + offs_m, word, mask=mask_m)


@triton.jit
def _unpack_kernel(packed, count_ptr, labels_out, scores_out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < tl.load(count_ptr)
    word = tl.load(packed + offs, mask=mask, other=0)
    key = (word >> 32).to(tl.int32)
    bits = tl.where(key < 0, key ^ 0x7FFFFFFF, key)
    tl.store(labels_out + offs, (word & 0xFFFFFFFF), mask=mask)
    tl.store(scores_out + offs, bits.to(tl.float32, bitcast=True), mask=mask)


def _search_geometry(k, d):
    """(one d tile?, BLOCK_D, BLOCK_K, SLICE_TILES, BLOCK_M, warps). Measured on H100
    (scripts/core/masked_bench.py): 128 x 128 tiles, 4 warps, 2 stages, 8 tiles per slice
    were fastest at d = 96, 128 and 256 (1.4-1.9x the 8-warp geometry)."""
    one_tile = d <= 256
    block_d = max(16, triton.next_power_of_2(d)) if one_tile else 64
    block_k = 128 if one_tile else 64
    block_m = 128 if one_tile else 64
    slice_tiles = max(1, min(8, triton.cdiv(k, block_k)))
    return one_tile, block_d, block_k, slice_tiles, block_m, 4 if one_tile else 8


def _masked_search(rows, positions, count, centroids, norms, full, packed, max_rows,
                   held=None, cap=None, exclude=None):
    """Launch the search; ``packed`` holds INT64_MAX for the first ``count`` rows.

    Masked clusters: ``full`` (k,) int8 non-zero, or ``held`` (k,) counts >= ``cap``.
    With ``exclude`` (int64 per batch row), each row also skips the centroid
    ``exclude[row]``: the search then returns the runner-up of that centroid.
    """
    if rows.dtype != centroids.dtype or rows.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("masked_assign needs matching BF16 or FP16 rows and centroids")
    if rows.stride(1) != 1 or centroids.stride(1) != 1:
        raise ValueError("masked_assign needs unit stride along d")
    k, d = centroids.shape
    if rows.shape[1] < d:
        raise ValueError("rows are narrower than the centroids")
    if k >= 2**31:
        raise ValueError("masked_assign needs k < 2**31")
    one_tile, block_d, block_k, slice_tiles, block_m, warps = _search_geometry(k, d)
    n_slices = triton.cdiv(k, block_k * slice_tiles)
    sms = torch.cuda.get_device_properties(rows.device).multi_processor_count
    items = triton.cdiv(max(1, max_rows), block_m) * n_slices
    grid = (max(1, min(4 * sms, items)),)
    with torch.cuda.device(rows.device):
        _masked_assign_kernel[grid](
            rows, rows.stride(0), positions, count, centroids, centroids.stride(0), norms,
            full if full is not None else held if held is not None else norms,
            cap if cap is not None else count, exclude if exclude is not None else positions,
            packed, k, d, n_slices,
            BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_D=block_d, SLICE_TILES=slice_tiles,
            ONE_D_TILE=one_tile, HAS_FULL=full is not None, FULL_FROM_COUNTS=held is not None,
            EXCLUDE=exclude is not None, num_warps=warps, num_stages=2)


def masked_assign(rows, positions, count, centroids, norms, labels_out, scores_out, max_rows=None,
                  packed=None):
    """Nearest centroid of ``rows[positions[i]]`` for i < ``count`` (a device int).

    ``rows`` (B, d) and ``centroids`` (k, d) share one 16-bit dtype (unit
    stride along d); ``norms`` (k,) FP32 holds the biased norms, +inf for a
    masked cluster. Writes int64 labels and the FP32 ranking score
    norm_j - 2 x.c_j (row norm dropped) for the first ``count`` entries;
    entries past it are untouched. Ties go to the lower index. ``max_rows``
    bounds ``count`` (default ``len(positions)``); ``packed`` is optional
    int64 scratch of that length.
    """
    max_rows = len(positions) if max_rows is None else max_rows
    if packed is None:
        packed = torch.empty(max_rows, dtype=torch.int64, device=rows.device)
    packed[:max_rows].fill_(torch.iinfo(torch.int64).max)
    _masked_search(rows, positions, count, centroids, norms, None, packed, max_rows)
    with torch.cuda.device(rows.device):
        _unpack_kernel[(triton.cdiv(max(1, max_rows), 1024),)](
            packed, count, labels_out, scores_out, max_rows, BLOCK=1024)


# --------------------------------------------------------------------------
# Fused bookkeeping of the cap: the same decisions as ``cap_first_pass`` and
# ``cap_round_evict`` (e_means_balanced.py) without sorting the batch. Only the
# rows of over-full clusters are ranked: they are grouped by cluster with a
# counting sort (atomic slots, a scan of the per-cluster counts), and each
# row's rank is the number of rows of its cluster ahead of it, counted
# pairwise. "Ahead" is the sort order of the pure functions (larger margin,
# or own row then smaller score; ties by lower batch position), so the
# decisions are identical and deterministic. The work is the sum over
# over-full clusters of their squared row counts.
#
# The rows each cluster holds are counted once, before the first pass; every
# move then updates the counts of the two clusters involved, so a repair round
# never recounts the batch. A cluster's own rows (first choices it kept) do not
# change after the first pass.
# --------------------------------------------------------------------------

# Columns of FusedCap.stats, per step: pair work of the first pass, of round 1,
# of round 2, rows evicted in round 1, in round 2, rows moved by the first pass.
_STATS_COLUMNS = 6


@triton.jit
def _ordered(values):
    """Order-preserving int64 view of FP32 values (as ``_sortable``)."""
    bits = values.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)


@triton.jit
def _gather_rows_kernel(x, x_stride, index, out, norms, n, d, BLOCK_M: tl.constexpr,
                        BLOCK_D: tl.constexpr, GATHER: tl.constexpr):
    """norms[i] = ||x[index[i]]||^2 summed in FP32 (and, with GATHER, out[i] = x[index[i]]),
    in one read of the rows. Without GATHER, ``index`` is unused: row i is x[i]."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n
    if GATHER:
        source = tl.load(index + offs_m, mask=mask_m, other=0).to(tl.int64)
    else:
        source = offs_m.to(tl.int64)
    total = tl.zeros((BLOCK_M,), tl.float32)
    for d0 in range(0, d, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = mask_m[:, None] & (offs_d[None, :] < d)
        value = tl.load(x + source[:, None] * x_stride + offs_d[None, :], mask=mask, other=0.0)
        if GATHER:
            # Norms of the rows as stored (FP32 input rounded to the 16-bit copy).
            value = value.to(out.dtype.element_ty)
            tl.store(out + offs_m.to(tl.int64)[:, None] * d + offs_d[None, :], value, mask=mask)
        value = value.to(tl.float32)
        total += tl.sum(value * value, axis=1)
    tl.store(norms + offs_m, total, mask=mask_m)


def _launch_rows(x, index, out, norms, n, d, gather):
    if x.stride(1) != 1:
        raise ValueError("the rows need unit stride along d")
    block_d = min(128, max(16, triton.next_power_of_2(d)))
    with torch.cuda.device(x.device):
        _gather_rows_kernel[(triton.cdiv(n, 32),)](
            x, x.stride(0), index, out, norms, n, d, BLOCK_M=32, BLOCK_D=block_d,
            GATHER=gather, num_warps=4)


def gather_rows(x, index, out, norms):
    """``out[i] = x[index[i]]`` (contiguous ``out``) and FP32 squared row norms into ``norms``."""
    if not out.is_contiguous() or out.shape[1] != x.shape[1]:
        raise ValueError("gather_rows needs a contiguous output as wide as the rows")
    _launch_rows(x, index, out, norms, out.shape[0], out.shape[1], True)


def row_norms(rows, norms):
    """FP32 squared norms of ``rows``, summed in FP32 as the Triton assignment does."""
    _launch_rows(rows, norms, norms, norms, rows.shape[0], rows.shape[1], False)


@triton.jit
def _first_count_kernel(codes, dist, second_d, row_norm, margin, score, arrived, counts, n,
                        KEYS: tl.constexpr, BLOCK: tl.constexpr):
    """Counts per cluster; the margin and score of every row unless KEYS (given)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    code = tl.load(codes + offs, mask=mask, other=0)
    if not KEYS:
        d2 = tl.load(second_d + offs, mask=mask, other=0.0)
        tl.store(margin + offs, d2 - tl.load(dist + offs, mask=mask, other=0.0), mask=mask)
        tl.store(score + offs, d2 - tl.load(row_norm + offs, mask=mask, other=0.0), mask=mask)
    tl.store(arrived + offs, tl.zeros((BLOCK,), tl.int8), mask=mask)
    tl.atomic_add(counts + code, tl.full((BLOCK,), 1, tl.int32), mask=mask)


@triton.jit
def _group_kernel(codes, arrived, counts, cap_ptr, seglen, local, n, ARRIVALS: tl.constexpr,
                  BLOCK: tl.constexpr):
    """Atomic slot of each candidate row inside its cluster (rows of over-full clusters;
    with ARRIVALS only the rows that arrived by a move)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    code = tl.load(codes + offs, mask=mask, other=0)
    cand = mask & (tl.load(counts + code, mask=mask, other=0) > tl.load(cap_ptr))
    if ARRIVALS:
        cand = cand & (tl.load(arrived + offs, mask=mask, other=0) != 0)
    slot = tl.atomic_add(seglen + code, tl.full((BLOCK,), 1, tl.int32), mask=cand)
    tl.store(local + offs, slot, mask=cand)


@triton.jit
def _place_kernel(codes, arrived, counts, cap_ptr, starts, local, members, n,
                  ARRIVALS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    code = tl.load(codes + offs, mask=mask, other=0)
    cand = mask & (tl.load(counts + code, mask=mask, other=0) > tl.load(cap_ptr))
    if ARRIVALS:
        cand = cand & (tl.load(arrived + offs, mask=mask, other=0) != 0)
    start = tl.load(starts + code, mask=cand, other=0)
    tl.store(members + start + tl.load(local + offs, mask=cand, other=0), offs.to(tl.int32),
             mask=cand)


@triton.jit
def _rank_tile(members, tile, total, codes, key, starts, ends, BLOCK_I: tl.constexpr,
               BLOCK_J: tl.constexpr, ASCENDING: tl.constexpr):
    """(row, its cluster, rank among its cluster's candidates, segment length, mask)."""
    offs = tile * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = offs < total
    row = tl.load(members + offs, mask=mask, other=0)
    code = tl.load(codes + row, mask=mask, other=0)
    start = tl.load(starts + code, mask=mask, other=0)
    length = tl.load(ends + code, mask=mask, other=0) - start
    # Compare the order-preserving integer views, as the sort keys do (so -0.0
    # ranks before +0.0 and every tie is broken by position alone).
    mine = _ordered(tl.load(key + row, mask=mask, other=0.0))
    rank = tl.zeros((BLOCK_I,), tl.int32)
    longest = tl.max(tl.where(mask, length, 0), axis=0)
    for j0 in range(0, longest, BLOCK_J):
        jj = j0 + tl.arange(0, BLOCK_J)
        valid = mask[:, None] & (jj[None, :] < length[:, None])
        other = tl.load(members + start[:, None] + jj[None, :], mask=valid, other=0)
        theirs = _ordered(tl.load(key + other, mask=valid, other=0.0))
        if ASCENDING:
            ahead = (theirs < mine[:, None]) | ((theirs == mine[:, None]) & (other < row[:, None]))
        else:
            ahead = (theirs > mine[:, None]) | ((theirs == mine[:, None]) & (other < row[:, None]))
        rank += tl.sum((ahead & valid).to(tl.int32), axis=1)
    return row, code, rank, tl.where(mask, length, 0), mask


@triton.jit
def _emit_rows(row, hit, flags, positions, packed, counter, tally, FLAGS: tl.constexpr,
               BLOCK: tl.constexpr):
    """Flag the ``hit`` rows (compacted in batch order later), or append them at atomic
    slots of ``positions`` with their ``packed`` search word reset; count them in ``tally``."""
    count = tl.sum(hit.to(tl.int32), axis=0)
    if FLAGS:
        tl.store(flags + row, tl.full((BLOCK,), 1, tl.int32), mask=hit)
    else:
        base = tl.atomic_add(counter, count)
        slot = base + tl.cumsum(hit.to(tl.int32), axis=0) - 1
        tl.store(positions + slot, row.to(tl.int32), mask=hit)
        tl.store(packed + slot, tl.full((BLOCK,), 0x7FFFFFFFFFFFFFFF, tl.int64), mask=hit)
        tl.atomic_add(tally, count.to(tl.int64))


@triton.jit
def _move_first(row, code, moved, dist, second_d, second_l, codes, arrived, own, held,
                positions, packed, counter, tally, LAZY: tl.constexpr, BLOCK: tl.constexpr):
    """The first pass's moves: each moved row takes its runner-up, or (LAZY) is queued for
    the runner-up search; the counts of the clusters involved follow."""
    tl.store(dist + row, tl.load(second_d + row, mask=moved, other=0.0), mask=moved)
    tl.store(arrived + row, tl.full((BLOCK,), 1, tl.int8), mask=moved)
    minus = tl.full((BLOCK,), -1, tl.int32)
    tl.atomic_add(own + code, minus, mask=moved)
    tl.atomic_add(held + code, minus, mask=moved)
    if LAZY:
        _emit_rows(row, moved, positions, positions, packed, counter, tally, False, BLOCK)
    else:
        target = tl.load(second_l + row, mask=moved, other=0).to(tl.int64)
        tl.store(codes + row, target, mask=moved)
        tl.atomic_add(held + target, tl.full((BLOCK,), 1, tl.int32), mask=moved)
        tl.atomic_add(tally, tl.sum(moved.to(tl.int64), axis=0))


@triton.jit
def _first_rank_kernel(members, total_ptr, codes, margin, starts, ends, cap_ptr, dist, second_d,
                       second_l, arrived, own, held, positions, packed, counter, moved_total,
                       stats, step_ptr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
                       LAZY: tl.constexpr):
    work = stats + tl.load(step_ptr) * 6
    total = tl.load(total_ptr)
    cap = tl.load(cap_ptr)
    for tile in range(tl.program_id(0), tl.cdiv(total, BLOCK_I), tl.num_programs(0)):
        row, code, rank, length, mask = _rank_tile(members, tile, total, codes, margin, starts,
                                                   ends, BLOCK_I, BLOCK_J, False)
        moved = mask & (rank >= cap)
        _move_first(row, code, moved, dist, second_d, second_l, codes, arrived, own, held,
                    positions, packed, counter, work + 5, LAZY, BLOCK_I)
        tl.atomic_add(moved_total, tl.sum(moved.to(tl.int64), axis=0))
        tl.atomic_add(work, tl.sum(length.to(tl.int64), axis=0))


@triton.jit
def _lazy_apply_kernel(positions, count_ptr, packed, codes, held, n, BLOCK: tl.constexpr):
    """The moved rows take the runner-up the search found."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < tl.load(count_ptr)) & (offs < n)
    word = tl.load(packed + offs, mask=mask, other=0)
    row = tl.load(positions + offs, mask=mask, other=0)
    target = word & 0xFFFFFFFF
    tl.store(codes + row, target, mask=mask)
    tl.atomic_add(held + target, tl.full((BLOCK,), 1, tl.int32), mask=mask)


@triton.jit
def _round_rank_kernel(members, total_ptr, codes, score, starts, ends, own, held, cap_ptr, flags,
                       positions, packed, counter, stats, step_ptr, column,
                       BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, FLAGS: tl.constexpr):
    row_stats = stats + tl.load(step_ptr) * 6
    total = tl.load(total_ptr)
    cap = tl.load(cap_ptr)
    for tile in range(tl.program_id(0), tl.cdiv(total, BLOCK_I), tl.num_programs(0)):
        row, code, rank, length, mask = _rank_tile(members, tile, total, codes, score, starts,
                                                   ends, BLOCK_I, BLOCK_J, True)
        evict = mask & (tl.load(own + code, mask=mask, other=0) + rank >= cap)
        # An over-full cluster keeps exactly cap rows, so it stays full for the search.
        tl.atomic_add(held + code, tl.full((BLOCK_I,), -1, tl.int32), mask=evict)
        _emit_rows(row, evict, flags, positions, packed, counter, row_stats + 2 + column,
                   FLAGS, BLOCK_I)
        tl.atomic_add(row_stats + column, tl.sum(length.to(tl.int64), axis=0))


@triton.jit
def _sorted_keys_kernel(codes, arrived, key_values, keys, n, ROUND: tl.constexpr,
                        BLOCK: tl.constexpr):
    """Radix-sort keys: (cluster, larger margin first) or (cluster, own first, smaller score)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    code = tl.load(codes + offs, mask=mask, other=0).to(tl.int64)
    value = _ordered(tl.load(key_values + offs, mask=mask, other=0.0))
    if ROUND:
        late = tl.load(arrived + offs, mask=mask, other=0).to(tl.int64)
        key = (code << 34) + (late << 33) + (value + (1 << 31))
    else:
        key = (code << 34) + ((1 << 31) - value)
    tl.store(keys + offs, key, mask=mask)


@triton.jit
def _sorted_first_kernel(sorted_keys, order, starts, cap_ptr, codes, dist, second_d, second_l,
                         arrived, own, held, positions, packed, counter, moved_total, stats,
                         step_ptr, n, BLOCK: tl.constexpr, LAZY: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    code = tl.load(sorted_keys + offs, mask=mask, other=0) >> 34
    start = tl.load(starts + code, mask=mask, other=0)
    row = tl.load(order + offs, mask=mask, other=0)
    moved = mask & (offs - start >= tl.load(cap_ptr))
    _move_first(row, code, moved, dist, second_d, second_l, codes, arrived, own, held,
                positions, packed, counter, stats + tl.load(step_ptr) * 6 + 5, LAZY, BLOCK)
    tl.atomic_add(moved_total, tl.sum(moved.to(tl.int64), axis=0))


@triton.jit
def _sorted_round_kernel(sorted_keys, order, starts, held, cap_ptr, flags, positions, packed,
                         counter, stats, step_ptr, column, n, BLOCK: tl.constexpr,
                         FLAGS: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    code = tl.load(sorted_keys + offs, mask=mask, other=0) >> 34
    start = tl.load(starts + code, mask=mask, other=0)
    evict = mask & (offs - start >= tl.load(cap_ptr))
    row = tl.load(order + offs, mask=mask, other=0)
    tl.atomic_add(held + code, tl.full((BLOCK,), -1, tl.int32), mask=evict)
    _emit_rows(row, evict, flags, positions, packed, counter,
               stats + tl.load(step_ptr) * 6 + 2 + column, FLAGS, BLOCK)


@triton.jit
def _compact_kernel(flags, slots, positions, packed, stats, step_ptr, column, n,
                    BLOCK: tl.constexpr):
    """Evicted rows to positions[slot] in batch order (slots = inclusive prefix sum)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    hit = mask & (tl.load(flags + offs, mask=mask, other=0) != 0)
    slot = tl.load(slots + offs, mask=hit, other=1) - 1
    tl.store(positions + slot, offs.to(tl.int32), mask=hit)
    tl.store(packed + slot, tl.full((BLOCK,), 0x7FFFFFFFFFFFFFFF, tl.int64), mask=hit)
    count = tl.sum(hit.to(tl.int64), axis=0)
    tl.atomic_add(stats + tl.load(step_ptr) * 6 + 2 + column, count)


@triton.jit
def _round_apply_kernel(positions, count_ptr, begin, packed, codes, score, dist, row_norm,
                        arrived, held, max_rows, ADD_HELD: tl.constexpr, BLOCK: tl.constexpr):
    offs = begin + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < tl.load(count_ptr)
    word = tl.load(packed + offs, mask=mask, other=0)
    key = (word >> 32).to(tl.int32)
    best = tl.where(key < 0, key ^ 0x7FFFFFFF, key).to(tl.float32, bitcast=True)
    row = tl.load(positions + offs, mask=mask, other=0)
    target = word & 0xFFFFFFFF
    tl.store(codes + row, target, mask=mask)
    tl.store(score + row, best, mask=mask)
    tl.store(dist + row, best + tl.load(row_norm + row, mask=mask, other=0.0), mask=mask)
    tl.store(arrived + row, tl.full((BLOCK,), 1, tl.int8), mask=mask)
    if ADD_HELD:
        tl.atomic_add(held + target, tl.full((BLOCK,), 1, tl.int32), mask=mask)


@triton.jit
def _bucket_apply_kernel(positions, count_ptr, bucket, labels, distances, codes, score, dist,
                         row_norm, arrived, held, ADD_HELD: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < tl.load(count_ptr)) & (offs < bucket)
    row = tl.load(positions + offs, mask=mask, other=0)
    d1 = tl.load(distances + offs, mask=mask, other=0.0)
    target = tl.load(labels + offs, mask=mask, other=0)
    tl.store(codes + row, target, mask=mask)
    tl.store(score + row, d1 - tl.load(row_norm + row, mask=mask, other=0.0), mask=mask)
    tl.store(dist + row, d1, mask=mask)
    tl.store(arrived + row, tl.full((BLOCK,), 1, tl.int8), mask=mask)
    if ADD_HELD:
        tl.atomic_add(held + target, tl.full((BLOCK,), 1, tl.int32), mask=mask)


@triton.jit
def _held_add_kernel(positions, count_ptr, codes, held, n, BLOCK: tl.constexpr):
    """After a round's searches: each re-assigned row counts in its new cluster."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < tl.load(count_ptr)) & (offs < n)
    row = tl.load(positions + offs, mask=mask, other=0)
    tl.atomic_add(held + tl.load(codes + row, mask=mask, other=0),
                  tl.full((BLOCK,), 1, tl.int32), mask=mask)


class FusedCap:
    """Workspace and launches of the fused rounds cap for one batch size.

    Two exact ways to rank the rows of a cluster, with identical decisions:
    "pair" (group the rows of over-full clusters, count pairwise; cheap while
    those clusters are small) and "sort" (one stable radix sort of the batch;
    fixed cost). A repair round can score its first ``bucket`` evicted rows
    with a caller-given search (CuTe), the rest with ``_masked_search``.
    The first pass takes each moved row's runner-up from the assignment, or,
    without one, searches it for the moved rows only. ``stats`` (steps, 6)
    int64 records per step [pair work of the first pass, of round 1, of
    round 2, evicted rows in round 1, in round 2, moved rows], for the
    caller's choice of method, bucket and runner-up source.
    """

    BLOCK = 1024
    BLOCK_I = 64
    BLOCK_J = 64

    def __init__(self, n, k, device, steps, step_ptr):
        self.n, self.k = n, k
        self.step_ptr = step_ptr  # (1,) int64 on the device: the global step
        # [counts | seglen | counter]: one memset per pass; a round resets seglen and counter.
        self.ints = torch.zeros(2 * k + 1, dtype=torch.int32, device=device)
        self.counts, self.seglen = self.ints[:k], self.ints[k:2 * k]
        self.counter = self.ints[2 * k:2 * k + 1]
        # [own | held]: the rows each cluster keeps from its first choices, and holds now.
        self.own_held = torch.zeros(2, k, dtype=torch.int32, device=device)
        self.own, self.held = self.own_held[0], self.own_held[1]
        self.ends = torch.empty(k, dtype=torch.int32, device=device)
        self.starts = torch.empty(k, dtype=torch.int32, device=device)
        self.local = torch.empty(n, dtype=torch.int32, device=device)
        self.members = torch.empty(n, dtype=torch.int32, device=device)
        self.margin = torch.empty(n, dtype=torch.float32, device=device)
        self.score = torch.empty(n, dtype=torch.float32, device=device)
        self.arrived = torch.empty(n, dtype=torch.int8, device=device)
        self.flags = torch.zeros(n, dtype=torch.int32, device=device)
        self.slots = torch.empty(n, dtype=torch.int32, device=device)
        self.keys = torch.empty(n, dtype=torch.int64, device=device)
        self.positions = torch.arange(n, dtype=torch.int32, device=device)
        self.packed = torch.empty(n, dtype=torch.int64, device=device)
        self.rest = torch.empty(1, dtype=torch.int32, device=device)
        self.totals = torch.zeros(1, dtype=torch.int64, device=device)  # moved rows
        self.stats = torch.zeros(max(1, steps), _STATS_COLUMNS, dtype=torch.int64, device=device)
        self.grid = (triton.cdiv(n, self.BLOCK),)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.rank_grid = (max(1, min(8 * sms, triton.cdiv(n, self.BLOCK_I))),)
        self.bucket_rows = self.bucket_dist = self.bucket_labels = None

    def _bucket_buffers(self, rows):
        if self.bucket_rows is None:
            self.bucket_rows = torch.empty_like(rows)
            self.bucket_dist = torch.empty(self.n, dtype=torch.float32, device=rows.device)
            self.bucket_labels = torch.empty(self.n, dtype=torch.int64, device=rows.device)

    def _group(self, codes, counts, cap, arrivals):
        """Members of over-full clusters grouped by cluster; returns (starts, total) views."""
        n, grid, block = self.n, self.grid, self.BLOCK
        _group_kernel[grid](codes, self.arrived, counts, cap, self.seglen, self.local, n,
                            ARRIVALS=arrivals, BLOCK=block)
        torch.cumsum(self.seglen, 0, dtype=torch.int32, out=self.ends)
        torch.sub(self.ends, self.seglen, out=self.starts)
        _place_kernel[grid](codes, self.arrived, counts, cap, self.starts, self.local,
                            self.members, n, ARRIVALS=arrivals, BLOCK=block)
        return self.starts, self.ends[-1:]

    def _sorted(self, codes, values, counts, round_):
        """Stable radix sort of the batch by cluster then rank key; ``starts`` gets each
        cluster's first sorted position (from ``counts``, read before any update)."""
        _sorted_keys_kernel[self.grid](codes, self.arrived, values, self.keys, self.n,
                                       ROUND=round_, BLOCK=self.BLOCK)
        torch.cumsum(counts, 0, dtype=torch.int32, out=self.ends)
        torch.sub(self.ends, counts, out=self.starts)
        return torch.sort(self.keys, stable=True)

    def work(self, cap, column):
        """Pairwise work the "pair" ranking would do now (sum of squared candidate counts),
        added to this step's ``column`` of ``stats``. Call before ranking."""
        if column == 0:
            candidates, over = self.counts, self.counts > cap
        else:
            candidates, over = self.held - self.own, self.held > cap
        work = torch.where(over, candidates.long() * candidates.long(), 0).sum()
        self.stats.view(-1).index_add_(0, self.step_ptr * _STATS_COLUMNS + column,
                                       work.reshape(1))

    def first_pass(self, codes, dist, second_d, second_l, row_norm, cap, method="pair",
                   rows=None, mirror=None, norms=None, keys=False):
        """In place on ``codes`` and ``dist``; fills the score and arrived buffers.

        ``second_l`` None: the moved rows' runner-ups are searched here, among the
        biased ``norms`` of ``mirror`` for the batch ``rows``. ``keys``: the caller
        has filled ``margin`` and ``score`` (``row_norm`` is then unused).
        """
        n, grid, block = self.n, self.grid, self.BLOCK
        lazy = second_l is None
        self.ints.zero_()
        _first_count_kernel[grid](codes, dist, second_d, row_norm, self.margin, self.score,
                                  self.arrived, self.counts, n, KEYS=keys, BLOCK=block)
        self.own_held.copy_(self.counts.expand(2, -1))
        labels = codes if lazy else second_l
        if method == "sort":
            self.work(cap, 0)
            sorted_keys, order = self._sorted(codes, self.margin, self.counts, False)
            _sorted_first_kernel[grid](sorted_keys, order, self.starts, cap, codes, dist,
                                       second_d, labels, self.arrived, self.own, self.held,
                                       self.positions, self.packed, self.counter,
                                       self.totals[0:1], self.stats, self.step_ptr, n,
                                       BLOCK=block, LAZY=lazy)
        else:
            starts, total = self._group(codes, self.counts, cap, False)
            _first_rank_kernel[self.rank_grid](
                self.members, total, codes, self.margin, starts, self.ends, cap, dist, second_d,
                labels, self.arrived, self.own, self.held, self.positions, self.packed,
                self.counter, self.totals[0:1], self.stats, self.step_ptr,
                BLOCK_I=self.BLOCK_I, BLOCK_J=self.BLOCK_J, LAZY=lazy)
        if lazy:
            _masked_search(rows, self.positions, self.counter, mirror, norms, None, self.packed,
                           n, exclude=codes)
            _lazy_apply_kernel[grid](self.positions, self.counter, self.packed, codes, self.held,
                                     n, BLOCK=block)

    def full_mask(self, cap):
        """(k,) bool: the clusters a repair round's search masks out (holding >= cap rows)."""
        return self.held >= cap

    def round(self, codes, dist, row_norm, cap, rows, mirror, norms, index=0, method="pair",
              bucket=0, bucket_search=None, search=None, ordered=False):
        """One repair round, in place on ``codes`` and ``dist``. Rounds past the second
        record their statistics in the second round's columns. ``ordered`` lists the
        evicted rows in batch order (the same list in every process of a group)."""
        n, grid, block = self.n, self.grid, self.BLOCK
        column = 1 + min(index, 1)
        self.ints[self.k:].zero_()
        # A bucket splits the evicted rows between two searches by slot: slots must then
        # follow the batch order (prefix sum), not the order the atomics happened in.
        flags = ordered or (bucket > 0 and bucket_search is not None)
        if flags:
            self.flags.zero_()
        if method == "sort":
            self.work(cap, column)
            sorted_keys, order = self._sorted(codes, self.score, self.held, True)
            _sorted_round_kernel[grid](sorted_keys, order, self.starts, self.held, cap,
                                       self.flags, self.positions, self.packed, self.counter,
                                       self.stats, self.step_ptr, column, n, BLOCK=block,
                                       FLAGS=flags)
        else:
            starts, total = self._group(codes, self.held, cap, True)
            _round_rank_kernel[self.rank_grid](
                self.members, total, codes, self.score, starts, self.ends, self.own, self.held,
                cap, self.flags, self.positions, self.packed, self.counter, self.stats,
                self.step_ptr, column, BLOCK_I=self.BLOCK_I, BLOCK_J=self.BLOCK_J,
                FLAGS=flags)
        if flags:
            torch.cumsum(self.flags, 0, dtype=torch.int32, out=self.slots)
            _compact_kernel[grid](self.flags, self.slots, self.positions, self.packed,
                                  self.stats, self.step_ptr, column, n, BLOCK=block)
            count = self.slots[-1:]
        else:
            count = self.counter
        positions = self.positions
        begin = 0
        if bucket < 0 and bucket_search is not None:
            # Exact: read the evicted count (synchronizes) and let the caller's search score
            # exactly those rows (it pads and splits them into a few fixed shapes).
            evicted = int(count)
            if evicted:
                self._bucket_buffers(rows)
                bucket_search(rows, positions, evicted, self.held, cap,
                              self.bucket_dist, self.bucket_labels, self.bucket_rows)
                _bucket_apply_kernel[(triton.cdiv(evicted, block),)](
                    positions, count, evicted, self.bucket_labels, self.bucket_dist, codes,
                    self.score, dist, row_norm, self.arrived, self.held, ADD_HELD=True,
                    BLOCK=block)
            return
        if bucket > 0 and bucket_search is not None:
            bucket = min(bucket, n)
            self._bucket_buffers(rows)
            torch.index_select(rows, 0, positions[:bucket], out=self.bucket_rows[:bucket])
            bucket_search(self.bucket_rows[:bucket], self.held, cap,
                          self.bucket_dist[:bucket], self.bucket_labels[:bucket])
            _bucket_apply_kernel[(triton.cdiv(bucket, block),)](
                positions, count, bucket, self.bucket_labels, self.bucket_dist, codes,
                self.score, dist, row_norm, self.arrived, self.held, ADD_HELD=False,
                BLOCK=block)
            torch.clamp(count - bucket, min=0, out=self.rest)
            begin = bucket
            rest = self.rest
        else:
            rest = count
        if search is not None:
            # Caller's search (multi-GPU): fills packed[:count] as _masked_search would.
            search(positions[begin:], rest, self.packed[begin:])
        else:
            _masked_search(rows, positions[begin:], rest, mirror, norms, None,
                           self.packed[begin:], n - begin, held=self.held, cap=cap)
        # Every search of the round saw the same full clusters; only now do the
        # re-assigned rows count in their new clusters.
        _round_apply_kernel[(triton.cdiv(n - begin, block),)](
            positions, count, begin, self.packed, codes, self.score, dist, row_norm,
            self.arrived, self.held, n, ADD_HELD=begin == 0, BLOCK=block)
        if begin:
            _held_add_kernel[grid](positions, count, codes, self.held, n, BLOCK=block)


# --------------------------------------------------------------------------
# Repair-round search split over the processes of a data-parallel group: each
# process holds a slice of the global batch, searches the evicted rows of its
# slice and writes their answers at their slots; an all-reduce (minimum) of the
# (score, label) words then gives every process every answer.
# --------------------------------------------------------------------------

@triton.jit
def _own_rows_kernel(positions, count_ptr, lo, local_n, index, local_rows, local_slots,
                     local_packed, local_count, n, HAS_INDEX: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < tl.load(count_ptr)) & (offs < n)
    pos = tl.load(positions + offs, mask=mask, other=0) - lo
    own = mask & (pos >= 0) & (pos < local_n)
    if HAS_INDEX:
        row = tl.load(index + pos, mask=own, other=0)
    else:
        row = pos
    base = tl.atomic_add(local_count, tl.sum(own.to(tl.int32), axis=0))
    slot = base + tl.cumsum(own.to(tl.int32), axis=0) - 1
    tl.store(local_rows + slot, row.to(tl.int32), mask=own)
    tl.store(local_slots + slot, offs.to(tl.int32), mask=own)
    tl.store(local_packed + slot, tl.full((BLOCK,), 0x7FFFFFFFFFFFFFFF, tl.int64), mask=own)


@triton.jit
def _scatter_packed_kernel(local_slots, local_packed, local_count, packed, local_n,
                           BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < tl.load(local_count)) & (offs < local_n)
    slot = tl.load(local_slots + offs, mask=mask, other=0)
    tl.store(packed + slot, tl.load(local_packed + offs, mask=mask, other=0), mask=mask)


class GroupSearch:
    """Buffers and launches of the split repair-round search for one local batch size."""

    BLOCK = 1024

    def __init__(self, local_n, n, device):
        self.local_n, self.n = local_n, n
        self.rows = torch.empty(local_n, dtype=torch.int32, device=device)
        self.slots = torch.empty(local_n, dtype=torch.int32, device=device)
        self.packed = torch.empty(local_n, dtype=torch.int64, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)

    def search(self, positions, count, packed, lo, rows, index, mirror, norms, held, cap):
        """Answers for this process's rows among ``positions[:count]`` (global batch
        positions; ``rows[index[p - lo]]`` or ``rows[p - lo]`` is row p) into ``packed``."""
        self.count.zero_()
        with torch.cuda.device(rows.device):
            _own_rows_kernel[(triton.cdiv(self.n, self.BLOCK),)](
                positions, count, lo, self.local_n, index if index is not None else positions,
                self.rows, self.slots, self.packed, self.count, self.n,
                HAS_INDEX=index is not None, BLOCK=self.BLOCK)
        _masked_search(rows, self.rows, self.count, mirror, norms, None, self.packed,
                       self.local_n, held=held, cap=cap)
        with torch.cuda.device(rows.device):
            _scatter_packed_kernel[(triton.cdiv(self.local_n, self.BLOCK),)](
                self.slots, self.packed, self.count, packed, self.local_n, BLOCK=self.BLOCK)


# --------------------------------------------------------------------------
# Fused batch distance, counts and bias step (the exact band).
# --------------------------------------------------------------------------

@triton.jit
def _batch_distance_kernel(dist, codes, bias, shift_ptr, batch_sum, epoch_sum, counts, n,
                           COUNT: tl.constexpr, EPOCH: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    code = tl.load(codes + offs, mask=mask, other=0)
    true = (tl.load(dist + offs, mask=mask, other=0.0) - tl.load(shift_ptr)
            + tl.load(bias + code, mask=mask, other=0.0))
    total = tl.sum(tl.where(mask, true, 0.0).to(tl.float64), axis=0)
    tl.atomic_add(batch_sum, total)
    if EPOCH:
        tl.atomic_add(epoch_sum, total)
    if COUNT:
        tl.atomic_add(counts + code, tl.full((BLOCK,), 1.0, tl.float32), mask=mask)


@triton.jit
def _band_kernel(counts, counts_stride, lam, mu, bias, distance, distance_next, batch_sum, inv_n,
                 scale_ptr, u, share, rate, k, FIRST: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < k
    mean = (tl.load(batch_sum) * inv_n).to(tl.float32)
    if FIRST:
        d_new = mean
    else:
        d_new = tl.load(distance) * (1.0 - rate) + rate * mean
    eta = (u * d_new) * tl.load(scale_ptr)
    m = tl.load(counts + offs * counts_stride, mask=mask, other=0.0)
    relative = (m - share) / share
    lam_new = tl.maximum(tl.load(lam + offs, mask=mask, other=0.0) + eta * relative, 0.0)
    mu_new = tl.maximum(tl.load(mu + offs, mask=mask, other=0.0) - eta * relative, 0.0)
    tl.store(lam + offs, lam_new, mask=mask)
    tl.store(mu + offs, mu_new, mask=mask)
    tl.store(bias + offs, mu_new - lam_new, mask=mask)
    if pid == 0:
        tl.store(distance_next, d_new)


class FusedBias:
    """Batch distance sum, polish counts and the band step in three launches."""

    BLOCK = 1024

    def __init__(self, n, k, device, share_rows=None):
        self.n, self.k = n, k
        # The band's fair share and the mean distance use the global batch (data parallel).
        self.share_rows = n if share_rows is None else int(share_rows)
        self.batch_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.distance_next = torch.zeros((), dtype=torch.float32, device=device)
        self.counts = torch.zeros(k, dtype=torch.float32, device=device)

    def distances(self, dist, codes, bias, shift, epoch_sum, count):
        """Sum of dist - shift + bias[codes] into ``batch_sum`` (and the epoch's
        FP64 sum); with ``count``, per-cluster counts into ``counts``."""
        self.batch_sum.zero_()
        if count:
            self.counts.zero_()
        _batch_distance_kernel[(triton.cdiv(self.n, self.BLOCK),)](
            dist, codes, bias, shift, self.batch_sum,
            self.batch_sum if epoch_sum is None else epoch_sum, self.counts, self.n,
            COUNT=count, EPOCH=epoch_sum is not None, BLOCK=self.BLOCK)
        return self.batch_sum

    def band(self, counts, lam, mu, bias, distance, scale, u, first, rate):
        """lam, mu, b <- the exact band step; ``distance`` (D) updated in place."""
        _band_kernel[(triton.cdiv(self.k, self.BLOCK),)](
            counts, counts.stride(0), lam, mu, bias, distance, self.distance_next,
            self.batch_sum, 1.0 / self.share_rows, scale, float(u),
            float(self.share_rows) / self.k, float(rate),
            self.k, FIRST=first, BLOCK=self.BLOCK)
        distance.copy_(self.distance_next)
