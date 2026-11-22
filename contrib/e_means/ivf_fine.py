# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""IVF fine search and exact probe selection for IndexIVFFlat.

search_2nn returns, for each row, the two best candidates of its probed
lists: score = norm - 2 * inner over the represented operands (FP32
accumulation), best = smallest score with ties to the lower ID, second =
smallest score among the other IDs, and score inf / ID -1 for missing entries
(non-finite scores, from overflowing or NaN inputs, count as missing). On CUDA,
routed (row, probe) pairs are grouped by list with a stable device sort; a
work item covers up to 64 (Triton) or 128 (CuTe) pairs of one list and scans
the whole list in candidate tiles, accumulating 16-wide contraction steps in
order; one merge over the nprobe partial results gives each row its two
candidates. Pair results depend neither on the filing order
nor on the tiling, so the CuTe SM90 kernel (ivf_fine_cute, BF16 operands, used
with its required CuTe/quack dependencies) and the Triton kernel return identical
bits; the Triton kernel serves other GPUs and FP16 / FP32 operands. Other
devices and dtypes use the same search written with PyTorch operations.

``route`` keeps the coarse scores of IndexIVFFlat.route (the same matmul per
query chunk, dis = norm - 2 * dot) and replaces torch.topk with an exact radix
select, one kernel for any nlist and nprobe: four passes over each row's
scores fix the nprobe-th score one 8-bit digit at a time, and a last pass
writes the selected lists (rows of up to 2,048 lists stay in registers across
the passes; longer rows are read in chunks of 512 lists per pass). Each row gets the same set of nprobe
lists as torch.topk, the lower list ID winning equal scores at the boundary,
ordered by score, then by list ID.
"""

from functools import lru_cache

import torch

# None tunes the CuTe candidate-tile width per shape class; 128 or 256 fixes it.
DEFAULT_TILE = None
_IMAX = 0x7FFFFFFFFFFFFFFF
_ROUTE_RESIDENT = 2048  # up to this many lists, the selection keeps each row's scores in registers
_ROUTE_CHUNK = 512  # otherwise it reads them in chunks of this many lists per pass


@lru_cache(maxsize=1)
def _triton_kernels():
    try:
        import triton
        from .ivf_fine_triton import _search_kernel, _merge_kernel, _select_kernel
    except ImportError as exc:
        raise ImportError("IVF CUDA search and routing require Triton; install it in the selected environment") from exc
    return triton, _search_kernel, _merge_kernel, _select_kernel


def route_eligible(index, x):
    """CUDA rows (CPU routing keeps torch.topk)."""
    return x.is_cuda


def route(index, x):
    """IndexIVFFlat.route's probe lists, selected without torch.topk."""
    with torch.cuda.device(x.device):
        return _route(index, x)


def _route(index, x):
    from .ivf_flat import _dot_fp32

    triton, _, _, _select_kernel = _triton_kernels()

    nq = x.shape[0]
    P, nl = int(index.nprobe), int(index.nlist)
    norms = index._current_routing_norms()
    dt, xb_t = index._lowp_operands(index.centroids, "coarse")
    chunk = index.query_chunk if index.implem >= 2 else nq
    resident = nl <= _ROUTE_RESIDENT
    CT, BR = (triton.next_power_of_2(nl), 2) if resident else (_ROUTE_CHUNK, 1)
    cols = torch.empty((nq, P), dtype=torch.int32, device=x.device)
    keys = torch.empty((nq, P), dtype=torch.int32, device=x.device)
    for i0 in range(0, nq, max(1, chunk)):
        i1 = min(i0 + chunk, nq)
        rows = i1 - i0
        dot = _dot_fp32(x[i0:i1] if dt is None else x[i0:i1].to(dt), xb_t)
        if dot.stride(1) != 1:
            dot = dot.contiguous()
        _select_kernel[(triton.cdiv(rows, BR),)](dot, norms, cols[i0:i1], keys[i0:i1], rows, nl, dot.stride(0), P,
                                                 CT=CT, BLOCK_R=BR, RESIDENT=resident, num_warps=4, enable_fp_fusion=False)
    # Nearest first: back to the scores (inverse of the key map), then a stable sort keeps lower list IDs first.
    scores = torch.where(keys < 0, keys & 0x7FFFFFFF, ~keys).view(torch.float32)
    order = torch.sort(scores, dim=1, stable=True).indices
    return torch.gather(cols, 1, order).long()


def _file_pairs(index, coarse_idx, BM):
    """Group the routed pairs by list and cut each list's pairs into work items, without host reads.

    The item tables cover an upper bound of item slots; slots past the
    device-side total have count 0. Returns (item_list, item_start,
    item_count, pair_row, pair_id, slots, total).
    """
    n, P = coarse_idx.shape
    device = coarse_idx.device
    nl = int(index.nlist)
    lists = coarse_idx.reshape(-1)
    # A stable device sort groups the pairs by list for any number of lists; list bounds come from a search.
    sorted_lists, order = torch.sort(lists.int(), stable=True)
    bounds = torch.searchsorted(sorted_lists, torch.arange(nl + 1, dtype=torch.int32, device=device))
    starts, counts = bounds[:-1], bounds[1:] - bounds[:-1]
    pair_id, pair_row = order.int(), (order // P).int()
    per = (counts + BM - 1) // BM
    cum = torch.cumsum(per, 0)
    slots = (n * P + BM - 1) // BM + nl
    i = torch.arange(slots, device=device)
    owner = torch.searchsorted(cum, i, right=True)
    valid = owner < nl
    item_list = owner.clamp(max=nl - 1)
    local = i - (cum - per)[item_list]
    item_start = torch.where(valid, starts[item_list] + local * BM, 0)
    item_count = torch.where(valid, torch.minimum(counts[item_list] - local * BM, torch.full_like(local, BM)), 0)
    return item_list, item_start, item_count, pair_row, pair_id, slots, cum[-1:]


def _merge(p_sc, p_id, n, P):
    triton, _, _merge_kernel, _ = _triton_kernels()
    scores = torch.empty((n, 2), dtype=torch.float32, device=p_sc.device)
    ids = torch.empty((n, 2), dtype=torch.int32, device=p_sc.device)
    BP = triton.next_power_of_2(2 * P)
    BR = max(1, min(64, 1024 // BP))
    _merge_kernel[(triton.cdiv(n, BR),)](p_sc, p_id, scores, ids, n, P, BR=BR, BP=BP, num_warps=4)
    return scores, ids


def _partials(n, P, device):
    return (torch.full((n * P, 2), torch.inf, dtype=torch.float32, device=device),
            torch.full((n * P, 2), _IMAX, dtype=torch.int64, device=device))


def _operands(index, x):
    """Represented operands of the fine search: (queries, list storage)."""
    fine = index._mm_dtype("fine")
    vecs = index.vectors_lowp if index.vectors_lowp is not None else index.vectors
    return (x if fine is None else x.to(fine)), vecs


def _search_triton(index, xq, vecs, coarse_idx):
    triton, _search_kernel, _, _ = _triton_kernels()
    n, d = xq.shape
    P = int(coarse_idx.shape[1])
    ieee = vecs.dtype == torch.float32
    BM, BN, BK, stages = (64, 64, 32, 3) if ieee else (64, 128, 64, 4)
    if torch.version.hip:
        stages = 2
    xq = xq.contiguous()
    vecs = vecs.contiguous()
    lims = index.xb_lims.to(torch.int64).contiguous()
    item_list, item_start, item_count, pair_row, pair_id, slots, _ = _file_pairs(index, coarse_idx, BM)
    p_sc, p_id = _partials(n, P, xq.device)
    if slots:
        _search_kernel[(slots,)](xq, vecs, index.vectors_norms, index.ids, lims, pair_row, pair_id, item_list,
                                 item_start, item_count, p_sc, p_id, d, xq.stride(0), vecs.stride(0),
                                 BM=BM, BN=BN, BK=BK, IEEE=ieee, num_warps=4, num_stages=stages)
    return _merge(p_sc, p_id, n, P)


def _top2_merge(scores, ids, new_scores, new_ids):
    """Best two (score, ID) of each row over the running pair and new entries; lower IDs first among equal scores."""
    scores, ids = torch.cat((scores, new_scores), 1), torch.cat((ids, new_ids), 1)
    by_id = torch.argsort(ids, dim=1, stable=True)
    scores, ids = scores.gather(1, by_id), ids.gather(1, by_id)
    first = torch.sort(scores, dim=1, stable=True).indices[:, :1]
    first_score, first_id = scores.gather(1, first), ids.gather(1, first)
    others = torch.where(ids == first_id, torch.inf, scores)  # a list probed twice repeats its candidates
    second = torch.sort(others, dim=1, stable=True).indices[:, :1]
    return torch.cat((first_score, others.gather(1, second)), 1), torch.cat((first_id, ids.gather(1, second)), 1)


def _search_torch(index, xq, vecs, coarse_idx, block=1 << 22):
    """The same search with PyTorch operations (CPU and other dtypes): one probed list at a time, in blocks of at most
    ``block`` scores, keeping a running top-2 per routed pair."""
    from .ivf_flat import _dot_fp32

    n, P = coarse_idx.shape
    device = xq.device
    pairs = coarse_idx.reshape(-1)
    order = torch.argsort(pairs, stable=True)
    counts = torch.bincount(pairs, minlength=int(index.nlist)).tolist()
    lims = index.xb_lims.tolist()
    p_sc = torch.full((n * P, 2), torch.inf, dtype=torch.float32, device=device)
    p_id = torch.full((n * P, 2), _IMAX, dtype=torch.int64, device=device)
    start = 0
    for lst, count in enumerate(counts):
        if not count:
            continue
        sel = order[start:start + count]
        start += count
        b0, b1 = lims[lst], lims[lst + 1]
        if b1 == b0:
            continue
        cand = min(b1 - b0, 4096)
        rows = max(1, block // cand)
        for r0 in range(0, count, rows):
            part = sel[r0:r0 + rows]
            q = xq[part // P]
            best = torch.full((len(part), 2), torch.inf, dtype=torch.float32, device=device)
            best_id = torch.full((len(part), 2), _IMAX, dtype=torch.int64, device=device)
            for c0 in range(b0, b1, cand):
                c1 = min(c0 + cand, b1)
                dis = (index.vectors_norms[c0:c1][None] - 2 * _dot_fp32(q, vecs[c0:c1].T)).float()
                dis = torch.where(torch.isfinite(dis), dis, torch.inf)  # non-finite: missing
                top = min(2, c1 - c0)
                values, cols = torch.sort(dis, dim=1, stable=True)  # positions follow IDs within a list
                best, best_id = _top2_merge(best, best_id, values[:, :top], index.ids[c0:c1].long()[cols[:, :top]])
            p_sc[part], p_id[part] = best, best_id
    # Merge the 2 * nprobe entries of each row.
    scores, ids = p_sc.view(n, 2 * P), p_id.view(n, 2 * P)
    scores, ids = _top2_merge(scores[:, :0], ids[:, :0], scores, ids)
    ids = torch.where(torch.isfinite(scores) & (ids != _IMAX), ids, -1)
    return scores, ids.int()


@lru_cache(maxsize=None)
def _cute_ready(device_index):
    """Select the SM90 route by capability and require its CuTe dependencies."""
    if torch.cuda.get_device_capability(device_index) != (9, 0):
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from quack import copy_utils, layout_utils, sm90_utils  # noqa: F401
        from quack import pipeline as quack_pipeline  # noqa: F401
        from quack.compile_utils import make_fake_tensor  # noqa: F401
    except ImportError as exc:
        raise ImportError("IVF BF16 search on SM90 requires CuTe and quack; install their supported versions") from exc
    return True


def search_2nn(index, x, coarse_idx, tile=None):
    """Scores (n, 2) FP32 and IDs (n, 2) int32 of the two best candidates of each row of x in its routed lists.

    ``tile`` (128 or 256) fixes the CuTe candidate-tile width; None uses
    DEFAULT_TILE, tuning it once per shape class when that is None too.
    """
    if x.is_cuda:
        with torch.cuda.device(x.device):  # kernels launch on the current device
            return _search_2nn(index, x, coarse_idx, tile)
    return _search_2nn(index, x, coarse_idx, tile)


def _search_2nn(index, x, coarse_idx, tile=None):
    if not len(x) or not len(index.ids):  # no rows, or no candidates anywhere
        return (torch.full((len(x), 2), torch.inf, dtype=torch.float32, device=x.device),
                torch.full((len(x), 2), -1, dtype=torch.int32, device=x.device))
    xq, vecs = _operands(index, x)
    kernel = (xq.is_cuda and xq.dtype == vecs.dtype
              and vecs.dtype in (torch.bfloat16, torch.float16, torch.float32)
              and index.vectors_norms.dtype == torch.float32)
    if not kernel:
        return _search_torch(index, xq, vecs, coarse_idx)
    if vecs.dtype == torch.bfloat16 and _cute_ready(xq.device.index if xq.device.index is not None
                                                    else torch.cuda.current_device()):
        from .ivf_fine_cute import search_2nn as cute_search_2nn

        return cute_search_2nn(index, xq, coarse_idx, tile if tile is not None else DEFAULT_TILE)
    return _search_triton(index, xq, vecs, coarse_idx)
