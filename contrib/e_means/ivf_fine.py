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
when its optional dependencies import) and the Triton kernel return identical
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

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CUDA builds ship Triton
    triton = tl = None

# None tunes the CuTe candidate-tile width per shape class; 128 or 256 fixes it.
DEFAULT_TILE = None
_IMAX = 0x7FFFFFFFFFFFFFFF
_ROUTE_RESIDENT = 2048  # up to this many lists, the selection keeps each row's scores in registers
_ROUTE_CHUNK = 512  # otherwise it reads them in chunks of this many lists per pass


if triton is not None:

    @triton.jit
    def _search_kernel(x16, vecs, norms, ids, lims, pair_row, pair_out, item_list, item_start, item_count,
                       o_sc, o_id, d, sx, sv, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       IEEE: tl.constexpr):
        """One work item: up to BM pairs of one list against the whole list."""
        imax = 0x7FFFFFFFFFFFFFFF
        it = tl.program_id(0)
        L = tl.load(item_list + it)
        p0 = tl.load(item_start + it)
        pc = tl.load(item_count + it)
        rm = tl.arange(0, BM)
        live = rm < pc
        prow = tl.load(pair_row + p0 + rm, mask=live, other=0).to(tl.int64)
        v0 = tl.load(lims + L).to(tl.int64)
        v1 = tl.load(lims + L + 1).to(tl.int64)
        v1 = tl.where(pc > 0, v1, v0)  # unused slots of the upper-bound grid
        best = tl.full([BM], float("inf"), tl.float32)
        best_id = tl.full([BM], imax, tl.int64)
        second = tl.full([BM], float("inf"), tl.float32)
        second_id = tl.full([BM], imax, tl.int64)
        for c0 in range(v0, v1, BN):
            cn = c0 + tl.arange(0, BN)
            cl = cn < v1
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k0 in range(0, d, BK):
                rk = k0 + tl.arange(0, BK)
                km = rk < d
                a = tl.load(x16 + prow[:, None] * sx + rk[None, :], mask=live[:, None] & km[None, :], other=0.0)
                b = tl.load(vecs + cn[:, None] * sv + rk[None, :], mask=cl[:, None] & km[None, :], other=0.0)
                if IEEE:  # FP32 operands: full-precision products, no TF32
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
            nm = tl.load(norms + cn, mask=cl, other=float("inf"))
            cid = tl.load(ids + cn, mask=cl, other=imax).to(tl.int64)
            score = nm[None, :] - 2.0 * acc
            score = tl.where(cl[None, :] & (tl.abs(score) < float("inf")), score, float("inf"))  # non-finite: missing
            m1 = tl.min(score, axis=1)
            i1 = tl.min(tl.where((score == m1[:, None]) & (tl.abs(score) < float("inf")), cid[None, :], imax), axis=1)
            s2 = tl.where(cid[None, :] == i1[:, None], float("inf"), score)
            m2 = tl.min(s2, axis=1)
            i2 = tl.min(tl.where((s2 == m2[:, None]) & (tl.abs(s2) < float("inf")), cid[None, :], imax), axis=1)
            # update_probes' merge of [tile first, tile second, old first, old second]; IDs differ across tiles.
            nb = tl.minimum(tl.minimum(m1, m2), tl.minimum(best, second))
            nbid = tl.minimum(tl.minimum(tl.where(m1 == nb, i1, imax), tl.where(m2 == nb, i2, imax)),
                              tl.minimum(tl.where(best == nb, best_id, imax), tl.where(second == nb, second_id, imax)))
            e1 = tl.where(i1 == nbid, float("inf"), m1)
            e2 = tl.where(i2 == nbid, float("inf"), m2)
            e3 = tl.where(best_id == nbid, float("inf"), best)
            e4 = tl.where(second_id == nbid, float("inf"), second)
            ot = tl.minimum(tl.minimum(e1, e2), tl.minimum(e3, e4))
            otid = tl.minimum(
                tl.minimum(tl.where((e1 == ot) & (i1 != nbid), i1, imax), tl.where((e2 == ot) & (i2 != nbid), i2, imax)),
                tl.minimum(tl.where((e3 == ot) & (best_id != nbid), best_id, imax),
                           tl.where((e4 == ot) & (second_id != nbid), second_id, imax)))
            best, best_id, second, second_id = nb, nbid, ot, otid
        po = tl.load(pair_out + p0 + rm, mask=live, other=0).to(tl.int64)
        tl.store(o_sc + po * 2, best, mask=live)
        tl.store(o_id + po * 2, best_id, mask=live)
        tl.store(o_sc + po * 2 + 1, second, mask=live)
        tl.store(o_id + po * 2 + 1, second_id, mask=live)

    @triton.jit
    def _merge_kernel(p_sc, p_id, scores, out_ids, n, P, BR: tl.constexpr, BP: tl.constexpr):
        """merge_probes over the 2 * nprobe (score, ID) entries of each row."""
        imax = 0x7FFFFFFFFFFFFFFF
        row = tl.program_id(0) * BR + tl.arange(0, BR)
        lane = tl.arange(0, BP)
        live = (row < n)[:, None] & (lane < 2 * P)[None, :]
        off = row[:, None].to(tl.int64) * (2 * P) + lane[None, :]
        value = tl.load(p_sc + off, mask=live, other=float("inf"))
        cand = tl.load(p_id + off, mask=live, other=imax)
        first = tl.min(value, axis=1)
        fid = tl.min(tl.where(value == first[:, None], cand, imax), axis=1)
        second = tl.min(tl.where(cand == fid[:, None], float("inf"), value), axis=1)
        sid = tl.min(tl.where((value == second[:, None]) & (cand != fid[:, None]), cand, imax), axis=1)
        rr = row < n
        out = row.to(tl.int64) * 2
        tl.store(scores + out, first, mask=rr)
        tl.store(scores + out + 1, second, mask=rr)
        tl.store(out_ids + out, tl.where(fid == imax, -1, fid).to(tl.int32), mask=rr)
        tl.store(out_ids + out + 1, tl.where(sid == imax, -1, sid).to(tl.int32), mask=rr)


    @triton.jit
    def _keys(dot, norms, r64, m, col, NCOL, s_d):
        """Order-preserving uint32 images of dis = norm - 2 * dot for one chunk of columns, and the valid lanes."""
        cm = col < NCOL
        mask = m[:, None] & cm[None, :]
        d = tl.load(dot + r64[:, None] * s_d + col[None, :], mask=mask, other=0.0)
        nrm = tl.load(norms + col, mask=cm, other=float("inf"))
        bits = (nrm[None, :] - 2.0 * d).to(tl.uint32, bitcast=True)
        return tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000), mask

    @triton.jit
    def _count(key, mask, prefix, pmask, shift, rl, BLOCK_R: tl.constexpr, CT: tl.constexpr):
        """Per-row counts of the 8-bit digit at ``shift`` among the keys that match the prefix found so far."""
        match = mask & ((key & pmask[:, None]) == prefix[:, None])
        digit = ((key >> shift) & 255).to(tl.int32)
        binid = tl.where(match, rl[:, None] * 256 + digit, BLOCK_R * 256)
        hist = tl.histogram(tl.reshape(binid, (BLOCK_R * CT,)), 2 * BLOCK_R * 256)
        hh = tl.reshape(hist, (2, BLOCK_R, 256))
        return tl.sum(tl.where(tl.arange(0, 2)[:, None, None] == 0, hh, 0), axis=0)

    @triton.jit
    def _pick(h, krem, prefix, pmask, shift, BLOCK_R: tl.constexpr):
        """Fix the digit of the krem-th key; krem becomes its rank among the keys with the longer prefix."""
        d256 = tl.arange(0, 256)
        cum = tl.cumsum(h, axis=1)
        dsel = tl.min(tl.where(cum >= krem[:, None], d256[None, :], 256), axis=1)
        krem = krem - tl.sum(tl.where(d256[None, :] < dsel[:, None], h, 0), axis=1)
        prefix = prefix | (dsel.to(tl.uint32) << shift)
        pmask = pmask | (tl.full((BLOCK_R,), 255, tl.uint32) << shift)
        return krem, prefix, pmask

    @triton.jit
    def _emit(key, mask, col, prefix, krem, eq_seen, sel_seen, out_col, out_key, r64, P,
              BLOCK_R: tl.constexpr, CT: tl.constexpr):
        """Write the selected columns of one chunk after those of the previous chunks, in column order."""
        less = mask & (key < prefix[:, None])
        eq = (mask & (key == prefix[:, None])).to(tl.int32)
        eq_rank = eq_seen[:, None] + tl.cumsum(eq, axis=1) - eq
        sel = (less | ((eq != 0) & (eq_rank < krem[:, None]))).to(tl.int32)
        pos = sel_seen[:, None] + tl.cumsum(sel, axis=1) - sel
        keep = (sel != 0) & (pos < P)
        dst = r64[:, None] * P + pos
        tl.store(out_col + dst, col[None, :] + tl.zeros((BLOCK_R, CT), tl.int32), mask=keep)
        tl.store(out_key + dst, key.to(tl.int32, bitcast=True), mask=keep)
        return eq_seen + tl.sum(eq, axis=1), sel_seen + tl.sum(sel, axis=1)

    @triton.jit
    def _select_kernel(dot, norms, out_col, out_key, R, NCOL, s_d, P, CT: tl.constexpr, BLOCK_R: tl.constexpr,
                       RESIDENT: tl.constexpr):
        """The P smallest keys of each row, lower columns first among equal keys, for any NCOL and P.

        Four counting passes over the row, in CT-wide column chunks, fix the P-th key one 8-bit digit at a time; a
        last pass writes the selected columns and keys in column order. RESIDENT (NCOL <= CT): the keys are loaded
        once and kept in registers across the passes.
        """
        rl = tl.arange(0, BLOCK_R)
        r = tl.program_id(0) * BLOCK_R + rl
        m = r < R
        r64 = r.to(tl.int64)
        prefix = tl.zeros((BLOCK_R,), tl.uint32)
        pmask = tl.zeros((BLOCK_R,), tl.uint32)
        krem = tl.zeros((BLOCK_R,), tl.int32) + P
        zero = tl.zeros((BLOCK_R,), tl.int32)
        if RESIDENT:
            col = tl.arange(0, CT)
            key, mask = _keys(dot, norms, r64, m, col, NCOL, s_d)
            for shift in tl.static_range(24, -1, -8):
                krem, prefix, pmask = _pick(_count(key, mask, prefix, pmask, shift, rl, BLOCK_R, CT), krem, prefix,
                                            pmask, shift, BLOCK_R)
            _emit(key, mask, col, prefix, krem, zero, zero, out_col, out_key, r64, P, BLOCK_R, CT)
        else:
            for shift in tl.static_range(24, -1, -8):
                h = tl.zeros((BLOCK_R, 256), tl.int32)
                for c0 in range(0, NCOL, CT):
                    col = c0 + tl.arange(0, CT)
                    key, mask = _keys(dot, norms, r64, m, col, NCOL, s_d)
                    h += _count(key, mask, prefix, pmask, shift, rl, BLOCK_R, CT)
                krem, prefix, pmask = _pick(h, krem, prefix, pmask, shift, BLOCK_R)
            eq_seen, sel_seen = zero, zero
            for c0 in range(0, NCOL, CT):
                col = c0 + tl.arange(0, CT)
                key, mask = _keys(dot, norms, r64, m, col, NCOL, s_d)
                eq_seen, sel_seen = _emit(key, mask, col, prefix, krem, eq_seen, sel_seen, out_col, out_key, r64, P,
                                          BLOCK_R, CT)


def route_eligible(index, x):
    """CUDA rows (CPU routing keeps torch.topk)."""
    return triton is not None and x.is_cuda


def route(index, x):
    """IndexIVFFlat.route's probe lists, selected without torch.topk."""
    with torch.cuda.device(x.device):
        return _route(index, x)


def _route(index, x):
    from .ivf_flat import _dot_fp32

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
    n, d = xq.shape
    P = int(coarse_idx.shape[1])
    ieee = vecs.dtype == torch.float32
    BM, BN, BK, stages = (64, 64, 32, 3) if ieee else (64, 128, 64, 4)
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
    """Whether the CuTe SM90 kernel can run on this device (optional dependencies importable)."""
    if torch.cuda.get_device_capability(device_index) != (9, 0):
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from quack import copy_utils, layout_utils, sm90_utils  # noqa: F401
        from quack import pipeline as quack_pipeline  # noqa: F401
        from quack.compile_utils import make_fake_tensor  # noqa: F401
    except ImportError:
        return False
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
    kernel = (triton is not None and xq.is_cuda and xq.dtype == vecs.dtype
              and vecs.dtype in (torch.bfloat16, torch.float16, torch.float32)
              and index.vectors_norms.dtype == torch.float32)
    if not kernel:
        return _search_torch(index, xq, vecs, coarse_idx)
    if vecs.dtype == torch.bfloat16 and _cute_ready(xq.device.index if xq.device.index is not None
                                                    else torch.cuda.current_device()):
        from .ivf_fine_cute import search_2nn as cute_search_2nn

        return cute_search_2nn(index, xq, coarse_idx, tile if tile is not None else DEFAULT_TILE)
    return _search_triton(index, xq, vecs, coarse_idx)
