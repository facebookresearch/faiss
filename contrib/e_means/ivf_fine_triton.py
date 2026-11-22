# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lazily imported Triton kernels for IVF fine search and probe selection."""

import triton
import triton.language as tl

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

