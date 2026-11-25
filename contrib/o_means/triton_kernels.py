# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels for the dense parts of o-means training.

Layout shared by every kernel here and in triton_two_level.py:

    x16  [n, dp]  float16 copy of the data, columns >= d are zero
    C    [kp, dp] float32 centroids, rows >= k unused
    c16  [kp, dp] float16 centroids used for assignment (C rounded)
    cn   [kp]     float32 squared norms of the c16 rows, +inf for rows >= k

Batch sums are kept in int64 fixed point: every FP16 value is an integer
multiple of 2**-24, so adding v * 2**24 as an integer is exact and the result
does not depend on the order of the atomic additions. This makes training
bitwise deterministic for a given seed.
"""

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:
    raise ImportError("o-means requires the triton package") from error

_FIXED_SCALE = 2.0**24

# FP16 keeps 11 significant bits for |v| in [2**-14, 65504]. The data is
# rescaled, exactly, by a power of two only when its largest magnitude leaves
# this comfortable part of that range.
_CAST_HI, _CAST_LO = 2.0**15, 2.0**-8


def _round_up(v, m):
    return (v + m - 1) // m * m


@dataclass(frozen=True)
class DenseLayout:
    """Padded sizes and tiles of the dense assignment kernel."""

    d: int
    k: int
    dp: int
    kp: int
    parts: int
    block_b: int = 64
    block_k: int = 256
    block_d: int = 32
    num_warps: int = 4
    num_stages: int = 2


def dense_layout(d, k, parts=8, block_b=64, block_k=256, block_d=32):
    # Fewer partitions for small k keep the padding below one partition.
    parts = max(1, min(parts, -(-k // block_k)))
    return DenseLayout(d=d, k=k, dp=_round_up(d, block_d),
                       kp=_round_up(k, parts * block_k), parts=parts,
                       block_b=block_b, block_k=block_k, block_d=block_d)


def scale_exponent(max_abs):
    """Power of two e such that max_abs * 2**e stays well inside FP16."""
    m = max_abs
    if not math.isfinite(m) or m == 0 or _CAST_LO <= m <= _CAST_HI:
        return 0
    return 7 - math.frexp(m)[1]          # max_abs * 2**e in [2**6, 2**7)


@triton.jit
def _cast_pad_kernel(x_ptr, out_ptr, amax_ptr, n, d, dp, sx, so, scale,
                     BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    r = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    rm = r < n
    amax = tl.zeros((BLOCK_R,), tl.float32)
    for d0 in range(0, dp, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        v = tl.load(x_ptr + r[:, None] * sx + col[None, :], mask=rm[:, None] & (col < d)[None, :],
                    other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(v), axis=1))
        tl.store(out_ptr + r[:, None] * so + col[None, :], (v * scale).to(out_ptr.dtype.element_ty),
                 mask=rm[:, None] & (col < dp)[None, :])
    tl.atomic_max(amax_ptr, tl.max(amax, axis=0), sem="relaxed")


def cast_pad_rows(x, dp):
    """Return (x16, e) with x16 = fp16(x * 2**e), zero-padded to dp columns.

    One fused pass casts and measures max|x|; reading that maximum back is
    the only host synchronization of a fit. A second pass runs only in the
    rare case where the data has to be rescaled.
    """
    n, d = x.shape
    assert x.stride(1) == 1 and dp >= d
    out = torch.empty((n, dp), dtype=torch.float16, device=x.device)
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    block_d = min(128, triton.next_power_of_2(dp))
    grid = (triton.cdiv(n, 16),)

    def run(scale):
        _cast_pad_kernel[grid](x, out, amax, n, d, dp, x.stride(0), out.stride(0), scale,
                               BLOCK_R=16, BLOCK_D=block_d, num_warps=4)
    run(1.0)
    e = scale_exponent(float(amax.item()))
    if e:
        run(2.0**e)
    return out, e


@triton.jit(do_not_specialize=["R"])
def _assign_ksplit_kernel(x_ptr, idx_ptr, c_ptr, cn_ptr, best_ptr, lab_ptr,
                          R, D, KPART, sx, sc, sp,
                          HAS_IDX: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
                          BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    part = tl.program_id(1)
    r = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = r < R
    if HAS_IDX:
        rows = tl.load(idx_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    best = tl.full((BLOCK_B,), float("inf"), tl.float32)
    lab = tl.zeros((BLOCK_B,), tl.int32)
    kbeg = part * KPART
    for k0 in range(kbeg, kbeg + KPART, BLOCK_K):
        cid = k0 + tl.arange(0, BLOCK_K)
        cbase = c_ptr + cid[:, None].to(tl.int64) * sc
        acc = tl.zeros((BLOCK_B, BLOCK_K), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            xt = tl.load(xbase + col[None, :], mask=rmask[:, None], other=0.0)
            ct = tl.load(cbase + col[None, :])
            acc = tl.dot(xt, tl.trans(ct), acc=acc, out_dtype=tl.float32)
        score = tl.load(cn_ptr + cid)[None, :] - 2.0 * acc
        bs, bl = tl.min(score, axis=1, return_indices=True, return_indices_tie_break_left=True)
        better = bs < best
        best = tl.where(better, bs, best)
        lab = tl.where(better, bl.to(tl.int32) + k0, lab)
    tl.store(best_ptr + part * sp + r, best, mask=rmask)
    tl.store(lab_ptr + part * sp + r, lab, mask=rmask)


@triton.jit(do_not_specialize=["R"])
def _reduce_parts_kernel(best_ptr, lab_ptr, out_ptr, R, sp, PARTS: tl.constexpr,
                         BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    b = tl.load(best_ptr + r, mask=m, other=float("inf"))
    lab = tl.load(lab_ptr + r, mask=m, other=0)
    for p in tl.static_range(1, PARTS):
        bp = tl.load(best_ptr + p * sp + r, mask=m, other=float("inf"))
        lp = tl.load(lab_ptr + p * sp + r, mask=m, other=0)
        better = bp < b                      # strict: the lower partition wins ties
        b = tl.where(better, bp, b)
        lab = tl.where(better, lp, lab)
    tl.store(out_ptr + r, lab, mask=m)


class DenseWorkspace:
    """Partial winners and labels for batches of up to `rows` rows."""

    def __init__(self, rows, parts, device):
        self.rows, self.parts = rows, parts
        self.best = torch.empty((parts, rows), dtype=torch.float32, device=device)
        self.lab = torch.empty((parts, rows), dtype=torch.int32, device=device)
        self.labels = torch.empty(rows, dtype=torch.int32, device=device)


def assign_dense(x16, c16, cn, layout, ws, idx=None, rows=None):
    """Exact nearest-centroid labels of x16[idx] (or x16[:rows]), int32.

    The score cn - 2 <x, c> is never materialized: the grid is (row tiles,
    centroid partitions), each program keeps a running minimum over its
    slice, and a second kernel reduces the partitions. Ties go to the lowest
    centroid index. The result is a view into ws.labels.
    """
    R = int(idx.shape[0]) if idx is not None else int(rows if rows is not None else x16.shape[0])
    dp = x16.shape[1]
    kp = c16.shape[0]
    assert R <= ws.rows and dp == layout.dp and kp == layout.kp and c16.shape[1] == dp
    assert x16.stride(1) == 1 and c16.stride(1) == 1 and x16.dtype == c16.dtype
    if R == 0:
        return ws.labels[:0]
    parts = layout.parts
    grid = (triton.cdiv(R, layout.block_b), parts)
    _assign_ksplit_kernel[grid](
        x16, idx if idx is not None else x16, c16, cn, ws.best, ws.lab,
        R, dp, kp // parts, x16.stride(0), c16.stride(0), ws.rows,
        HAS_IDX=idx is not None, BLOCK_B=layout.block_b, BLOCK_K=layout.block_k,
        BLOCK_D=layout.block_d, num_warps=layout.num_warps, num_stages=layout.num_stages)
    if parts == 1:
        return ws.lab[0, :R]
    _reduce_parts_kernel[(triton.cdiv(R, 1024),)](ws.best, ws.lab, ws.labels, R, ws.rows,
                                                  PARTS=parts, BLOCK=1024, num_warps=4)
    return ws.labels[:R]


@triton.jit(do_not_specialize=["R"])
def _accumulate_kernel(x_ptr, idx_ptr, lab_ptr, sum_ptr, cnt_ptr, R, d, sx, ss,
                       HAS_IDX: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < R
    if HAS_IDX:
        rows = tl.load(idx_ptr + r, mask=rm, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    lab = tl.load(lab_ptr + r, mask=rm, other=0).to(tl.int64)
    for d0 in range(0, d, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        m = rm[:, None] & (col < d)[None, :]
        v = tl.load(x_ptr + rows[:, None] * sx + col[None, :], mask=m, other=0.0).to(tl.float32)
        v = (v * 16777216.0).to(tl.int64)
        tl.atomic_add(sum_ptr + lab[:, None] * ss + col[None, :], v, mask=m, sem="relaxed")
    tl.atomic_add(cnt_ptr + lab, tl.full((BLOCK_R,), 1, tl.int32), mask=rm, sem="relaxed")


def accumulate(x16, labels, sums, cnt, d, idx=None):
    """sums[labels[i]] += x16[i] (int64, units of 2**-24), cnt[labels[i]] += 1.

    sums and cnt must be zero on entry; ema_update leaves them zeroed.
    """
    R = int(labels.shape[0])
    if R == 0:
        return
    assert sums.dtype == torch.int64 and cnt.dtype == torch.int32
    block_d = min(128, triton.next_power_of_2(d))
    _accumulate_kernel[(triton.cdiv(R, 16),)](
        x16, idx if idx is not None else labels, labels, sums, cnt, R, d,
        x16.stride(0), sums.stride(0),
        HAS_IDX=idx is not None, BLOCK_R=16, BLOCK_D=block_d, num_warps=4)


@triton.jit
def _ema_update_kernel(C_ptr, c16_ptr, cn_ptr, usage_ptr, sum_ptr, cnt_ptr, psum_ptr, pcnt_ptr,
                       k, d, sC, s16, sS, sP, alpha, decay, omega,
                       CLOSE: tl.constexpr, REPLACE: tl.constexpr, WEIGHTED: tl.constexpr,
                       BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    valid = rows < k
    cnt = tl.load(cnt_ptr + rows, mask=valid, other=0)
    touched = cnt > 0
    tm = valid & touched
    count = cnt.to(tl.float32)
    old = tl.load(usage_ptr + rows, mask=valid, other=0.0)
    if REPLACE:
        retained = tl.zeros((BLOCK_K,), tl.float32)
    else:
        retained = old * decay
    step = alpha * count
    mass = retained + step
    w = step / tl.where(touched, mass, 1.0)
    div = tl.where(touched, count, 1.0)
    r64 = rows.to(tl.int64)
    norm = tl.zeros((BLOCK_K,), tl.float32)
    for d0 in range(0, d, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        m = tm[:, None] & (col < d)[None, :]
        s = tl.load(sum_ptr + r64[:, None] * sS + col[None, :], mask=m, other=0)
        sf = s.to(tl.float32) * (1.0 / 16777216.0)
        mean = sf / div[:, None]
        coff = r64[:, None] * sC + col[None, :]
        c = tl.load(C_ptr + coff, mask=m, other=0.0)
        newc = tl.where((w == 1.0)[:, None], mean, c + w[:, None] * (mean - c))
        tl.store(C_ptr + coff, newc, mask=m)
        h = newc.to(c16_ptr.dtype.element_ty)
        tl.store(c16_ptr + r64[:, None] * s16 + col[None, :], h, mask=m)
        hf = h.to(tl.float32)
        norm += tl.sum(hf * hf, axis=1)
        if CLOSE:
            poff = r64[:, None] * sP + col[None, :]
            ps = tl.load(psum_ptr + poff, mask=m, other=0)
            if WEIGHTED:
                tl.store(psum_ptr + poff, ps + omega * sf, mask=m)
            else:
                tl.store(psum_ptr + poff, ps + s, mask=m)
        tl.store(sum_ptr + r64[:, None] * sS + col[None, :], tl.zeros_like(s), mask=m)
    tl.store(cn_ptr + rows, norm, mask=tm)
    if REPLACE:
        newu = tl.where(touched, mass, old)
    else:
        newu = tl.where(touched, mass, retained)
    tl.store(usage_ptr + rows, newu, mask=valid)
    if CLOSE:
        pc = tl.load(pcnt_ptr + rows, mask=tm, other=0)
        if WEIGHTED:
            tl.store(pcnt_ptr + rows, pc + omega * count, mask=tm)
        else:
            tl.store(pcnt_ptr + rows, pc + cnt.to(pc.dtype), mask=tm)
    tl.store(cnt_ptr + rows, tl.zeros_like(cnt), mask=tm)


def ema_update(C, c16, cn, usage, sums, cnt, k, d, alpha, close_sums=None, close_cnt=None,
               omega=1.0):
    """One e-means update of every centroid that received rows, in place.

    For a centroid j with cnt[j] > 0 (Algorithm 1, line 8 of the e-means
    paper): mass = (1 - alpha) * usage[j] + alpha * cnt[j], the centroid
    moves to the batch mean by alpha * cnt[j] / mass, and usage[j] = mass.
    Untouched centroids only decay their usage. The FP16 table and norms of
    moved centroids are refreshed, and the batch sums are cleared for the
    next batch, all in the same pass.

    When close_sums / close_cnt are given, the batch is also added to the
    closing-mean accumulators: exactly, when they are int64 / int32 like the
    batch sums, or with weight omega when they are float32. alpha == 1 uses
    e-means' rule of replacing the usage instead of decaying it.
    """
    close = close_sums is not None
    weighted = close and close_sums.dtype != sums.dtype
    assert weighted or omega == 1.0
    if weighted:
        assert close_sums.dtype == torch.float32 and close_cnt.dtype == torch.float32
    block_d = min(128, triton.next_power_of_2(d))
    block_k = max(1, 1024 // block_d)
    _ema_update_kernel[(triton.cdiv(k, block_k),)](
        C, c16, cn, usage, sums, cnt,
        close_sums if close else sums, close_cnt if close else cnt,
        k, d, C.stride(0), c16.stride(0), sums.stride(0),
        close_sums.stride(0) if close else 0, float(alpha), float(1.0 - alpha), float(omega),
        CLOSE=close, REPLACE=alpha == 1.0, WEIGHTED=weighted,
        BLOCK_K=block_k, BLOCK_D=block_d, num_warps=4)


def refresh_table(C, c16, cn, k, d):
    c16[:k, :d] = C[:k, :d]
    cn[:k] = c16[:k, :d].float().square().sum(1)


@triton.jit
def _split_kernel(C_ptr, c16_ptr, cn_ptr, usage_ptr, src_ptr, tgt_ptr, m_ptr, d, sC, s16, eps,
                  wsum_ptr, wcnt_ptr, sW,
                  SPLIT_CLOSE: tl.constexpr, BLOCK_D: tl.constexpr):
    i = tl.program_id(0)
    m = tl.load(m_ptr)
    if i < m:
        s = tl.load(src_ptr + i).to(tl.int64)
        t = tl.load(tgt_ptr + i).to(tl.int64)
        nlo = tl.zeros((BLOCK_D,), tl.float32)
        nhi = tl.zeros((BLOCK_D,), tl.float32)
        ndiff = tl.zeros((BLOCK_D,), tl.int32)
        for d0 in range(0, d, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            cm = col < d
            v = tl.load(C_ptr + s * sC + col, mask=cm, other=0.0)
            off = eps * tl.where(col % 2 == 0, 1.0, -1.0)
            lo = v * (1.0 - off)
            hi = v * (1.0 + off)
            hlo = lo.to(c16_ptr.dtype.element_ty)
            hhi = hi.to(c16_ptr.dtype.element_ty)
            ndiff += tl.where(cm & (hlo != hhi), 1, 0)
            tl.store(C_ptr + s * sC + col, lo, mask=cm)
            tl.store(C_ptr + t * sC + col, hi, mask=cm)
            tl.store(c16_ptr + s * s16 + col, hlo, mask=cm)
            tl.store(c16_ptr + t * s16 + col, hhi, mask=cm)
            flo = hlo.to(tl.float32)
            fhi = hhi.to(tl.float32)
            nlo += flo * flo
            nhi += fhi * fhi
            if SPLIT_CLOSE:
                hw = tl.load(wsum_ptr + s * sW + col, mask=cm, other=0.0) * 0.5
                tl.store(wsum_ptr + s * sW + col, hw, mask=cm)
                tl.store(wsum_ptr + t * sW + col, hw, mask=cm)
        norm_lo = tl.sum(nlo, axis=0)
        norm_hi = tl.sum(nhi, axis=0)
        if tl.sum(ndiff, axis=0) == 0:
            # Donor and child round to the same FP16 row: move the child's
            # first coordinate one FP16 step (outwards, or inwards if that
            # would overflow) so that the two can separate.
            h0 = tl.load(c16_ptr + t * s16)
            b = h0.to(tl.uint16, bitcast=True).to(tl.int32)
            mag = b & 0x7FFF
            neg = (b & 0x8000) != 0
            up = tl.where(mag == 0, 1, tl.where(neg, b - 1, b + 1))
            inward = tl.where(mag == 0, 0x8001, tl.where(neg, b + 1, b - 1))
            moved = tl.where((up & 0x7FFF) < 0x7C00, up, inward)
            moved = tl.where(mag > 0x7C00, b, moved)          # NaN stays NaN
            nb = moved.to(tl.uint16).to(c16_ptr.dtype.element_ty, bitcast=True)
            f_old = h0.to(tl.float32)
            f_new = nb.to(tl.float32)
            norm_hi = norm_hi + f_new * f_new - f_old * f_old
            tl.store(c16_ptr + t * s16, nb)
            tl.store(C_ptr + t * sC, f_new)
        tl.store(cn_ptr + s, norm_lo)
        tl.store(cn_ptr + t, norm_hi)
        half = tl.load(usage_ptr + s) * 0.5
        tl.store(usage_ptr + s, half)
        tl.store(usage_ptr + t, half)
        if SPLIT_CLOSE:
            hc = tl.load(wcnt_ptr + s) * 0.5
            tl.store(wcnt_ptr + s, hc)
            tl.store(wcnt_ptr + t, hc)


def apply_splits(C, c16, cn, usage, src, tgt, m, d, eps, max_pairs, wsum=None, wcnt=None):
    """Split donor src[i] into itself and child tgt[i], for i < m.

    m is a 0-dim device tensor, so no value is read back to the host: the
    grid has max_pairs programs and those with i >= m return at once. As in
    e-means, donor and child move apart by a relative eps with alternating
    signs over the coordinates and share the donor's usage. With wsum / wcnt
    (float32 closing accumulators), the two also share the donor's
    accumulated closing mass.
    """
    block_d = min(256, triton.next_power_of_2(d))
    split_close = wsum is not None
    if split_close:
        assert wsum.dtype == torch.float32 and wcnt.dtype == torch.float32
    _split_kernel[(max(1, max_pairs),)](C, c16, cn, usage, src, tgt, m, d, C.stride(0),
                                        c16.stride(0), float(eps),
                                        wsum if split_close else C,
                                        wcnt if split_close else usage,
                                        wsum.stride(0) if split_close else 0,
                                        SPLIT_CLOSE=split_close, BLOCK_D=block_d, num_warps=4)
