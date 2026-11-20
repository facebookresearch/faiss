# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Two-level nearest-centroid assignment: coarse cells, then the centroids
of the probed cells only.

The centroid table is kept sorted by coarse cell, so that cell c owns the
contiguous table rows off[c] .. off[c + 1] - 1. A batch of R rows is
assigned in three launches, with no value read back to the host:

1. _coarse_kernel scores every row against the G cell centres, takes its
   p best cells, and appends the pair (row, probe rank) to the bucket of
   each of those cells. Slots are taken with an atomic counter per cell, so
   the pairs end up grouped by cell without a sort.
2. _fine_kernel runs one program per tile of pairs of one cell and scans
   that cell's centroid block. It writes the best (score, label) of each
   pair. The grid is sized for the worst case; programs past the last tile
   return at once.
3. _reduce_child_kernel takes, for every row, the best of its p partial
   results, then checks the children of the winner (see record_splits),
   and finally clears the cell counters for the next batch.

A bucket can hold R pairs: the p cells probed by one row are distinct, so a
cell receives at most one pair per row.
"""

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:
    raise ImportError("o-means requires the triton package") from error

# Empty cells get a huge but finite norm so that every row probes the
# non-empty cells first and can never end up with only empty probes. Padding
# cells (G <= index < GP) get +inf, which the coarse kernel uses as the
# "already taken" mark, so they are never probed.
EMPTY_NORM = 1e37


@triton.jit
def _pack_keys(score, gid, GP: tl.constexpr):
    # Order-preserving int32 image of the FP32 scores with the cell id in
    # the low log2(GP) bits, so that each probe round is one plain min
    # instead of a (value, index) reduction. The truncation ranks cells to
    # about 2**-(23 - log2 GP) relative; near-ties go to the lower id.
    bits = score.to(tl.int32, bitcast=True)
    key = tl.where(bits < 0, bits ^ 2147483647, bits)
    return (key & (-GP)) | gid[None, :]


@triton.jit(do_not_specialize=["R"])
def _coarse_kernel(x_ptr, idx_ptr, g_ptr, gn_ptr, cnt_ptr, bucket_ptr,
                   R, D, sx, sg, cap,
                   HAS_IDX: tl.constexpr, P: tl.constexpr, GP: tl.constexpr,
                   BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr):
    # One atomic per (row, probe).
    pid = tl.program_id(0)
    r = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = r < R
    if HAS_IDX:
        rows = tl.load(idx_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    gid = tl.arange(0, GP)
    gbase = g_ptr + gid[:, None].to(tl.int64) * sg
    acc = tl.zeros((BLOCK_B, GP), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        xt = tl.load(xbase + col[None, :], mask=rmask[:, None], other=0.0)
        gt = tl.load(gbase + col[None, :])
        acc = tl.dot(xt, tl.trans(gt), acc=acc, out_dtype=tl.float32)
    score = tl.load(gn_ptr + gid)[None, :] - 2.0 * acc
    pair0 = r * P
    key = _pack_keys(score, gid, GP)
    for s in tl.static_range(P):
        kmin = tl.min(key, axis=1)
        bj = kmin & (GP - 1)
        key = tl.where(key == kmin[:, None], 2147483647, key)
        pos = tl.atomic_add(cnt_ptr + bj, tl.full((BLOCK_B,), 1, tl.int32), mask=rmask,
                            sem="relaxed")
        tl.store(bucket_ptr + bj.to(tl.int64) * cap + pos, pair0 + s, mask=rmask)


@triton.jit(do_not_specialize=["R"])
def _coarse_agg_kernel(x_ptr, idx_ptr, g_ptr, gn_ptr, cnt_ptr, bucket_ptr,
                       R, D, sx, sg, cap,
                       HAS_IDX: tl.constexpr, P: tl.constexpr, GP: tl.constexpr,
                       BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr):
    # As _coarse_kernel, but one atomic per (program, cell): the program
    # reserves a slot range per cell for all its rows at once, and a row's
    # slot is its rank among the program's rows that probe that cell.
    pid = tl.program_id(0)
    r = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = r < R
    if HAS_IDX:
        rows = tl.load(idx_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    gid = tl.arange(0, GP)
    gbase = g_ptr + gid[:, None].to(tl.int64) * sg
    acc = tl.zeros((BLOCK_B, GP), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        xt = tl.load(xbase + col[None, :], mask=rmask[:, None], other=0.0)
        gt = tl.load(gbase + col[None, :])
        acc = tl.dot(xt, tl.trans(gt), acc=acc, out_dtype=tl.float32)
    score = tl.load(gn_ptr + gid)[None, :] - 2.0 * acc
    rank1 = tl.zeros((BLOCK_B, GP), tl.int32)
    key = _pack_keys(score, gid, GP)
    for s in tl.static_range(P):
        kmin = tl.min(key, axis=1)
        hit = key == kmin[:, None]
        key = tl.where(hit, 2147483647, key)
        rank1 = tl.where(hit, s + 1, rank1)
    sel = (rank1 > 0) & rmask[:, None]
    one = sel.to(tl.int32)
    ccount = tl.sum(one, axis=0)
    base = tl.atomic_add(cnt_ptr + gid, ccount, mask=ccount > 0, sem="relaxed")
    slot = base[None, :] + tl.cumsum(one, axis=0) - one
    pair = r[:, None] * P + rank1 - 1
    tl.store(bucket_ptr + gid[None, :].to(tl.int64) * cap + slot, pair, mask=sel)


@triton.jit
def _fine_kernel(x_ptr, idx_ptr, c_ptr, cn_ptr, off_ptr, cnt_ptr, bucket_ptr, best_ptr, lab_ptr,
                 G, D, sx, sc, cap, sp,
                 HAS_IDX: tl.constexpr, P: tl.constexpr, GP: tl.constexpr,
                 BT: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    # Program t finds its (cell, tile) from a prefix sum of the tiles per
    # cell, which it computes itself from the G counters.
    gid = tl.arange(0, GP)
    gm = gid < G
    cnt = tl.load(cnt_ptr + gid, mask=gm, other=0)
    nt = (cnt + BT - 1) // BT
    incl = tl.cumsum(nt, axis=0)
    c = tl.minimum(tl.sum((incl <= t).to(tl.int32), axis=0), G - 1)
    sel = gid == c
    excl_c = tl.sum(tl.where(sel, incl - nt, 0), axis=0)
    cnt_c = tl.sum(tl.where(sel, cnt, 0), axis=0)
    total = tl.sum(nt, axis=0)
    valid = t < total
    q = (t - excl_c) * BT + tl.arange(0, BT)
    qm = (q < cnt_c) & valid
    pair = tl.load(bucket_ptr + c.to(tl.int64) * cap + q, mask=qm, other=0)
    rl = pair // P
    s = pair - rl * P
    if HAS_IDX:
        rows = tl.load(idx_ptr + rl, mask=qm, other=0).to(tl.int64)
    else:
        rows = rl.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    k_lo = tl.load(off_ptr + c)
    k_hi = tl.load(off_ptr + c + 1)
    k_hi = tl.where(valid, k_hi, k_lo)
    best = tl.full((BT,), float("inf"), tl.float32)
    lab = tl.zeros((BT,), tl.int32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        cid = k0 + tl.arange(0, BLOCK_K)
        cm = cid < k_hi
        cbase = c_ptr + cid[:, None].to(tl.int64) * sc
        acc = tl.zeros((BT, BLOCK_K), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            xt = tl.load(xbase + col[None, :], mask=qm[:, None], other=0.0)
            ct = tl.load(cbase + col[None, :], mask=cm[:, None], other=0.0)
            acc = tl.dot(xt, tl.trans(ct), acc=acc, out_dtype=tl.float32)
        score = tl.load(cn_ptr + cid, mask=cm, other=float("inf"))[None, :] - 2.0 * acc
        bs, bl = tl.min(score, axis=1, return_indices=True, return_indices_tie_break_left=True)
        better = bs < best
        best = tl.where(better, bs, best)
        lab = tl.where(better, bl.to(tl.int32) + k0, lab)
    dst = s.to(tl.int64) * sp + rl
    tl.store(best_ptr + dst, best, mask=qm)
    tl.store(lab_ptr + dst, lab, mask=qm)


@triton.jit(do_not_specialize=["R"])
def _reduce_child_kernel(best_ptr, lab_ptr, out_ptr, child_ptr, x_ptr, idx_ptr, c_ptr, cn_ptr,
                         cnt_ptr, R, G, D, sx, sc, sp,
                         HAS_IDX: tl.constexpr, P: tl.constexpr, GP: tl.constexpr,
                         BLOCK: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    b = tl.load(best_ptr + r, mask=m, other=float("inf"))
    lab = tl.load(lab_ptr + r, mask=m, other=0)
    for p in tl.static_range(1, P):
        bp = tl.load(best_ptr + p * sp + r, mask=m, other=float("inf"))
        lp = tl.load(lab_ptr + p * sp + r, mask=m, other=0)
        better = bp < b                    # strict: the lower probe rank wins ties
        b = tl.where(better, bp, b)
        lab = tl.where(better, lp, lab)
    # Centroids created by a split since the last rebuild may sit in a cell
    # the row does not probe. Score the winner's child c1 and grandchild c2
    # explicitly and keep the best of the three.
    c1 = tl.load(child_ptr + lab, mask=m, other=-1)
    c1 = tl.where(c1 < 0, lab, c1)
    c2 = tl.load(child_ptr + c1, mask=m, other=-1)
    c2 = tl.where(c2 < 0, c1, c2)
    n1 = m & (c1 != lab)
    n2 = m & (c2 != c1)
    need = n1 | n2
    if tl.sum(need.to(tl.int32), axis=0) > 0:
        if HAS_IDX:
            rows = tl.load(idx_ptr + r, mask=need, other=0).to(tl.int64)
        else:
            rows = r.to(tl.int64)
        a1 = tl.zeros((BLOCK,), tl.float32)
        a2 = tl.zeros((BLOCK,), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            xv = tl.load(x_ptr + rows[:, None] * sx + col[None, :], mask=need[:, None],
                         other=0.0).to(tl.float32)
            v1 = tl.load(c_ptr + c1[:, None].to(tl.int64) * sc + col[None, :],
                         mask=n1[:, None], other=0.0).to(tl.float32)
            v2 = tl.load(c_ptr + c2[:, None].to(tl.int64) * sc + col[None, :],
                         mask=n2[:, None], other=0.0).to(tl.float32)
            a1 += tl.sum(xv * v1, axis=1)
            a2 += tl.sum(xv * v2, axis=1)
        s1 = tl.load(cn_ptr + c1, mask=n1, other=float("inf")) - 2.0 * a1
        s2 = tl.load(cn_ptr + c2, mask=n2, other=float("inf")) - 2.0 * a2
        t1 = n1 & (s1 < b)
        b = tl.where(t1, s1, b)
        lab = tl.where(t1, c1, lab)
        t2 = n2 & (s2 < b)
        b = tl.where(t2, s2, b)
        lab = tl.where(t2, c2, lab)
    tl.store(out_ptr + r, lab, mask=m)
    if pid == 0:
        gid = tl.arange(0, GP)
        tl.store(cnt_ptr + gid, tl.zeros((GP,), tl.int32), mask=gid < G)


@dataclass
class Cells:
    """Coarse cells over a cell-sorted centroid table."""

    G: int
    GP: int                    # G rounded up to a power of two, at least 16
    g32: torch.Tensor          # [G, dp] float32 cell centres, warm start of the next build
    g16: torch.Tensor          # [GP, dp] float16 cell centres, rows >= G zero
    gn: torch.Tensor           # [GP] float32 squared norms of g16 (see EMPTY_NORM)
    off: torch.Tensor          # [G + 1] int32 table offsets of the cells
    size: torch.Tensor         # [G] int64 centroids per cell
    child: torch.Tensor        # [kp + 1] int32 latest split child, -1 if none


class TwoLevelWorkspace:
    """Buffers for batches of up to `rows` rows with up to `max_p` probes."""

    def __init__(self, rows, G, max_p, device):
        self.rows, self.G, self.max_p = rows, G, max_p
        GP = max(16, triton.next_power_of_2(G))
        self.cnt = torch.zeros(GP, dtype=torch.int32, device=device)
        self.bucket = torch.empty((G, rows), dtype=torch.int32, device=device)
        self.best = torch.empty((max_p, rows), dtype=torch.float32, device=device)
        self.lab = torch.empty((max_p, rows), dtype=torch.int32, device=device)
        self.labels = torch.empty(rows, dtype=torch.int32, device=device)


def _launch_params(GP, dp):
    """(coarse kernel, rows per coarse program, warps, centroids per fine slice).

    Measured on an RTX 5090: with up to 128 cells the per-cell atomics of
    _coarse_agg_kernel are 12-20% faster per pass; at 512 cells its
    [rows, GP] prefix sum dominates and the per-pair kernel is about twice
    as fast. 128-centroid slices are 22% faster than 64 at d = 768.
    """
    if GP <= 128:
        coarse, rows, warps = _coarse_agg_kernel, 16, 4
    else:
        coarse, rows, warps = _coarse_kernel, 32, 8
    return coarse, rows, warps, 64 if dp <= 128 else 128


def two_level_assign(x16, c16, cn, cells, ws, p, idx=None, rows=None):
    """Labels of rows x16[idx] (or x16[:rows]) as positions in the sorted table.

    Each row is compared with the centroids of its p nearest cells, and with
    the split children of its best match. The result is an int32 view into
    ws.labels.
    """
    R = int(idx.shape[0]) if idx is not None else int(rows if rows is not None else x16.shape[0])
    dp = x16.shape[1]
    G, GP = cells.G, cells.GP
    assert R <= ws.rows and p <= ws.max_p and G == ws.G and 1 <= p <= G
    assert x16.stride(1) == 1 and c16.stride(1) == 1 and dp % 32 == 0
    if R == 0:
        return ws.labels[:0]
    xi = idx if idx is not None else x16
    coarse, coarse_rows, coarse_warps, block_k = _launch_params(GP, dp)
    coarse[(triton.cdiv(R, coarse_rows),)](
        x16, xi, cells.g16, cells.gn, ws.cnt, ws.bucket, R, dp, x16.stride(0),
        cells.g16.stride(0), ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BLOCK_B=coarse_rows, BLOCK_D=32,
        num_warps=coarse_warps)
    block_t = 64
    _fine_kernel[(triton.cdiv(R * p, block_t) + G,)](
        x16, xi, c16, cn, cells.off, ws.cnt, ws.bucket, ws.best, ws.lab,
        G, dp, x16.stride(0), c16.stride(0), ws.rows, ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BT=block_t, BLOCK_K=block_k,
        BLOCK_D=32, num_warps=4, num_stages=2)
    _reduce_child_kernel[(triton.cdiv(R, 128),)](
        ws.best, ws.lab, ws.labels, cells.child, x16, xi, c16, cn, ws.cnt,
        R, G, dp, x16.stride(0), c16.stride(0), ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BLOCK=128, BLOCK_D=32, num_warps=4)
    return ws.labels[:R]


def build_cells(C, k, G, iters, prev=None, generator=None):
    """Group the centroids C[:k] into G cells by Lloyd iterations on them.

    The first build starts from G centroids drawn with `generator`; later
    builds start from the previous cell centres. Returns (order, cells):
    `order` sorts the table by cell (stably) and `cells` describes the table
    after it has been permuted by `order`.
    """
    dev = C.device
    dp = C.shape[1]
    Ck = C[:k]
    if prev is None:
        pick = torch.randperm(k, generator=generator, device=dev)[:G]
        g = Ck[pick].clone()
    else:
        g = prev.g32.clone()
    ones = torch.ones(k, dtype=torch.float32, device=dev)
    for _ in range(iters):
        a = torch.addmm((g * g).sum(1)[None, :], Ck, g.t(), alpha=-2.0).argmin(1)
        sums = torch.zeros((G, dp), dtype=torch.float32, device=dev).index_add_(0, a, Ck)
        cnt = torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, a, ones)
        g = torch.where((cnt > 0)[:, None], sums / cnt.clamp_min(1)[:, None], g)
    a = torch.addmm((g * g).sum(1)[None, :], Ck, g.t(), alpha=-2.0).argmin(1)
    order = torch.argsort(a, stable=True)
    size = torch.bincount(a, minlength=G)
    off = torch.zeros(G + 1, dtype=torch.int32, device=dev)
    off[1:] = torch.cumsum(size, 0)
    GP = max(16, triton.next_power_of_2(G))
    g16 = torch.zeros((GP, dp), dtype=torch.float16, device=dev)
    g16[:G] = g
    gn = torch.full((GP,), float("inf"), dtype=torch.float32, device=dev)
    gn[:G] = torch.where(size > 0, g16[:G].float().square().sum(1), EMPTY_NORM)
    # One spare slot past the table absorbs the writes of unused split pairs.
    child = torch.full((C.shape[0] + 1,), -1, dtype=torch.int32, device=dev)
    return order, Cells(G=G, GP=GP, g32=g, g16=g16, gn=gn, off=off, size=size, child=child)


def record_splits(child, src, tgt, m, max_pairs):
    """Record the splits of one revival check: child[t] = child[s]; child[s] = t.

    src, tgt and m are as passed to apply_splits (m a 0-dim device tensor).
    The shapes are fixed and nothing is read back: pairs i >= m write to the
    spare slot child[-1], which is reset afterwards. The new child takes over
    the donor's previous child, so a row whose best match is the donor
    reaches both its newest child and the one before.
    """
    spare = child.shape[0] - 1
    s, t = src[:max_pairs], tgt[:max_pairs]
    live = torch.arange(s.shape[0], device=s.device) < m
    s = torch.where(live, s, spare)
    t = torch.where(live, t, spare)
    inherited = child[s]
    child[t] = inherited
    child[s] = t.to(child.dtype)
    child[spare] = -1


def permute_rows(order, k, tensors):
    """t[:k] = t[:k][order] in place for each tensor; rows >= k are untouched."""
    for t in tensors:
        t[:k] = t[:k].index_select(0, order)
