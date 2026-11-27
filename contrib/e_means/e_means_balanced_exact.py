# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Exact balancing after balanced E-means, on PyTorch tensors (CPU and CUDA).

Balanced E-means leaves clusters nearly, not exactly, even. This step takes
the trained centroids c_j and bias b_j and assigns every row so that each
cluster gets exactly floor(n / k) or ceil(n / k) rows, at a total squared
distance close to the smallest possible: a capacitated assignment
(transportation) problem, solved as an auction with prices.

1. Candidates. Each row keeps m centroids: the best by biased distance
   ||x - c_j||^2 - b_j among the ``graph`` centroids nearest to its anchor,
   its biased nearest centroid (``assign``'s labels, or one search), with
   the true squared distances to them. Rows only ever move between their
   candidates, until the last step.
2. Capacities. With n = q k + r, the r clusters that the biased assignment
   fills most get q + 1 rows and the others q, so the capacities sum to n.
3. Auction. Each cluster has a price p_j, started at -b_j, and every row
   sits in the candidate of smallest distance + price: the biased
   assignment. Over-full clusters must send rows to clusters with room,
   possibly through chains of moves (a row of A moves to B, a row of B moves
   to C, ...). Each batch:
     - lists the cheap moves: a row moving from its cluster to another of
       its candidates, priced at how much worse that candidate is under the
       current prices (the reduced cost);
     - computes, for every cluster, the cheapest chain of moves to a cluster
       with room (Bellman-Ford over the move list);
     - raises every cluster's price by that distance (capped). Every row
       stays at its best candidate, and every cheapest chain becomes free;
     - sends the surplus down the free chains, only towards clusters that
       can still take rows (a capacity computed level by level from the
       clusters with room).
   Prices only rise. Every row the auction places stays within ``eps`` of
   its best candidate under the final prices, so that part of the partition
   is optimal for these capacities among the candidates, up to ``eps`` per
   moved row.
4. Exits (``finish``). When the batches stop, every row also gets as a
   candidate its nearest cluster with room, when that is within
   ``exit_tau`` of where it sits, and up to 4 more batches run: chains can
   end with one longer move.
5. Completion. Stopping rule: while the surplus (rows above capacity) is
   above ``target`` x n (0.1%), the batches go on unless one resolves less
   than ``min_progress`` (1%) of it; below that, they stop as soon as one
   resolves less than ``stall_progress`` (5%). Each over-full cluster then
   gives up its rows cheapest to move elsewhere, and they go greedily to the
   nearest clusters with room, by exact search over those clusters.

The result is exactly balanced. It is a partition of the given rows: the
final prices (returned as ``stats["bias"]`` = -prices) are optimal only on
each row's candidates, and routing new rows with them over all centroids
balances them worse than the trained bias. Keep the trained bias for new rows.
"""

import math
import time

import numpy as np
import torch

from .e_means_balanced_exact_numpy import _push as _numpy_push

try:
    from .balanced_exact_triton import exact_costs as _triton_exact_costs
    from .balanced_exact_triton import push as _triton_push
    from .balanced_exact_triton import relax_csr as _triton_relax
    from .balanced_exact_triton import top_m as _triton_top_m
except ImportError:  # No Triton: PyTorch candidates and sweeps.
    _triton_top_m = _triton_exact_costs = _triton_relax = _triton_push = None


__all__ = [
    "exact_balance",
    "exact_capacities",
    "balance_candidates",
    "biased_anchors",
    "auction_balance",
    "greedy_complete",
]


# Candidates per row, chosen among the _GRAPH centroids nearest to the row's
# biased nearest centroid.
_M = 8
_GRAPH = 32
# Moves of reduced cost above _TAU x (mean distance) are left out of a batch's
# move list; the price rise of a batch is capped there.
_TAU = 0.25
# A move whose reduced cost after the price rise is at most _EPS x (mean
# distance) counts as free (the auction's eps).
_EPS = 0.05
# Stopping rule, scale-free. While the surplus (rows above capacity) is above
# _TARGET x n, the batches go on unless one resolves less than _MIN_PROGRESS of
# the surplus (a stall); below it, they stop as soon as one resolves less than
# _STALL_PROGRESS (slow tail). The rows left go to the greedy completion.
_TARGET = 1e-3
_MIN_PROGRESS = 0.01
_STALL_PROGRESS = 0.05
_MAX_BATCHES = 256
# After the batches stop, each row whose nearest cluster with room is within
# _EXIT_TAU x (mean distance) of where it sits gets that cluster as an extra
# candidate, and at most _EXIT_BATCHES more batches run.
_EXIT_TAU = 0.5
_EXIT_BATCHES = 4
_EXITS_PER_CLUSTER = 64
# Bellman-Ford sweeps per batch, at most.
_MAX_SWEEPS = 4096
# Rows per chunk of the candidate search (CUDA) and of the move list.
_CHUNK_ROWS = 1 << 20
# Moves per row in a batch's move list (the cheapest ones).
_MOVES_PER_ROW = 3
# Moves per chunk when testing which moves are free.
_EDGE_CHUNK = 1 << 26
_I32_MAX = 2**31 - 1
# Scores per chunk of a dense search against the clusters with room.
_SCORES = 1 << 27
_UNREACHED = (0x7F800000 << 32) | 0x7FFFFFFF


def exact_capacities(counts, n):
    """Capacities summing to n: q + 1 for the r most filled clusters, else q.

    n = q k + r. ``counts`` (k,) are the starting counts; ties go to the lower
    index.
    """
    k = counts.shape[0]
    base, extra = divmod(int(n), k)
    cap = torch.full((k,), base, dtype=torch.int64, device=counts.device)
    if extra:
        order = torch.sort(counts.to(torch.int64), descending=True, stable=True).indices
        cap[order[:extra]] += 1
    return cap


def _dense_top_m(rows, centroids, norms, m):
    score = torch.addmm(norms[None, :], rows.float(), centroids.t(), alpha=-2.0)   # (r, k)
    return torch.topk(score, m, dim=1, largest=False, sorted=False).indices.to(torch.int32)


def _top_m(rows, centroids, c16, norms, m):
    """Labels (r, m) int32 of the m smallest norms_j - 2 x.c_j (unordered)."""
    if c16 is not None:
        return _triton_top_m(rows.to(c16.dtype), c16, norms, m)[0]
    return _dense_top_m(rows, centroids, norms, m)


def _exact_costs(rows, centroids, labels):
    """||x_i - c_{labels[i, l]}||^2 in float32, (r, g)."""
    if rows.is_cuda and _triton_exact_costs is not None:
        return _triton_exact_costs(rows, centroids, labels)
    return (rows.float()[:, None, :] - centroids[labels.long()]).square().sum(2)


def _search_centroids(centroids, bias, dtype, use_triton):
    """(16-bit copy or None, norms - bias): the norms of the centroids as represented."""
    if not use_triton:
        return None, (centroids.double() ** 2).sum(1).float() - bias
    c16 = centroids.to(dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16)
    return c16, (c16.double() ** 2).sum(1).float() - bias


def biased_anchors(x, centroids, bias, chunk_rows=_CHUNK_ROWS):
    """Each row's centroid of smallest biased distance (the deployed routing), (n,) int64."""
    centroids = centroids.float().contiguous()
    c16, norms = _search_centroids(centroids, bias.float(), x.dtype,
                                   x.is_cuda and _triton_top_m is not None)
    out = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    for begin in range(0, x.shape[0], chunk_rows):
        rows = x[begin:begin + chunk_rows]
        out[begin:begin + len(rows)] = _top_m(rows, centroids, c16, norms, 1)[:, 0]
    return out


def balance_candidates(x, centroids, bias, m=_M, anchors=None, graph=_GRAPH,
                       chunk_rows=_CHUNK_ROWS):
    """Each row's m candidate centroids, and the true squared distances to them.

    The candidates are the m of smallest biased distance ||x - c_j||^2 - b_j
    among the ``graph`` centroids nearest (in biased distance) to the row's
    anchor, its biased nearest centroid. ``anchors`` (n,) can be passed (for
    instance the labels of ``assign``); otherwise one search computes them.
    Distances are recomputed exactly in float32. Returns (cost (n, m)
    float32, labels (n, m) int32), each row ordered by biased distance.
    """
    n, d = x.shape
    k = centroids.shape[0]
    m = min(int(m), k)
    g = max(m, min(int(graph), k))
    centroids = centroids.float().contiguous()
    bias = bias.float()
    if anchors is None:
        anchors = biased_anchors(x, centroids, bias, chunk_rows)
    anchors = torch.as_tensor(anchors, device=x.device).long()
    # The centroid graph: each centroid's g nearest centroids in biased distance.
    c16, norms = _search_centroids(centroids, bias, torch.float32,
                                   x.is_cuda and _triton_top_m is not None)
    graph_ids = torch.empty((k, g), dtype=torch.int32, device=x.device)
    for begin in range(0, k, chunk_rows):
        rows = centroids[begin:begin + chunk_rows]
        graph_ids[begin:begin + len(rows)] = _top_m(rows, centroids, c16, norms, g)
    labels = torch.empty((n, m), dtype=torch.int32, device=x.device)
    cost = torch.empty((n, m), dtype=torch.float32, device=x.device)
    bias64 = bias.double()
    for begin in range(0, n, chunk_rows):
        end = min(n, begin + chunk_rows)
        cand = graph_ids[anchors[begin:end]]                                  # (r, g)
        dist = _exact_costs(x[begin:end], centroids, cand)                    # (r, g)
        biased = dist.double() - bias64[cand.long()]
        pick = torch.argsort(biased, dim=1, stable=True)[:, :m]                # (r, m)
        labels[begin:end] = cand.gather(1, pick)
        cost[begin:end] = dist.gather(1, pick)
    return cost, labels


def _kth_smallest(w, q):
    """The q-th smallest value of each row of w (r, m), (r, 1); q small."""
    w = w.clone()
    for _ in range(q - 1):
        w.scatter_(1, w.argmin(1, keepdim=True), math.inf)
    return w.min(1, keepdim=True).values


def _move_list(cost, labels, slot, assigned, prices, tau, keep_from=None,
               moves_per_row=None, chunk_rows=_CHUNK_ROWS):
    """The moves of reduced cost <= tau, at most ``moves_per_row`` per row (any
    finite cost from column ``keep_from`` on): (row, slot, from, to, reduced cost)."""
    out = []
    moves_per_row = _MOVES_PER_ROW if moves_per_row is None else moves_per_row
    for begin in range(0, cost.shape[0], chunk_rows):
        end = min(cost.shape[0], begin + chunk_rows)
        lab = labels[begin:end].long()
        net = cost[begin:end].double() + prices[lab]                        # (r, m)
        s = slot[begin:end, None]
        w = (net - net.gather(1, s)).float()
        w.scatter_(1, s, math.inf)
        keep = w <= tau
        if moves_per_row < keep.shape[1] - 1:
            # Only each row's cheapest moves: shortest chains rarely need more.
            cut = _kth_smallest(w[:, :keep_from], moves_per_row)
            keep[:, :keep_from] &= w[:, :keep_from] <= cut
        if keep_from is not None:
            keep[:, keep_from:] = torch.isfinite(w[:, keep_from:])
        row, col = torch.nonzero(keep, as_tuple=True)
        out.append(((row + begin).to(torch.int32), col.to(torch.uint8),
                    assigned[row + begin].to(torch.int32), lab[row, col].to(torch.int32),
                    w[row, col].clamp_(min=0)))
        del net, w, row, col
    return [torch.cat(parts) for parts in zip(*out)]


def _by_source(src, dst, w, k, chunk=1 << 24):
    """The moves grouped by source cluster (CSR): (offsets (k + 1,), dst, w).

    A counting sort by chunks, so the memory stays at one chunk's sort on top
    of the output (a single argsort of 1e8-scale move lists does not fit next
    to the rows on an 80 GB GPU).
    """
    offsets = torch.zeros(k + 1, dtype=torch.int64, device=src.device)
    offsets[1:] = torch.cumsum(torch.bincount(src, minlength=k), 0)
    fill = offsets[:-1].clone()
    dst_s = torch.empty(len(src), dtype=torch.int32, device=src.device)
    w_s = torch.empty(len(src), dtype=w.dtype, device=src.device)
    for begin in range(0, len(src), chunk):
        part = src[begin:begin + chunk].long()
        order = torch.argsort(part, stable=True)
        part = part[order]
        count = torch.bincount(part, minlength=k)
        rank = torch.arange(len(part), device=src.device) - (torch.cumsum(count, 0) - count)[part]
        pos = fill[part] + rank
        dst_s[pos] = dst[begin + order].to(torch.int32)
        w_s[pos] = w[begin + order]
        fill += count
    return offsets, dst_s, w_s


def _distances(src, dst, w, holes, k):
    """Cheapest chain of moves from every cluster to a cluster with room.

    Bellman-Ford from ``holes`` over the moves (src -> dst, length w).
    Returns (distance (k,) float32, inf if unreachable; hops (k,) int64,
    strictly larger than the hops of the next cluster on a cheapest chain;
    sweeps).
    """
    device = src.device
    if src.is_cuda and _triton_relax is not None:
        key = torch.full((k,), _UNREACHED, dtype=torch.int64, device=device)
        key[holes] = 0
        flag = torch.zeros(1, dtype=torch.int32, device=device)
        offsets, dst_s, w_s = _by_source(src, dst, w, k)
        sweeps = 0
        while sweeps < _MAX_SWEEPS:
            step = 4 if sweeps == 0 else 8
            for _ in range(step):
                _triton_relax(offsets, dst_s, w_s, key, flag)
            sweeps += step
            if int(flag) == 0:
                break
            flag.zero_()
        dist = (key >> 32).to(torch.int32).view(torch.float32)
        hops = key & 0xFFFFFFFF
        return dist.clone(), hops, sweeps
    dist = torch.full((k,), math.inf, dtype=torch.float32, device=device)
    dist[holes] = 0
    hops = torch.full((k,), 1 << 31, dtype=torch.int64, device=device)
    hops[holes] = 0
    sweeps = 0
    while sweeps < _MAX_SWEEPS:
        sweeps += 1
        new = dist.scatter_reduce(0, src.long(), w + dist[dst.long()], "amin", include_self=True)
        changed = new < dist
        if not bool(changed.any()):
            break
        hops[changed] = sweeps
        dist = new
    return dist, hops, sweeps


def _push_device(rows, slots, src, dst, units, room, hops, k):
    """``e_means_balanced_exact_numpy._push`` as one Triton program
    (``balanced_exact_triton.push``).

    Within a source cluster, its moves claim its units with atomics, so which
    of its free moves carry them is not fixed by preference order.
    """
    level = hops[src]
    order = torch.argsort(level, stable=True)
    levels, counts = torch.unique_consecutive(level[order], return_counts=True)
    offsets = torch.zeros(len(counts) + 1, dtype=torch.int64, device=rows.device)
    offsets[1:] = torch.cumsum(counts, 0)
    units32 = units.to(torch.int32)
    cap32 = room.to(torch.int32)
    moved = _triton_push(offsets, src[order].to(torch.int32), dst[order].to(torch.int32),
                         units32, cap32).bool()
    picked = order[moved]
    return rows[picked], slots[picked]


def _push_host(rows, slots, src, dst, units, room, hops, k):
    """``e_means_balanced_exact_numpy._push`` on host copies (CPU tensors, or CUDA without
    Triton): send the surplus down free moves towards clusters that still have
    room. Returns (moved rows, slots)."""
    device = rows.device
    host = [t.cpu().numpy() for t in (rows, slots, src, dst, units, room, hops)]
    moved, new_slot = _numpy_push(*host, k)
    return torch.from_numpy(moved).to(device), torch.from_numpy(new_slot).to(device)


def auction_balance(cost, labels, prices, cap, tau=_TAU, eps=_EPS,
                    stall_progress=_STALL_PROGRESS, max_batches=_MAX_BATCHES,
                    target=_TARGET, min_progress=_MIN_PROGRESS, slot=None, keep_from=None):
    """The auction's batches on the candidate graph; see the module docstring.

    cost (n, m) float32, labels (n, m) int32, prices (k,) float64 (modified
    in place), cap (k,) int64. ``tau`` and ``eps`` are fractions of the mean
    distance from a row to its first candidate. Returns (slot (n,) int64
    into the candidate lists, counts (k,) int64, stats dict).
    """
    n, m = cost.shape
    k = cap.shape[0]
    if slot is None:
        slot = torch.zeros(n, dtype=torch.int64, device=cost.device)
        for begin in range(0, n, _CHUNK_ROWS):
            part = (cost[begin:begin + _CHUNK_ROWS].double()
                    + prices[labels[begin:begin + _CHUNK_ROWS].long()])
            slot[begin:begin + _CHUNK_ROWS] = part.argmin(1)
    assigned = labels.gather(1, slot[:, None])[:, 0].long()
    counts = torch.bincount(assigned, minlength=k)
    scale = float(cost.gather(1, slot[:, None]).double().mean())
    tau_abs, eps_abs = tau * scale, eps * scale
    history = []
    batches = []
    stop = "balanced"
    while True:
        excess = counts - cap
        surplus = int(excess.clamp(min=0).sum())
        history.append(surplus)
        if surplus == 0:
            break
        if len(batches) >= max_batches:
            stop = "max_batches"
            break
        if len(history) > 1:
            gain = history[-2] - surplus
            if gain < min_progress * history[-2]:
                stop = "stalled"
                break
            if surplus <= target * n and gain < stall_progress * history[-2]:
                stop = "slow"
                break
        holes = excess < 0
        row, col, src, dst, w = _move_list(cost, labels, slot, assigned, prices, tau_abs,
                                           keep_from)
        dist, hops, sweeps = _distances(src, dst, w, holes, k)
        over = excess > 0
        reach = over & torch.isfinite(dist)
        unreachable = int(excess[over & ~reach].sum())
        batches.append(dict(surplus=surplus, moves=len(row), sweeps=sweeps,
                            unreachable=unreachable))
        if unreachable == surplus:
            stop = "unreachable"
            break
        rise = torch.clamp(dist, max=min(float(dist[reach].max()), tau_abs))
        prices += rise.double()
        # Free moves: reduced cost <= eps after the rise, to fewer hops (by chunks: the
        # move list can hold 3e8 entries next to the rows on the GPU).
        hops32 = hops.clamp(max=_I32_MAX).to(torch.int32)
        free = torch.empty(len(w), dtype=torch.bool, device=w.device)
        for begin in range(0, len(w), _EDGE_CHUNK):
            sl = slice(begin, begin + _EDGE_CHUNK)
            d, s_ = dst[sl], src[sl]
            free[sl] = ((w[sl] + rise[d] - rise[s_]) <= eps_abs) & (hops32[d] < hops32[s_])
        free = torch.nonzero(free)[:, 0]
        row, col, src, dst, w = row[free], col[free], src[free], dst[free], w[free]
        reduced = w + rise[dst] - rise[src]
        units, room = excess.clamp(min=0), (-excess).clamp(min=0)
        if cost.is_cuda and _triton_push is not None:
            # One move per row, its cheapest (ties to the first listed): no sort needed.
            ids = torch.arange(len(row), device=cost.device)
            row64 = row.long()
            best = torch.full((n,), math.inf, device=cost.device).scatter_reduce(
                0, row64, reduced, "amin")
            cheapest = reduced == best[row64]
            first = torch.full((n,), len(row), device=cost.device).scatter_reduce(
                0, row64[cheapest], ids[cheapest], "amin")
            keep = ids == first[row64]
            moved, new_slot = _push_device(row64[keep], col[keep].long(), src[keep],
                                           dst[keep].long(), units, room, hops, k)
        else:
            # One move per row (its cheapest), grouped by cluster, cheapest first.
            order = torch.argsort(reduced, stable=True)
            order = order[torch.argsort(row[order], stable=True)]
            keep = torch.ones(len(order), dtype=torch.bool, device=cost.device)
            keep[1:] = row[order][1:] != row[order][:-1]
            order = order[keep]
            order = order[torch.argsort(src[order], stable=True)]
            moved, new_slot = _push_host(row[order].long(), col[order].long(), src[order],
                                         dst[order].long(), units, room, hops, k)
        old = assigned[moved]
        new = labels[moved, new_slot].long()
        slot[moved] = new_slot
        assigned[moved] = new
        counts.index_add_(0, old, torch.full_like(old, -1))
        counts.index_add_(0, new, torch.ones_like(new))
        batches[-1]["moved"] = len(moved)
        # Free this batch's move list before the next one is built (memory at 1e8 rows).
        del row, col, src, dst, w, reduced, free, moved, new_slot
    return slot, counts, dict(batches=batches, surplus_history=history, stop=stop,
                              rounds=len(batches), moves=sum(b.get("moved", 0) for b in batches),
                              eps=eps_abs, tau=tau_abs)


def _nearest_open(x, rows, centroids, prices, open_mask, chunk_rows=1 << 18):
    """For x[rows], the cluster of smallest distance + price among ``open_mask``:
    (true squared distance (r,) float32, label (r,) int64). Rows are gathered
    chunk by chunk."""
    open_ids = torch.nonzero(open_mask)[:, 0]
    c_open = centroids[open_ids].float().contiguous()
    c16, key = _search_centroids(c_open, -prices[open_ids].float(), x.dtype,
                                 x.is_cuda and _triton_top_m is not None)
    dist = torch.empty(len(rows), dtype=torch.float32, device=x.device)
    lab = torch.empty(len(rows), dtype=torch.int64, device=x.device)
    if c16 is None:
        chunk_rows = max(1024, min(chunk_rows, _SCORES // max(1, len(open_ids))))
    for begin in range(0, len(rows), chunk_rows):
        xr = x[rows[begin:begin + chunk_rows]]
        arg = _top_m(xr, c_open, c16, key, 1)[:, 0].long()
        lab[begin:begin + len(xr)] = open_ids[arg]
        dist[begin:begin + len(xr)] = (xr.float() - c_open[arg]).square().sum(1)
    return dist, lab


def _surplus_rows(x, cost, labels, slot, centroids, prices, counts, cap, chunk_rows=1 << 16):
    """The rows the over-full clusters give up: those cheapest to send elsewhere.

    For every row of an over-full cluster, the exit cost is how much worse
    (distance + price) its best cluster with room is than where it sits; each
    over-full cluster gives up its surplus of rows of smallest exit cost.
    """
    assigned = labels.gather(1, slot[:, None])[:, 0].long()
    excess = counts - cap
    over = excess > 0
    rows = torch.nonzero(over[assigned])[:, 0]
    own = cost[rows].gather(1, slot[rows, None])[:, 0].double() + prices[assigned[rows]]
    open_ids = torch.nonzero(excess < 0)[:, 0]
    c_open = centroids[open_ids].float()
    key = ((centroids[open_ids].double() ** 2).sum(1) + prices[open_ids]).float()
    exit_cost = torch.empty(len(rows), dtype=torch.float64, device=rows.device)
    chunk_rows = max(1024, min(chunk_rows, _SCORES // max(1, len(open_ids))))
    for begin in range(0, len(rows), chunk_rows):
        xr = x[rows[begin:begin + chunk_rows]].float()
        best = torch.addmm(key[None, :], xr, c_open.t(), alpha=-2.0).min(1).values
        exit_cost[begin:begin + len(xr)] = best.double() + xr.double().square().sum(1)
    exit_cost -= own
    cluster = assigned[rows]
    order = torch.argsort(exit_cost, stable=True)
    order = order[torch.argsort(cluster[order], stable=True)]
    rows, cluster = rows[order], cluster[order]
    size = torch.where(over, counts, 0)
    start = torch.cumsum(size, 0) - size
    rank = torch.arange(len(rows), device=rows.device) - start[cluster]
    return rows[rank < excess.clamp(min=0)[cluster]]


def greedy_complete(x, rows, centroids, prices, room, width=16, chunk_rows=1 << 16):
    """Place ``rows`` greedily into the clusters with room.

    Each row lists its ``width`` clusters of smallest distance + price among
    those with room (one search on the device). Then, on the host, round
    after round, every unplaced row proposes to the first cluster of its list
    that still has room, and a cluster accepts its best proposals up to its
    room; rows whose list is used up search again. ``room`` (k,) int64 sums
    to len(rows). Returns (labels (r,) int64, distances (r,) float32 true
    squared).
    """
    device = x.device
    centroids = centroids.float()
    k = len(room)
    out_label = np.empty(len(rows), dtype=np.int64)
    pending = np.arange(len(rows))
    room = room.cpu().numpy().copy()
    norms = (centroids.double() ** 2).sum(1)
    while len(pending):
        open_ids = np.flatnonzero(room > 0)
        assert len(open_ids), "no room left for the remaining rows"
        g = min(width, len(open_ids))
        open_t = torch.from_numpy(open_ids).to(device)
        c_open = centroids[open_t]
        key = (norms[open_t] + prices[open_t]).float()
        rows_t = rows[torch.from_numpy(pending).to(device)]
        vals, lists = [], []
        chunk_rows = max(1024, min(chunk_rows, _SCORES // len(open_ids)))
        for begin in range(0, len(rows_t), chunk_rows):
            xr = x[rows_t[begin:begin + chunk_rows]].float()
            score = torch.addmm(key[None, :], xr, c_open.t(), alpha=-2.0)      # (r, open)
            v, i = torch.topk(score, g, dim=1, largest=False)                  # sorted
            vals.append(v)
            lists.append(i)
        vals = torch.cat(vals).cpu().numpy()
        lists = open_ids[torch.cat(lists).cpu().numpy()]                       # (p, g)
        while len(pending):
            ok = room[lists] > 0
            has = ok.any(1)
            if not has.any():
                break
            first = np.argmax(ok, axis=1)
            target = np.where(has, lists[np.arange(len(lists)), first], k)
            value = vals[np.arange(len(vals)), first]
            order = np.lexsort((value, target))
            target_s = target[order]
            size = np.bincount(target_s, minlength=k + 1)
            start = np.cumsum(size) - size
            rank = np.arange(len(order)) - start[target_s]
            accept = (target_s < k) & (rank < np.r_[room, 0][target_s])
            out_label[pending[order[accept]]] = target_s[accept]
            np.subtract.at(room, target_s[accept], 1)
            keep = np.ones(len(pending), dtype=bool)
            keep[order[accept]] = False
            pending, vals, lists = pending[keep], vals[keep], lists[keep]
    out_label = torch.from_numpy(out_label).to(device)
    out_dist = (x[rows].float() - centroids[out_label]).square().sum(1)
    return out_label, out_dist


def exact_balance(x, centroids, bias, m=_M, anchors=None, graph=_GRAPH, tau=_TAU, eps=_EPS,
                  stall_progress=_STALL_PROGRESS, max_batches=_MAX_BATCHES,
                  target=_TARGET, min_progress=_MIN_PROGRESS, finish=True,
                  exit_tau=_EXIT_TAU, chunk_rows=_CHUNK_ROWS, verbose=False):
    """Exactly balanced labels for x under trained centroids and bias.

    Every cluster receives floor(n / k) or ceil(n / k) rows. Returns
    (distances (n,) float32 true squared distances, labels (n,) int64,
    stats dict). ``stats["bias"]`` is the final bias (-prices), optimal on the
    candidate lists only (see the module docstring).
    """
    timings = {}
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    t0 = time.perf_counter()
    n = x.shape[0]
    centroids = centroids.float().to(x.device)
    bias = bias.float().to(x.device)
    k = centroids.shape[0]
    cost, labels = balance_candidates(x, centroids, bias, m, anchors, graph, chunk_rows)
    m0 = cost.shape[1]
    if finish:
        # A spare column for the exit candidates, unused (infinite cost) until then:
        # growing the arrays later would copy them when memory is fullest.
        cost = torch.cat([cost, torch.full_like(cost[:, :1], math.inf)], 1)
        labels = torch.cat([labels, labels[:, :1]], 1)
    sync()
    timings["candidates_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    prices = -bias.double()
    start_counts = torch.bincount(labels[:, 0].long(), minlength=k)
    cap = exact_capacities(start_counts, n)
    slot, counts, stats = auction_balance(cost, labels, prices, cap, tau, eps,
                                          stall_progress, max_batches, target, min_progress)
    sync()
    timings["auction_s"] = time.perf_counter() - t1
    t_exit = time.perf_counter()
    stats["exit_batches"] = 0
    if finish and bool((counts > cap).any()):
        # Exit candidates: every row also gets its nearest cluster with room, so the
        # cheapest chains can end with one long move instead of a greedy jump.
        # Keep only the exits within ``exit_tau`` (x mean distance) of the row's value.
        # A row's exit is not among its candidates, so it is about as far as its
        # last candidate or farther: only rows whose last candidate is within
        # reach are searched.
        here = cost.gather(1, slot[:, None])[:, 0].double() + prices[
            labels.gather(1, slot[:, None])[:, 0].long()]
        scale = float(cost.gather(1, slot[:, None]).double().mean())
        reach = exit_tau * scale
        near = torch.nonzero(cost[:, m0 - 1].double() + prices[labels[:, m0 - 1].long()] - here
                             <= reach)[:, 0]
        e_dist = torch.full((n,), math.inf, dtype=torch.float32, device=x.device)
        e_lab = torch.zeros(n, dtype=torch.int64, device=x.device)
        if len(near):
            d_near, l_near = _nearest_open(x, near, centroids, prices, counts < cap,
                                           chunk_rows)
            gap = d_near.double() + prices[l_near] - here[near]
            ok = gap <= reach
            # Each cluster with room keeps its _EXITS_PER_CLUSTER cheapest exits: only
            # those can be used, and it bounds the move list at 1e8 rows.
            order = torch.argsort(gap[ok], stable=True)
            order = order[torch.argsort(l_near[ok][order], stable=True)]
            dest = l_near[ok][order]
            size = torch.bincount(dest, minlength=k)
            start = torch.cumsum(size, 0) - size
            rank = torch.arange(len(dest), device=x.device) - start[dest]
            keep = order[rank < _EXITS_PER_CLUSTER]
            rows = near[ok][keep]
            e_dist[rows] = d_near[ok][keep]
            e_lab[rows] = l_near[ok][keep]
        stats["exit_moves"] = int(torch.isfinite(e_dist).sum())
        cost[:, m0] = e_dist
        labels[:, m0] = e_lab.to(torch.int32)
        slot, counts, more = auction_balance(cost, labels, prices, cap, tau, eps,
                                             stall_progress, _EXIT_BATCHES, target,
                                             min_progress, slot=slot, keep_from=m0)
        stats["exit_batches"] = more["rounds"]
        stats["exit_history"] = more["surplus_history"]
        stats["moves"] += more["moves"]
        stats["stop"] = more["stop"]
    sync()
    timings["exit_s"] = time.perf_counter() - t_exit
    t2 = time.perf_counter()
    out_label = labels.gather(1, slot[:, None])[:, 0].long()
    out_dist = cost.gather(1, slot[:, None])[:, 0]
    n_greedy = 0
    greedy_cost_gap = 0.0
    greedy_regret_q = None
    if bool((counts > cap).any()):
        rows = _surplus_rows(x, cost, labels, slot, centroids, prices, counts, cap)
        n_greedy = len(rows)
        counts.index_add_(0, out_label[rows], torch.full_like(rows, -1))
        before = float(out_dist[rows].double().sum())
        g_label, g_dist = greedy_complete(x, rows, centroids, prices, cap - counts)
        out_label[rows] = g_label
        out_dist[rows] = g_dist
        counts.index_add_(0, g_label, torch.ones_like(g_label))
        greedy_cost_gap = float(g_dist.double().sum()) - before
        regret = (g_dist.double() - cost.gather(1, slot[rows, None])[:, 0].double()) / float(
            cost[:, 0].double().mean())
        greedy_regret_q = [float(v) for v in torch.quantile(
            regret.float(), torch.tensor([0.5, 0.99, 1.0], device=regret.device))]
    sync()
    timings["greedy_s"] = time.perf_counter() - t2
    assert bool((counts == cap).all())
    # Rows whose current value beats every non-candidate (certified optimal over all k).
    value = out_dist.double() + prices[out_label]
    worst_start = cost[:, m0 - 1].double() - bias.double()[labels[:, m0 - 1].long()]
    certified = float((value <= worst_start).double().mean()) if m0 < k else 1.0
    timings["total_s"] = time.perf_counter() - t0
    stats.update(timings)
    stats.update(n=n, k=k, m=m0, n_greedy=n_greedy,
                 greedy_fraction=n_greedy / max(n, 1), greedy_cost_gap=greedy_cost_gap,
                 greedy_regret_q=greedy_regret_q,
                 certified_fraction=certified, start_surplus=stats["surplus_history"][0],
                 bias=(-prices).float())
    if verbose:
        print("exact_balance: n %d k %d m %d, %d batches, start surplus %d, greedy %d rows "
              "(%.4f%%), %.3f s (candidates %.3f, auction %.3f, greedy %.3f)"
              % (n, k, stats["m"], stats["rounds"], stats["start_surplus"], n_greedy,
                 100 * stats["greedy_fraction"], timings["total_s"], timings["candidates_s"],
                 timings["auction_s"], timings["greedy_s"]))
    return out_dist, out_label, stats
