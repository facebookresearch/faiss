# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Exact balancing after balanced E-means, on NumPy arrays, searched by Faiss.

The NumPy counterpart of ``e_means_balanced_exact`` (same steps, same definitions and
defaults; see that module's docstring). Every cluster receives floor(n / k)
or ceil(n / k) rows. The candidate search is ``faiss.knn_L2sqr`` with the
bias folded into the centroid norms, as in ``NumpyEmeansBalanced``. It never
imports PyTorch.
"""

import time

import numpy as np
import faiss


__all__ = [
    "exact_balance",
    "exact_capacities",
    "balance_candidates",
    "biased_anchors",
    "auction_balance",
    "greedy_complete",
]


_M = 8
_GRAPH = 32
_TAU = 0.25
_EPS = 0.05
_TARGET = 1e-3
_EXIT_TAU = 0.5
_MOVES_PER_ROW = 3
_EXIT_BATCHES = 4
_EXITS_PER_CLUSTER = 64
_MIN_PROGRESS = 0.01
_STALL_PROGRESS = 0.05
_MAX_BATCHES = 256
_MAX_SWEEPS = 4096
_CHUNK_ROWS = 1 << 18


def exact_capacities(counts, n):
    """Capacities summing to n: q + 1 for the r most filled clusters, else q.

    n = q k + r; ties go to the lower index.
    """
    k = len(counts)
    base, extra = divmod(int(n), k)
    cap = np.full(k, base, dtype=np.int64)
    if extra:
        order = np.argsort(-np.asarray(counts, dtype=np.int64), kind="stable")
        cap[order[:extra]] += 1
    return cap


def _knn(x, centroids, norms, m):
    """faiss.knn_L2sqr with precomputed (biased) centroid norms; see NumpyEmeansBalanced._knn."""
    n, d = x.shape
    distance = np.empty((n, m), dtype="float32")
    labels = np.empty((n, m), dtype="int64")
    if n == 0:
        return distance, labels
    x = np.ascontiguousarray(x, dtype="float32")
    cvar = faiss.cvar
    saved = cvar.distance_compute_blas_threshold, cvar.distance_compute_blas_database_bs
    cvar.distance_compute_blas_threshold = 0
    if n >= 4096:
        cvar.distance_compute_blas_database_bs = max(saved[1], len(centroids) // 8)
    try:
        faiss.knn_L2sqr(
            faiss.swig_ptr(x), faiss.swig_ptr(centroids), d, n, len(centroids), m,
            faiss.swig_ptr(distance), faiss.swig_ptr(labels), faiss.swig_ptr(norms))
    finally:
        cvar.distance_compute_blas_threshold, cvar.distance_compute_blas_database_bs = saved
    return distance, labels


def _biased_norms(centroids, bias):
    """||c_j||^2 + (max(b) - b_j) (>= ||c_j||^2: Faiss clamps negative distances) and max(b)."""
    norms = np.empty(len(centroids), dtype="float32")
    faiss.fvec_norms_L2sqr(faiss.swig_ptr(norms), faiss.swig_ptr(centroids),
                           centroids.shape[1], len(centroids))
    shift = np.float32(bias.max())
    return norms + (shift - bias), shift


def biased_anchors(x, centroids, bias, chunk_rows=_CHUNK_ROWS):
    """Each row's centroid of smallest biased distance (the deployed routing), (n,) int64."""
    centroids = np.ascontiguousarray(centroids, dtype="float32")
    norms, _ = _biased_norms(centroids, np.asarray(bias, dtype="float32"))
    out = np.empty(len(x), dtype=np.int64)
    for begin in range(0, len(x), chunk_rows):
        rows = x[begin:begin + chunk_rows]
        out[begin:begin + len(rows)] = _knn(rows, centroids, norms, 1)[1][:, 0]
    return out


def balance_candidates(x, centroids, bias, m=_M, anchors=None, graph=_GRAPH,
                       chunk_rows=_CHUNK_ROWS):
    """Each row's m candidate centroids and the true squared distances to them.

    The m of smallest biased distance among the ``graph`` centroids nearest
    (in biased distance) to the row's anchor, its biased nearest centroid
    (``e_means_balanced_exact.balance_candidates``). Distances by direct difference
    (``faiss.fvec_L2sqr_by_idx``). Returns (cost (n, m) float32, labels
    (n, m) int32), each row ordered by biased distance.
    """
    n, d = x.shape
    k = len(centroids)
    m = min(int(m), k)
    g = max(m, min(int(graph), k))
    centroids = np.ascontiguousarray(centroids, dtype="float32")
    bias = np.asarray(bias, dtype="float32")
    if anchors is None:
        anchors = biased_anchors(x, centroids, bias, chunk_rows)
    anchors = np.asarray(anchors, dtype=np.int64)
    norms, _ = _biased_norms(centroids, bias)
    graph_ids = _knn(centroids, centroids, norms, g)[1]                     # (k, g)
    cost = np.empty((n, m), dtype="float32")
    labels = np.empty((n, m), dtype="int32")
    for begin in range(0, n, chunk_rows):
        end = min(n, begin + chunk_rows)
        rows = np.ascontiguousarray(x[begin:end], dtype="float32")
        cand = np.ascontiguousarray(graph_ids[anchors[begin:end]])            # (r, g)
        dist = np.empty(cand.shape, dtype="float32")
        faiss.fvec_L2sqr_by_idx(faiss.swig_ptr(dist), faiss.swig_ptr(rows),
                                faiss.swig_ptr(centroids), faiss.swig_ptr(cand),
                                d, end - begin, g)
        biased = dist - bias[cand].astype(np.float64)
        ar = np.arange(end - begin)[:, None]
        pick = np.argpartition(biased, m - 1, axis=1)[:, :m] if m < g else np.tile(
            np.arange(g), (end - begin, 1))
        pick = pick[ar, np.argsort(biased[ar, pick], axis=1, kind="stable")]
        labels[begin:end] = cand[ar, pick]
        cost[begin:end] = dist[ar, pick]
    return cost, labels


def _move_list(cost, labels, slot, assigned, prices, tau, keep_from=None,
               chunk_rows=_CHUNK_ROWS):
    """The moves of reduced cost <= tau (any finite cost from column ``keep_from``
    on): (row, slot, from, to, reduced cost)."""
    out = []
    m = cost.shape[1]
    for begin in range(0, len(cost), chunk_rows):
        end = min(len(cost), begin + chunk_rows)
        lab = labels[begin:end]
        net = cost[begin:end] + prices[lab]                                 # (r, m) float64
        ar = np.arange(end - begin)
        s = slot[begin:end]
        w = net - net[ar, s][:, None]
        w[ar, s] = np.inf
        keep = w <= tau
        width = m if keep_from is None else keep_from
        if _MOVES_PER_ROW < width - 1:
            # Only each row's cheapest moves: shortest chains rarely need more. Only rows
            # with more cheap moves than that need the cut.
            q = _MOVES_PER_ROW
            many = np.flatnonzero(keep[:, :width].sum(1) > q)
            if len(many):
                sub = w[many, :width]
                cut = np.sort(sub, axis=1)[:, q - 1:q]
                keep[many, :width] &= sub <= cut
        if keep_from is not None:
            keep[:, keep_from:] = np.isfinite(w[:, keep_from:])
        flat = np.flatnonzero(keep)
        row, col = np.divmod(flat, m)
        out.append((row + begin, col, assigned[row + begin].astype(np.int32),
                    lab.ravel()[flat].astype(np.int64),
                    np.maximum(w.ravel()[flat], 0).astype(np.float32)))
    return [np.concatenate(parts) for parts in zip(*out)]


def _cheapest_pairs(src, dst, w, k):
    """The moves collapsed to cluster pairs, each with its cheapest row's cost:
    the chains' costs only depend on these. Returns (src, dst, w), by source."""
    if k <= 1 << 16 and len(src):
        # One sort of (pair, cost bits) packed in 64 bits: costs are >= 0 floats, whose
        # bits order like the values, so each pair's first entry is its cheapest.
        packed = ((src.astype(np.uint64) * np.uint64(k) + dst.astype(np.uint64)) << np.uint64(32)
                  | w.astype(np.float32).view(np.uint32).astype(np.uint64))
        packed.sort()
        pair = packed >> np.uint64(32)
        heads = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
        pair = pair[heads].astype(np.int64)
        cost = (packed[heads] & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.float32)
        return pair // k, pair % k, cost
    key = src.astype(np.int64) * k + dst
    order = np.argsort(key)
    key = key[order]
    heads = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else key
    return key[heads] // k, key[heads] % k, np.minimum.reduceat(w[order], heads) if len(
        key) else w


def _distances(src, dst, w, holes, k):
    """Cheapest chain of moves to a cluster with room (Bellman-Ford, Jacobi sweeps).

    The moves are first collapsed to cluster pairs (``_cheapest_pairs``). Each
    sweep only relaxes the pairs into clusters whose distance changed in the
    previous sweep. While many changed, the moves stay grouped by source
    and the minimum per source is one ``reduceat``; once few changed, their
    incoming moves are read through an index by destination. Returns
    (distance (k,) float32, hops (k,) int64: the sweep that last lowered it,
    sweeps); see ``e_means_balanced_exact._distances``.
    """
    dist = np.full(k, np.inf, dtype=np.float32)
    dist[holes] = 0
    hops = np.full(k, 1 << 31, dtype=np.int64)
    hops[holes] = 0
    src, dst, w = _cheapest_pairs(src, dst, w, k)
    by_src = np.argsort(src)
    src_s, dst_s, w_s = src[by_src], dst[by_src], w[by_src]
    by_dst = None
    changed = np.flatnonzero(holes)
    sweeps = 0
    while len(changed) and sweeps < _MAX_SWEEPS:
        sweeps += 1
        if len(changed) * 16 > k:
            mask = np.zeros(k, dtype=bool)
            mask[changed] = True
            edges = np.flatnonzero(mask[dst_s])
            if len(edges) == 0:
                break
            owner = src_s[edges]
            heads = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
            best = np.minimum.reduceat(w_s[edges] + dist[dst_s[edges]], heads)
            owner = owner[heads]
        else:
            if by_dst is None:
                by_dst = np.argsort(dst)
                count = np.bincount(dst, minlength=k)
                start = np.cumsum(count) - count
            lens = count[changed]
            total = int(lens.sum())
            if total == 0:
                break
            first = np.repeat(start[changed] - np.cumsum(lens) + lens, lens)
            edges = by_dst[first + np.arange(total)]
            owner, inverse = np.unique(src[edges], return_inverse=True)
            best = np.full(len(owner), np.inf, dtype=np.float32)
            np.minimum.at(best, inverse, w[edges] + dist[dst[edges]])
        better = best < dist[owner]
        changed = owner[better]
        dist[changed] = best[better]
        hops[changed] = sweeps
    return dist, hops, sweeps


def _ranks(groups):
    """Position of each element among the equal values before it (groups (e,) int64)."""
    if len(groups) == 0:
        return groups
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    heads = np.r_[True, g[1:] != g[:-1]]
    first = np.maximum.accumulate(np.where(heads, np.arange(len(g)), 0))
    rank = np.empty(len(g), dtype=np.int64)
    rank[order] = np.arange(len(g)) - first
    return rank


def _push(rows, slots, src, dst, units, room, hops, k):
    """Send the surplus down free moves, towards clusters that still have room.

    rows/slots/src/dst (e,): the free moves, at most one per row, each to a
    cluster of fewer hops, grouped by source in order of preference. units
    (k,): rows each cluster must shed; room (k,): rows a cluster can take.
    First, from the clusters with room up (by hops), each cluster's capacity:
    how many of its free moves lead to a cluster that has capacity. Then,
    from the top down, each cluster sends its units (its own and those that
    reached it) along its free moves whose target still has capacity, each
    target accepting up to its remaining capacity. Each row moves at most
    once. Returns (moved rows, their slots).
    """
    level = hops[src]
    by_level = np.argsort(level, kind="stable")
    bounds = np.flatnonzero(np.r_[True, level[by_level][1:] != level[by_level][:-1], True])
    groups = [by_level[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    cap = room.astype(np.int64).copy()
    for sel in groups:                                    # by increasing hops
        cap += np.bincount(src[sel], weights=cap[dst[sel]] > 0, minlength=k).astype(np.int64)
    units = units.astype(np.int64).copy()
    moved = np.zeros(len(rows), dtype=bool)
    for sel in reversed(groups):                          # by decreasing hops
        if not units[src[sel]].any():
            continue
        sel = np.sort(sel)                                # keep each source's preference order
        sel = sel[cap[dst[sel]] > 0]
        sel = sel[_ranks(src[sel]) < units[src[sel]]]
        sel = sel[_ranks(dst[sel]) < cap[dst[sel]]]
        if len(sel) == 0:
            continue
        moved[sel] = True
        np.subtract.at(units, src[sel], 1)
        np.add.at(units, dst[sel], 1)
        np.subtract.at(cap, dst[sel], 1)
    return rows[moved], slots[moved]


def auction_balance(cost, labels, prices, cap, tau=_TAU, eps=_EPS,
                    stall_progress=_STALL_PROGRESS, max_batches=_MAX_BATCHES,
                    target=_TARGET, min_progress=_MIN_PROGRESS, slot=None, keep_from=None):
    """The auction's batches on the candidate graph (``e_means_balanced_exact.auction_balance``).

    prices (k,) float64 is modified in place. Returns (slot (n,) int64,
    counts (k,) int64, stats dict).
    """
    n, m = cost.shape
    k = len(cap)
    if slot is None:
        slot = np.empty(n, dtype=np.int64)
        for begin in range(0, n, _CHUNK_ROWS):
            end = min(n, begin + _CHUNK_ROWS)
            slot[begin:end] = (cost[begin:end] + prices[labels[begin:end]]).argmin(1)
    assigned = labels[np.arange(n), slot].astype(np.int64)
    counts = np.bincount(assigned, minlength=k).astype(np.int64)
    scale = float(cost[np.arange(n), slot].mean(dtype=np.float64))
    tau_abs, eps_abs = tau * scale, eps * scale
    history = []
    batches = []
    stop = "balanced"
    while True:
        excess = counts - cap
        surplus = int(np.maximum(excess, 0).sum())
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
        reach = over & np.isfinite(dist)
        unreachable = int(excess[over & ~reach].sum())
        batches.append(dict(surplus=surplus, moves=len(row), sweeps=sweeps,
                            unreachable=unreachable))
        if unreachable == surplus:
            stop = "unreachable"
            break
        rise = np.minimum(dist, min(float(dist[reach].max()), tau_abs))
        prices += rise
        reduced = w + rise[dst] - rise[src]
        free = (reduced <= eps_abs) & (hops[dst] < hops[src])
        row, col, src, dst, reduced = row[free], col[free], src[free], dst[free], reduced[free]
        # One move per row (its cheapest), grouped by cluster, cheapest first.
        order = np.lexsort((reduced, row))
        keep = np.r_[True, row[order][1:] != row[order][:-1]]
        order = order[keep]
        order = order[np.argsort(src[order], kind="stable")]
        moved, new_slot = _push(row[order], col[order], src[order], dst[order],
                                np.maximum(excess, 0), np.maximum(-excess, 0), hops, k)
        old = assigned[moved]
        new = labels[moved, new_slot].astype(np.int64)
        slot[moved] = new_slot
        assigned[moved] = new
        np.subtract.at(counts, old, 1)
        np.add.at(counts, new, 1)
        batches[-1]["moved"] = len(moved)
    return slot, counts, dict(batches=batches, surplus_history=history, stop=stop,
                              rounds=len(batches), moves=sum(b.get("moved", 0) for b in batches),
                              eps=eps_abs, tau=tau_abs)


def _nearest_open(x, rows, centroids, prices, open_mask, chunk_rows=_CHUNK_ROWS):
    """For x[rows], the cluster of smallest distance + price among ``open_mask``:
    (true squared distance (r,) float32, label (r,) int64), chunk by chunk."""
    open_ids = np.flatnonzero(open_mask)
    c_open = np.ascontiguousarray(centroids[open_ids])
    low = prices[open_ids].min()
    norms = ((c_open.astype(np.float64) ** 2).sum(1) + (prices[open_ids] - low)).astype("float32")
    dist = np.empty(len(rows), dtype=np.float32)
    lab = np.empty(len(rows), dtype=np.int64)
    for begin in range(0, len(rows), chunk_rows):
        xr = np.ascontiguousarray(x[rows[begin:begin + chunk_rows]], dtype="float32")
        d1, l1 = _knn(xr, c_open, norms, 1)
        lab[begin:begin + len(xr)] = open_ids[l1[:, 0]]
        dist[begin:begin + len(xr)] = np.maximum(
            d1[:, 0] - (prices[open_ids[l1[:, 0]]] - low), 0)
    return dist, lab


def _surplus_rows(x, cost, labels, slot, centroids, prices, counts, cap):
    """The rows the over-full clusters give up: those cheapest to send elsewhere

    (``e_means_balanced_exact._surplus_rows``).
    """
    n = len(cost)
    assigned = labels[np.arange(n), slot]
    excess = counts - cap
    over = excess > 0
    rows = np.flatnonzero(over[assigned])
    own = cost[rows, slot[rows]] + prices[assigned[rows]]
    open_ids = np.flatnonzero(excess < 0)
    c_open = np.ascontiguousarray(centroids[open_ids])
    low = prices[open_ids].min()
    norms = ((c_open.astype(np.float64) ** 2).sum(1) + (prices[open_ids] - low)).astype("float32")
    best = _knn(np.ascontiguousarray(x[rows], dtype="float32"), c_open, norms, 1)[0][:, 0]
    exit_cost = best + low - own
    cluster = assigned[rows]
    order = np.lexsort((exit_cost, cluster))
    rows, cluster = rows[order], cluster[order]
    size = np.where(over, counts, 0)
    start = np.cumsum(size) - size
    rank = np.arange(len(rows)) - start[cluster]
    return rows[rank < np.maximum(excess, 0)[cluster]]


def greedy_complete(x, rows, centroids, prices, room):
    """Place ``rows`` greedily into the clusters with room
    (``e_means_balanced_exact.greedy_complete``).

    Returns (labels (r,) int64, distances (r,) float32 true squared).
    """
    centroids = np.ascontiguousarray(centroids, dtype="float32")
    out_label = np.empty(len(rows), dtype=np.int64)
    out_dist = np.empty(len(rows), dtype=np.float32)
    pending = np.arange(len(rows))
    room = room.copy()
    norms = (centroids.astype(np.float64) ** 2).sum(1)
    while len(pending):
        open_ids = np.flatnonzero(room > 0)
        assert len(open_ids), "no room left for the remaining rows"
        c_open = np.ascontiguousarray(centroids[open_ids])
        # ||x - c||^2 + price - min(price) >= 0: Faiss clamps negative distances.
        key = (norms[open_ids] + (prices[open_ids] - prices[open_ids].min())).astype("float32")
        xr = np.ascontiguousarray(x[rows[pending]], dtype="float32")
        dist, arg = _knn(xr, c_open, key, 1)
        best_val, target = dist[:, 0], open_ids[arg[:, 0]]
        order = np.lexsort((best_val, target))
        target_s = target[order]
        size = np.bincount(target_s, minlength=len(room))
        start = np.cumsum(size) - size
        rank = np.arange(len(order)) - start[target_s]
        accept = rank < room[target_s]
        chosen = pending[order[accept]]
        dest = target_s[accept]
        out_label[chosen] = dest
        diff = np.asarray(x[rows[chosen]], dtype=np.float32) - centroids[dest]
        out_dist[chosen] = (diff * diff).sum(1)
        np.subtract.at(room, dest, 1)
        keep = np.ones(len(pending), dtype=bool)
        keep[order[accept]] = False
        pending = pending[keep]
    return out_label, out_dist


def exact_balance(x, centroids, bias, m=_M, anchors=None, graph=_GRAPH, tau=_TAU, eps=_EPS,
                  stall_progress=_STALL_PROGRESS, max_batches=_MAX_BATCHES,
                  target=_TARGET, min_progress=_MIN_PROGRESS, finish=True,
                  exit_tau=_EXIT_TAU, chunk_rows=_CHUNK_ROWS, verbose=False):
    """Exactly balanced labels for x under trained centroids and bias.

    Returns (distances (n,) float32, labels (n,) int64, stats dict);
    ``stats["bias"]`` is the final bias (-prices).
    """
    timings = {}
    t0 = time.perf_counter()
    n = len(x)
    centroids = np.ascontiguousarray(centroids, dtype="float32")
    bias = np.asarray(bias, dtype="float32")
    k = len(centroids)
    cost, labels = balance_candidates(x, centroids, bias, m, anchors, graph, chunk_rows)
    m0 = cost.shape[1]
    if finish:
        # A spare column for the exit candidates, unused (infinite cost) until then.
        cost = np.concatenate([cost, np.full((n, 1), np.inf, dtype=cost.dtype)], axis=1)
        labels = np.concatenate([labels, labels[:, :1]], axis=1)
    timings["candidates_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    prices = -bias.astype(np.float64)
    cap = exact_capacities(np.bincount(labels[:, 0], minlength=k), n)
    slot, counts, stats = auction_balance(cost, labels, prices, cap, tau, eps,
                                          stall_progress, max_batches, target, min_progress)
    timings["auction_s"] = time.perf_counter() - t1
    t_exit = time.perf_counter()
    stats["exit_batches"] = 0
    if finish and (counts > cap).any():
        ar = np.arange(n)
        here = cost[ar, slot] + prices[labels[ar, slot]]
        scale = float(cost[ar, slot].mean(dtype=np.float64))
        reach = exit_tau * scale
        near = np.flatnonzero(cost[:, m0 - 1] + prices[labels[:, m0 - 1]] - here <= reach)
        e_dist = np.full(n, np.inf, dtype=np.float32)
        e_lab = np.zeros(n, dtype=np.int64)
        if len(near):
            d_near, l_near = _nearest_open(x, near, centroids, prices, counts < cap,
                                           chunk_rows)
            gap = d_near + prices[l_near] - here[near]
            ok = np.flatnonzero(gap <= reach)
            # Each cluster with room keeps its _EXITS_PER_CLUSTER cheapest exits.
            order = ok[np.lexsort((gap[ok], l_near[ok]))]
            dest = l_near[order]
            size = np.bincount(dest, minlength=k)
            rank = np.arange(len(dest)) - (np.cumsum(size) - size)[dest]
            keep = order[rank < _EXITS_PER_CLUSTER]
            e_dist[near[keep]] = d_near[keep]
            e_lab[near[keep]] = l_near[keep]
        stats["exit_moves"] = int(np.isfinite(e_dist).sum())
        cost[:, m0] = e_dist
        labels[:, m0] = e_lab.astype(np.int32)
        slot, counts, more = auction_balance(cost, labels, prices, cap, tau, eps,
                                             stall_progress, _EXIT_BATCHES, target,
                                             min_progress, slot=slot, keep_from=m0)
        stats["exit_batches"] = more["rounds"]
        stats["exit_history"] = more["surplus_history"]
        stats["moves"] += more["moves"]
        stats["stop"] = more["stop"]
    timings["exit_s"] = time.perf_counter() - t_exit
    t2 = time.perf_counter()
    ar = np.arange(n)
    out_label = labels[ar, slot].astype(np.int64)
    out_dist = cost[ar, slot]
    n_greedy = 0
    greedy_cost_gap = 0.0
    greedy_regret_q = None
    if (counts > cap).any():
        rows = _surplus_rows(x, cost, labels, slot, centroids, prices, counts, cap)
        n_greedy = len(rows)
        np.subtract.at(counts, out_label[rows], 1)
        before = float(out_dist[rows].sum(dtype=np.float64))
        g_label, g_dist = greedy_complete(x, rows, centroids, prices, cap - counts)
        out_label[rows] = g_label
        out_dist[rows] = g_dist
        np.add.at(counts, g_label, 1)
        greedy_cost_gap = float(g_dist.sum(dtype=np.float64)) - before
        regret = (g_dist - cost[rows, slot[rows]]) / float(cost[:, 0].mean(dtype=np.float64))
        greedy_regret_q = [float(v) for v in np.quantile(regret, [0.5, 0.99, 1.0])]
    timings["greedy_s"] = time.perf_counter() - t2
    assert (counts == cap).all()
    value = out_dist.astype(np.float64) + prices[out_label]
    worst_start = cost[:, m0 - 1].astype(np.float64) - bias[labels[:, m0 - 1]]
    certified = float((value <= worst_start).mean()) if m0 < k else 1.0
    timings["total_s"] = time.perf_counter() - t0
    stats.update(timings)
    stats.update(n=n, k=k, m=m0, n_greedy=n_greedy,
                 greedy_fraction=n_greedy / max(n, 1), greedy_cost_gap=greedy_cost_gap,
                 greedy_regret_q=greedy_regret_q,
                 certified_fraction=certified, start_surplus=stats["surplus_history"][0],
                 bias=(-prices).astype(np.float32))
    if verbose:
        print("exact_balance: n %d k %d m %d, %d batches, start surplus %d, greedy %d rows "
              "(%.4f%%), %.3f s (candidates %.3f, auction %.3f, greedy %.3f)"
              % (n, k, stats["m"], stats["rounds"], stats["start_surplus"], n_greedy,
                 100 * stats["greedy_fraction"], timings["total_s"], timings["candidates_s"],
                 timings["auction_s"], timings["greedy_s"]))
    return out_dist, out_label, stats
