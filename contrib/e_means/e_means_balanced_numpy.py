# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Balanced (auxiliary-loss-free) E-means on NumPy arrays, assigned by Faiss.

Each centroid j carries a bias b_j, and a row goes to the centroid that
minimises ||x - c_j||^2 - b_j. After every batch the bias moves every cluster
toward an equal share of the batch, so the clusters come out balanced without
an extra loss term. The bias rides in the centroid norms that
``faiss.knn_L2sqr`` already takes (``y_norm2`` = ||c_j||^2 - b_j), so the
search is Faiss's own exact BLAS search, with no extra pass.

``NumpyEmeansBalanced`` is the NumPy counterpart of the tensor class
``EmeansBalanced`` (``e_means_balanced``): same constructor, same defaults,
same step. It never imports PyTorch.
"""

import math
import time

import numpy as np
import faiss

from .base import _INIT_USAGE, _epoch_seed, _resolve_recipe
from .e_means import NumpyEmeans, _numpy_unused_threshold


__all__ = [
    "NumpyEmeansBalanced",
    "split_epochs",
    "cap_first_pass",
    "cap_round_evict",
]


# Smoothing of the running mean assigned squared distance D, which scales the
# bias step: D <- (1 - rate) * D + rate * (this batch's mean).
_DISTANCE_SCALE_RATE = 0.1

# Polish-rate cap: the polish's bias rate is min(u, this).
_POLISH_U_CAP = 0.02

# Share of the ``nepochs`` budget spent polishing when ``polish_epochs`` is
# not given: 10 of the default 25 passes, the paper's 15 + 10.
_POLISH_FRACTION = 0.4

# Multiplicative split perturbation (faiss split_clusters): the donor and its
# child become c (1 -/+ eps s), s = (+1, -1, +1, ...).
_SPLIT_EPS = 1e-4

# The cap's anneal divides the factor by the bias schedule, floored here.
_CAP_SCALE_FLOOR = 1e-6

# Updates between dead-cluster checks.
_CHECK_UNUSED_EVERY = 5

# Faiss's BLAS search processes the centroids in blocks of
# distance_compute_blas_database_bs (1024 by default). With at least this many
# rows, a search uses blocks of k / 8 centroids instead (never fewer than the
# default). Measured on 16 Sapphire Rapids cores, top-1, d = 128: 32768 rows x
# 16384 centroids 166 -> 115 ms, 65536 x 65536 1236 -> 923 ms, unchanged at k
# = 4096 and below. (The default block also lets Faiss pick its
# database-parallel search once k >= threads * 1024, which is the slower one
# for a full batch.) Fewer rows keep the default, where that path suits them.
_WIDE_SEARCH_ROWS = 4096
_DATABASE_BLOCK_DIVISOR = 8

# The rounds cap needs a runner-up only for the rows of over-full clusters.
# When fewer than this fraction of the previous step's rows were in one, the
# step searches top-1 and then top-2 for those rows only; otherwise top-2 for
# the whole batch (2-14% dearer than top-1, cheaper than a second search of
# more than a few percent of the batch). On fb-ssnpp-1M, K = 4096, the cap
# binds on 56% of the steps; late in training and in the polish it does not.
_LAZY_RUNNER_UP_FRACTION = 0.03

# A masked search (the repair rounds) runs on the open
# clusters only, copying them, when rows x full clusters > this x open
# clusters: skipping the full clusters then saves more search than the copy
# costs (about 2 flops per skipped row-centroid-dimension at ~1 TFLOP/s
# against 4 bytes per copied centroid-dimension at ~5 GB/s).
_COMPACT_RATIO = 400


# --------------------------------------------------------------------------
# Recipe math on NumPy arrays. Same definitions as the module-level tensor
# functions of ``e_means_balanced``; kept here so this module needs no PyTorch.
# --------------------------------------------------------------------------

def _cosine_scale(step, total):
    """0.5 * (1 + cos(pi * step / total)): 1 at step 0, down to 0, no floor."""
    fraction = min(step, total) / max(1, total)
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def _cap_rows(batch, k, factor, scale=1.0, anneal=True):
    """Rows one cluster may keep from a batch: ceil(factor / scale * batch / k)."""
    if anneal:
        factor = factor / max(scale, _CAP_SCALE_FLOOR)
    return max(1, int(math.ceil(min(factor, 1e9) * batch / k)))


def split_epochs(nepochs, polish_epochs=None):
    """(training passes, polish passes) of a budget of ``nepochs`` passes.

    ``polish_epochs=None`` takes round(0.4 * nepochs) polish passes (10 of
    the default 25); the rest train. The polish needs at least one training
    pass before it.
    """
    if float(nepochs) != int(nepochs) or int(nepochs) < 1:
        raise ValueError("nepochs must be a positive integer")
    nepochs = int(nepochs)
    if polish_epochs is None:
        polish_epochs = int(math.floor(_POLISH_FRACTION * nepochs + 0.5))
    if float(polish_epochs) != int(polish_epochs) or not 0 <= int(polish_epochs) < nepochs:
        raise ValueError("polish_epochs must be an integer in [0, nepochs)")
    return nepochs - int(polish_epochs), int(polish_epochs)


def _resolve_balanced_recipe(n, k, train_epochs, polish_epochs, batch_law,
                             batch_size, alpha0):
    """Batch, step and step counts of a balanced run.

    The batch and the initial EMA step come from the public laws at T =
    ``train_epochs``. Every pass drops its short last batch: a pass is
    floor(n / B) full batches.
    """
    batch, alpha0 = _resolve_recipe(n, k, train_epochs, batch_law, batch_size, alpha0)
    steps_per_pass = n // batch
    return dict(
        batch_size=batch, alpha0=alpha0, steps_per_pass=steps_per_pass,
        pass_rows=steps_per_pass * batch,
        train_steps=train_epochs * steps_per_pass,
        polish_steps=polish_epochs * steps_per_pass)


def _sortable(values):
    """int64 keys ordered like the FP32 ``values``, as the tensor class's."""
    bits = np.ascontiguousarray(values, dtype="float32").view("int32").astype("int64")
    # Negative floats: flip the magnitude bits so larger magnitude sorts lower.
    return np.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def _ranks_in_cluster(codes, key, k):
    """Sort rows by (cluster, key) stably, ties by position.

    ``key`` is a non-negative int64 below 2**34. Returns the visiting order,
    each sorted row's rank inside its cluster and the per-cluster counts.
    """
    codes = codes.astype("int64", copy=False)
    order = np.argsort(codes * (1 << 34) + key, kind="stable")  # (n,)
    counts = np.bincount(codes, minlength=k)  # (k,)
    start = np.cumsum(counts) - counts  # (k,)
    rank = np.arange(len(codes)) - start[codes[order]]  # (n,)
    return order, rank, counts


def cap_first_pass(codes, margin, second, cap, k):
    """One-pass capacity cap: over-full clusters send their weakest rows to the runner-up.

    ``codes`` (n,) are the biased first choices, ``margin`` (n,) FP32 the
    biased runner-up score minus the best score and ``second`` (n,) the
    runner-up index. A cluster keeps its ``cap`` rows of largest margin (ties
    by lower batch position); the others take their runner-up. Returns the new
    codes (int64) and a bool mask of the moved rows. A runner-up may end above
    the cap; the repair rounds handle that. Only the rows of over-full
    clusters are ranked.
    """
    codes = codes.astype("int64", copy=False)
    moved = np.zeros(len(codes), dtype=bool)
    over = np.flatnonzero(np.bincount(codes, minlength=k)[codes] > cap)
    if len(over):
        # Descending margin: negate the ordered key, offset into [0, 2**33).
        order, rank, _ = _ranks_in_cluster(
            codes[over], (1 << 31) - _sortable(margin[over]), k)
        moved[over[order[rank >= cap]]] = True
    return np.where(moved, second, codes).astype("int64"), moved


def cap_round_evict(codes, arrived, score, cap, k):
    """One repair round's eviction: which rows leave clusters above the cap.

    ``arrived`` (n,) bool marks rows that reached their cluster by a move;
    the cluster's own first-choice rows keep priority, then arrivals by lowest
    ``score`` (FP32, ties by lower batch position). Rows ranked at or past
    ``cap`` are evicted. Returns the eviction mask (n,) and the mask (k,) of
    full clusters (holding at least ``cap`` rows), which the evicted rows
    must avoid. Only the rows of over-full clusters are ranked.
    """
    codes = codes.astype("int64", copy=False)
    counts = np.bincount(codes, minlength=k)
    evict = np.zeros(len(codes), dtype=bool)
    over = np.flatnonzero(counts[codes] > cap)
    if len(over):
        key = arrived[over].astype("int64") * (1 << 33) + (
            _sortable(score[over]) + (1 << 31))
        order, rank, _ = _ranks_in_cluster(codes[over], key, k)
        evict[over[order[rank >= cap]]] = True
    return evict, counts >= cap


def _band_step(lam, mu, bias, counts, eta):
    """The exact band, both bounds at the fair share Q = mean count, in place."""
    counts = counts.astype("float32", copy=False)
    share = counts.mean(dtype="float32")
    relative = (counts - share) / max(share, np.float32(1e-6))  # (k,)
    step = np.float32(eta) * relative
    np.maximum(lam + step, 0, out=lam)
    np.maximum(mu - step, 0, out=mu)
    np.subtract(mu, lam, out=bias)


def _select_inplace(dead, bias, quantile=0.5):
    """In-place revival: dead clusters whose bias is below the live median.

    The median is the lower one, as ``torch.median`` gives it (the k-th
    smallest live bias, k = floor((n - 1) q) + 1). Returns (in-place mask,
    reference bias).
    """
    live = bias[~dead]
    if len(live):
        kth = int((len(live) - 1) * quantile)
        reference = np.partition(live, kth)[kth]
    else:
        reference = np.float32(0)
    return dead & (bias < reference), np.float32(reference)


def _largest_donors(weights, receivers, dead):
    """Donors with the largest ``weights``, one receiver each, no draw.

    Neither the ``receivers`` nor the ``dead`` clusters (every cluster found
    dead, including those revived in place) can donate. Each receiver, in
    index order, takes a distinct live donor, largest weight first, ties to
    the lower index. Returns (source, target) or (None, None).
    """
    donors = ~(receivers | dead) & (weights > 0)
    nsplit = min(int(receivers.sum()), int(donors.sum()))
    if nsplit == 0:
        return None, None
    keys = np.where(donors, weights, -np.inf)
    source = np.argsort(-keys, kind="stable")[:nsplit]
    target = np.flatnonzero(receivers)[:nsplit]
    return source, target


def _split_rows(rows, eps=_SPLIT_EPS):
    """faiss split_clusters' multiplicative split: (donor, child) rows."""
    sign = np.ones(rows.shape[1], dtype=rows.dtype)
    sign[1::2] = -1
    return (rows * (1 - np.float32(eps) * sign)).astype(rows.dtype), (
        rows * (1 + np.float32(eps) * sign)).astype(rows.dtype)


def _order_ties(distance, labels):
    """Top-2 results with equal distances: the lower label first, in place."""
    tie = distance[:, 0] == distance[:, 1]
    if tie.any():
        low = labels.min(axis=1)
        high = labels.max(axis=1)
        labels[:, 0] = np.where(tie, low, labels[:, 0])
        labels[:, 1] = np.where(tie, high, labels[:, 1])


# --------------------------------------------------------------------------
# The NumPy class.
# --------------------------------------------------------------------------

class NumpyEmeansBalanced(NumpyEmeans):
    """Balanced E-means on NumPy arrays: ``NumpyEmeans`` plus a per-centroid bias.

    Assignment is argmin_j (||x - c_j||^2 - b_j). After every batch the bias
    follows the exact band: with m_j the rows the batch sent to cluster j and
    Q = B / k the fair share,

        lam_j <- max(0, lam_j + eta (m_j - Q) / Q)    pushes rows out
        mu_j  <- max(0, mu_j  + eta (Q - m_j) / Q)    pulls rows in
        b_j    = mu_j - lam_j

    with eta = u * D * s(t): ``u`` the balancing rate, D the running mean
    squared distance from a row to its assigned centroid (rate 0.1, bias not
    included) and s(t) a cosine from 1 to 0 over training.

    One training step, in this order: biased assignment
    with each row's runner-up; the capacity cap; the batch's sums, counts
    and D; every fifth step the dead test and revival; the bias step; the
    centroid EMA at step alpha = alpha0 s(t) (``alpha_min_factor`` 0: down to
    zero, not to the public 0.1 floor).

    Capacity cap (``cap_mode="rounds"``, the default): no cluster keeps more
    than ceil(cap_factor / s(t) * B / k) rows of a batch. An over-full
    cluster keeps its rows of largest biased margin over their runner-up; the
    others move to their runner-up. Then up to ``cap_rounds`` repair rounds:
    clusters still over the cap evict their arrivals (own rows keep priority,
    then lowest score), and the evicted rows are searched again with every
    full cluster masked out of the norms. ``cap_mode=None`` turns the cap
    off. ``cap_phase`` is "both" (training and polish), "train" or "polish".

    Revival (every 5 training steps): a cluster is dead when its usage is
    below ``unused_threshold`` times the mean usage. A dead cluster whose
    bias is below the live clusters' median is revived in place: its
    bias set to that median (lam, mu rebuilt from it), its usage to the mean
    usage, its centroid kept. Every other dead cluster takes a split of a
    live donor (faiss's multiplicative split, eps 1e-4); the child copies the
    donor's bias, lam and mu and both keep half its usage. With
    ``donor_rule="proportional"`` (the default) the donors are drawn in
    proportion to usage; ``donor_rule="largest"`` takes the live clusters of
    largest usage, a distinct one per dead cluster, ties to the lower index.

    ``nepochs`` is the whole budget, in passes over the data (default 25).
    The last ``polish_epochs`` of them (default round(0.4 * nepochs), so 10
    of 25) are the polish: the centroids are frozen and there is no revival,
    and the bias keeps learning with its cosine restarted, at rate
    min(u, ``polish_u_cap``), with D still updated and the cap still on. The
    other ``train_epochs`` passes (15 of 25) train the centroids and the bias.
    The returned centroids are the moving averages (no closing mean).

    Every pass is a fresh shuffle and drops its short last batch. The batch
    and alpha0 come from the public laws at T = ``train_epochs`` with the
    quality law by default (``batch_law="quality"``), which is what the
    validated recipe used at every dimension.

    ``assign(x, biased=True)`` returns the deployed routing (bias kept) and
    the true squared distances to the chosen centroids; ``biased=False`` is
    plain nearest-centroid assignment. ``bias`` holds b after training.
    Training goes through ``train`` only: the cap ranks whole batches.
    """

    def __init__(
        self,
        d,
        k,
        nepochs=25,
        batch_size=None,
        alpha0=None,
        batch_law="quality",
        seed=1234,
        verbose=False,
        assignment_chunk_size=None,
        capture_assignments=True,
        spherical=False,
        *,
        u=0.02,
        polish_epochs=None,
        polish_u_cap=_POLISH_U_CAP,
        cap_mode="rounds",
        cap_factor=2.0,
        cap_rounds=2,
        cap_anneal=True,
        cap_phase="both",
        inplace_quantile=0.5,
        donor_rule="proportional",
        unused_threshold=0.1,
        split_eps=_SPLIT_EPS,
        alpha_min_factor=0.0,
        max_points_per_centroid=None,
    ):
        if spherical:
            raise ValueError("NumpyEmeansBalanced supports Euclidean clustering only")
        train_epochs, polish_epochs = split_epochs(nepochs, polish_epochs)
        if cap_mode not in ("rounds", None):
            raise ValueError('cap_mode must be "rounds" or None')
        if cap_phase not in ("both", "train", "polish"):
            raise ValueError('cap_phase must be "both", "train" or "polish"')
        if donor_rule not in ("largest", "proportional"):
            raise ValueError('donor_rule must be "largest" or "proportional"')
        if not u >= 0:
            raise ValueError("u must be >= 0")
        if not 0 <= alpha_min_factor < 1:
            raise ValueError("alpha_min_factor must be in [0, 1)")
        if unused_threshold is None or not float(unused_threshold) >= 0:
            raise ValueError("unused_threshold must be >= 0 (0 disables revival)")
        if inplace_quantile is not None and not 0 <= inplace_quantile <= 1:
            raise ValueError("inplace_quantile must be in [0, 1] or None")
        super().__init__(
            d, k, train_epochs, batch_size, alpha0, batch_law, seed, verbose,
            assignment_chunk_size, capture_assignments, False,
            unused_threshold=unused_threshold,
            max_points_per_centroid=max_points_per_centroid,
            closing_handover=False,
        )
        self.nepochs = int(nepochs)
        self.train_epochs = train_epochs
        self.polish_epochs = polish_epochs
        self.u = float(u)
        self.polish_u_cap = None if polish_u_cap is None else float(polish_u_cap)
        self.cap_mode = cap_mode
        self.cap_factor = float(cap_factor)
        self.cap_rounds = int(cap_rounds)
        self.cap_anneal = bool(cap_anneal)
        self.cap_phase = cap_phase
        self.inplace_quantile = inplace_quantile
        self.donor_rule = donor_rule
        self.split_eps = float(split_eps)
        self.alpha_min_factor = float(alpha_min_factor)
        self.bias = None
        self.n_inplace = 0
        self.cap_stats = None
        self.timings = None
        self._lam = self._mu = None
        self._distance_scale = self._over_fraction = None

    # ------------------------------------------------------------- recipe

    def _resolve_run_recipe(self, n):
        plan = _resolve_balanced_recipe(
            n, self.k, self.train_epochs, self.polish_epochs, self.batch_law,
            self.batch_size, self.alpha0)
        if plan["steps_per_pass"] < 1:
            raise ValueError("the batch (%d rows) exceeds the %d training rows"
                             % (plan["batch_size"], n))
        self.recipe = dict(
            name="balanced_v1",
            batch_law=None if self.batch_size is not None else self.batch_law,
            batch_law_requested=self.batch_law,
            batch_law_reason="requested",
            batch_size=plan["batch_size"],
            alpha0=plan["alpha0"],
            alpha0_source="given" if self.alpha0 is not None else "step_law",
            nepochs=self.nepochs,
            train_epochs=self.train_epochs,
            polish_epochs=self.polish_epochs,
            steps_per_pass=plan["steps_per_pass"],
            train_steps=plan["train_steps"],
            polish_steps=plan["polish_steps"],
            total_steps=plan["train_steps"] + plan["polish_steps"],
            epoch_rows=[plan["pass_rows"]] * self.nepochs,
            epoch_steps=[plan["steps_per_pass"]] * self.nepochs,
            tail_policy="drop",
            alpha_min_factor=self.alpha_min_factor,
            check_unused_every=_CHECK_UNUSED_EVERY,
            unused_threshold=self._unused_threshold,
            init_usage=_INIT_USAGE,
            seed=self.seed,
            backend=self._backend,
            assignment_chunk_size=self._get_assignment_chunk_size(),
            balance="exact_band",
            u=self.u,
            polish_u=(self.u if self.polish_u_cap is None
                      else min(self.u, self.polish_u_cap)),
            distance_scale_rate=_DISTANCE_SCALE_RATE,
            cap_mode=self.cap_mode,
            cap_factor=self.cap_factor,
            cap_rounds=self.cap_rounds if self.cap_mode == "rounds" else None,
            cap_anneal=self.cap_anneal,
            cap_phase=self.cap_phase if self.cap_mode is not None else None,
            revival="inplace" if self.inplace_quantile is not None else "split",
            inplace_quantile=self.inplace_quantile,
            donor_rule=self.donor_rule,
            split="faiss_mult",
            split_eps=self.split_eps,
            closing=None,
        )

    def _schedule(self, step):
        """(training?, bias schedule s, EMA step alpha) at a global step."""
        recipe = self.recipe
        if step < recipe["train_steps"]:
            s = _cosine_scale(step, recipe["train_steps"])
            alpha = recipe["alpha0"] * (
                self.alpha_min_factor + (1 - self.alpha_min_factor) * s)
            return True, s, alpha
        return False, _cosine_scale(
            step - recipe["train_steps"], recipe["polish_steps"]), 0.0

    def _cap_active(self, training):
        if self.cap_mode is None:
            return False
        return self.cap_phase == "both" or self.cap_phase == (
            "train" if training else "polish")

    def _keeps_removal_stats(self):
        return False  # Mass-based revival: no removal costs or axes.

    # -------------------------------------------------------------- state

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self.bias = np.zeros(self.k, dtype="float32")
        self._lam = np.zeros(self.k, dtype="float32")
        self._mu = np.zeros(self.k, dtype="float32")
        self._distance_scale = None
        self._over_fraction = None
        self.n_inplace = 0
        self.cap_stats = dict(steps=0, moved=0, evicted=0, rounds=0, left_over=0,
                              second_rows=0)
        # Seconds per part of the step; "cap" includes "repair".
        self.timings = dict(assign=0.0, cap=0.0, repair=0.0, stats=0.0,
                            revival=0.0, bias=0.0, update=0.0)

    def _start_epoch(self):
        # No closing sums: a balanced run returns its moving averages.
        self._epoch_steps = self._epoch_rows = 0
        self._splits_before = self.n_splits
        self._inplace_before = getattr(self, "n_inplace", 0)
        self._pass_counts = np.zeros(self.k, dtype="float64")
        self._objective = np.float64(0)
        self._pass_sums = None

    # ---------------------------------------------------------- assignment

    def _biased_norms(self, centroids, bias):
        """||c_j||^2 + (max(b) - b_j), and the shift max(b).

        The shift keeps every biased distance >= the true one, so the clamp
        of negative distances at 0 in Faiss's BLAS search never fires, and
        leaves the ranking and the margins unchanged. The norms come from
        Faiss's own routine, the one ``faiss.knn`` uses.
        """
        norms = np.empty(self.k, dtype="float32")
        faiss.fvec_norms_L2sqr(
            faiss.swig_ptr(norms), faiss.swig_ptr(centroids), self.d, self.k)
        shift = np.float32(bias.max()) if len(bias) else np.float32(0)
        return norms + (shift - bias), shift

    def _knn(self, x, centroids, norms, nearest):
        """faiss.knn_L2sqr with precomputed (biased) centroid norms.

        knn_L2sqr computes ||x||^2 + y_norm2_j - 2 x.c_j on its BLAS paths,
        so passing the biased norms gives the biased argmin at the cost of a
        plain search. Its sequential path computes ||x - c_j||^2 directly and
        ignores y_norm2; it is chosen when nx * d is below
        distance_compute_blas_threshold, so the threshold is set to 0 for this
        call (and the centroid block widened for a wide search, see
        _WIDE_SEARCH_ROWS). Returns (distance (n, nearest), labels (n,
        nearest)), biased distances plus the shift.
        """
        n = len(x)
        distance = np.empty((n, nearest), dtype="float32")
        labels = np.empty((n, nearest), dtype="int64")
        if n == 0:
            return distance, labels
        x = np.ascontiguousarray(x, dtype="float32")
        cvar = faiss.cvar
        saved = cvar.distance_compute_blas_threshold, cvar.distance_compute_blas_database_bs
        cvar.distance_compute_blas_threshold = 0
        if n >= _WIDE_SEARCH_ROWS:
            cvar.distance_compute_blas_database_bs = max(
                saved[1], len(centroids) // _DATABASE_BLOCK_DIVISOR)
        try:
            faiss.knn_L2sqr(
                faiss.swig_ptr(x), faiss.swig_ptr(centroids),
                self.d, n, len(centroids), nearest,
                faiss.swig_ptr(distance), faiss.swig_ptr(labels),
                faiss.swig_ptr(norms),
            )
        finally:
            cvar.distance_compute_blas_threshold, cvar.distance_compute_blas_database_bs = saved
        return distance, labels

    def _check_biased_search(self, x, centroids, labels, plain, rows=4):
        """Raise if Faiss returned plain distances, i.e. ignored the norms.

        With the norms honoured, ``plain`` (the returned distance minus the
        shift, plus b) is ||x - c||^2 up to rounding of the norm expansion.
        """
        x4 = x[:rows].astype("float64")
        c4 = centroids[labels[:rows]].astype("float64")
        direct = ((x4 - c4) ** 2).sum(1)
        scale = (x4 ** 2).sum(1) + (c4 ** 2).sum(1) + float(np.abs(self.bias).max())
        gap = np.abs(plain[:rows] - direct)
        if (gap > 1e-3 * scale + 1e-6).any():
            raise RuntimeError(
                "faiss.knn_L2sqr ignored the precomputed norms that carry the "
                "balancing bias (max gap %g); this Faiss build dispatches "
                "differently and the biased assignment would be wrong"
                % float(gap.max()))

    def _assign_batch(self, rows, centroids, norms, shift, bias, nearest):
        """Biased search of one batch, in assignment chunks.

        Returns (distance (n, nearest), labels (n, nearest)): biased
        distances plus the shift, ties to the lower label.
        """
        n = len(rows)
        distance = np.empty((n, nearest), dtype="float32")
        labels = np.empty((n, nearest), dtype="int64")
        chunk = self.recipe["assignment_chunk_size"]
        for begin in range(0, n, chunk):
            end = min(begin + chunk, n)
            distance[begin:end], labels[begin:end] = self._knn(
                rows[begin:end], centroids, norms, nearest)
        if n and labels[0, 0] >= 0:
            plain = distance[:, 0] - (shift - bias[labels[:, 0]])
            self._check_biased_search(rows, centroids, labels[:, 0], plain)
        if nearest == 2:
            _order_ties(distance, labels)
        return distance, labels

    # --------------------------------------------------------------- step

    def _train_step(self, x, index, step):
        """One full training or polish step on the rows ``x[index]``."""
        timings = self.timings
        t0 = time.perf_counter()
        training, scale, alpha = self._schedule(step)
        k, n = self.k, len(index)
        bias = self.bias.copy()
        centroids = self._centroids
        norms, shift = self._biased_norms(centroids, bias)
        rows = np.ascontiguousarray(x[index], dtype="float32")
        capping = self._cap_active(training)
        cap = _cap_rows(n, k, self.cap_factor, scale, self.cap_anneal) if capping else n
        capping = capping and cap < n
        lazy = (capping and self._over_fraction is not None
                and self._over_fraction < _LAZY_RUNNER_UP_FRACTION)
        nearest = 2 if capping and not lazy else 1
        top_d, top_l = self._assign_batch(rows, centroids, norms, shift, bias, nearest)
        if capping:
            if lazy:
                top_d, top_l = self._runner_up(rows, top_d[:, 0], top_l[:, 0], cap,
                                               centroids, norms)
            over = np.bincount(top_l[:, 0], minlength=k)[top_l[:, 0]] > cap
            self._over_fraction = float(over.mean())
        t1 = time.perf_counter()
        timings["assign"] += t1 - t0
        if capping:
            codes, dist = self._cap_rounds_step(rows, top_d, top_l, cap, centroids, norms)
        else:
            codes, dist = top_l[:, 0], top_d[:, 0]
        t2 = time.perf_counter()
        timings["cap"] += t2 - t1
        # Statistics from the capped codes, against the centroids the
        # assignment used; distances without the bias.
        true_dist = dist - (shift - bias[codes])
        objective = true_dist.sum(dtype="float64")
        if training:
            stats = self._batch_stats
            stats.fill(0)
            faiss.accumulate_cluster_sums(
                self.d, n, k, *map(faiss.swig_ptr, (rows, codes, stats)))
            counts = stats[:, -1]
        else:
            counts = np.bincount(codes, minlength=k).astype("float32")
        t3 = time.perf_counter()
        timings["stats"] += t3 - t2
        # Dead test and revival on the state the assignment used.
        if (training and self._unused_threshold > 0
                and (step + 1) % _CHECK_UNUSED_EVERY == 0):
            self._revive()
        t4 = time.perf_counter()
        timings["revival"] += t4 - t3
        self._bias_step(counts, objective / n, scale, training)
        t5 = time.perf_counter()
        timings["bias"] += t5 - t4
        self._objective += objective
        self._pass_counts += counts
        if training:
            self._ema_step(stats[:, :-1], counts, alpha)
        timings["update"] += time.perf_counter() - t5
        return codes

    def _runner_up(self, rows, dist, codes, cap, centroids, norms):
        """Best and runner-up for the rows of over-full clusters only.

        From a top-1 search of the batch, the rows whose cluster holds more
        than ``cap`` rows are searched again with two neighbors, and both
        their best and runner-up are taken from that search (one evaluation).
        If that moves a row into another over-full cluster, that cluster's
        rows are searched too, until every row of an over-full cluster has its
        runner-up. Returns (n, 2) distances and labels; the other rows' second
        column repeats their first (the first pass never moves them).
        """
        k = self.k
        top_d = np.repeat(dist[:, None], 2, axis=1)
        top_l = np.repeat(codes[:, None], 2, axis=1)
        known = np.zeros(len(codes), dtype=bool)
        while True:
            first = top_l[:, 0]
            need = np.flatnonzero((np.bincount(first, minlength=k)[first] > cap) & ~known)
            if len(need) == 0:
                return top_d, top_l
            d2, l2 = self._knn(rows[need], centroids, norms, 2)
            _order_ties(d2, l2)
            top_d[need], top_l[need] = d2, l2
            known[need] = True
            self.cap_stats["second_rows"] += len(need)

    def _ema_step(self, sums, counts, alpha):
        """Algorithm 1, line 8, in place: NumpyEmeans' update without temporaries.

        N_j <- (1 - alpha) N_j + alpha n_j, and for occupied clusters
        c_j <- (1 - w_j) c_j + (alpha / N_j') s_j with w_j = alpha n_j / N_j',
        the same algebra as ``_apply_statistics`` (c_j + w_j (s_j / n_j - c_j)),
        in three passes over the (k, d) table with no (k, d) allocation (the
        batch sums are scaled in place: they are not read again). Empty
        clusters keep their positions. A step of one replaces the occupied
        centroids by their batch means.
        """
        usage = self._usage
        touched = counts > 0
        if alpha == 1:
            usage[touched] = counts[touched]
            self._centroids[touched] = sums[touched] / counts[touched, None]
            self._invalidate_readout()
            return
        usage *= np.float32(1 - alpha)
        mass = usage + np.float32(alpha) * counts  # (k,)
        scale = np.where(touched, np.float32(alpha) / np.where(touched, mass, 1), 0)
        keep = (1 - counts * scale).astype("float32")  # 1 - w_j; 1 for empty clusters
        np.multiply(self._centroids, keep[:, None], out=self._centroids)
        np.multiply(sums, scale.astype("float32")[:, None], out=sums)
        np.add(self._centroids, sums, out=self._centroids)
        usage[touched] = mass[touched]
        self._invalidate_readout()

    def _cap_rounds_step(self, rows, top_d, top_l, cap, centroids, norms):
        """The one-pass cap, then up to ``cap_rounds`` masked repair rounds.

        ``top_d``, ``top_l`` (n, 2) are every row's best and runner-up (a
        top-2 search costs 2-14% more than a top-1 one on CPU, less than a
        second search of the rows of over-full clusters).
        """
        k, n = self.k, len(rows)
        dist, second_d = top_d[:, 0], top_d[:, 1]
        codes, moved = cap_first_pass(top_l[:, 0], second_d - dist, top_l[:, 1], cap, k)
        dist = np.where(moved, second_d, dist)
        # The score at the current cluster with the row norm dropped, which
        # ranks the arrivals. Only arrivals are ever ranked by
        # it (a cluster's own rows are within the cap after the first pass),
        # so the row norms are computed for the moved rows only.
        score = np.zeros(n, dtype="float32")
        ids = np.flatnonzero(moved)
        score[ids] = second_d[ids] - self._row_norms(rows[ids])
        arrived = moved
        stats = self.cap_stats
        stats["steps"] += 1
        stats["moved"] += len(ids)
        t = time.perf_counter()
        for _ in range(self.cap_rounds):
            evict, full = cap_round_evict(codes, arrived, score, cap, k)
            ids = np.flatnonzero(evict)
            if len(ids) == 0:
                break
            stats["rounds"] += 1
            stats["evicted"] += len(ids)
            sub = rows[ids]
            d1, l1 = self._masked_search(sub, centroids, norms, full)
            codes[ids] = l1
            dist[ids] = d1
            score[ids] = d1 - self._row_norms(sub)
            arrived[ids] = True
        if self.cap_rounds and np.bincount(codes, minlength=k).max() > cap:
            stats["left_over"] += 1
        self.timings["repair"] += time.perf_counter() - t
        return codes, dist

    def _masked_search(self, rows, centroids, norms, full):
        """Nearest cluster (bias included) among those not ``full``, per row.

        Either the full clusters' norms are set to +inf (Faiss's heap
        neutral is the largest float, so +inf is never admitted), or, when the
        rows are many and enough clusters are full that skipping them pays
        for the copy, the search runs on the open clusters only. Same argmin,
        ties to the lower index either way. Returns (distances, labels).
        """
        n_full = int(full.sum())
        if len(rows) * n_full > _COMPACT_RATIO * (self.k - n_full):
            open_ids = np.flatnonzero(~full)
            d1, l1 = self._knn(rows, np.ascontiguousarray(centroids[open_ids]),
                               np.ascontiguousarray(norms[open_ids]), 1)
            return d1[:, 0], open_ids[l1[:, 0]]
        d1, l1 = self._knn(rows, centroids, np.where(full, np.float32(np.inf), norms), 1)
        return d1[:, 0], l1[:, 0]

    def _row_norms(self, rows):
        out = np.empty(len(rows), dtype="float32")
        if len(rows):
            rows = np.ascontiguousarray(rows, dtype="float32")
            faiss.fvec_norms_L2sqr(
                faiss.swig_ptr(out), faiss.swig_ptr(rows), self.d, len(rows))
        return out

    # -------------------------------------------------------- bias, revival

    def _bias_step(self, counts, batch_mean, scale, training):
        batch_mean = np.float32(batch_mean)
        if self._distance_scale is None:
            self._distance_scale = batch_mean
        else:
            self._distance_scale = np.float32(
                self._distance_scale * np.float32(1 - _DISTANCE_SCALE_RATE)
                + batch_mean * np.float32(_DISTANCE_SCALE_RATE))
        u = self.u if training else self.recipe["polish_u"]
        eta = u * float(self._distance_scale) * scale
        _band_step(self._lam, self._mu, self.bias, counts, eta)

    def _revive(self):
        """Dead test, then in-place revival below the median, splits above."""
        usage = self._usage
        dead = usage < _numpy_unused_threshold(usage, self.k, self._unused_threshold)
        if not dead.any():
            return
        if self.inplace_quantile is not None:
            inplace, reference = _select_inplace(dead, self.bias, self.inplace_quantile)
            count = int(inplace.sum())
            if count:
                usage[inplace] = usage.mean(dtype="float32")
                self._lam[inplace] = max(-reference, 0)
                self._mu[inplace] = max(reference, 0)
                self.bias[inplace] = reference
                self.n_inplace += count
            rest = dead & ~inplace
        else:
            rest = dead
        if self.donor_rule == "largest":
            source, target = _largest_donors(usage, rest, dead)
        else:
            source, target = self._draw_donors(usage, rest)
        if source is None:
            return
        donor, child = _split_rows(self._centroids[source], self.split_eps)
        half = usage[source] / 2
        self._centroids[source] = donor
        self._centroids[target] = child
        usage[source] = half
        usage[target] = half
        for state in (self.bias, self._lam, self._mu):
            state[target] = state[source]
        self.n_splits += len(source)

    def _draw_donors(self, weights, receivers):
        """Donors drawn in proportion to ``weights``, without replacement.

        Gumbel-top-k over log-weights, from the model's split stream; the
        receivers are the first ones in index order.
        """
        donors = (~receivers) & (weights > 0)
        nsplit = min(int(receivers.sum()), int(donors.sum()))
        if nsplit == 0:
            return None, None
        uniform = np.maximum(
            self._split_rng.random_sample(self.k), np.finfo("float64").tiny)
        keys = np.full(self.k, -np.inf)
        keys[donors] = np.log(weights[donors]) - np.log(-np.log(uniform[donors]))
        source = np.argsort(keys, kind="stable")[-nsplit:][::-1]
        target = np.flatnonzero(receivers)[:nsplit]
        return source, target

    # ------------------------------------------------------------ training

    def _train_rows(self, x, init_centroids=None):
        """Train an FP32 matrix, returning centroids and last-pass labels.

        The labels are the last pass's capped, biased labels. Rows that pass
        dropped (the n mod B tail of its shuffle) take the final biased
        assignment.
        """
        start, given_init = time.perf_counter(), init_centroids is not None
        x = np.ascontiguousarray(x, dtype="float32")
        n = len(x)
        if init_centroids is None:
            init_centroids = (
                x if n == self.k
                else x[np.random.RandomState(self.seed).permutation(n)[: self.k]])
        self.begin(n, init_centroids)
        self._report_header_once()
        if self._reports():
            self._report_preprocessing(start, given_init, n)
        recipe = self.recipe
        batch = recipe["batch_size"]
        assignments = None
        try:
            step = 0
            for epoch, rows in enumerate(recipe["epoch_rows"]):
                order = np.random.RandomState(_epoch_seed(self.seed, epoch)).permutation(n)
                last = epoch == len(recipe["epoch_rows"]) - 1
                if last and self.capture_assignments:
                    assignments = np.full(n, -1, dtype="int64")
                for begin in range(0, rows, batch):
                    index = order[begin:begin + batch]
                    codes = self._train_step(x, index, step)
                    if assignments is not None and last:
                        assignments[index] = codes
                    step += 1
                    self._step = step
                    self._epoch_steps += 1
                    self._epoch_rows += len(index)
                self.end_epoch()
            if assignments is not None and rows < n:
                tail = order[rows:]
                _, assignments[tail] = self._assign_biased(x[tail], self._centroids)
            self.finalize()
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        self.assignments = assignments
        return self.centroids, self.assignments

    def end_epoch(self):
        inplace = self.n_inplace - self._inplace_before
        epoch = self._epoch
        verbose, self.verbose = self.verbose, False
        try:
            super().end_epoch()
        finally:
            self.verbose = verbose
        stat = self.iteration_stats[-1]
        stat["ninplace"] = inplace
        stat["phase"] = "train" if epoch < self.train_epochs else "polish"
        if self._reports():
            self._report_epoch(stat, epoch)
            self._say("    %s pass, imbalance %.4f, %d splits, %d in place"
                      % (stat["phase"], stat["imbalance_factor"], stat["nsplit"], inplace))

    def accumulate(self, x):
        raise NotImplementedError(
            "NumpyEmeansBalanced trains through train(): the capacity cap ranks whole batches")

    def update(self, stats):
        raise NotImplementedError(
            "NumpyEmeansBalanced trains through train(): the capacity cap ranks whole batches")

    def finalize(self):
        """Return the trained centroids (moving averages); the bias stays in ``bias``."""
        if self._failed:
            raise RuntimeError("create a new object after execution failure")
        if self.centroids is not None:
            return self.centroids
        self._failed = True
        self.centroids = np.ascontiguousarray(self._centroids, dtype="float32").copy()
        self.obj = np.asarray(
            [stat["obj"] for stat in self.iteration_stats], dtype="float64")
        self._centroids = self._usage = self._batch_stats = None
        self._pass_sums = self._pass_counts = self._objective = None
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None
        self._failed = False
        return self.centroids

    # ------------------------------------------------------------ queries

    def _assign_biased(self, x, centroids):
        """True squared distances and biased labels (the deployed routing)."""
        x = np.ascontiguousarray(x, dtype="float32")
        norms, shift = self._biased_norms(centroids, self.bias)
        distance = np.empty(len(x), dtype="float32")
        labels = np.empty(len(x), dtype="int64")
        chunk = self._get_assignment_chunk_size()
        for begin in range(0, len(x), chunk):
            end = min(begin + chunk, len(x))
            d1, l1 = self._knn(x[begin:end], centroids, norms, 1)
            l1 = l1[:, 0]
            d1 = d1[:, 0] - (shift - self.bias[l1])
            if end > begin:
                self._check_biased_search(x[begin:end], centroids, l1, d1)
            distance[begin:end] = np.maximum(d1, 0)
            labels[begin:end] = l1
        return distance, labels

    def assign(self, x, biased=True):
        """Distances and labels with the bias (deployed routing) or without.

        Distances are the true squared distances to the chosen centroids
        (bias not included). ``biased=False`` is NumpyEmeans' nearest-centroid
        assignment.
        """
        if self._failed:
            raise RuntimeError("create a new object after execution failure")
        if not biased or self.bias is None:
            return super().assign(x)
        return self._assign_biased(x, self.centroids)

    def exact_balance(self, x, labels=None, **kwargs):
        """Exactly balanced labels: every cluster gets floor(n / k) or ceil(n / k) rows of x.

        A post-processing step after training (module ``e_means_balanced_exact_numpy``):
        each row moves among its m nearest centroids under the bias, by an
        auction whose prices start at -bias, and the few rows the auction
        cannot place go greedily to the nearest clusters with room.
        ``labels`` are the biased labels of x when already known (from
        ``assign``, or from ``train`` when it labelled every row); otherwise
        one ``assign`` computes them. Keyword arguments go to that module's
        ``exact_balance``. Returns (distances, labels): the true squared
        distances and the balanced labels, a partition of these rows only
        (keep using ``assign`` and the trained bias for new rows).
        ``exact_balance_stats`` keeps the step's statistics.
        """
        from .e_means_balanced_exact_numpy import exact_balance

        if self.bias is None:
            raise RuntimeError("exact_balance needs a trained model")
        if labels is None:
            labels = self.assign(x)[1]
        distances, labels, self.exact_balance_stats = exact_balance(
            x, self.centroids, self.bias, anchors=labels, **kwargs)
        return distances, labels

    def _report_header(self):
        super()._report_header()
        recipe = self.recipe
        self._say("  Balanced: u %g, %d + %d passes, cap %s, revival %s, %s donor"
                  % (recipe["u"], recipe["train_epochs"], recipe["polish_epochs"],
                     "off" if recipe["cap_mode"] is None else "%s x%g (%s)" % (
                         recipe["cap_mode"], recipe["cap_factor"], recipe["cap_phase"]),
                     recipe["revival"], recipe["donor_rule"]))
