# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Balanced (auxiliary-loss-free) E-means on PyTorch tensors, CPU and CUDA.

Each centroid j carries a bias b_j, and a row goes to the centroid that
minimises ||x - c_j||^2 - b_j. After every batch the bias moves every cluster
toward an equal share of the batch, so the clusters come out balanced without
an extra loss term. The bias rides in the centroid norms the assignment
kernels already take (||c_j||^2 - b_j), so the Torch, Triton and CuTe kernels
of ``Emeans`` are reused unchanged.

The defaults are the recipe of the e-means paper's balanced variant: the
bias at rate u = 0.02, a capacity cap of twice the fair share with two repair
rounds, in training and in the polish, 15 training passes and 10 polish passes
at the quality batch law. Two rules complete it: in-place revival of dead
clusters whose bias is below the live median, and the polish's bias rate capped
at 0.02 (inactive at the default u).

The recipe math that does not depend on how rows are assigned (the bias step,
the cap's ranking, the in-place selection, the donor choice and the split) lives in
module-level functions on plain tensors, so the NumPy and the multi-GPU
classes can call the same code.
"""

import json
import math
import time

import torch

from .base import _INIT_USAGE, _epoch_seed, _resolve_recipe
from .e_means_balanced_numpy import split_epochs
from .e_means_torch import Emeans, _torch_generator

try:
    from .balanced_triton import FusedBias, FusedCap, gather_rows, row_norms
except ImportError:  # No Triton: the host path only.
    FusedBias = FusedCap = gather_rows = row_norms = None


__all__ = [
    "EmeansBalanced",
    "cosine_scale",
    "cap_rows",
    "cap_first_pass",
    "cap_round_evict",
    "cap_reference",
    "band_step",
    "select_inplace",
    "draw_donors",
    "largest_donors",
    "split_rows",
    "resolve_balanced_recipe",
    "split_epochs",
    "select_inplace_fixed",
    "draw_donors_fixed",
    "largest_donors_fixed",
]


# Smoothing of the running mean assigned squared distance D, which scales the
# bias step: D <- (1 - rate) * D + rate * (this batch's mean).
_DISTANCE_SCALE_RATE = 0.1

# Polish-rate cap: the polish's bias rate is min(u, this).
_POLISH_U_CAP = 0.02

# Multiplicative split perturbation (faiss split_clusters): the donor and its
# child become c (1 -/+ eps s), s = (+1, -1, +1, ...).
_SPLIT_EPS = 1e-4

# The cap's anneal divides the factor by the bias schedule, floored here.
_CAP_SCALE_FLOOR = 1e-6

# Padded row counts for the repair rounds' masked assignment: powers of two
# from this size up, so a run compiles and binds a handful of shapes.
_REPAIR_MIN_ROWS = 1024

# Device path, chosen per epoch from the last steps' statistics (speed only;
# every choice gives the same decisions): rank with the pairwise kernels while
# the mean pairwise work per step (sum of squared candidate counts over the
# over-full clusters) stays below this, else radix-sort the batch. On H100 the
# two cost the same near 2**26 comparisons per step, at 32k and 262k rows.
_PAIR_WORK_LIMIT = 2**26
# Score the first evicted rows of a repair round with CuTe, as many as this
# quantile of the last steps' evicted counts (rounded up to 2^j or 1.5 * 2^j),
# when that is at least _BUCKET_MIN_ROWS; Triton scores the rest. Triton costs
# about 1.6-1.9x CuTe per row at 8k+ rows (and less below ~4k rows, where CuTe's
# floor is ~90 us), so the best quantile is about 1 - 1/1.8.
_BUCKET_QUANTILE = 0.5
_BUCKET_MIN_ROWS = 8192
# CuTe re-scores the evicted rows at their exact count (a host read per round,
# no CUDA graph for those steps) when batch * k * d is at least this (10M-row
# cells and up) and the last steps evicted at least _BUCKET_MIN_ROWS rows.
_EXACT_ROUNDS_WORK = 2**38
# Exact re-search in chunks of 1024 * 4^j rows (True) or one call padded to a
# quarter of a power of two (False).
_EXACT_LADDER = False
# The first pass needs each moved row's runner-up. While the last steps moved
# more than this fraction of the batch on average, the assignment returns
# every row's runner-up label (about 10% dearer than the second distance alone
# with CuTe on H100, k = 16k-65k); below it, the assignment returns only the
# second distance and the moved rows' runner-ups are searched afterwards
# (Triton, about 1.8x CuTe per row). Late in training and in the polish the
# cap rarely moves a row, so this second way is the common one.
_LAZY_MOVED_FRACTION = 0.03
# The first choice is made after this many steps, then at every pass.
_FIRST_PLAN_STEP = 5


# --------------------------------------------------------------------------
# Recipe math on plain tensors (shared with the NumPy and multi-GPU classes).
# --------------------------------------------------------------------------

def cosine_scale(step, total):
    """0.5 * (1 + cos(pi * step / total)): 1 at step 0, down to 0, no floor."""
    fraction = min(step, total) / max(1, total)
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def cap_rows(batch, k, factor, scale=1.0, anneal=True):
    """Rows one cluster may keep from a batch: ceil(factor / scale * batch / k).

    With ``anneal`` the factor is divided by the bias schedule ``scale``
    (floored at 1e-6), so the largest bias step the cap allows stays about
    factor * u * D while the step anneals.
    """
    if anneal:
        factor = factor / max(scale, _CAP_SCALE_FLOOR)
    return max(1, int(math.ceil(min(factor, 1e9) * batch / k)))


def _sortable(values):
    """int64 keys ordered like the FP32 ``values`` (NaN sorts last)."""
    bits = values.float().contiguous().view(torch.int32).long()  # (n,)
    # Negative floats: flip the magnitude bits so larger magnitude sorts lower.
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def _ranks_in_cluster(codes, key, k):
    """Sort rows by (cluster, key) stably, ties by position.

    ``key`` is a non-negative int64 below 2**34. Returns the visiting order,
    each sorted row's rank inside its cluster and the per-cluster counts.
    No host synchronization.
    """
    codes = codes.long()
    composite = codes * (1 << 34) + key  # (n,)
    order = torch.sort(composite, stable=True)[1]  # (n,)
    counts = torch.zeros(k, dtype=torch.int64, device=codes.device)
    counts.index_add_(0, codes, torch.ones_like(codes))
    start = torch.cumsum(counts, 0) - counts  # (k,)
    rank = torch.arange(len(codes), device=codes.device) - start[codes[order]]
    return order, rank, counts


def cap_first_pass(codes, margin, second, cap, k):
    """One-pass capacity cap: over-full clusters send their weakest rows to the runner-up.

    ``codes`` (n,) are the biased first choices, ``margin`` (n,) FP32 the
    biased runner-up score minus the best score and ``second`` (n,) the
    runner-up index. A cluster keeps its ``cap`` rows of largest margin (ties
    by lower batch position); the others take their runner-up. Returns the new
    codes (int64) and a bool mask of the moved rows. A runner-up may end above
    the cap; the repair rounds handle that. No host synchronization.
    """
    # Descending margin: negate the ordered key, offset into [0, 2**33).
    key = (1 << 31) - _sortable(margin)
    order, rank, _ = _ranks_in_cluster(codes, key, k)
    moved_sorted = rank >= cap
    codes = codes.long()
    new = torch.where(moved_sorted, second.long()[order], codes[order])
    out = torch.empty_like(codes).scatter_(0, order, new)
    moved = torch.zeros_like(moved_sorted).scatter_(0, order, moved_sorted)
    return out, moved


def cap_round_evict(codes, arrived, score, cap, k):
    """One repair round's eviction: which rows leave clusters above the cap.

    ``arrived`` (n,) bool marks rows that reached their cluster by a move;
    the cluster's own first-choice rows keep priority, then arrivals by lowest
    ``score`` (FP32, the kernel score at the current cluster, ties by lower
    batch position). Rows ranked at or past ``cap`` are evicted. Returns the
    eviction mask (n,) and the mask (k,) of full clusters (holding at least
    ``cap`` kept rows), which the evicted rows must avoid. No host
    synchronization.
    """
    key = arrived.long() * (1 << 33) + (_sortable(score) + (1 << 31))
    order, rank, counts = _ranks_in_cluster(codes, key, k)
    evict = torch.zeros(len(codes), dtype=torch.bool, device=codes.device)
    evict.scatter_(0, order, rank >= cap)
    return evict, counts >= cap


def cap_reference(scores, cap, rounds):
    """Slow sequential reference of the rounds cap, for tests.

    ``scores`` (n, k) FP64 biased scores (lower is better); ties go to the
    lower centroid index. Mirrors ``cap_first_pass`` then ``rounds`` repair
    rounds with explicit Python loops. Returns int64 codes.
    """
    n, k = scores.shape
    scores = scores.double().cpu()
    order = torch.argsort(scores, dim=1, stable=True)
    best, second = order[:, 0], order[:, 1] if k > 1 else order[:, 0]
    margin = (scores[torch.arange(n), second] - scores[torch.arange(n), best]).float()
    codes = best.clone()
    # Score at the current cluster, as the kernel reports it.
    score = scores[torch.arange(n), second].float().clone()
    arrived = torch.zeros(n, dtype=torch.bool)
    for j in range(k):
        rows = [i for i in range(n) if int(best[i]) == j]
        rows.sort(key=lambda i: (-float(margin[i]), i))
        for i in rows[cap:]:
            codes[i] = second[i]
            arrived[i] = True
    for _ in range(rounds):
        count = torch.bincount(codes, minlength=k)
        if int(count.max()) <= cap:
            break
        evicted = []
        for j in range(k):
            rows = [i for i in range(n) if int(codes[i]) == j]
            rows.sort(key=lambda i: (bool(arrived[i]), float(score[i]), i))
            evicted += rows[cap:]
        full = torch.bincount(codes, minlength=k) >= cap
        for i in evicted:
            masked = torch.where(full, float("inf"), scores[i])
            j = int(torch.argmin(masked))
            codes[i] = j
            arrived[i] = True
            score[i] = float(scores[i, j])
    return codes


def band_step(lam, mu, bias, counts, eta):
    """The exact band, both bounds at the fair share Q = mean count, in place.

        lam <- max(0, lam + eta (m - Q) / Q)
        mu  <- max(0, mu  + eta (Q - m) / Q)
        b    = mu - lam

    ``eta`` is a float or a 0-dim tensor (u * D * schedule).
    """
    counts = counts.float()
    share = counts.mean()
    relative = (counts - share) / share.clamp_min(1e-6)  # (k,)
    lam.add_(eta * relative).clamp_(min=0.0)
    mu.sub_(eta * relative).clamp_(min=0.0)
    torch.sub(mu, lam, out=bias)


def select_inplace(dead, bias, quantile=0.5):
    """Split the dead clusters into in-place and split revivals.

    A dead cluster whose bias is below the live median (``quantile`` 0.5,
    the lower median as ``torch.median`` gives it) is revived in place.
    Returns (in-place mask, reference bias as a 0-dim tensor).
    """
    live = ~dead
    if bool(live.any()):
        values = bias[live]
        reference = values.kthvalue(int((len(values) - 1) * quantile) + 1).values
    else:
        reference = bias.new_zeros(())
    return dead & (bias < reference), reference


def draw_donors(weights, receivers, generator):
    """Donors drawn in proportion to ``weights``, without replacement.

    ``receivers`` (k,) bool marks the clusters to revive; they cannot donate.
    Gumbel-top-k over log-weights (the same distribution as
    ``torch.multinomial(replacement=False)``). The receivers are the first
    ones in index order. Returns (source, target) int64 tensors or
    (None, None) when nothing can be split.
    """
    w = weights.clone().clamp_(min=0.0)
    w[receivers] = 0.0
    nsplit = int(torch.minimum(receivers.sum(), (w > 0).sum()))
    if nsplit == 0:
        return None, None
    uniform = torch.rand(len(w), dtype=w.dtype, device=w.device, generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(w.dtype).tiny)))
    keys = torch.where(w > 0, w.log() + gumbel, float("-inf"))
    source = torch.topk(keys, nsplit).indices
    target = receivers.nonzero().flatten()[:nsplit]
    return source, target


def _largest_keys(weights, receivers, dead):
    """Donor keys of the largest rule: the weight, -inf where a cluster cannot donate."""
    blocked = receivers if dead is None else receivers | dead
    w = torch.where(blocked, 0.0, weights.clamp(min=0.0))
    eligible = w > 0
    return torch.where(eligible, w, float("-inf")), eligible


def largest_donors(weights, receivers, dead=None):
    """Donors with the largest ``weights``, one receiver each, no draw.

    ``receivers`` (k,) bool marks the clusters to revive; they cannot
    donate, nor can the clusters in ``dead`` (every cluster found dead,
    including those revived in place). Each receiver, in index order, takes
    a distinct live donor, largest weight first, ties to the lower index.
    Returns (source, target) int64 tensors or (None, None) when nothing can
    be split.
    """
    keys, eligible = _largest_keys(weights, receivers, dead)
    nsplit = int(torch.minimum(receivers.sum(), eligible.sum()))
    if nsplit == 0:
        return None, None
    source = torch.sort(keys, descending=True, stable=True).indices[:nsplit]
    target = receivers.nonzero().flatten()[:nsplit]
    return source, target


def split_rows(rows, eps=_SPLIT_EPS):
    """faiss split_clusters' multiplicative split: (donor, child) rows."""
    sign = torch.ones(rows.shape[1], dtype=rows.dtype, device=rows.device)
    sign[1::2] = -1.0
    return rows * (1.0 - eps * sign), rows * (1.0 + eps * sign)


def resolve_balanced_recipe(n, k, train_epochs, polish_epochs, batch_law, batch_size, alpha0):
    """Batch, step and step counts of a balanced run.

    The batch and the initial EMA step come from the public laws at T =
    ``train_epochs`` (the polish moves no centroid). Every pass drops its
    short last batch: a pass is floor(n / B) full batches.
    """
    batch, alpha0 = _resolve_recipe(n, k, train_epochs, batch_law, batch_size, alpha0)
    steps_per_pass = n // batch
    return dict(
        batch_size=batch, alpha0=alpha0, steps_per_pass=steps_per_pass,
        pass_rows=steps_per_pass * batch,
        train_steps=train_epochs * steps_per_pass,
        polish_steps=polish_epochs * steps_per_pass)


def select_inplace_fixed(dead, bias, quantile=0.5):
    """``select_inplace`` with fixed shapes and no host synchronization (CUDA graphs)."""
    live = (~dead).sum()
    ordered = torch.where(dead, float("inf"), bias).sort().values  # (k,) live ascending first
    index = ((live - 1).clamp(min=0).double() * quantile).floor().long()
    reference = torch.where(live > 0, ordered.gather(0, index.reshape(1))[0], 0.0)
    return dead & (bias < reference), reference


def draw_donors_fixed(weights, receivers, generator):
    """``draw_donors`` with fixed shapes: (source (k,), target (k,), count).

    The first ``count`` (a device int) entries of ``source`` and ``target``
    are the pairs ``draw_donors`` returns; the rest are other indices, in an
    order that keeps every write a permutation. No host synchronization.
    """
    w = torch.where(receivers, 0.0, weights.clamp(min=0.0))
    eligible = w > 0
    count = torch.minimum(receivers.sum(), eligible.sum())
    uniform = torch.rand(len(w), dtype=w.dtype, device=w.device, generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(w.dtype).tiny)))
    keys = torch.where(eligible, w.log() + gumbel, float("-inf"))
    source = torch.sort(keys, descending=True).indices
    target = torch.sort(receivers.to(torch.int8), descending=True, stable=True).indices
    return source, target, count


def largest_donors_fixed(weights, receivers, dead=None):
    """``largest_donors`` with fixed shapes: (source (k,), target (k,), count).

    Same convention as ``draw_donors_fixed``: the first ``count`` entries
    are the pairs ``largest_donors`` returns. No host synchronization.
    """
    keys, eligible = _largest_keys(weights, receivers, dead)
    count = torch.minimum(receivers.sum(), eligible.sum())
    source = torch.sort(keys, descending=True, stable=True).indices
    target = torch.sort(receivers.to(torch.int8), descending=True, stable=True).indices
    return source, target, count


# --------------------------------------------------------------------------
# The tensor class.
# --------------------------------------------------------------------------

class EmeansBalanced(Emeans):
    """Balanced E-means on tensors: ``Emeans`` plus a per-centroid bias.

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
    cluster keeps its rows of largest biased margin over their runner-up
    (ties by batch position); the others move to their runner-up. Then
    ``cap_rounds`` repair rounds: clusters still over the cap evict their
    arrivals (own rows keep priority, then lowest kernel score
    ||c||^2 - b - 2 x.c), and the evicted rows are assigned again with every
    full cluster masked out. ``cap_mode=None`` turns the cap off.
    ``cap_phase`` is "both" (training and polish), "train" or "polish".

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

    On CUDA with 16-bit operands a step never waits for the GPU (the repair
    rounds read their row count on the device), and ``cuda_graphs=True``
    replays each kind of step (training, training with the revival check,
    polish) as one CUDA graph.

    ``assign(x, biased=True)`` returns the deployed routing (bias kept) and
    the true squared distances to the chosen centroids; ``biased=False`` is
    plain nearest-centroid assignment. ``bias`` holds b after training.
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
        centroid_chunk_size=4096,
        *,
        assignment_backend="auto",
        capture_assignments=True,
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
        cuda_graphs=True,
        max_points_per_centroid=None,
        spherical=False,
    ):
        if spherical:
            raise ValueError("EmeansBalanced supports Euclidean clustering only")
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
            assignment_chunk_size, centroid_chunk_size,
            assignment_backend=assignment_backend,
            capture_assignments=capture_assignments,
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
        self.cuda_graphs = bool(cuda_graphs)
        self.bias = None
        self.cap_stats = None
        self._lam = self._mu = None
        self._distance_scale = None
        self._base_norm = self._base_token = None
        self._mirror = None
        self._device_mode = False
        self._exact_rounds = False
        self._inplace_host = 0
        self._inplace_device = None
        # Phase timing (``profile_phases``): CUDA events between the step's phases.
        self._phase_events = None

    # ------------------------------------------------------------- recipe

    def _resolve_run_recipe(self, n):
        plan = resolve_balanced_recipe(
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
            check_unused_every=5,
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
        """(training?, bias schedule s, EMA step alpha, cap rows) at a global step."""
        recipe = self.recipe
        batch = recipe["batch_size"]
        training = step < recipe["train_steps"]
        if training:
            s = cosine_scale(step, recipe["train_steps"])
            alpha = recipe["alpha0"] * (self.alpha_min_factor + (1 - self.alpha_min_factor) * s)
        else:
            s, alpha = cosine_scale(step - recipe["train_steps"], recipe["polish_steps"]), 0.0
        cap = cap_rows(batch, self.k, self.cap_factor, s, self.cap_anneal)
        return training, s, alpha, cap

    def _cap_active(self, training):
        if self.cap_mode is None:
            return False
        return self.cap_phase == "both" or self.cap_phase == ("train" if training else "polish")

    def _revives(self, step):
        return (step < self.recipe["train_steps"] and self._unused_threshold > 0
                and (step + 1) % 5 == 0)

    # -------------------------------------------------------------- state

    @property
    def n_inplace(self):
        """In-place revivals so far; device counts are read (synchronizing) on access."""
        device = 0 if self._inplace_device is None else int(self._inplace_device)
        return self._inplace_host + device

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        zeros = self._usage.new_zeros(self.k)
        self.bias, self._lam, self._mu = zeros.clone(), zeros.clone(), zeros.clone()
        self._distance_scale = None
        self._base_norm = self._base_token = None
        self._mirror = None
        self.cap_stats = dict(steps=0, moved=0, evicted=0)
        self._device_counts = None

    def _start_epoch(self):
        # No closing sums: a balanced run returns its moving averages.
        self._epoch_steps = self._epoch_rows = 0
        self._splits_before = self.n_splits
        self._inplace_before = self.n_inplace
        if self._pass_counts is None:
            self._pass_counts = self._usage.new_zeros(self.k, dtype=torch.float64)
            self._objective = self._usage.new_zeros((), dtype=torch.float64)
        else:
            self._pass_counts.zero_()
            self._objective.zero_()
        self._pass_sums = None

    # ---------------------------------------------------------- assignment

    def _prepared_mirror(self, centroids, frozen):
        """Represented centroids and unbiased norms (CuTe: norms None).

        ``frozen`` reuses the last mirror (polish: the centroids do not move).
        """
        if frozen and self._mirror is not None:
            return self._mirror
        prepared, norm = self._execution_backend.prepare_centroids(centroids, self._input_dtype)
        if norm is None:
            # CuTe keeps the norms in the prepared object; keep our own copy
            # of the unbiased norms, refreshed only when the mirror was rebuilt.
            token = (id(prepared), prepared.source_version)
            if self._base_norm is None:
                self._base_norm = prepared.norms.clone()
            elif token != self._base_token:
                self._base_norm.copy_(prepared.norms)
            self._base_token = token
        self._mirror = (prepared, norm)
        return self._mirror

    def _biased_assignment(self, centroids, bias, frozen=False):
        """Prepared centroids and norms with the bias folded in.

        Norm_j = ||c_j||^2 + (max(b) - b_j): the shift keeps every biased
        distance >= the true one, so the kernels' clamp at zero never fires,
        and leaves the ranking and the margins unchanged. Returns
        (prepared, norm, shift); on CuTe the biased norm is written into the
        prepared object and ``norm`` is None.
        """
        shift = bias.max()
        prepared, norm = self._prepared_mirror(centroids, frozen)
        if norm is not None:
            return prepared, norm + (shift - bias), shift
        torch.add(self._base_norm, shift - bias, out=prepared.norms)
        return prepared, None, shift

    def _biased_norm(self, prepared, norm):
        return prepared.norms if norm is None else norm

    def _mirror_rows(self, prepared, norm):
        """The 16-bit represented centroids as a (k, d) view, for ``masked_assign``."""
        mirror = prepared.centroids if norm is None else prepared
        return mirror[:, :self.d]

    def _assign_chunk(self, rows, prepared, norm, second=True, out=None):
        """Biased distances, labels and (optionally) runner-up distances and labels.

        ``out`` (distance, labels, second distance or None, second label or None)
        receives the results instead of new tensors.
        """
        n = len(rows)
        if out is None:
            distance = torch.empty(n, dtype=torch.float32, device=rows.device)
            labels = torch.empty(n, dtype=torch.int64, device=rows.device)
            second_d = second_l = None
            if second:
                second_d = torch.empty(n, dtype=torch.float32, device=rows.device)
                second_l = torch.empty(n, dtype=torch.int32, device=rows.device)
        else:
            distance, labels, second_d, second_l = out
        result = self._execution_backend.assign(
            rows, prepared, norm, False, distance, labels, second_d,
            **({} if second_l is None else {"second_labels_out": second_l}))
        if not self._execution_backend.writes_outputs:
            if out is None:
                distance, labels = result
            else:
                distance.copy_(result[0])
                labels.copy_(result[1])
        return distance, labels, second_d, second_l

    def _row_norms(self, rows):
        """Squared row norms summed in FP32 (on CUDA without a copy of the rows)."""
        if rows.is_cuda and row_norms is not None:
            norms = torch.empty(len(rows), dtype=torch.float32, device=rows.device)
            row_norms(rows, norms)
            return norms
        if rows.is_cuda:
            return torch.linalg.vector_norm(rows, dim=1, dtype=torch.float32).square_()
        return rows.float().square().sum(dim=1)

    # ------------------------------------------------------------ profiling

    def profile_phases(self):
        """Record CUDA events between the phases of every step (no host sync).

        Not with CUDA graphs: a replayed step records no events.
        """
        self._phase_events = []

    def _mark(self, name):
        if self._phase_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._phase_events.append((name, event))

    def phase_times(self):
        """Milliseconds per phase name, summed over the run (after ``profile_phases``)."""
        torch.cuda.synchronize()
        totals = {}
        events = self._phase_events or []
        for (_, start), (name, end) in zip(events, events[1:]):
            if name != "begin":
                totals[name] = totals.get(name, 0.0) + start.elapsed_time(end)
        return totals

    # --------------------------------------------------------------- step

    def _train_step(self, x, index, training, scale, alpha, cap, revive, state):
        """One training or polish step on the rows ``x[index]``; returns the capped codes.

        ``scale``, ``alpha`` and ``cap`` are host numbers, or 0-dim device
        tensors on the device path (``state`` holds its persistent buffers).
        """
        self._mark("begin")
        n = len(index)
        bias = self.bias
        prepared, norm, shift = self._biased_assignment(self._centroids, bias, frozen=not training)
        self._mark("prepare")
        chunk = self.recipe["assignment_chunk_size"]
        capping = self._cap_active(training)
        if state is not None:
            rows_all = state["rows"]
            gather_rows(x, index, rows_all, state["row_norm"])
            codes, dist, second_d, second_l, row_norm = self._assign_batch_device(
                rows_all, prepared, norm, capping, state)
        else:
            rows_all = self._prepare_rows(x[index]) if n <= chunk else None
            codes, dist, second_d, second_l, row_norm = self._assign_batch(
                x, index, rows_all, prepared, norm, capping)
        self._mark("assign")
        # The cap binds on the global batch (``_batch_shards`` slices of n rows).
        if capping and (state is not None or cap < n * self._batch_shards):
            codes, dist = self._cap_rounds_step(
                x, index, rows_all, codes, dist, second_d, second_l, row_norm, cap,
                prepared, norm, shift, bias, state)
        self._mark("cap")
        # Statistics from the capped codes, against the centroids the
        # assignment used. D reads the kernel's distances without the bias.
        if state is None:
            true_dist = dist - (shift - bias[codes])
            stats, counts, objective, batch_mean = self._batch_statistics(
                x, index, rows_all, codes, true_dist, training)
            self._objective += objective
        else:
            stats, counts = self._batch_statistics_device(
                x, index, rows_all, codes, dist, shift, training, state)
        self._pass_counts += counts
        self._mark("stats")
        # Dead test and revival on the state the assignment used.
        if revive:
            if state is not None:
                self._revive_fixed()
            else:
                self._revive()
        self._mark("revive")
        if state is None:
            self._bias_step(counts, batch_mean, scale, training)
        else:
            self._bias_step_device(counts, scale, training, state)
        self._mark("bias")
        if training:
            self._execution_backend.update(
                self._centroids, self._usage, None, stats[:, :-1], counts,
                alpha, 1.0 - alpha if state is None else state["decay"],
                state is None and alpha == 1)
            self._mirror = None
        self._mark("ema")
        return codes

    def _assign_batch(self, x, index, rows_all, prepared, norm, second):
        """Biased assignment of the batch, chunk by chunk: (codes, dist, second_d, second_l,
        row_norm); the last three are None without ``second``."""
        n, device = len(index), self._centroids.device
        chunk = self.recipe["assignment_chunk_size"]
        codes = torch.empty(n, dtype=torch.int64, device=device)
        dist = torch.empty(n, dtype=torch.float32, device=device)
        second_d = second_l = row_norm = None
        if second:
            second_d = torch.empty(n, dtype=torch.float32, device=device)
            second_l = torch.empty(n, dtype=torch.int64, device=device)
            row_norm = torch.empty(n, dtype=torch.float32, device=device)
        for begin in range(0, n, chunk):
            end = min(begin + chunk, n)
            rows = (rows_all[begin:end] if rows_all is not None
                    else self._prepare_rows(x[index[begin:end]]))
            d1, l1, d2, l2 = self._assign_chunk(rows, prepared, norm, second)
            dist[begin:end], codes[begin:end] = d1, l1
            if second:
                second_d[begin:end], second_l[begin:end] = d2, l2
                row_norm[begin:end] = self._row_norms(rows)
        return codes, dist, second_d, second_l, row_norm

    def _assign_batch_device(self, rows_all, prepared, norm, second, state):
        """Device path of ``_assign_batch``, into the persistent buffers of ``state``.

        The row norms were computed with the gather. The runner-up labels are
        returned only when the cap plan asks for them (``"second": "labels"``).
        """
        n = len(rows_all)
        chunk = self.recipe["assignment_chunk_size"]
        labels = second and self._cap_plan["second"] == "labels"
        codes, dist = state["codes"], state["dist"]
        second_d = state["second_d"] if second else None
        second_l = state["second_l"] if labels else None
        for begin in range(0, n, chunk):
            end = min(begin + chunk, n)
            self._assign_chunk(rows_all[begin:end], prepared, norm, out=(
                dist[begin:end], codes[begin:end],
                None if second_d is None else second_d[begin:end],
                None if second_l is None else second_l[begin:end]))
        return codes, dist, second_d, second_l, state["row_norm"]

    def _batch_statistics(self, x, index, rows_all, codes, true_dist, training):
        """(stats (k, d+1) or None, counts (k,), FP64 objective, mean distance) of a batch.

        Training accumulates the sums and counts of the capped codes; the
        polish only counts. ``true_dist`` is the distance without the bias.
        """
        k, n = self.k, len(index)
        chunk = self.recipe["assignment_chunk_size"]
        objective = true_dist.sum(dtype=torch.float64)
        stats = None
        if training:
            stats = self._batch_stats.zero_()
            self._execution_backend.begin_batch(stats, min(chunk, n))
            for begin in range(0, n, chunk):
                end = min(begin + chunk, n)
                rows = (rows_all[begin:end] if rows_all is not None
                        else self._prepare_rows(x[index[begin:end]]))
                self._execution_backend.accumulate(rows, codes[begin:end], stats, n)
            counts = stats[:, -1]
        else:
            counts = torch.zeros(k, dtype=torch.float32, device=codes.device)
            counts.index_add_(0, codes, torch.ones(n, dtype=torch.float32, device=codes.device))
        return stats, counts, objective, objective / n

    def _batch_statistics_device(self, x, index, rows_all, codes, dist, shift, training, state):
        """Device path of ``_batch_statistics``: (stats or None, counts). The batch's distance
        sum (bias removed) stays on the device for the bias step (``state["bias"]``) and is
        added to the epoch objective in the same launch."""
        fused = state["bias"]
        fused.distances(dist, codes, self.bias, shift, None, not training)
        if training:
            n = len(index)
            chunk = self.recipe["assignment_chunk_size"]
            stats = self._batch_stats.zero_()
            self._execution_backend.begin_batch(stats, min(chunk, n))
            for begin in range(0, n, chunk):
                self._execution_backend.accumulate(rows_all[begin:begin + chunk],
                                                   codes[begin:begin + chunk], stats, n)
            counts = stats[:, -1]
        else:
            stats, counts = None, fused.counts
        self._reduce_device_batch(stats if training else counts, fused.batch_sum)
        self._objective += fused.batch_sum
        return stats, counts

    def _reduce_device_batch(self, statistics, batch_sum):
        """Hook (identity here): a data-parallel subclass all-reduces the batch's (k, d+1)
        statistics (or the polish counts) and its FP64 distance sum in place."""

    def _bias_step_device(self, counts, scale, training, state):
        first = self._distance_scale is None
        if first:
            self._distance_scale = torch.zeros((), dtype=torch.float32, device=counts.device)
        u = self.u if training else self.recipe["polish_u"]
        state["bias"].band(counts, self._lam, self._mu, self.bias, self._distance_scale, scale,
                           u, first, _DISTANCE_SCALE_RATE)

    def _cap_rounds_step(self, x, index, rows_all, codes, dist, second_d, second_l, row_norm,
                         cap, prepared, norm, shift, bias, state):
        """The one-pass cap, then ``cap_rounds`` masked repair rounds."""
        k = self.k
        if state is not None:
            fused = state["fused"]
            plan = self._cap_plan
            mirror = self._mirror_rows(prepared, norm)
            biased = self._biased_norm(prepared, norm)
            fused.first_pass(codes, dist, second_d, second_l, row_norm, cap, plan["first"],
                             state["rows"], mirror, biased)
            self._mark("cap_first")
            search = None
            if norm is None:
                search = (self._exact_search if -1 in plan["bucket"] else self._bucket_search)
            self._bucket_context = (prepared, shift, bias)
            for r in range(self.cap_rounds):
                fused.round(codes, dist, row_norm, cap, state["rows"], mirror, biased, r,
                            plan["round"][r], plan["bucket"][r], search)
                self._mark("cap_round")
            self._bucket_context = None
            return codes, dist
        codes, moved = cap_first_pass(codes, second_d - dist, second_l, cap, k)
        dist = torch.where(moved, second_d, dist)
        self._mark("cap_first")
        # The kernel score at the current cluster (row norm dropped), which
        # ranks the arrivals: the runner-up's for every row.
        score = second_d - row_norm
        arrived = moved
        self.cap_stats["steps"] += 1
        for _ in range(self.cap_rounds):
            evict, full = cap_round_evict(codes, arrived, score, cap, k)
            ids = evict.nonzero().flatten()  # One host synchronization per round.
            if len(ids) == 0:
                break
            self.cap_stats["evicted"] += len(ids)
            masked = torch.where(full, float("inf"), self._biased_norm(prepared, norm))
            if norm is None:
                saved = prepared.norms.clone()
                prepared.norms.copy_(masked)
            width = max(_REPAIR_MIN_ROWS, 1 << (len(ids) - 1).bit_length())
            padded = torch.cat((ids, ids[-1:].expand(width - len(ids))))
            rows = self._prepare_rows(x[index[padded]])
            d1, l1, _, _ = self._assign_chunk(rows, prepared, None if norm is None else masked,
                                              second=False)
            if norm is None:
                prepared.norms.copy_(saved)
            d1, l1 = d1[:len(ids)], l1[:len(ids)]
            codes[ids] = l1
            dist[ids] = d1
            score[ids] = d1 - self._row_norms(rows[:len(ids)])
            arrived[ids] = True
            self._mark("cap_round")
        return codes, dist

    def _masked_prepared(self, prepared, counts, cap, shift, bias):
        """A CuTe prepared view of the same mirror whose norms are biased, with the
        clusters holding >= cap rows at +inf (own norms and bindings: the main ones stay)."""
        import dataclasses
        from collections import OrderedDict

        masked = getattr(self, "_masked_view", None)
        if masked is None or masked.centroids is not prepared.centroids:
            storage = torch.full_like(prepared.padded_norms.reshape(-1), float("inf"))
            masked = dataclasses.replace(prepared, norms=storage[:self.k],
                                         padded_norms=storage.view(1, -1),
                                         bindings=OrderedDict())
            self._masked_view = masked
        torch.add(self._base_norm, shift - bias, out=masked.norms)
        masked.norms.masked_fill_(counts >= cap, float("inf"))
        return masked

    def _bucket_search(self, rows, counts, cap, distances, labels):
        """CuTe top-1 of ``rows`` with the clusters holding >= cap rows masked out."""
        prepared, shift, bias = self._bucket_context
        masked = self._masked_prepared(prepared, counts, cap, shift, bias)
        self._execution_backend.assign(rows, masked, None, False, distances, labels)

    def _exact_search(self, rows, positions, evicted, counts, cap, distances, labels, buffer):
        """CuTe top-1 of rows[positions[:evicted]], clusters holding >= cap masked out.

        The rows go in chunks of 1024 * 4^j rows (largest that fits, the last one padded with
        other batch rows), so a run binds at most a handful of CuTe shapes.
        """
        prepared, shift, bias = self._bucket_context
        masked = self._masked_prepared(prepared, counts, cap, shift, bias)
        n = len(positions)
        if not _EXACT_LADDER:
            # One call, padded to a quarter of a power of two.
            width = max(1024, 1 << (evicted - 1).bit_length())
            width = min(n, max(1024, -(-evicted // (width // 4)) * (width // 4)))
            torch.index_select(rows, 0, positions[:width], out=buffer[:width])
            self._execution_backend.assign(buffer[:width], masked, None, False,
                                           distances[:width], labels[:width])
            return
        begin = 0
        while begin < evicted:
            width = 1024
            while width * 4 <= min(evicted - begin, n):
                width *= 4
            width = min(width, n)
            if begin + width > n:
                begin = n - width  # Re-score a few rows rather than read past the batch.
            part = positions[begin:begin + width]
            torch.index_select(rows, 0, part, out=buffer[:width])
            self._execution_backend.assign(buffer[:width], masked, None, False,
                                           distances[begin:begin + width],
                                           labels[begin:begin + width])
            begin += width

    def _update_cap_plan(self, state, begin, end):
        """Choose ranking methods, CuTe buckets and the runner-up source from the steps
        [begin, end) (synchronizes).

        Both rankings give identical decisions; the choice is speed only. The bucket moves
        rows between two searches (CuTe and Triton), and so does the runner-up source, so
        they change only here, at fixed steps.
        """
        if end <= begin or not self._device_mode:
            return
        stats = state["fused"].stats[begin:end].double().cpu()
        n = self.recipe["batch_size"] // self._batch_shards
        cute = getattr(self._execution_backend, "name", None) == "cute"
        # The cap's work drifts within a pass (it grows early in training, then
        # fades): the last quarter of the steps predicts the next pass best.
        recent = stats[-max(1, len(stats) // 4):]
        work = recent[:, :3].mean(dim=0)
        moved = float(recent[:, 5].mean())
        plan = dict(first="pair" if work[0] <= _PAIR_WORK_LIMIT else "sort",
                    round=[], bucket=[],
                    second="labels" if moved > _LAZY_MOVED_FRACTION * n else "lazy")
        for r in range(self.cap_rounds):
            plan["round"].append("pair" if work[1 + min(r, 1)] <= _PAIR_WORK_LIMIT else "sort")
            evicted = float(stats[:, 3 + min(r, 1)].quantile(_BUCKET_QUANTILE))
            bucket = 0
            if cute and self._exact_rounds and evicted >= _BUCKET_MIN_ROWS:
                bucket = -1
            elif cute and evicted >= _BUCKET_MIN_ROWS:
                # 2^j or 1.5 * 2^j rows: a few CuTe shapes, at most 1.5x the quantile.
                low = 1 << int(math.floor(math.log2(evicted)))
                bucket = min(n, low if evicted <= low else 3 * low // 2 if evicted <= 3 * low // 2
                             else 2 * low)
            plan["bucket"].append(bucket)
        self._cap_plan = plan
        self._cap_plans.append((end, plan))

    # -------------------------------------------------------- bias, revival

    def _bias_step(self, counts, batch_mean, scale, training):
        if self._distance_scale is None:
            self._distance_scale = batch_mean.float().clone()
        else:
            self._distance_scale.mul_(1.0 - _DISTANCE_SCALE_RATE).add_(
                batch_mean.float(), alpha=_DISTANCE_SCALE_RATE)
        u = self.u if training else self.recipe["polish_u"]
        eta = u * self._distance_scale * scale
        band_step(self._lam, self._mu, self.bias, counts, eta)

    def _select_donors(self, usage, receivers, dead):
        """(source, target) of the splits that revive ``receivers``, by ``donor_rule``."""
        if self.donor_rule == "largest":
            return largest_donors(usage, receivers, dead)
        return draw_donors(usage, receivers, self._split_rng)

    def _revive(self):
        """Dead test, then in-place revival below the median, splits above."""
        usage = self._usage
        threshold = usage.sum() * self._unused_threshold / self.k
        dead = usage < threshold
        if not bool(dead.any()):
            return
        if self.inplace_quantile is not None:
            inplace, reference = select_inplace(dead, self.bias, self.inplace_quantile)
            count = int(inplace.sum())
            if count:
                usage[inplace] = usage.mean()
                self._lam[inplace] = (-reference).clamp(min=0.0)
                self._mu[inplace] = reference.clamp(min=0.0)
                self.bias[inplace] = reference
                self._inplace_host += count
            rest = dead & ~inplace
        else:
            rest = dead
        source, target = self._select_donors(usage, rest, dead)
        if source is None:
            return
        donor, child = split_rows(self._centroids[source], self.split_eps)
        half = usage[source] / 2
        self._centroids[source] = donor
        self._centroids[target] = child
        usage[source] = half
        usage[target] = half
        for state in (self.bias, self._lam, self._mu):
            state[target] = state[source]
        self._n_splits += len(source)
        torch.autograd.graph.increment_version((self._centroids, self._usage))

    def _revive_fixed(self):
        """``_revive`` with fixed shapes and no host synchronization (CUDA graphs)."""
        usage, k = self._usage, self.k
        dead = usage < usage.sum() * self._unused_threshold / k
        if self.inplace_quantile is not None:
            inplace, reference = select_inplace_fixed(dead, self.bias, self.inplace_quantile)
            usage.copy_(torch.where(inplace, usage.mean(), usage))
            self._lam.copy_(torch.where(inplace, (-reference).clamp(min=0.0), self._lam))
            self._mu.copy_(torch.where(inplace, reference.clamp(min=0.0), self._mu))
            self.bias.copy_(torch.where(inplace, reference, self.bias))
            self._inplace_device += inplace.sum()
            rest = dead & ~inplace
        else:
            rest = dead
        if self.donor_rule == "largest":
            source, target, count = largest_donors_fixed(usage, rest, dead)
        else:
            source, target, count = draw_donors_fixed(usage, rest, self._split_rng)
        valid = torch.arange(k, device=usage.device) < count
        donor, child = split_rows(self._centroids[source], self.split_eps)
        half = usage[source] / 2
        # Donors first, then children: every write is a permutation, and the
        # entries past ``count`` write back the value they read.
        self._centroids.index_copy_(0, source, torch.where(valid[:, None], donor,
                                                           self._centroids[source]))
        usage.index_copy_(0, source, torch.where(valid, half, usage[source]))
        self._centroids.index_copy_(0, target, torch.where(valid[:, None], child,
                                                           self._centroids[target]))
        usage.index_copy_(0, target, torch.where(valid, half, usage[target]))
        for state in (self.bias, self._lam, self._mu):
            state.index_copy_(0, target, torch.where(valid, state[source], state[target]))
        self._check_splits.add_(count)
        torch.autograd.graph.increment_version((self._centroids, self._usage))

    # ------------------------------------------------------------ training

    def train(self, x, init_centroids=None):
        """Train on x; return centroids and last-pass labels (-1 outside the subset).

        The labels are the last pass's capped, biased labels. Rows that pass
        dropped (the n mod B tail of its shuffle) take the final biased
        assignment.
        """
        keep = self._training_subset(len(x))
        if keep is None:
            return self._train_rows(x, init_centroids)
        index = torch.as_tensor(keep, device=x.device)
        _, sampled = self._train_rows(x[index], init_centroids)
        if sampled is not None:
            labels = torch.full((len(x),), -1, dtype=sampled.dtype, device=sampled.device)
            labels[index.to(sampled.device)] = sampled
            self.assignments = labels
        return self.centroids, self.assignments

    # A data-parallel subclass trains on a rank-local shard of the rows: it
    # returns the global row count here and splits every batch into shards.
    _batch_shards = 1

    def _global_rows(self, local_rows):
        return local_rows

    def _draw_initial_centroids(self, x):
        """Random start: the first k rows of a seeded permutation (public Emeans' draw)."""
        n = len(x)
        rows = x if n == self.k else x[torch.randperm(
            n, generator=_torch_generator(x.device, self.seed), device=x.device)[:self.k]]
        return self._initial_centroid_values(rows)

    def _device_state(self, n, device):
        """Persistent buffers and per-step tables of the device path."""
        recipe = self.recipe
        batch = recipe["batch_size"] // self._batch_shards
        steps = [self._schedule(s) for s in range(recipe["total_steps"])]
        table = torch.tensor([[s, a, 1.0 - a] for _, s, a, _ in steps], dtype=torch.float32,
                             device=device)
        caps = torch.tensor([c for _, _, _, c in steps], dtype=torch.int64, device=device)
        self._inplace_device = torch.zeros((), dtype=torch.int64, device=device)
        self._check_splits = torch.zeros((), dtype=torch.int64, device=device)
        step = torch.zeros(1, dtype=torch.int64, device=device)
        self._cap_plan = dict(first="pair", round=["pair"] * self.cap_rounds,
                              bucket=[0] * self.cap_rounds, second="labels")
        self._cap_plans = [(0, self._cap_plan)]
        return dict(
            rows=torch.empty((batch, self.d), dtype=self._input_dtype, device=device),
            row_norm=torch.empty(batch, dtype=torch.float32, device=device),
            codes=torch.empty(batch, dtype=torch.int64, device=device),
            dist=torch.empty(batch, dtype=torch.float32, device=device),
            second_d=torch.empty(batch, dtype=torch.float32, device=device),
            second_l=torch.empty(batch, dtype=torch.int32, device=device),
            fused=FusedCap(batch, self.k, device, recipe["total_steps"], step),
            bias=FusedBias(batch, self.k, device, share_rows=recipe["batch_size"]),
            table=table, caps=caps[:, None],
            step=step,
            position=torch.zeros(1, dtype=torch.int64, device=device),
            index=torch.empty(batch, dtype=torch.int64, device=device),
        )

    def _device_step(self, x, batches, training, revive, state):
        """One step of the device path: every input read from device tables."""
        torch.index_select(batches, 0, state["position"], out=state["index"].unsqueeze(0))
        row = state["table"].index_select(0, state["step"])[0]
        state["decay"] = row[2]
        cap = state["caps"].index_select(0, state["step"])[0]
        codes = self._train_step(x, state["index"], training, row[0], row[1], cap, revive, state)
        state["position"].add_(1)
        state["step"].add_(1)
        return codes

    def _capture(self, x, batches, training, revive, state):
        """Record ``_device_step`` as a CUDA graph (public Emeans' capture pattern)."""
        device = x.device
        backend = self._execution_backend
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.MemPool()
        if training:
            self._mirror = None
            cached = backend.capture_state()
        else:
            cached = getattr(backend, "cache", None)
        captured = False
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                backend.initialize_capture()
                graph.capture_begin(pool=pool.id, capture_error_mode="thread_local")
                try:
                    codes = self._device_step(x, batches, training, revive, state)
                finally:
                    graph.capture_end()
            captured = True
            current.wait_stream(stream)
        finally:
            if training:
                backend.finish_capture(cached, captured)
            if not captured:
                graph.reset()
        if training:
            self._mirror = None
        return dict(graph=graph, pool=pool, codes=codes, cached=cached,
                    bindings=backend.capture_bindings(cached))

    def _train_rows(self, x, init_centroids=None):
        start, given_init = time.perf_counter(), init_centroids is not None
        x = x.detach()
        local_n = len(x)
        n = self._global_rows(local_n)
        device = x.device
        self._resolve_input_dtype(x)
        self._resolve_assignment_backend(x)
        self._execution_backend.require_training(device)
        if init_centroids is None:
            init_centroids = self._draw_initial_centroids(x)
        self.begin(n, init_centroids)
        x = self._prepare_input(x)
        self._report_header_once()
        if self._reports():
            self._report_preprocessing(start, given_init, n)
        recipe = self.recipe
        shards = self._batch_shards
        batch = recipe["batch_size"] // shards
        self._device_mode = (x.is_cuda and self._input_dtype in (torch.bfloat16, torch.float16)
                             and FusedCap is not None
                             and getattr(self, "_allow_device_path", True))
        state = self._device_state(n, device) if self._device_mode else None
        graphs = {}
        self._plan_from = 0
        # Large steps with many evicted rows re-score them with CuTe at their exact count (one
        # host read per round, no graph for those steps): cheaper there than a padded bucket.
        cute = getattr(self._execution_backend, "name", None) == "cute"
        self._exact_rounds = (self._device_mode and cute and self.cap_mode == "rounds"
                              and batch * self.k * self.d >= _EXACT_ROUNDS_WORK)
        use_graphs = self._device_mode and self.cuda_graphs and self._phase_events is None
        if self._exact_rounds:
            self._cap_plan = dict(self._cap_plan, bucket=[-1] * self.cap_rounds)
            self._cap_plans = [(0, self._cap_plan)]
        assignments = order = None
        try:
            step = 0
            for epoch, global_rows in enumerate(recipe["epoch_rows"]):
                rows = global_rows // shards
                shuffled = torch.randperm(
                    local_n, generator=_torch_generator(device, _epoch_seed(self.seed, epoch)),
                    device=device)
                if order is None:
                    order = shuffled
                else:
                    order.copy_(shuffled)  # Graphs read this storage.
                del shuffled
                last = epoch == len(recipe["epoch_rows"]) - 1
                if last and self.capture_assignments:
                    assignments = torch.full((local_n,), -1, dtype=torch.int64, device=device)
                batches = order[:rows].view(-1, batch)
                if state is not None:
                    state["position"].zero_()
                for begin in range(0, rows, batch):
                    training, scale, alpha, cap = self._schedule(step)
                    revive = self._revives(step)
                    if state is None:
                        index = order[begin:begin + batch]
                        codes = self._train_step(x, index, training, scale, alpha, cap,
                                                 revive, None)
                    else:
                        if step == _FIRST_PLAN_STEP or (begin == 0 and epoch > 0):
                            before = self._cap_plan
                            self._update_cap_plan(state, self._plan_from, step)
                            self._plan_from = step
                            if self._cap_plan != before:  # Free the graphs of the old plan.
                                for entry in graphs.values():
                                    if entry is not None:
                                        entry["graph"].reset()
                                graphs.clear()
                        kind = (training, revive, json.dumps(self._cap_plan, sort_keys=True))
                        # The first step of each kind runs eagerly (compiling, warming
                        # caches), as does the first step (D starts there) and the first
                        # polish step (it rebuilds the frozen mirror).
                        first_polish = step == recipe["train_steps"]
                        exact = -1 in self._cap_plan["bucket"]
                        if (not use_graphs or exact or step == 0 or first_polish
                                or kind not in graphs):
                            if use_graphs and step > 0 and not first_polish and not exact:
                                graphs[kind] = None  # Warmed; capture at the next one.
                            codes = self._device_step(x, batches, training, revive, state)
                        else:
                            entry = graphs[kind]
                            if entry is None:
                                entry = graphs[kind] = self._capture(
                                    x, batches, training, revive, state)
                            if training:
                                self._execution_backend.validate_capture(entry["cached"])
                            entry["graph"].replay()
                            torch.autograd.graph.increment_version((self._centroids, self._usage))
                            codes = entry["codes"]
                        index = state["index"]
                    if assignments is not None and last:
                        assignments[index] = codes
                    step += 1
                    self._step = step
                    self._epoch_steps += 1
                    self._epoch_rows += batch * shards
                self.end_epoch()
            if assignments is not None and rows < local_n:
                tail = order[rows:]
                _, labels = self._assign_biased(x[tail])
                assignments[tail] = labels
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        finally:
            for entry in graphs.values():
                if entry is not None:
                    entry["graph"].reset()
            graphs = None
        if state is not None:
            fused = state["fused"]
            self.cap_stats.update(moved=int(fused.totals[0]),
                                  evicted=int(fused.stats[:, 3:5].sum()),
                                  steps=recipe["total_steps"])
        self.finalize()
        self.assignments = assignments
        return self.centroids, self.assignments

    def end_epoch(self):
        inplace = self.n_inplace - self._inplace_before
        epoch = self._epoch
        super().end_epoch()
        stat = self.iteration_stats[-1]
        stat["ninplace"] = inplace
        stat["phase"] = "train" if epoch < self.train_epochs else "polish"
        if self._reports():
            Emeans._report_epoch(self, stat, epoch)
            self._say("    %s pass, imbalance %.4f, %d splits, %d in place"
                      % (stat["phase"], stat["imbalance_factor"], stat["nsplit"], inplace))

    def _report_epoch(self, stat, epoch):
        pass  # end_epoch reports, once the balanced fields are filled in.

    def _keeps_removal_stats(self):
        return False  # Mass-based revival: no removal costs or axes.

    def accumulate(self, x):
        raise NotImplementedError(
            "EmeansBalanced trains through train(): the capacity cap ranks whole batches")

    def update(self, stats):
        raise NotImplementedError(
            "EmeansBalanced trains through train(): the capacity cap ranks whole batches")

    def finalize(self):
        if self.centroids is not None:
            return self.centroids
        if self._failed:
            raise RuntimeError("create a new object after execution failure")
        self._failed = True
        self.centroids = self._centroids.clone()
        self.obj = torch.tensor([s["obj"] for s in self.iteration_stats], dtype=torch.float64,
                                device=self.centroids.device)
        self._inplace_host, self._inplace_device = self.n_inplace, None
        self._n_splits, self._check_splits = self.n_splits, None
        self._centroids = self._usage = self._batch_stats = None
        self._pass_sums = self._pass_counts = self._objective = None
        self._pending_stats = None
        self._base_norm = self._base_token = None
        self._mirror = None
        self._execution_backend.clear_statistics()
        self._failed = False
        return self.centroids

    # ------------------------------------------------------------ queries

    def assign(self, x, biased=True):
        """Distances and labels with the bias (deployed routing) or without.

        Distances are the true squared distances to the chosen centroids
        (bias not included), in the assignment representation.
        """
        self._query_biased = bool(biased)
        try:
            return super().assign(x)
        finally:
            self._query_biased = True

    def _assign_biased(self, x):
        """Biased assignment against the live training state (used for the tail)."""
        prepared, norm, shift = self._biased_assignment(self._centroids, self.bias)
        chunk = self._get_assignment_chunk_size()
        distance = torch.empty(len(x), dtype=torch.float32, device=x.device)
        labels = torch.empty(len(x), dtype=torch.int64, device=x.device)
        for begin in range(0, len(x), chunk):
            rows = self._prepare_rows(x[begin:begin + chunk])
            d1, l1, _, _ = self._assign_chunk(rows, prepared, norm, second=False)
            distance[begin:begin + len(rows)] = d1 - (shift - self.bias[l1])
            labels[begin:begin + len(rows)] = l1
        return distance, labels

    def _prepare_assignment_query(self, centroids):
        if not getattr(self, "_query_biased", True) or self.bias is None:
            self._query_shift = None
            if getattr(self._execution_backend, "cache", None) is not None:
                self._execution_backend.cache.source_version = None  # May hold biased norms.
            return super()._prepare_assignment_query(centroids)
        shift = self.bias.max()
        backend = self._execution_backend
        if getattr(backend, "cache", None) is not None:
            backend.cache.source_version = None  # Recompute the plain norms.
        prepared, norm = backend.prepare_centroids(centroids, self._input_dtype)
        self._query_shift = shift
        if norm is not None:
            return prepared, norm + (shift - self.bias)
        prepared.norms.add_(shift - self.bias)
        prepared.source_version = None  # The next query recomputes the plain norms.
        return prepared, None

    def _assign(self, x):
        distance, labels = super()._assign(x)
        shift = getattr(self, "_query_shift", None)
        if shift is not None:
            distance -= shift - self.bias[labels]
        return distance, labels

    def _report_header(self):
        super()._report_header()
        recipe = self.recipe
        self._say("  Balanced: u %g, %d + %d passes, cap %s, revival %s, %s donor"
                  % (recipe["u"], recipe["train_epochs"], recipe["polish_epochs"],
                     "off" if recipe["cap_mode"] is None else "%s x%g (%s)" % (
                         recipe["cap_mode"], recipe["cap_factor"], recipe["cap_phase"]),
                     recipe["revival"], recipe["donor_rule"]))
