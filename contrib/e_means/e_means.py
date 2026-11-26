# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""FP32 E-means clustering with NumPy and Faiss assignment."""

import contextlib
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss

from .base import (
    _FLOAT32_COUNT_LIMIT,
    _EmeansBase,
    _INIT_USAGE,
    _UNUSED_THRESHOLD,
    _compact_label_dtype,
    _epoch_rows,
    _epoch_seed,
    _format_passes as _format_passes,
    _resolve_recipe,
    _resolve_unused_threshold,
    _round_power_of_two as _round_power_of_two,
    _same_operation,
    _scheduled_alpha as _scheduled_alpha,
    _step_law,
    _twin_reassign,
)

# Rows per step of the closing-pass relabelling, bounding its temporaries.
_HANDOVER_CHUNK = 1 << 16

# Threaded host passes give each thread a block of at least this many
# elements (the GIL is taken between NumPy calls, so small blocks only wait);
# smaller passes stay on the calling thread.
_BLOCK_ELEMENTS = 1 << 18

# Elements per step of the twin reassignment within a block: its FP64
# temporaries stay a few MB, recycled by the allocator, not page-faulted.
_TWIN_STEP_ELEMENTS = 1 << 21

# e-means-ivf on CPU: k-means iterations of the coarse cells, trained once
# on the first centroids, and logical updates between two coarse
# re-assignments of every centroid. In between, the lists keep each
# centroid's cell (split centroids are re-assigned at once) and are refilled
# with the current positions before every search, so listed distances are
# exact. Re-assigning every 1, 5 or 20 updates gave the same test error
# (DINO k = 16384, 65536, DEEP k = 65536, 1M rows); each re-assignment costs
# a k x nlist search (0.23 s at k = 65536, nlist = 4096, d = 1024).
_IVF_COARSE_NITER = 10
_IVF_REASSIGN_EVERY = 20

# nlist="auto" on CPU: exact search below _IVF_AUTO_MIN_K centroids, else
# nlist = _IVF_AUTO_LISTS * sqrt(k) rounded up to a power of 2 and nprobe =
# _IVF_AUTO_NPROBE (1024 / 16 at k = 16384, 2048 / 16 at k = 65536). Fitted
# on DINO (d = 1024), DEEP (96) and LAION-CLIP (512) at 1M rows, k = 2**10 to
# 2**16, 32 threads, for the time to reach exact e-means' test error after 1,
# 2 and 3 epochs: the near-best rule that never missed that error; IVF was
# slower than exact at k <= 4096 on 2 of the 3. On 9 other datasets (d = 100
# to 3072) at k >= 16384 it reached exact's error 1.2-7.9x faster, except
# SSNPP at k = 16384 (even or slower), and 0.4-2.0x at k = 8192 (slower on SSNPP
# and SPACEV), for +0.0-0.5% test error at equal epochs.
_IVF_AUTO_MIN_K = 8192
_IVF_AUTO_LISTS = 8
_IVF_AUTO_NPROBE = 16


__all__ = [
    "NumpyEmeans",
    "_epoch_rows",
    "_epoch_seed",
    "_format_passes",
    "_resolve_recipe",
    "_round_power_of_two",
    "_scheduled_alpha",
    "_step_law",
]


def _numpy_unused_threshold(usage, k, coefficient=_UNUSED_THRESHOLD):
    """Compute the threshold consistently across NumPy scalar rules."""
    total = usage.sum(dtype="float32")
    scaled = np.multiply(total, np.float32(coefficient), dtype="float32")
    return np.divide(scaled, np.float32(k), dtype="float32")


def _add_rows_at(out, labels, rows):
    """out[labels[i]] += rows[i] for every i, in row order, as np.add.at.
    The labels are in [0, len(out)).

    The sparse product [I | one-hot] @ [sums; rows] starts the sum of each
    centroid at its row of out and adds its rows in row order, so the result
    is the same as np.add.at, and much faster on 2-D rows. With fewer rows
    than centroids, the sums are only the rows of out that the labels touch.
    Without SciPy, or when rows cannot be cast to the dtype of out without
    a loss (np.add.at then rounds to that dtype after each add), fall back to
    np.add.at.
    """
    if not np.can_cast(rows.dtype, out.dtype, "safe"):
        np.add.at(out, labels, rows)
        return
    try:
        import scipy.sparse
    except ImportError:
        np.add.at(out, labels, rows)
        return
    k, n = len(out), len(labels)
    touched = None
    if n < k:
        touched, labels = np.unique(labels, return_inverse=True)
    sums = out if touched is None else out[touched]
    m = len(sums)
    # shape: an empty chunk has no index to infer it from
    product = scipy.sparse.csr_matrix(
        (
            np.ones(m + n, dtype=out.dtype),
            (np.concatenate([np.arange(m), labels]), np.arange(m + n)),
        ),
        shape=(m, m + n),
    )
    result = product @ np.concatenate([sums, rows])
    if touched is None:
        out[...] = result
    else:
        out[touched] = result


def _separate_splits(donor, child):
    """Separate identical donor and child centroids along one coordinate.

    Zero donors become distinct subnormal operands; their squared distances
    can still tie. No value scan or dynamic collapsed-row list is needed.
    """
    collapsed = (donor == child).all(axis=1)
    value = child[:, 0]
    with np.errstate(over="ignore"):
        neighbor = np.nextafter(value, np.float32(np.inf))
    neighbor = np.where(
        np.isfinite(neighbor),
        neighbor,
        np.nextafter(value, np.float32(-np.inf)),
    )
    child[:, 0] = np.where(collapsed, neighbor, value)


def _resolve_ivf_size(k, nlist, nprobe):
    """Return (nlist, nprobe, rule) for e-means-ivf over k CPU centroids.

    nlist=None keeps exact assignment (nprobe must stay "auto"), and so does
    nlist="auto" below _IVF_AUTO_MIN_K centroids, where IVF did not pay off
    on CPU; then nlist and nprobe are None and ``rule`` says why. Otherwise
    "auto" gives nlist = _IVF_AUTO_LISTS * sqrt(k) rounded up to a power of 2
    and nprobe = _IVF_AUTO_NPROBE. Explicit integers must satisfy
    1 <= nprobe <= nlist <= k.
    """
    if nlist is None:
        if not _is_auto(nprobe):
            raise ValueError("nprobe requires nlist")
        return None, None, None
    if _is_auto(nlist) and k < _IVF_AUTO_MIN_K:
        if not _is_auto(nprobe):
            _ivf_count(nprobe, "nprobe")
        return None, None, "auto: exact search, IVF needs k >= %d on CPU (k = %d)" % (
            _IVF_AUTO_MIN_K, k)
    rules = []
    if _is_auto(nlist):
        nlist = min(k, 2 ** math.ceil(math.log2(_IVF_AUTO_LISTS * math.sqrt(k))))
        rules.append("nlist = %g * sqrt(k) rounded up to a power of 2" % _IVF_AUTO_LISTS)
    else:
        nlist = _ivf_count(nlist, "nlist")
    if _is_auto(nprobe):
        nprobe = min(nlist, _IVF_AUTO_NPROBE)
        rules.append("nprobe = %d" % _IVF_AUTO_NPROBE)
    else:
        nprobe = _ivf_count(nprobe, "nprobe")
    if not 1 <= nprobe <= nlist <= k:
        raise ValueError("require 1 <= nprobe <= nlist <= k")
    return nlist, nprobe, "; ".join(rules) or "given"


def _is_auto(value):
    return isinstance(value, str) and value == "auto"


def _ivf_count(value, name):
    if isinstance(value, (bool, str)) or int(value) != value or value < 1:
        raise ValueError('%s must be a positive integer or "auto"' % name)
    return int(value)


@contextlib.contextmanager
def _without_interrupt_checks():
    """Run faiss calls without the Python interrupt callback, then restore it.

    faiss's Python module installs a callback that takes the GIL to look for
    Ctrl-C; IndexIVF searches call it after every query from every OpenMP
    thread, which serializes them. Like faiss.TimeoutGuard, this swaps the
    process-wide callback for the duration of the block and installs the
    Python callback again on exit, unless no callback was installed (faiss
    then hints a period of 2**30). Ctrl-C is handled when the block returns.
    """
    installed = faiss.InterruptCallback.get_period_hint(0) != 1 << 30
    if installed:
        faiss.InterruptCallback.clear_instance()
    try:
        yield
    finally:
        if installed:
            faiss.PythonInterruptCallback.reset()


class _CentroidIVF:
    """faiss IndexIVFFlat over the k current centroids (e-means-ivf on CPU).

    The nlist coarse cells are trained once with faiss.Kmeans on the first
    centroids. ``refill`` rebuilds the lists from the current positions:
    every ``_IVF_REASSIGN_EVERY`` calls it re-assigns every centroid to its
    nearest cell, otherwise only the ``moved`` ones, so most refills are a
    copy of the centroids into their lists (IndexIVF.add_core with the
    stored cells). Ids in the lists are centroid indices.
    """

    def __init__(self, centroids, nlist, nprobe, seed):
        k, d = centroids.shape
        kmeans = faiss.Kmeans(d, nlist, niter=_IVF_COARSE_NITER, seed=seed,
                              min_points_per_centroid=1)
        kmeans.train(centroids)
        self.quantizer = faiss.IndexFlatL2(d)
        self.quantizer.add(kmeans.centroids)
        self.index = faiss.IndexIVFFlat(self.quantizer, d, nlist, faiss.METRIC_L2)
        self.index.is_trained = True
        self.index.nprobe = nprobe
        self.ids = np.arange(k, dtype="int64")
        self.cells = None
        self.refills = self.reassignments = 0

    def refill(self, centroids, moved):
        if self.cells is None or self.refills % _IVF_REASSIGN_EVERY == 0:
            self.cells = self.quantizer.assign(centroids, 1).ravel().astype("int64")
            self.reassignments += 1
        elif len(moved):
            self.cells[moved] = self.quantizer.assign(
                np.ascontiguousarray(centroids[moved]), 1).ravel()
        self.refills += 1
        self.index.reset()
        self.index.add_core(len(centroids), *map(faiss.swig_ptr, (centroids, self.ids, self.cells)))

    def search(self, x, nearest):
        """Squared distances and ids of the nearest centroids in the probed
        lists, (n, nearest); missing results are inf and -1."""
        with _without_interrupt_checks():
            distance, labels = self.index.search(x, nearest)
        distance[labels < 0] = np.inf
        return distance, labels


class NumpyEmeans(_EmeansBase):
    """FP32 NumPy E-means with exact Faiss assignment.

    Implements Algorithm 1 of "E-Means: Scaling Laws for Mini-Batch K-Means":
    https://arxiv.org/abs/2611.09501

    ``nepochs`` is the epoch budget T and ``batch_size`` is B. ``alpha0`` is
    the initial EMA step, alpha = 1 - gamma, annealed with the cosine schedule
    of Equation 5. The batch law named by ``batch_law`` (``"throughput"``, or
    ``"quality"`` for a batch about 8 times smaller) chooses B, and the step law
    then sets alpha0 for that B. The default ``"auto"`` uses the quality law
    when d <= 16 and the throughput law otherwise. A ``batch_size`` given
    alone keeps the step law, and giving ``alpha0`` too sets both.
    ``recipe["batch_law"]`` and ``recipe["batch_law_reason"]`` record the
    resolved law.

    Each object represents one run. Call ``train`` or ``begin`` once;
    create a fresh object for another run or after execution failure.

    ``train`` samples initial centroids by default and shuffles each epoch
    (lines 1-4). For caller-driven training, ``begin`` initializes the state,
    ``accumulate`` assigns rows and collects batch counts n_j and sums s_j
    (lines 6-7), and ``update`` applies the EMA (line 8). ``_usage`` stores
    masses N_j and ``_centroids`` stores positions c_j = S_j / N_j, an
    algebraically equivalent form of the sum-and-count updates. Every five
    updates, the default ``unused_threshold=None`` moves centroids whose
    removal cost (EMA of d2 - d1 over their rows) is zero or below 0.3 times
    the median to split the donors with the longest axes, until the end of
    training. A split puts the two copies at c_j -/+ s_j u_j along the
    donor's axis u_j, s_j = sqrt(2 ||U_j|| / (pi N_j)): the half-means of a
    Gaussian along u_j. With ``closing_handover`` (default) and captured
    assignments, ``train`` hands the last-pass rows such a revived centroid held
    before moving to their second-nearest centroid from the same assignment,
    unless it moved after them too: those rows are assigned again like a
    training batch, at their share of one epoch's cost. A donor's other rows
    from before its first split after them then go to the nearer of that
    split's two centroids. Every centroid returns the mean of its last-pass
    rows (without captured labels, of those assigned to it after its last
    split). A value revives clusters below that
    fraction of the mean mass instead, splitting mass-sampled donors (line 9,
    Appendix J); 0 disables revival.

    ``finalize`` computes unweighted means from assignments collected while
    the centroids move during the last epoch (line 12). Unassigned centroids
    retain their positions. ``train`` returns centroids and last-pass labels.
    For Euclidean clustering, ``obj`` reports moving-pass sums of squared
    distances; use ``assign`` to
    evaluate the final centroids.

    Implementation details: the initial mass is 2**-16 (the paper uses 1e-10),
    remainder batches are retained, and fractional epoch budgets are supported.
    ``spherical=True`` extends the algorithm to projected centroid directions.

    ``max_points_per_centroid`` caps the training set as in ``faiss.Kmeans``:
    when the input has more than ``max_points_per_centroid * k`` rows,
    ``train`` uses a random subset of that many rows (drawn from ``seed``) and
    returns label -1 for the other rows. The default None trains on all rows.

    ``nlist`` turns on e-means-ivf: during training, each row's nearest and
    second-nearest centroids are searched only in ``nprobe`` of ``nlist``
    cells of a faiss IndexIVFFlat over the current centroids, which is much
    faster with many centroids and approximate. ``nlist="auto"`` and the
    default ``nprobe="auto"`` size the index from k (``recipe["ivf_rule"]``
    records the rule); the default ``nlist=None`` keeps the exact search.
    Rows whose probed cells hold no centroid are searched exactly
    (``ivf_stats["exact_rows"]``); ``assign`` stays exact. Not available with
    ``spherical=True``.

    Input arrays are converted to contiguous FP32. Model coordinates,
    statistics and closing centroids use FP32; objectives use FP64.
    Use Emeans or QuantizedEmeans for tensor input.
    """

    _backend = "faiss"

    def __init__(
        self,
        d,
        k,
        nepochs=5,
        batch_size=None,
        alpha0=None,
        batch_law="auto",
        seed=1234,
        verbose=False,
        assignment_chunk_size=None,
        capture_assignments=True,
        spherical=False,
        *,
        unused_threshold=None,
        max_points_per_centroid=None,
        closing_handover=True,
        nlist=None,
        nprobe="auto",
    ):
        self.d = int(d)
        self.k = int(k)
        self._set_ivf_assignment(nlist, nprobe, spherical)
        self._set_max_points_per_centroid(max_points_per_centroid)
        self.closing_handover = bool(closing_handover)
        nepochs = float(nepochs)
        self.nepochs = int(nepochs) if nepochs.is_integer() else nepochs
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.batch_law = batch_law
        self.unused_threshold = unused_threshold
        self._unused_threshold = _resolve_unused_threshold(
            unused_threshold, self.d
        )
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.assignment_chunk_size = (
            None
            if assignment_chunk_size is None
            else int(assignment_chunk_size)
        )
        self.capture_assignments = bool(capture_assignments)
        # Closing-pass relabelling of rows held by revived centroids (train() only).
        self._handover = None
        self.closing_handover_stats = None
        self.spherical = bool(spherical)
        self.centroids = None
        self.assignments = None
        self.obj = None
        self.iteration_stats = None
        self.recipe = None
        self.n_splits = 0
        self._usage = None
        self._centroids = None
        self._removal_stats = self._axes = None
        self._pending_stats = None
        self._failed = False
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None
        self._batch_stats = None
        self._pass_counts = self._pass_sums = self._objective = None
        self._split_rng = None
        self._unit_centroids = self._unit_version = None
        self._unit_generation = 0
        self._pool = self._pool_threads = None

    def _parallel(self, fn, n, width):
        """Call ``fn(begin, end)`` on row blocks covering range(n), on up to
        faiss.omp_get_max_threads() threads.

        ``width`` is the number of elements per row: each block has at least
        _BLOCK_ELEMENTS elements, so small passes run in one call on the
        calling thread. Calls must write separate rows; NumPy releases the
        GIL inside its loops. Every row is computed on its own, so results do
        not depend on the blocks.
        """
        blocks = min(faiss.omp_get_max_threads(), n * width // _BLOCK_ELEMENTS)
        if blocks <= 1:
            fn(0, n)
            return
        if self._pool is None or self._pool_threads < blocks:
            self._pool, self._pool_threads = ThreadPoolExecutor(blocks), blocks
        rows = -(-n // blocks)
        for job in [self._pool.submit(fn, begin, min(begin + rows, n))
                    for begin in range(0, n, rows)]:
            job.result()

    def _take_rows(self, x, index):
        """Copy x[index] with threads; the row ids must be in range."""
        if type(x) is not np.ndarray:
            return x[index]    # other row containers: their own indexing, same rows
        out = np.empty((len(index),) + x.shape[1:], dtype=x.dtype)
        # mode="clip" never clips in-range ids; "raise" would buffer out.
        self._parallel(lambda begin, end: np.take(
            x, index[begin:end], axis=0, out=out[begin:end], mode="clip"), len(index), self.d)
        return out

    def _set_ivf_assignment(self, nlist, nprobe, spherical):
        """Resolve the IVF assignment options (class docstring)."""
        self._ivf = None
        self._ivf_moved = []
        self.ivf_stats = None
        self._ivf_requested = None if nlist is None else (nlist, nprobe)
        self.nlist, self.nprobe, self._ivf_rule = _resolve_ivf_size(self.k, nlist, nprobe)
        if self._ivf_requested is None:
            return
        if spherical:
            raise ValueError("IVF assignment does not support spherical=True")
        if not all(_same_operation(getattr(self, name), getattr(NumpyEmeans, name), self)
                   for name in ("_assign_rows", "_accumulate", "train")):
            raise ValueError("nlist is a NumpyEmeans option; %s replaces its assignment or "
                             "training loop" % type(self).__name__)

    def _invalidate_readout(self):
        super()._invalidate_readout()
        if self._ivf is not None:
            self._ivf_stale = True

    def _describe_assignment(self):
        if self._ivf_requested is not None and self.nlist is None:
            return "faiss on CPU, %s" % self._ivf_rule
        return super()._describe_assignment()

    def _readout(self, force=False):
        """Return effective FP32 centroids for assignment and inspection.

        "Readout" means converting the internal EMA positions into the
        coordinates used by assignment and ``current_centroids``. Euclidean
        clustering uses the stored positions directly. Spherical clustering
        uses their unit directions, keeping the EMA positions unchanged;
        zero vectors remain zero.

        Directions are cached and refreshed after model updates or when
        ``force=True``. The returned array borrows model storage; copy it to
        retain an independent snapshot.
        """
        if not self.spherical:
            return self._centroids
        from .e_means_spherical import normalize_into

        version = None
        token = (self._unit_generation, version)
        if self._unit_centroids is None:
            self._unit_centroids = np.empty_like(self._centroids)
        if force or self._unit_version != token:
            normalize_into(self._centroids, self._unit_centroids)
            self._unit_version = token
        return self._unit_centroids

    def current_centroids(self):
        """Borrow a read-only view of the current FP32 coordinates.

        Acquire a new view after update, end_epoch or finalize.
        Copy the view for an independent snapshot. Spherical mode borrows
        normalized directions from a reusable cache; zero stays zero.
        """
        if self._failed:
            raise RuntimeError(
                "create a new object after execution failure"
            )
        view = self._readout().view()
        view.flags.writeable = False
        return view

    def _get_assignment_chunk_size(self):
        return self.assignment_chunk_size or 65536

    def _choose_splits(self, rng):
        if self._unused_threshold is None:
            stats, d = self._removal_stats, self.d
            norm = np.empty(self.k, dtype="float64")

            def axis_norms(begin, end):
                norm[begin:end] = np.linalg.norm(stats[begin:end, :d].astype("float64"), axis=1)

            self._parallel(axis_norms, self.k, d)
            return self._choose_removal_splits(
                stats[:, d].astype("float64"), self._usage.astype("float64"),
                self._axis_scale * norm)
        # Sample donors without replacement in proportion to their usage.
        threshold = _numpy_unused_threshold(
            self._usage, self.k, self._unused_threshold
        )
        unused = self._usage < threshold
        donors = (~unused) & (self._usage > 0)
        nsplit = min(int(unused.sum()), int(donors.sum()))
        if nsplit == 0:
            return None, None

        uniform = np.maximum(
            rng.random_sample(self.k), np.finfo("float64").tiny
        )
        keys = np.full(self.k, -np.inf)
        keys[donors] = np.log(self._usage[donors]) - np.log(
            -np.log(uniform[donors])
        )
        source = np.argsort(keys)[-nsplit:][::-1]
        target = np.flatnonzero(unused)[:nsplit]

        return source, target

    def _apply_splits(self, source, target):
        # c_j -/+ s_j u_j, s_j = sqrt(2 ||U_j|| / (pi N_j)) with N_j before
        # halving, in FP64 rounded once; s_j = 0 (coincident copies, separated
        # below) for a zero or non-finite ||U_j|| or N_j, or without U.
        centroids = self._centroids[source].astype("float64")
        offset = 0.0
        if self._removal_stats is not None:
            norm = self._axis_scale * np.linalg.norm(self._removal_stats[source, :-1].astype("float64"), axis=1)
            mass = self._usage[source].astype("float64")
            valid = (norm > 0) & (mass > 0) & np.isfinite(norm) & np.isfinite(mass)
            spread = np.sqrt(2.0 * np.where(valid, norm / np.where(valid, mass, 1.0), 0.0) / np.pi)
            offset = spread[:, None] * self._axes[source].astype("float64")
        donor_centroids = (centroids - offset).astype("float32")
        child_centroids = (centroids + offset).astype("float32")
        _separate_splits(donor_centroids, child_centroids)
        half_usage = self._usage[source] / 2.0

        self._usage[source] = half_usage
        self._centroids[source] = donor_centroids
        self._usage[target] = half_usage
        self._centroids[target] = child_centroids
        if self._removal_stats is not None:
            self._removal_stats[source] /= 2
            self._removal_stats[target] = self._removal_stats[source]
            self._axes[target] = self._axes[source]
        if self._pass_sums is not None and self._unused_threshold is None and self._handover is None:
            # Cost-based revival in the last pass, no handover: neither end keeps rows from before the split.
            for v in (self._pass_sums, self._pass_counts):
                v[source] = v[target] = 0
        if self._handover is not None:
            # O(k): when each revived centroid moved, in closing-pass rows.
            self._handover["moved_at"][np.asarray(target)] = self._handover["done"]
            self._handover["events"].append((np.asarray(target), np.asarray(source), self._handover["done"]))
        if self._ivf is not None:
            self._ivf_moved += [np.asarray(source), np.asarray(target)]
        self.n_splits += len(source)
        self._invalidate_readout()
        return len(source)

    def _handover_start(self, labels, second, x):
        """Begin the closing pass's handover: ``labels`` (n,) are the captured
        assignments, ``second`` (n,) each row's second-nearest index from the
        same assignment (-1 if unknown). A revival only records, per revived
        centroid, how many closing-pass rows were assigned when it moved
        (``moved_at``, O(k)); the rows are sorted out once, at the end."""
        self._handover = dict(labels=labels, second=second, x=x, done=0, events=[],
                              moved_at=np.full(self.k, -1, dtype="int64"))

    def _handover_finish(self, order, rows):
        """One pass over the ``rows`` closing-pass rows in visiting ``order``: a
        row assigned before its centroid moved takes its second-nearest unless
        that centroid moved after the row's assignment too; those rows are
        assigned again like a training batch (against the current centroids),
        so this costs their share of one epoch. ``_close_split_centroids`` then
        rebuilds the closing sums."""
        h, self._handover = self._handover, None
        labels, second, x, moved_at = h["labels"], h["second"], h["x"], h["moved_at"]
        handed, search = [], []
        if not (moved_at >= 0).any():  # O(k): nothing moved in this pass.
            rows = 0
        for begin in range(0, rows, _HANDOVER_CHUNK):
            index = order[begin:min(begin + _HANDOVER_CHUNK, rows)]
            rank = np.arange(begin, begin + len(index))
            stale = rank < moved_at[labels[index]]
            if not stale.any():
                continue
            nearest2 = second[index].astype("int64")
            usable = stale & (nearest2 >= 0) & (moved_at[np.maximum(nearest2, 0)] <= rank)
            labels[index[usable]] = nearest2[usable]
            handed.append(index[usable])
            search.append(index[stale & ~usable])
        handed = np.concatenate(handed) if handed else order[:0]
        search = np.concatenate(search) if search else order[:0]
        if len(search):
            centroids = self._readout()
            chunk_size = self.recipe["assignment_chunk_size"]
            for begin in range(0, len(search), chunk_size):
                chunk = search[begin:begin + chunk_size]
                _, labels[chunk] = self._assign_rows(np.ascontiguousarray(x[chunk]), centroids)
        self._close_split_centroids(order, rows, labels, x, moved_at, h["events"],
                                    np.concatenate((handed, search)))
        self.closing_handover_stats = dict(second=len(handed), searched=len(search))

    def _close_split_centroids(self, order, rows, labels, x, moved_at, events, stale_rows):
        """Twin reassignment and closing sums, as ``Emeans._close_split_centroids``."""
        if not rows:
            return
        target, source, born = (
            np.concatenate(v) for v in zip(
                *((t, s, np.full(len(t), b)) for t, s, b in [([0], [self.k], 0)] + events)
            )
        )  # Donor k: no row.
        span, order = rows + 1, order[:rows]
        by_donor = np.argsort(source * span + born, kind="stable")
        target, source, born = target[by_donor], source[by_donor], born[by_donor]
        stale, rebuild = np.zeros(len(labels), dtype=bool), np.zeros(self.k + 1, dtype=bool)
        stale[stale_rows] = True
        _twin_reassign(order, labels, lambda i: x[i], self._readout(), moved_at, source * span + born,
                       target, source, born, stale, span,
                       max(1, min(_HANDOVER_CHUNK, _TWIN_STEP_ELEMENTS // self.d)),
                       lambda fn, n: self._parallel(fn, n, self.d))
        rebuild[source] = rebuild[np.where(source < self.k, target, self.k)] = rebuild[labels[stale_rows]] = True
        rebuild = rebuild[:self.k]
        self._pass_sums[rebuild] = 0
        self._pass_counts[rebuild] = 0
        index = order[rebuild[labels[order]]]
        # The rebuilt sums start from zero and add their rows in visiting
        # order, as np.add.at would, in one table for faiss: [sums | count].
        local = np.cumsum(rebuild) - 1
        table = np.zeros((local[-1] + 1, self.d + 1), dtype="float32")
        chunk = self.recipe["assignment_chunk_size"]
        for begin in range(0, len(index), chunk):
            part = index[begin:begin + chunk]
            # Keep the operands referenced: swig_ptr does not own them.
            rows, ids = self._take_rows(x, part), np.ascontiguousarray(local[labels[part]])
            faiss.accumulate_cluster_sums(self.d, len(part), len(table), *map(
                faiss.swig_ptr, (rows, ids, table)))
            self._pass_counts += np.bincount(labels[part], minlength=self.k).astype(self._pass_counts.dtype)
        self._pass_sums[rebuild] = table[:, :-1]

    def _assign_rows(self, x, centroids, second_out=None, second_labels_out=None):
        nearest = 1 if second_out is None else 2
        if self._ivf is not None and centroids is self._centroids:
            distance, labels = self._search_ivf(x, nearest)
        elif self.spherical:
            similarity, labels = faiss.knn(
                x, centroids, nearest, metric=faiss.METRIC_INNER_PRODUCT
            )
            distance = -2 * similarity
        else:
            distance, labels = faiss.knn(x, centroids, nearest)
        if second_out is not None:
            second_out[:] = distance[:, -1]
        # Without a second candidate (IVF), the second is inf and -1.
        tie = distance[:, 0] == distance[:, -1]
        if second_labels_out is not None:
            second_labels_out[:] = np.where(tie, labels.max(axis=1), labels[:, -1])
        # Ties keep the lowest label, as with one neighbor.
        return distance[:, 0], np.where(tie, labels.min(axis=1), labels[:, 0])

    def _search_ivf(self, x, nearest):
        """Training search through the IVF lists, refilled first if the
        centroids changed. Rows that find no centroid are searched exactly."""
        if self._ivf_stale:
            moved = np.unique(np.concatenate(self._ivf_moved)) if self._ivf_moved else []
            self._ivf.refill(self._centroids, moved)
            self._ivf_moved, self._ivf_stale = [], False
        distance, labels = self._ivf.search(x, nearest)
        missing = np.flatnonzero(labels[:, 0] < 0)
        if len(missing):
            distance[missing], labels[missing] = faiss.knn(
                np.ascontiguousarray(x[missing]), self._centroids, nearest)
            self.ivf_stats["exact_rows"] += len(missing)
        return distance, labels

    def begin(self, global_n, init_centroids):
        """Copy initial centroids and start the caller-driven loop.

        Call once on a fresh object.
        Initial centroids have shape (k, d). Each epoch supplies its recipe's
        row count in logical batches, including the final remainder, then
        calls end_epoch. The caller owns loading and ordering rows.
        """
        if self._failed or self.recipe is not None:
            raise RuntimeError("create a new object for another run or after failure")
        n = int(global_n)
        centroids = np.ascontiguousarray(init_centroids, dtype="float32")
        self._failed = True
        self._centroids = self._initial_centroid_state(centroids)
        self._usage = np.full(self.k, _INIT_USAGE, dtype="float32")
        self._batch_stats = np.empty((self.k, self.d + 1), dtype="float32")
        self._split_rng = np.random.RandomState(self.seed)
        self._resolve_removal_revival()
        if self._keeps_removal_stats():
            # Rows are the EMAs [U / _axis_scale | R] and the batch's
            # [sum p y | sum d2 - d1].
            self._removal_stats = np.zeros((self.k, self.d + 1), dtype="float32")
            self._axis_scale = 1.0
            self._batch_removal = np.zeros((self.k, self.d + 1), dtype="float32")
            self._axes = self._initial_removal_axes()
        self._resolve_run_recipe(n)
        if self.spherical:
            self.recipe.update(
                spherical=True,
                assignment_score="represented_inner_product",
                objective="sum(2-2*represented_inner_product)",
            )
        if self._ivf_requested is not None:
            self.recipe.update(
                nlist_requested=self._ivf_requested[0],
                nprobe_requested=self._ivf_requested[1],
                ivf_rule=self._ivf_rule,
            )
        if self.nlist is not None:
            self.recipe.update(index_factory="IVF%d,Flat" % self.nlist, nprobe=self.nprobe,
                               ivf_reassign_every=_IVF_REASSIGN_EVERY,
                               ivf_coarse_niter=_IVF_COARSE_NITER)
            self._ivf = _CentroidIVF(self._centroids, self.nlist, self.nprobe, self.seed)
            self._ivf_stale = True
            self.ivf_stats = dict(exact_rows=0)
        self._n = n
        self._step = self._epoch = 0
        self.iteration_stats = []
        self._start_epoch()
        self._start_time = time.perf_counter()
        self._failed = False
        self._header_due, self._quiet = True, False

    def _initial_centroid_state(self, centroids):
        state = centroids.copy()
        if self.spherical:
            from .e_means_spherical import normalize_into

            normalize_into(state, state)
        return state

    def _initial_removal_axes(self):
        axes = self._split_rng.standard_normal((self.k, self.d))
        return (axes / np.linalg.norm(axes, axis=1, keepdims=True)).astype("float32")

    def _start_epoch(self):
        self._epoch_steps = self._epoch_rows = 0
        self._splits_before = self.n_splits
        self._pass_counts = np.zeros(self.k, dtype="float64")
        self._objective = np.float64(0)
        self._pass_sums = (
            np.zeros_like(self._centroids)
            if self._epoch == len(self.recipe["epoch_rows"]) - 1
            else None
        )

    def accumulate(self, x):
        """Accumulate rows against one fixed centroid snapshot.

        Complete statistics contain FP32 sums/counts, an FP64 objective and
        the supplied row count. Supply exactly the recipe's logical batch
        size before update. Partial calls return sums/counts as None.
        Complete totals borrow workspace; copy them before the next call
        if they must be retained. Final-pass labels, when enabled, are owned
        by the caller and describe only these supplied rows.
        """
        self._require_running()
        try:
            x = np.ascontiguousarray(x, dtype="float32")
        except BaseException:
            self._failed = True
            raise
        return self._accumulate(x)

    def _accumulate(self, x, index=None, labels_out=None, second_labels_out=None):
        # Indexed matrix training gathers only assignment chunks, never B*d.
        n = len(x) if index is None else len(index)
        if n == 0 and self._pending_stats is not None:
            stats = dict(self._pending_stats)
            if (
                self.capture_assignments
                and self._epoch == len(self.recipe["epoch_rows"]) - 1
            ):
                stats["labels"] = np.empty(0, dtype="int64")
            return stats
        try:
            # Only plain models hand rows over; subclasses keep their signature.
            stats = self._accumulate_core(
                x, index, labels_out=labels_out,
                **({} if second_labels_out is None else {"second_labels_out": second_labels_out}))
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        self._pending_stats = stats
        return stats

    def _accumulate_core(self, x, index=None, labels_out=None, second_labels_out=None):
        n = len(x) if index is None else len(index)
        pending_n = (
            0 if self._pending_stats is None else self._pending_stats["n"]
        )
        target = self._local_batch_size()
        wide = n > 0 and self._logical_batch_size() > _FLOAT32_COUNT_LIMIT
        if pending_n == 0:
            # Bound the whole contribution, not each supplied/assignment chunk.
            self._parallel(lambda begin, end: self._batch_stats[begin:end].fill(0),
                           self.k, self.d + 1)
            accumulation = (
                np.zeros(self._batch_stats.shape, dtype="float64")
                if wide
                else self._batch_stats
            )
            self._batch_objective = np.float64(0)
            self._batch_centroids = self._readout()
            self._batch_accumulation = accumulation
        accumulation = self._batch_accumulation
        objective = self._batch_objective
        centroids = self._batch_centroids
        sums, counts = accumulation[:, :-1], accumulation[:, -1]
        captured_labels = None if labels_out is None else labels_out[:n]
        if (
            captured_labels is None
            and self.capture_assignments
            and self._epoch == len(self.recipe["epoch_rows"]) - 1
        ):
            captured_labels = np.empty(n, dtype="int64")
        chunk_size = self.recipe["assignment_chunk_size"]
        for begin in range(0, n, chunk_size):
            end = min(begin + chunk_size, n)
            rows = x[begin:end] if index is None else self._take_rows(x, index[begin:end])
            # Mass-based revival reads only U: no second-nearest search, and R stays 0.
            track = self._removal_stats is not None
            second = (np.empty(len(rows), dtype="float32")
                      if track and self._unused_threshold is None else None)
            second_labels = None
            if second_labels_out is not None:
                second_labels = second_labels_out[begin:end]
                if second is None:
                    second_labels[:] = -1  # Mass-based revival: no handover.
                    second_labels = None
            distance, labels = self._assign_rows(rows, centroids, second, second_labels)
            if track:
                # Keep the contiguous copy referenced: swig_ptr does not own it.
                nearest = np.ascontiguousarray(distance)
                faiss.accumulate_removal_stats(
                    self.d, len(rows), self.k, *map(faiss.swig_ptr, (
                        rows, labels, nearest, nearest if second is None else second,
                        centroids, self._axes, self._batch_removal)))
            if captured_labels is not None:
                captured_labels[begin:end] = labels
            if self.spherical:
                objective += 2 * len(rows)
            objective += distance.sum(dtype="float64")
            if accumulation.dtype == np.float32:
                if len(labels) and labels.min() < 0:
                    raise ValueError("faiss found no nearest centroid for some rows")
                faiss.accumulate_cluster_sums(self.d, len(rows), self.k, *map(
                    faiss.swig_ptr, (rows, labels, accumulation)))
            else:
                counts += np.bincount(labels, minlength=self.k)
                _add_rows_at(sums, labels, rows)
        self._batch_objective = objective
        total_n = pending_n + n
        if total_n == target and accumulation is not self._batch_stats:
            accumulation = self._complete_statistics(accumulation)
            sums, counts = accumulation[:, :-1], accumulation[:, -1]
        complete = total_n == target or total_n == 0
        stats = {
            "sums": sums if complete else None,
            "counts": counts if complete else None,
            "objective": objective,
            "n": total_n,
        }
        if self._removal_stats is not None:
            stats["removal"] = self._batch_removal
        if captured_labels is not None:
            stats["labels"] = captured_labels
        return stats

    def _prepare_statistics(self, stats, expected_n):
        if "stat_schema" in stats:
            raise ValueError("stat_schema is only supported by Tensor statistics")
        return None, stats["sums"], stats["counts"], stats["objective"]

    def _apply_statistics(
        self, ids, sums, counts, objective, alpha, decay, replace_usage=False
    ):
        """Apply Algorithm 1, line 8 (Equation 2), to prepared batch statistics.

        In the paper's notation, ``sums`` is s_j, ``counts`` is n_j,
        ``decay`` is gamma and ``alpha`` is 1 - gamma. For occupied clusters,
        the sum-and-count EMA is expressed using mass N_j and centroid c_j:

            N_new = decay * N_j + alpha * n_j
            c_new = c_j + (alpha * n_j / N_new) * (s_j / n_j - c_j)

        ``_usage`` stores N_j and ``_centroids`` stores c_j = S_j / N_j.
        Empty clusters keep their positions. ``replace_usage`` bypasses the
        previous mass for occupied clusters when the step is one. ``ids``
        maps sparse statistics to centroid rows; None denotes the full table.

        Preserve batch counts and sums in epoch totals before the EMA for
        Algorithm 1's last-epoch finalization (line 12), and add the batch
        objective for reporting. The caller handles revival afterward (line 9).
        See the class docstring for the paper reference.
        """
        self._objective += objective
        if self._removal_stats is not None:
            # U decays as one scalar, _axis_scale; touched rows add sum p y
            # and renew V.
            stats, d = self._removal_stats, self.d
            stats[:, d] = decay * stats[:, d] + alpha * self._batch_removal[:, d]
            scale = self._axis_scale * decay
            if not scale > 2.0**-32:
                stats[:, :d] *= scale
                scale = 1.0
            self._axis_scale = scale
            touched = np.flatnonzero(counts)
            if ids is not None:
                touched = ids[touched]
            faiss.update_removal_axes(
                d, len(touched), faiss.swig_ptr(touched), alpha / scale, *map(faiss.swig_ptr, (
                    self._batch_removal, stats, self._axes)))
        if ids is None:
            self._pass_counts += counts
            if self._pass_sums is not None:
                pass_sums = self._pass_sums
                self._parallel(lambda begin, end: np.add(
                    pass_sums[begin:end], sums[begin:end], out=pass_sums[begin:end]),
                    self.k, self.d)
        else:
            np.add.at(self._pass_counts, ids, counts)
            if self._pass_sums is not None:
                np.add.at(self._pass_sums, ids, sums)
        if not replace_usage:
            self._usage *= decay
        if ids is None:
            # Row blocks of the model's own arrays, updated in place.
            self._parallel(lambda begin, end: self._move_centroids(
                self._centroids[begin:end], self._usage[begin:end], sums[begin:end],
                counts[begin:end], alpha, replace_usage), self.k, self.d)
        else:
            centroids, usage = self._centroids[ids], self._usage[ids]
            self._move_centroids(centroids, usage, sums, counts, alpha, replace_usage)
            touched = counts > 0
            self._centroids[ids[touched]] = centroids[touched]
            self._usage[ids[touched]] = usage[touched]
        self._invalidate_readout()

    @staticmethod
    def _move_centroids(centroids, usage, sums, counts, alpha, replace_usage):
        """Equation 2 on the rows with counts, in place on centroids and usage
        (the model's c_j and N_j for these rows, after decay)."""
        touched = counts > 0
        retained = (
            usage[touched]
            if not replace_usage
            else np.zeros_like(counts[touched])
        )
        mass = retained + alpha * counts[touched]
        mean = sums[touched] / counts[touched, None]
        weight = alpha * counts[touched] / mass
        position = centroids[touched] + weight[:, None] * (
            mean - centroids[touched]
        )
        position[weight == 1] = mean[weight == 1]
        centroids[touched] = position
        usage[touched] = mass

    def _epoch_totals(self):
        return float(self._epoch_objective()), float(
            np.dot(self._pass_counts, self._pass_counts)
        )

    def _spherical_closing(self):
        from .e_means_spherical import normalize_into

        result = np.empty_like(self._centroids)
        # Bound wide means separately from projection scratch; normalize before
        # F32.
        rows = max(1, (1 << 25) // (8 * self.d))
        for begin in range(0, self.k, rows):
            end = min(begin + rows, self.k)
            sums, counts = (
                self._pass_sums[begin:end],
                self._pass_counts[begin:end],
            )
            raw = self._centroids[begin:end]
            mean = sums / np.maximum(counts, 1)[:, None].astype(sums.dtype)
            mean = np.where((counts > 0)[:, None], mean, raw)
            normalize_into(mean, result[begin:end])
        return result

    def finalize(self):
        """Return closing means after the final epoch.

        Centroids without rows in that epoch retain their positions. Repeated
        calls return the same result.
        """
        if self._failed:
            raise RuntimeError(
                "create a new object after execution failure"
            )
        if self.centroids is not None:
            return self.centroids
        self._failed = True
        nonempty = self._pass_counts > 0
        centroids = (
            self._centroids if self.spherical else self._centroids.copy()
        )
        objectives = [stat["obj"] for stat in self.iteration_stats]
        if self.spherical:
            self.centroids = self._spherical_closing()
        else:
            centroids[nonempty] = self._pass_sums[nonempty] / self._pass_counts[
                nonempty, None
            ].astype("float32")
            self.centroids = np.ascontiguousarray(centroids, dtype="float32")
        self.obj = np.asarray(objectives, dtype="float64")
        self._centroids = self._usage = self._batch_stats = None
        self._removal_stats = self._axes = self._batch_removal = None
        self._unit_centroids = self._unit_version = None
        self._pass_sums = self._pass_counts = self._objective = None
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None
        self._pool = self._pool_threads = None
        if self._ivf is not None:
            self.ivf_stats.update(refills=self._ivf.refills,
                                  reassignments=self._ivf.reassignments)
        self._ivf = None
        self._failed = False
        return self.centroids

    def train(self, x, init_centroids=None):
        """Train on x, or on a random subset when max_points_per_centroid binds.

        Return centroids and last-pass labels for every row of x (None with
        capture disabled); rows left out of the subset get label -1. Only the
        subset is read from x, so a memory-mapped array is not loaded whole.
        """
        keep = self._training_subset(len(x))
        if keep is None:
            return self._train_rows(x, init_centroids)
        _, sampled = self._train_rows(x[keep], init_centroids)
        if sampled is not None:
            try:
                labels = np.full(len(x), -1, dtype=sampled.dtype)
                labels[keep] = sampled
            except BaseException:
                self._failed = True
                self.assignments = None
                raise
            self.assignments = labels
        return self.centroids, self.assignments

    def _train_rows(self, x, init_centroids=None):
        """Train an FP32 matrix, returning centroids and captured assignments."""
        start, given_init = time.perf_counter(), init_centroids is not None
        x = np.ascontiguousarray(x, dtype="float32")
        n = len(x)
        if init_centroids is None:
            init_centroids = (
                x
                if n == self.k
                else x[
                    np.random.RandomState(self.seed).permutation(n)[: self.k]
                ]
            )
        self.begin(n, init_centroids)
        self._report_header_once()
        if self._reports():
            self._report_preprocessing(start, given_init, n)
        assignments = second_labels = None
        if self.capture_assignments:
            assignments = np.full(n, -1, dtype="int64")
            if (self._unused_threshold is None and self.closing_handover
                    and type(self)._accumulate_core is NumpyEmeans._accumulate_core):
                # Cost-based revival can move a centroid during the closing
                # pass; captured rows it held then take their second-nearest.
                second_labels = np.full(n, -1, dtype=_compact_label_dtype(self.k))
        self.recipe["closing_handover"] = (
            "second_nearest" if second_labels is not None else None)
        try:
            for epoch, rows in enumerate(self.recipe["epoch_rows"]):
                if second_labels is not None and epoch == len(self.recipe["epoch_rows"]) - 1:
                    self._handover_start(assignments, second_labels, x)
                order = np.random.RandomState(
                    _epoch_seed(self.seed, epoch)
                ).permutation(n)
                for begin in range(0, rows, self.recipe["batch_size"]):
                    batch = order[
                        begin : min(begin + self.recipe["batch_size"], rows)
                    ]
                    if (
                        assignments is not None
                        and epoch == len(self.recipe["epoch_rows"]) - 1
                    ):
                        chunk_size = self.recipe["assignment_chunk_size"]
                        for offset in range(0, len(batch), chunk_size):
                            chunk = batch[offset : offset + chunk_size]
                            second = (None if second_labels is None
                                      else np.empty(len(chunk), dtype=second_labels.dtype))
                            stats = self._accumulate(x, chunk, None, second)
                            assignments[chunk] = stats.pop("labels")
                            if second is not None:
                                second_labels[chunk] = second
                        if self._handover is not None:
                            self._handover["done"] += len(batch)
                        self.update(stats)
                    else:
                        self.update(self._accumulate(x, batch))
                self.end_epoch()
            if self._handover is not None:
                self._handover_finish(order, self.recipe["epoch_rows"][-1])
            self.finalize()
        except BaseException:
            self._failed = True
            self.assignments = None
            self._handover = None
            raise
        self.assignments = assignments
        return self.centroids, self.assignments

    def assign(self, x):
        """Return squared L2 distances and nearest centroid IDs.

        With spherical=True, return inner-product similarities instead.
        Input rows must already be normalized for spherical assignment.
        """
        if self._failed:
            raise RuntimeError(
                "create a new object after execution failure"
            )
        x = np.ascontiguousarray(x, dtype="float32")
        distance = np.empty(len(x), dtype="float32")
        labels = np.empty(len(x), dtype="int64")
        chunk_size = self._get_assignment_chunk_size()
        for begin in range(0, len(x), chunk_size):
            end = min(begin + chunk_size, len(x))
            distance[begin:end], labels[begin:end] = self._assign_rows(
                x[begin:end], self.centroids
            )
        if self.spherical:
            distance *= -0.5
        return distance, labels
