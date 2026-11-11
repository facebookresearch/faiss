# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""FP32 E-means clustering with NumPy and Faiss assignment."""

import time

import numpy as np
import faiss

from .base import (
    _EmeansBase,
    _INIT_USAGE,
    _SPLIT_EPSILON,
    _UNUSED_THRESHOLD,
    _epoch_rows,
    _epoch_seed,
    _format_passes as _format_passes,
    _resolve_recipe,
    _resolve_unused_threshold,
    _round_power_of_two as _round_power_of_two,
    _scheduled_alpha as _scheduled_alpha,
    _step_law,
)


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
    updates,
    clusters below ``unused_threshold`` times the mean mass are revived by
    splitting donors sampled in proportion to their masses (line 9 and
    Appendix J). The default ``unused_threshold=None`` is 0.1, or 0 (no
    revival) when d <= 2. Pass a value to override it.

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
    ):
        self.d = int(d)
        self.k = int(k)
        self._set_max_points_per_centroid(max_points_per_centroid)
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
        self.spherical = bool(spherical)
        self.centroids = None
        self.assignments = None
        self.obj = None
        self.iteration_stats = None
        self.recipe = None
        self.n_splits = 0
        self._usage = None
        self._centroids = None
        self._pending_stats = None
        self._failed = False
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None
        self._batch_stats = None
        self._pass_counts = self._pass_sums = self._objective = None
        self._split_rng = None
        self._unit_centroids = self._unit_version = None
        self._unit_generation = 0

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
        centroids = self._centroids[source]
        sign = np.ones(self.d, dtype="float32")
        sign[1::2] = -1.0
        offset = _SPLIT_EPSILON * sign
        donor_centroids = centroids * (1.0 - offset)
        child_centroids = centroids * (1.0 + offset)
        _separate_splits(donor_centroids, child_centroids)
        half_usage = self._usage[source] / 2.0

        self._usage[source] = half_usage
        self._centroids[source] = donor_centroids
        self._usage[target] = half_usage
        self._centroids[target] = child_centroids
        self.n_splits += len(source)
        self._invalidate_readout()
        return len(source)

    def _assign_rows(self, x, centroids):
        if self.spherical:
            similarity, labels = faiss.knn(
                x, centroids, 1, metric=faiss.METRIC_INNER_PRODUCT
            )
            return -2 * similarity.ravel(), labels.ravel()
        distance, labels = faiss.knn(x, centroids, 1)
        return distance.ravel(), labels.ravel()

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
        self._centroids = centroids.copy()
        if self.spherical:
            from .e_means_spherical import normalize_into

            normalize_into(self._centroids, self._centroids)
        self._usage = np.full(self.k, _INIT_USAGE, dtype="float32")
        self._batch_stats = np.empty((self.k, self.d + 1), dtype="float32")
        self._split_rng = np.random.RandomState(self.seed)
        self._resolve_run_recipe(n)
        if self.spherical:
            self.recipe.update(
                spherical=True,
                assignment_score="represented_inner_product",
                objective="sum(2-2*represented_inner_product)",
            )
        self._n = n
        self._step = self._epoch = 0
        self.iteration_stats = []
        self._start_epoch()
        self._start_time = time.perf_counter()
        self._failed = False
        self._header_due, self._quiet = True, False

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

    def accumulate(self, x, *, labels_out=None):
        """Accumulate rows against one fixed centroid snapshot.

        Complete statistics contain FP32 sums/counts, an FP64 objective and
        the supplied row count. Supply exactly the recipe's logical batch
        size before update. Partial calls return sums/counts as None.
        Complete totals borrow workspace; copy them before the next call
        if they must be retained.

        Optionally write this call's assignments into caller-owned
        `labels_out`: a writable, contiguous int64 array of length len(x).
        The returned stats also borrow its labels. Each call describes only
        the supplied rows, even when it is part of a larger logical batch.
        No extra nearest-centroid search is performed. Omitting labels_out
        preserves the default last-pass capture policy.
        """
        self._require_running()
        try:
            source = x
            x = np.ascontiguousarray(x, dtype="float32")
        except BaseException:
            self._failed = True
            raise
        if labels_out is not None:
            if (
                not isinstance(labels_out, np.ndarray)
                or labels_out.dtype != np.dtype("int64")
                or labels_out.shape != (len(x),)
                or not labels_out.flags.c_contiguous
                or not labels_out.flags.writeable
            ):
                raise ValueError(
                    "labels_out must be a writable, contiguous int64 "
                    "array with one entry per supplied row"
                )
            if np.may_share_memory(labels_out, np.asarray(source)):
                raise ValueError("labels_out must not alias input rows")
        return self._accumulate(x, labels_out=labels_out)

    def _accumulate(self, x, index=None, labels_out=None):
        # Indexed matrix training gathers only assignment chunks, never B*d.
        n = len(x) if index is None else len(index)
        if n == 0 and self._pending_stats is not None:
            stats = dict(self._pending_stats)
            if labels_out is not None:
                stats["labels"] = labels_out[:0]
            elif (
                self.capture_assignments
                and self._epoch == len(self.recipe["epoch_rows"]) - 1
            ):
                stats["labels"] = np.empty(0, dtype="int64")
            else:
                stats.pop("labels", None)
            return stats
        try:
            stats = self._accumulate_core(x, index, labels_out=labels_out)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        self._pending_stats = stats
        return stats

    def _accumulate_core(self, x, index=None, labels_out=None):
        n = len(x) if index is None else len(index)
        pending_n = (
            0 if self._pending_stats is None else self._pending_stats["n"]
        )
        target = self._local_batch_size()
        wide = n > 0 and target > 2**24
        if pending_n == 0:
            # Bound the whole contribution, not each supplied/assignment chunk.
            self._batch_stats.fill(0)
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
            rows = x[begin:end] if index is None else x[index[begin:end]]
            distance, labels = self._assign_rows(rows, centroids)
            if captured_labels is not None:
                captured_labels[begin:end] = labels
            if self.spherical:
                objective += 2 * len(rows)
            objective += distance.sum(dtype="float64")
            counts += np.bincount(labels, minlength=self.k)
            _add_rows_at(sums, labels, rows)
        self._batch_objective = objective
        total_n = pending_n + n
        if total_n == target and accumulation is not self._batch_stats:
            np.copyto(self._batch_stats, accumulation, casting="unsafe")
            sums, counts = self._batch_stats[:, :-1], self._batch_stats[:, -1]
        complete = total_n == target or total_n == 0
        stats = {
            "sums": sums if complete else None,
            "counts": counts if complete else None,
            "objective": objective,
            "n": total_n,
        }
        if captured_labels is not None:
            stats["labels"] = captured_labels
        return stats

    def _prepare_statistics(self, stats, expected_n):
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
        if ids is None:
            self._pass_counts += counts
            if self._pass_sums is not None:
                self._pass_sums += sums
        else:
            np.add.at(self._pass_counts, ids, counts)
            if self._pass_sums is not None:
                np.add.at(self._pass_sums, ids, sums)
        touched = counts > 0
        selected = touched if ids is None else ids[touched]
        if not replace_usage:
            self._usage *= decay
        retained = (
            self._usage[selected]
            if not replace_usage
            else np.zeros_like(counts[touched])
        )
        mass = retained + alpha * counts[touched]
        mean = sums[touched] / counts[touched, None]
        weight = alpha * counts[touched] / mass
        position = self._centroids[selected] + weight[:, None] * (
            mean - self._centroids[selected]
        )
        position[weight == 1] = mean[weight == 1]
        self._centroids[selected] = position
        self._usage[selected] = mass
        self._invalidate_readout()

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

        Never-assigned centroids retain their positions. No new assignment
        is performed. Repeated calls return the same result.
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
        self._unit_centroids = self._unit_version = None
        self._pass_sums = self._pass_counts = self._objective = None
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None
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
        assignments = None
        if self.capture_assignments:
            assignments = np.full(n, -1, dtype="int64")
        try:
            for epoch, rows in enumerate(self.recipe["epoch_rows"]):
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
                            stats = self._accumulate(x, chunk)
                            assignments[chunk] = stats.pop("labels")
                        self.update(stats)
                    else:
                        self.update(self._accumulate(x, batch))
                self.end_epoch()
            self.finalize()
        except BaseException:
            self._failed = True
            self.assignments = None
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
