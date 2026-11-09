# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared recipes and lifecycle for E-means implementations.

Algorithm 1 and equations below refer to "E-Means: Scaling Laws for Mini-Batch
K-Means": https://arxiv.org/abs/XXXX 
"""

import math
import time

# Epsilon in Equation 5: the cosine schedule's minimum step / initial step.
_ALPHA_MIN_FACTOR = 0.1

# Number of logical batch updates between revival checks (Algorithm 1, line 9).
_CHECK_UNUSED_EVERY = 5

# Initial EMA mass N_j (Algorithm 1, line 1); c_j represents S_j / N_j.
# This implementation uses 2**-16; the paper initializes N_j to 1e-10.
_INIT_USAGE = 2.0**-16

# Relative coordinate perturbation delta for donor splits (Appendix J, Eq. 8).
_SPLIT_EPSILON = 1e-4

# Revival threshold tau: revive N_j < tau * mean(N) (Algorithm 1, line 9).
_UNUSED_THRESHOLD = 0.1

# unused_threshold=None (the default) turns revival off at or below this
# dimension and uses _UNUSED_THRESHOLD above it. On 2-D data, reviving dead
# clusters costs held-out error: on the foursquare and geonames corpora,
# switching it off lowered the median held-out error by 7-10% (36 paired
# runs, 3 passes). By d=3 and d=14 it made no difference.
_NO_REVIVAL_MAX_D = 2

# batch_law="auto" (the default) picks the quality law at or below this
# dimension and the throughput law above it. At 3 passes, the throughput
# batch makes 8 times fewer updates. At d <= 16 that cost E-means 11-20
# percentage points of wins against Lloyd's algorithm at 25 iterations, and
# 0-6 points above (30,000 synthetic configurations, n in [1e5, 1e6], K in
# [10, 1e4]).
_AUTO_QUALITY_MAX_D = 16

# Batch law B = C_B * k**b_k * n**b_n * T**b_T (Section 4, Equation 7).
# Tuples contain (C_B, b_k, b_n, b_T), with T = nepochs. Quality uses one
# eighth of the throughput coefficient before power-of-two rounding and batch
# caps.
# These shipped fits used full passes with 100K-10M training rows; their
# coefficient values differ from those reported in the paper's Table 2.
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

# Step law alpha0 = C_alpha * k**a_k * B**a_B * n**a_n * T**a_T (Equation 6).
# Tuple order is (C_alpha, a_k, a_B, a_n, a_T); B is the resolved batch size.
# This shipped fit also included 100M rows and used a zero-ending cosine;
# execution uses the 0.1 floor in _ALPHA_MIN_FACTOR. Its coefficients also
# differ from Table 2. _step_law clamps the result to [1e-4, 1].
_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _round_power_of_two(value):
    """Round a positive value to the nearest power of two."""
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def _step_law(n, k, nepochs, batch_size):
    """Initial EMA step coupled to a batch size by the fitted step law."""
    coefficient, k_exp, batch_exp, n_exp, iter_exp = _ALPHA_LAW
    alpha0 = (
        coefficient
        * k**k_exp
        * batch_size**batch_exp
        * n**n_exp
        * nepochs**iter_exp
    )
    return min(max(alpha0, 1e-4), 1.0)


def _resolve_batch_law(batch_law, d):
    """Resolve ``batch_law="auto"`` to a fitted law, returning (law, reason).

    Any other value is returned unchanged (an unknown name fails in
    ``_resolve_recipe``).
    """
    if batch_law != "auto":
        return batch_law, "requested"
    if d <= _AUTO_QUALITY_MAX_D:
        return "quality", "d <= %d" % _AUTO_QUALITY_MAX_D
    return "throughput", "d > %d" % _AUTO_QUALITY_MAX_D


def _resolve_unused_threshold(unused_threshold, d):
    """Revival threshold tau for a run, picked from ``d`` if ``None``."""
    if unused_threshold is None:
        return 0.0 if d <= _NO_REVIVAL_MAX_D else _UNUSED_THRESHOLD
    unused_threshold = float(unused_threshold)
    if not unused_threshold >= 0:
        raise ValueError("unused_threshold must be >= 0 (0 disables revival)")
    return unused_threshold


def _divisibility_error(recipe, n, batch, processes):
    """The error raised when a multi-process run cannot split its batches.

    Names the batch law that chose the batch (an automatic choice can pick the
    small quality batch at low dimension) and how to make it divisible.
    """
    law = recipe.get("batch_law")
    if law is None:
        source = "the given batch_size"
    elif recipe.get("batch_law_requested") == "auto":
        source = 'batch_law="auto", which chose the %s law because %s' % (
            law, recipe.get("batch_law_reason"))
    else:
        source = 'batch_law="%s"' % law
    return ValueError(
        "the global batch (%d rows, from %s), the global row count (%d) and "
        "every epoch's rows (%s) must be divisible by the %d processes. Pass "
        'batch_law="throughput", or a batch_size divisible by %d.'
        % (batch, source, n, list(recipe["epoch_rows"]), processes, processes))


def _resolve_recipe(n, k, nepochs, batch_law, batch_size, alpha0):
    """Resolve the recipe's statistical batch size B and initial EMA step.

    The batch comes from ``batch_law`` unless ``batch_size`` is given. The
    step comes from the step law at that batch unless ``alpha0`` is given too.
    """
    if batch_size is None:
        coefficient, k_exp, n_exp, iter_exp = _BATCH_LAWS[batch_law]
        raw_batch = coefficient * k**k_exp * n**n_exp * nepochs**iter_exp
        batch_size = _round_power_of_two(raw_batch)
        # These intentionally differ: they reproduce the two fitted recipes.
        if batch_law == "throughput":
            batch_size = min(batch_size, _round_power_of_two(n / 2))
        else:
            batch_size = min(batch_size, n // 2)
        batch_size = max(1, min(batch_size, n))
    else:
        batch_size = min(int(batch_size), n)
    if alpha0 is None:
        alpha0 = _step_law(n, k, nepochs, batch_size)
    return batch_size, float(alpha0)


def _epoch_rows(n, nepochs):
    """Whole passes, a rounded fractional prefix, then moving closing.

    Round the fractional row count with Python's ties-to-even convention.
    Below one pass, only that prefix contributes to closing means.
    """
    whole = math.floor(nepochs)
    remainder = int(round((nepochs - whole) * n))
    if whole == 0:
        return [remainder]
    return [n] * (whole - 1) + ([remainder] if remainder else []) + [n]


def _epoch_seed(seed, iteration):
    """Derive one reproducible permutation seed without ambient RNG state."""
    return (seed * 1000003 + iteration) % 2**32


def _scheduled_alpha(alpha0, step, total_steps):
    """Cosine-anneal an EMA step towards one tenth of its initial value."""
    fraction = min(step, total_steps) / max(1, total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return alpha0 * (_ALPHA_MIN_FACTOR + (1.0 - _ALPHA_MIN_FACTOR) * cosine)


_CITATION = ("This is e-means. If you use our work or find it useful, please cite the e-means "
             "paper (Bessa et al., see faiss/contrib/e_means/README.md).")


def _format_passes(passes):
    """Format a number of passes for verbose output: 3.0 -> '3', 1.4 -> '1.4'."""
    return "%.4g" % passes


def _same_operation(method, implementation, receiver):
    """Check that a bound callback uses this function and receiver."""
    return (getattr(method, "__func__", None) is implementation
            and getattr(method, "__self__", None) is receiver)


class _EmeansBase:
    """Shared batch accounting and update orchestration."""

    def _invalidate_readout(self):
        if self.spherical:
            self._unit_generation += 1
            self._unit_version = None

    def _revive_unused(self, rng):
        source, target = self._choose_splits(rng)
        return 0 if source is None else self._apply_splits(source, target)

    def _require_running(self):
        if self._failed:
            raise RuntimeError(
                "create a new object after execution failure"
            )
        self._report_header_once()

    def _local_batch_size(self):
        return min(
            self.recipe["batch_size"],
            self.recipe["epoch_rows"][self._epoch] - self._epoch_rows,
        )

    def update(self, stats):
        """Apply one complete batch's statistics, then revive unused clusters.

        The caller supplies matching sums, counts, objective and row count
        for the recipe's next logical batch. Shapes, dtypes, values and
        sequencing are caller responsibilities.
        """
        self._require_running()
        expected_n = min(
            self.recipe["batch_size"],
            self.recipe["epoch_rows"][self._epoch] - self._epoch_rows,
        )
        ids, sums, counts, objective = self._prepare_statistics(
            stats, expected_n
        )
        alpha = _scheduled_alpha(
            self.recipe["alpha0"], self._step, self.recipe["total_steps"]
        )
        try:
            self._apply_statistics(
                ids, sums, counts, objective, alpha, 1.0 - alpha, alpha == 1
            )
            self._finish_step(expected_n)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise

    def _finish_step(self, expected_n):
        if (self._step + 1) % _CHECK_UNUSED_EVERY == 0:
            self._revive_unused(self._split_rng)
        self._invalidate_readout()
        self._step += 1
        self._epoch_steps += 1
        self._epoch_rows += expected_n
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None

    def _epoch_objective(self):
        return self._objective

    def _resolve_run_recipe(self, n):
        """Build ``self.recipe``, the resolved training plan for this run.

        ``begin`` turns constructor options and the dataset size into this
        dictionary. It records the statistical batch size B, initial EMA step
        alpha0, rows and update counts per epoch, and total updates. The shared
        lifecycle uses these to size each update, end epochs and anneal alpha.

        The recipe also records revival settings, the seed and execution
        metadata. Assignment chunks subdivide a statistical batch without
        changing its update schedule. Implementations add backend-specific
        fields, such as the device or represented dtype.

        ``batch_law="auto"`` is resolved here, from the dimension. The recipe
        records the resolved law and why, and the revival threshold the
        constructor resolved from ``unused_threshold`` and the dimension.
        """
        batch_law, batch_law_reason = _resolve_batch_law(
            self.batch_law, self.d
        )
        self._batch_law_resolved = batch_law
        batch_size, alpha0 = _resolve_recipe(
            n,
            self.k,
            self.nepochs,
            batch_law,
            self.batch_size,
            self.alpha0,
        )
        steps_per_pass = (n + batch_size - 1) // batch_size
        epoch_rows = _epoch_rows(n, self.nepochs)
        epoch_steps = [
            (rows + batch_size - 1) // batch_size for rows in epoch_rows
        ]
        batch_given = self.batch_size is not None
        alpha_given = self.alpha0 is not None
        self.recipe = {
            "name": (("manual" if alpha_given else "manual_batch")
                     if batch_given else "canonical_v1"),
            "batch_law": None if batch_given else batch_law,
            "batch_law_requested": None if batch_given else self.batch_law,
            "batch_law_reason": None if batch_given else batch_law_reason,
            "batch_size": batch_size,
            "batch_size_requested": (int(self.batch_size) if batch_given
                                     else None),
            "alpha0": alpha0,
            "alpha0_source": "given" if alpha_given else "step_law",
            "nepochs": self.nepochs,
            "steps_per_pass": steps_per_pass,
            "total_steps": sum(epoch_steps),
            "epoch_rows": epoch_rows,
            "epoch_steps": epoch_steps,
            "tail_policy": "keep",
            "alpha_min_factor": _ALPHA_MIN_FACTOR,
            "check_unused_every": _CHECK_UNUSED_EVERY,
            "unused_threshold": self._unused_threshold,
            "split_epsilon": _SPLIT_EPSILON,
            "init_usage": _INIT_USAGE,
            "seed": self.seed,
            "backend": self._backend,
            "assignment_chunk_size": self._get_assignment_chunk_size(),
        }

    def _reports(self):
        """Whether this process prints verbose output (one rank per group)."""
        return self.verbose

    def _say(self, text):
        # Like faiss's printf, a closed stdout (e.g. piped into head) never
        # stops training: printing just stops for the rest of the run.
        if getattr(self, "_quiet", False):
            return
        try:
            print(text, flush=True)
        except (OSError, ValueError):  # broken pipe, or a closed file object
            self._quiet = True

    def _describe_assignment(self):
        recipe = self.recipe
        if recipe["backend"] == "faiss":
            return "faiss on CPU"
        backend = recipe.get("assignment_backend")
        name = {"native": "PyTorch", "triton": "Triton", "cute": "CuTe"}.get(backend, str(backend))
        representation = recipe.get("representation")
        return "%s%s on %s" % (name, " " + representation if representation else "",
                               recipe.get("device"))

    def _describe_parallel(self):
        """The devices and class of a multi-process run, or None."""
        return None

    def _report_header(self):
        recipe = self.recipe
        law = recipe["batch_law"]
        if law is None:
            extras = ["batch given"]
        elif recipe.get("batch_law_requested") == "auto":
            extras = ["%s batch law, auto for %s"
                      % (law, recipe["batch_law_reason"])]
        else:
            extras = ["%s batch law" % law]
        if self.spherical:
            extras.append("spherical")
        if "index_factory" in recipe:
            extras.append("IVF assignment, %d lists, %d probed"
                          % (self.nlist, recipe["nprobe"]))
        self._say(_CITATION)
        parallel = self._describe_parallel()
        if parallel:
            self._say(parallel)
        self._say(
            "Clustering %d points in %dD to %d clusters, %s epochs "
            "(batch %d, alpha0 %.3g, %s)"
            % (self._n, self.d, getattr(self, "global_k", self.k),
               _format_passes(sum(recipe["epoch_rows"]) / self._n),
               recipe["batch_size"], recipe["alpha0"], ", ".join(extras))
        )

    def _report_header_once(self):
        # After begin(), once subclasses have finished the recipe (sharded runs
        # re-derive it for the global k), and only for runs that really start.
        if getattr(self, "_header_due", False):
            self._header_due = False
            if self._reports():
                self._report_header()

    def _report_preprocessing(self, start, given_init, n):
        # Up to the start of the epoch clock, so no second is counted twice.
        self._say("  Preprocessing in %.2f s (%s; assignment: %s)" % (
            self._start_time - start,
            "initial centroids given" if given_init
            else "every row used as an initial centroid" if n == self.k
            else "random initialization from the data, seed %d" % self.seed,
            self._describe_assignment()))

    def _report_epoch(self, stat, epoch):
        recipe = self.recipe
        rows = recipe["epoch_rows"][epoch]
        partial = "" if rows == self._n else " (%d of %d rows)" % (rows, self._n)
        self._say(
            "  Epoch %s / %s%s, iteration %d / %d (%.2f s): "
            "objective=%g"
            % (_format_passes(sum(recipe["epoch_rows"][:epoch + 1]) / self._n),
               _format_passes(sum(recipe["epoch_rows"]) / self._n), partial,
               sum(recipe["epoch_steps"][:epoch + 1]), recipe["total_steps"],
               stat["time"], stat["obj"])
        )

    def end_epoch(self):
        """Finish one complete pass and record its moving-pass objective."""
        self._require_running()
        rows = self.recipe["epoch_rows"][self._epoch]
        self._failed = True
        objective, count_square = self._epoch_totals()
        stat = {
            "obj": objective,
            "time": time.perf_counter() - self._start_time,
            "imbalance_factor": self.k * count_square / (rows * rows),
            "nsplit": self.n_splits - self._splits_before,
        }
        self.iteration_stats.append(stat)
        self._epoch += 1
        if self._epoch < len(self.recipe["epoch_rows"]):
            self._start_epoch()
        self._failed = False
        if self._reports():
            self._report_epoch(stat, self._epoch - 1)
