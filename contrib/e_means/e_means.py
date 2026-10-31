# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""E-means clustering with NumPy arrays or PyTorch tensors."""

import math
import time

import numpy as np

import faiss


_ALPHA_MIN_FACTOR = 0.1
_CHECK_UNUSED_EVERY = 5
_INIT_USAGE = 2.0**-16
_SPLIT_EPSILON = 1e-4
_UNUSED_THRESHOLD = 0.1

# The default profile couples fitted batch-size and initial-step laws. The
# batch laws were fitted on full passes with 100K to 10M training vectors; the
# step law also includes 100M. That law was fitted with a zero-ending cosine;
# the reference algorithm uses a 0.1 floor, which raises the mean step.
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _validate_params(d, k, niter, operating_point, batch_size, alpha0, seed):
    """Validate constructor arguments."""
    if d < 1:
        raise ValueError("d must be positive")
    if k < 1:
        raise ValueError("k must be positive")
    if not math.isfinite(niter) or niter <= 0:
        raise ValueError("niter must be finite and positive")
    if operating_point not in _BATCH_LAWS:
        raise ValueError("operating_point must be 'throughput' or 'quality'")
    if (batch_size is None) != (alpha0 is None):
        raise ValueError("batch_size and alpha0 must be set together")
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < float(alpha0) <= 1.0:
            raise ValueError("alpha0 must be in (0, 1]")
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")


def _round_power_of_two(value):
    """Round a positive value to the nearest power of two."""
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def _resolve_recipe(n, k, niter, operating_point, batch_size, alpha0):
    """Return the coupled statistical batch size and initial EMA step."""
    if batch_size is None:
        coefficient, k_exp, n_exp, iter_exp = _BATCH_LAWS[operating_point]
        raw_batch = coefficient * k**k_exp * n**n_exp * niter**iter_exp
        batch_size = _round_power_of_two(raw_batch)
        # These intentionally differ: they reproduce the two fitted recipes.
        if operating_point == "throughput":
            batch_size = min(batch_size, _round_power_of_two(n / 2))
        else:
            batch_size = min(batch_size, n // 2)
        batch_size = max(1, min(batch_size, n))

        coefficient, k_exp, batch_exp, n_exp, iter_exp = _ALPHA_LAW
        alpha0 = (
            coefficient
            * k**k_exp
            * batch_size**batch_exp
            * n**n_exp
            * niter**iter_exp
        )
        alpha0 = min(max(alpha0, 1e-4), 1.0)
    else:
        batch_size = min(int(batch_size), n)
        alpha0 = float(alpha0)

    return batch_size, alpha0


def _epoch_rows(n, niter):
    """Whole passes, a rounded fractional prefix, then moving closing.

    Round the fractional row count with Python's ties-to-even convention.
    Below one pass, only that prefix contributes to closing means.
    """
    whole = math.floor(niter)
    remainder = int(round((niter - whole) * n))
    if whole == 0:
        if remainder < 1:
            raise ValueError("niter * global_n rounds to zero rows")
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


def _is_torch_tensor(x):
    """Detect tensors; other inputs remain NumPy-compatible array-likes."""
    if isinstance(x, np.ndarray):
        return False
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def _as_float32_matrix(x, name, is_torch, allow_empty=False, prepare=True):
    if is_torch:
        import torch

        if not isinstance(x, torch.Tensor):
            raise TypeError("%s must be a torch.Tensor" % name)
        if x.layout != torch.strided:
            raise ValueError("%s must be a dense strided tensor" % name)
        if x.ndim != 2:
            raise ValueError("%s must be a 2-D tensor" % name)
        if (x.shape[0] == 0 and not allow_empty) or x.shape[1] == 0:
            raise ValueError("%s must not be empty" % name)
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("%s must be on a CPU or CUDA device" % name)
        if x.dtype != torch.float32:
            raise ValueError("%s must have dtype torch.float32" % name)
    else:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("%s must be a 2-D array" % name)
        if (x.shape[0] == 0 and not allow_empty) or x.shape[1] == 0:
            raise ValueError("%s must not be empty" % name)
    if not prepare:
        return x
    x, finite = _prepare_float32_matrix(x, is_torch)
    if not finite:
        raise ValueError("%s contains NaN or infinity" % name)
    return x


def _prepare_float32_matrix(x, is_torch):
    if is_torch:
        import torch

        x = x.detach().contiguous()
        return x, x.device.type == "cuda" or bool(torch.isfinite(x).all())
    x = np.ascontiguousarray(x, dtype="float32")
    return x, bool(np.isfinite(x).all())


def _numpy_unused_threshold(usage, k):
    """Compute the threshold consistently across NumPy scalar rules."""
    total = usage.sum(dtype="float32")
    scaled = np.multiply(
        total, np.float32(_UNUSED_THRESHOLD), dtype="float32"
    )
    return np.divide(scaled, np.float32(k), dtype="float32")


def _torch_generator(device, seed):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _randperm(n, seed, is_torch, device=None):
    if is_torch:
        import torch

        return torch.randperm(
            n,
            generator=_torch_generator(device, seed),
            device=device,
        )
    return np.random.RandomState(seed).permutation(n)


def _clone(x):
    return x.clone() if _is_torch_tensor(x) else x.copy()


def _separate_splits(donor, child):
    """Move one coordinate only when the FP32 split operands coincide.

    Zero donors become distinct subnormal operands; their squared distances
    can still tie. No value scan or dynamic collapsed-row list is needed.
    """
    if _is_torch_tensor(child):
        import torch

        collapsed = (donor == child).all(dim=1)
        value = child[:, 0]
        neighbor = torch.nextafter(value, torch.full_like(value, float("inf")))
        inward = torch.nextafter(value, torch.full_like(value, float("-inf")))
        neighbor = torch.where((neighbor.abs() < float("inf")), neighbor, inward)
        child[:, 0] = torch.where(collapsed, neighbor, value)
    else:
        collapsed = (donor == child).all(axis=1)
        value = child[:, 0]
        with np.errstate(over="ignore"):
            neighbor = np.nextafter(value, np.float32(np.inf))
        neighbor = np.where(
            np.isfinite(neighbor), neighbor,
            np.nextafter(value, np.float32(-np.inf)),
        )
        child[:, 0] = np.where(collapsed, neighbor, value)


def _synchronize(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


class Emeans:
    """Cluster vectors with mini-batch exponential moving averages.

    Parameters
    ----------
    d : int
        Dimension of the vectors to cluster.
    k : int
        Number of clusters.
    niter : int or float, optional
        Training passes. A fractional prefix precedes the full moving closing
        pass; below one pass, closing covers only that rounded prefix. Each
        prefix uses its own seeded permutation. See recipe["epoch_rows"].
    batch_size, alpha0 : optional
        Coupled statistical batch size and initial EMA step. If omitted, both
        are selected by the fitted recipe.
    operating_point : {"throughput", "quality"}, optional
        Recipe used when ``batch_size`` and ``alpha0`` are omitted.
        ``throughput`` takes the largest fitted batch within the recipe's
        tolerance; ``quality`` takes the smaller one.
    seed : int, optional
        Seed for initialization, batch permutations, and cluster splitting.
    verbose : bool, optional
        Print one line of statistics after each pass.
    assignment_chunk_size : int, optional
        Maximum rows assigned at once. This does not alter the statistical
        batch size or the EMA schedule. Defaults are 65,536 rows for NumPy,
        8,192 for PyTorch CPU, and dimension-dependent 8,192 to 1,048,576
        for CUDA. With NumPy input, it can change floating-point
        assignment results if Faiss selects a different exact-search kernel
        for a different query size.
        CUDA chunks can also change floating assignment results through
        GEMM or tiling selection.
    centroid_chunk_size : int, optional
        Maximum centroids examined at once by the PyTorch assignment path.
    capture_assignments : bool, optional
        Return owned chunk labels during the final caller-driven epoch and
        retain last-pass labels in original input order after resident train.
        Defaults to True; False opts out without disabling CUDA graph replay.

    Attributes
    ----------
    centroids : ndarray or Tensor
        Trained centroids with shape ``(k, d)``.
    obj : ndarray or Tensor
        Sum of squared distances observed while centroids move during each
        pass. The last value does not score the closing-mean centroids.
    iteration_stats : list of dict
        Per-pass objective, cumulative loop time, imbalance, and split count.
        Time starts after initialization, setup allocations, and the initial
        synchronization. Caller-driven time also includes caller loading between
        begin and end_epoch; it is not a compute-only measurement.
    recipe : dict
        Resolved recipe used by the most recent call to :meth:`begin`.
    n_splits : int
        Total number of dead clusters revived during the training run.
    assignments : ndarray or Tensor or None
        Last training pass's INT64 labels after successful resident train with
        capture enabled. These need not be nearest to the closing centroids.
        Caller-driven runs do not retain dataset-wide labels.

    Notes
    -----
    NumPy input uses Faiss for exact L2 assignment. PyTorch input stays on its
    CPU or CUDA device and does not call Faiss. NumPy arrays do not import
    PyTorch.

    Every epoch includes its ragged final batch. CPU input conversion and
    finite-value validation scan the input. CUDA accepts native FP32 and requires
    caller-validated finite values; metadata checks do not scan CUDA values.
    Random permutations and
    cluster splitting use local seeded generators. Results need not agree
    across NumPy, Torch CPU, and Torch CUDA. Each epoch allocates an int64
    permutation with one entry per input row.

    The final epoch also collects per-cluster sums. Every centroid that
    receives at least one row is replaced by the mean of those rows.

    With spherical=True, callers normalize incoming rows before conversion.
    Raw FP32 EMA coordinates are projected only for assignment/readout/closing;
    zero stays zero. Assignment returns represented IP (approximate cosine),
    while objectives sum 2-2*IP without clamping represented rounding error.
    """

    def __init__(
        self,
        d,
        k,
        niter=25,
        batch_size=None,
        alpha0=None,
        operating_point="throughput",
        seed=1234,
        verbose=False,
        assignment_chunk_size=None,
        centroid_chunk_size=4096,
        capture_assignments=True,
        spherical=False,
    ):
        self.d = int(d)
        self.k = int(k)
        niter = float(niter)
        self.niter = int(niter) if niter.is_integer() else niter
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.operating_point = operating_point
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.assignment_chunk_size = (
            None
            if assignment_chunk_size is None
            else int(assignment_chunk_size)
        )
        self.centroid_chunk_size = int(centroid_chunk_size)
        self.capture_assignments = bool(capture_assignments)
        self.spherical = bool(spherical)

        _validate_params(
            self.d,
            self.k,
            self.niter,
            self.operating_point,
            self.batch_size,
            self.alpha0,
            self.seed,
        )
        if (
            self.assignment_chunk_size is not None
            and self.assignment_chunk_size < 1
        ):
            raise ValueError("assignment_chunk_size must be positive")
        if self.centroid_chunk_size < 1:
            raise ValueError("centroid_chunk_size must be positive")

        self.reset()

    def reset(self):
        """Clear the result of a previous training run."""
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
        self._batch_norm = self._batch_operand = None
        self._batch_stats = self._stats_operand = None
        self._pass_counts = self._pass_sums = self._objective = None
        self._split_rng = None
        self._unit_centroids = self._unit_version = None
        self._unit_generation = 0

    def _invalidate_readout(self):
        if self.spherical:
            self._unit_generation += 1
            self._unit_version = None

    def _readout(self, force=False):
        if not self.spherical:
            return self._centroids
        from .e_means_spherical import normalize_into

        version = None
        if self._is_torch:
            try:
                version = self._centroids._version
            except RuntimeError:  # Inference tensors use explicit mutation generation.
                pass
        token = (self._unit_generation, version)
        if self._unit_centroids is None:
            self._unit_centroids = (self._centroids.new_empty(self._centroids.shape)
                                    if self._is_torch else np.empty_like(self._centroids))
        if force or self._unit_version != token:
            normalize_into(self._centroids, self._unit_centroids)
            self._unit_version = token
        return self._unit_centroids

    def current_centroids(self):
        """Borrow the active FP32 model coordinates without copying data.

        Available after begin, including completed epochs, until finalize.
        NumPy returns a non-writeable view; Torch returns a detached alias on
        the model device. Neither is immutable or safe for concurrent mutation.
        Do not write through the result. Finish readers before update,
        end_epoch, finalize, reset or a new begin, then reacquire. Old handles
        are not valid current-state reads after reset, finalize or a new begin.
        Copy/clone explicitly for an independent snapshot. CUDA callers must
        order producer, reader and subsequent mutation streams themselves;
        this method adds no synchronization or collective. These are FP32
        model coordinates, not rounded or encoded assignment operands. In
        spherical mode they are effective unit directions (zero stays zero),
        borrowed from one reusable FP32 cache with the same read window.
        """
        if self._centroids is None or self._failed:
            raise RuntimeError("current centroids require an active run")
        centroids = self._readout()
        if self._is_torch:
            return centroids.detach()
        view = centroids.view()
        view.flags.writeable = False
        return view

    def _get_assignment_chunk_size(self, is_torch, cuda=False):
        if self.assignment_chunk_size is not None:
            return self.assignment_chunk_size
        if cuda:
            # Fewer, larger chunks amortize per-chunk host work and fill the
            # device. Target roughly 256 MiB per FP32 statistics matrix or
            # native Torch score tile.
            width = max(self.d, min(self.k, self.centroid_chunk_size))
            return min(2**20, max(8192, 2**26 // width))
        return 8192 if is_torch else 65536

    def _prepare_torch_assignment(self, centroids):
        if self.spherical and centroids is self._centroids:
            centroids = self._readout()
        norm = (centroids.new_zeros(len(centroids)) if self.spherical
                else (centroids * centroids).sum(dim=1))
        return centroids, norm

    def _revive_unused(self, rng):
        # Sample donors without replacement in proportion to their usage.
        if _is_torch_tensor(self._usage):
            import torch

            threshold = self._usage.sum() * _UNUSED_THRESHOLD / self.k
            unused = self._usage < threshold
            donors = (~unused) & (self._usage > 0)
            nunused, ndonors = torch.stack(
                (unused.sum(), donors.sum())
            ).tolist()
            nsplit = min(nunused, ndonors)
            if nsplit == 0:
                return 0

            uniform = torch.rand(
                self.k,
                dtype=self._usage.dtype,
                device=self._usage.device,
                generator=rng,
            ).clamp_min(torch.finfo(self._usage.dtype).tiny)
            if self._usage.is_cuda:
                keys = torch.where(donors, self._usage.log() - (-uniform.log()).log(),
                                   float("-inf"))
                target = torch.sort(unused.byte(), descending=True, stable=True).indices[:nsplit]
            else:
                keys = torch.full_like(self._usage, float("-inf"))
                keys[donors] = self._usage[donors].log() - (
                    -uniform[donors].log()
                ).log()
                target = unused.nonzero(as_tuple=False).flatten()[:nsplit]
            source = torch.topk(keys, nsplit).indices
            centroids = self._centroids[source]
            sign = torch.ones(
                self.d, dtype=centroids.dtype, device=centroids.device
            )
        else:
            threshold = _numpy_unused_threshold(self._usage, self.k)
            unused = self._usage < threshold
            donors = (~unused) & (self._usage > 0)
            nsplit = min(int(unused.sum()), int(donors.sum()))
            if nsplit == 0:
                return 0

            uniform = np.maximum(
                rng.random_sample(self.k), np.finfo("float64").tiny
            )
            keys = np.full(self.k, -np.inf)
            keys[donors] = np.log(self._usage[donors]) - np.log(
                -np.log(uniform[donors])
            )
            source = np.argsort(keys)[-nsplit:][::-1]
            target = np.flatnonzero(unused)[:nsplit]
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
        self.n_splits += nsplit
        self._invalidate_readout()
        return nsplit

    def _assign_rows(self, x, centroids, centroid_norm=None):
        if _is_torch_tensor(x):
            import torch

            x_norm = None if self.spherical else (x * x).sum(dim=1)
            best_score = torch.full(
                (len(x),), float("inf"), dtype=x.dtype, device=x.device
            )
            labels = torch.zeros(
                len(x), dtype=torch.int64, device=x.device
            )

            for begin in range(0, self.k, self.centroid_chunk_size):
                end = min(begin + self.centroid_chunk_size, self.k)
                block = centroids[begin:end]
                block_norm = (
                    (block * block).sum(dim=1)
                    if centroid_norm is None
                    else centroid_norm[begin:end]
                )
                score = torch.addmm(
                    block_norm[None, :],
                    x,
                    block.t(),
                    alpha=-2.0,
                )
                block_score, block_label = score.min(dim=1)
                better = block_score < best_score
                torch.where(better, block_score, best_score, out=best_score)
                torch.where(better, block_label + begin, labels, out=labels)

            return (best_score if self.spherical else (best_score + x_norm).clamp_min(0)), labels

        if self.spherical:
            similarity, labels = faiss.knn(x, centroids, 1, metric=faiss.METRIC_INNER_PRODUCT)
            return -2 * similarity.ravel(), labels.ravel()
        distance, labels = faiss.knn(x, centroids, 1)
        return distance.ravel(), labels.ravel()

    def begin(self, global_n, init_centroids):
        """Initialize a run with caller-owned batches and explicit FP32 state.

        ``init_centroids`` has shape (k, d); its array type and device select
        execution. It is copied. Each epoch must supply the resolved batch
        sizes for ``recipe["epoch_rows"][epoch]``, including the final remainder,
        then call :meth:`end_epoch`.
        Loading, ordering, transfers and prefetch are the caller's responsibility.
        """
        n = int(global_n)
        if n != global_n or n < self.k:
            raise ValueError("global_n must be an integer at least k")
        is_torch = _is_torch_tensor(init_centroids)
        centroids = _as_float32_matrix(init_centroids, "init_centroids", is_torch)
        if centroids.shape != (self.k, self.d):
            raise ValueError("init_centroids must have shape (k, d)")
        self.reset()
        self._failed = True
        self._centroids = _clone(centroids)
        self._is_torch = is_torch
        if self.spherical:
            from .e_means_spherical import normalize_into

            normalize_into(self._centroids, self._centroids)
        if is_torch:
            self._usage = centroids.new_full((self.k,), _INIT_USAGE)
            self._batch_stats = centroids.new_empty((self.k, self.d + 1))
            self._split_rng = _torch_generator(centroids.device, self.seed)
        else:
            self._usage = np.full(self.k, _INIT_USAGE, dtype="float32")
            self._batch_stats = np.empty((self.k, self.d + 1), dtype="float32")
            self._split_rng = np.random.RandomState(self.seed)
        batch_size, alpha0 = _resolve_recipe(
            n, self.k, self.niter, self.operating_point,
            self.batch_size, self.alpha0,
        )
        steps_per_pass = (n + batch_size - 1) // batch_size
        epoch_rows = _epoch_rows(n, self.niter)
        epoch_steps = [(rows + batch_size - 1) // batch_size for rows in epoch_rows]
        manual = self.batch_size is not None
        self.recipe = {
            "name": "manual" if manual else "canonical_v1",
            "operating_point": None if manual else self.operating_point,
            "batch_size": batch_size,
            "batch_size_requested": int(self.batch_size) if manual else None,
            "alpha0": alpha0,
            "niter": self.niter,
            "steps_per_pass": steps_per_pass,
            "total_steps": sum(epoch_steps),
            "epoch_rows": epoch_rows,
            "epoch_steps": epoch_steps,
            "tail_policy": "keep",
            "alpha_min_factor": _ALPHA_MIN_FACTOR,
            "check_unused_every": _CHECK_UNUSED_EVERY,
            "unused_threshold": _UNUSED_THRESHOLD,
            "split_epsilon": _SPLIT_EPSILON,
            "init_usage": _INIT_USAGE,
            "seed": self.seed,
            "backend": "torch" if is_torch else "faiss",
            "assignment_chunk_size": self._get_assignment_chunk_size(is_torch, is_torch and centroids.is_cuda),
        }
        if is_torch:
            self.recipe.update(
                device=str(centroids.device),
                centroid_chunk_size=self.centroid_chunk_size,
            )
            self._stats_operand = None
        if self.spherical:
            self.recipe.update(spherical=True, assignment_score="represented_inner_product",
                               objective="sum(2-2*represented_inner_product)")
        self._n = n
        self._step = self._epoch = 0
        self.iteration_stats = []
        self._start_epoch()
        if is_torch:
            _synchronize(centroids.device)
        self._start_time = time.perf_counter()
        self._failed = False

    def _require_running(self):
        if (self._failed or self._centroids is None
                or self._epoch >= len(self.recipe["epoch_rows"])):
            raise RuntimeError("begin a run before accumulating or updating")

    def _local_batch_size(self):
        return min(self.recipe["batch_size"],
                   self.recipe["epoch_rows"][self._epoch] - self._epoch_rows)

    def _check_accumulate(self, n):
        self._require_running()
        if self._epoch_steps == self.recipe["epoch_steps"][self._epoch]:
            raise RuntimeError("call end_epoch before the next batch")
        pending_n = 0 if self._pending_stats is None else self._pending_stats["n"]
        target = self._local_batch_size()
        if n and pending_n == target:
            raise RuntimeError("consume completed statistics before accumulating again")
        if pending_n + n > target:
            raise ValueError("batch exceeds the prescribed logical batch size")

    def _start_epoch(self):
        self._epoch_steps = self._epoch_rows = 0
        self._splits_before = self.n_splits
        if self._is_torch:
            import torch

            if self._pass_counts is None:
                self._pass_counts = self._usage.new_zeros(self.k, dtype=torch.float64)
                self._objective = self._usage.new_zeros((), dtype=torch.float64)
            else:
                self._pass_counts.zero_()
                self._objective.zero_()
            self._pass_sums = (
                self._centroids.new_zeros((self.k, self.d))
                if self._epoch == len(self.recipe["epoch_rows"]) - 1 else None
            )
        else:
            self._pass_counts = np.zeros(self.k, dtype="float64")
            self._objective = np.float64(0)
            self._pass_sums = (
                np.zeros_like(self._centroids)
                if self._epoch == len(self.recipe["epoch_rows"]) - 1 else None
            )

    def accumulate(self, x):
        """Extend one local statistical batch against fixed centroids.

        Return a mapping with FP32 ``sums`` (k,d), FP32 ``counts`` (k),
        scalar FP64 ``objective`` and host integer ``n`` at completion. Before
        completion, sums/counts are None and objective/n report cumulative
        progress. Complete sums/counts are borrowed workspace, invalidated by
        extension/update; copy retained values. CPU callers may
        combine matching frozen-snapshot statistics themselves, summing all
        four fields, then call the same local update.

        Empty calls validate metadata without changing a pending contribution.
        With no positive input they return zeros and permit external update.
        With capture enabled, the final epoch also returns caller-owned INT64
        ``labels`` for only this supplied chunk (empty for an empty call).
        CUDA FP32 values must be finite;
        value validation is the caller's responsibility and does not run here.
        """
        self._require_running()
        self._check_accumulate(len(x))
        if _is_torch_tensor(x) != self._is_torch:
            raise TypeError("x and init_centroids must use the same array type")
        x = _as_float32_matrix(x, "x", self._is_torch, allow_empty=True, prepare=False)
        if x.shape[1] != self.d:
            raise ValueError("x must have dimension d")
        if self._is_torch and x.device != self._centroids.device:
            raise ValueError("x and centroids must be on the same device")
        try:
            x, finite = _prepare_float32_matrix(x, self._is_torch)
        except BaseException:
            self._failed = True
            raise
        if not finite:
            raise ValueError("x contains NaN or infinity")
        return self._accumulate(x)

    def _accumulate(self, x, index=None, labels_out=None):
        # Indexed matrix training gathers only assignment chunks, never B*d.
        n = len(x) if index is None else len(index)
        self._check_accumulate(n)
        if n == 0 and self._pending_stats is not None:
            stats = dict(self._pending_stats)
            if self.capture_assignments and self._epoch == len(self.recipe["epoch_rows"]) - 1:
                if self._is_torch:
                    import torch

                    stats["labels"] = x.new_empty(0, dtype=torch.int64)
                else:
                    stats["labels"] = np.empty(0, dtype="int64")
            return stats
        try:
            stats = self._accumulate_core(x, index, labels_out)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        self._pending_stats = stats
        return stats

    def _accumulate_core(self, x, index=None, labels_out=None):
        n = len(x) if index is None else len(index)
        pending_n = 0 if self._pending_stats is None else self._pending_stats["n"]
        target = self._local_batch_size()
        if self._is_torch:
            import torch
        if pending_n == 0:
            # Bound the whole contribution, not each supplied/assignment chunk.
            wide = n > 0 and target > 2**24
            if self._is_torch:
                accumulation = (self._batch_stats.new_zeros(
                    self._batch_stats.shape, dtype=torch.float64)
                    if wide else self._batch_stats.zero_())
                self._batch_objective = self._objective.new_zeros(())
                if self.spherical:
                    self._readout(force=self._centroids.is_cuda
                                  and torch.cuda.is_current_stream_capturing())
                self._batch_centroids, self._batch_norm = (
                    self._prepare_torch_assignment(self._centroids)
                )
            else:
                self._batch_stats.fill(0)
                accumulation = (np.zeros(self._batch_stats.shape, dtype="float64")
                                if wide else self._batch_stats)
                self._batch_objective = np.float64(0)
                self._batch_centroids = self._readout()
            self._batch_accumulation = accumulation
            self._batch_operand = None
        accumulation = self._batch_accumulation
        objective, centroid_norm = self._batch_objective, self._batch_norm
        operand = self._batch_operand
        sums, counts = accumulation[:, :-1], accumulation[:, -1]
        captured_labels = None if labels_out is None else labels_out[:n]
        if (captured_labels is None and self.capture_assignments
                and self._epoch == len(self.recipe["epoch_rows"]) - 1):
            if self._is_torch:
                captured_labels = x.new_empty(n, dtype=torch.int64)
            else:
                captured_labels = np.empty(n, dtype="int64")
        chunk_size = self.recipe["assignment_chunk_size"]
        for begin in range(0, n, chunk_size):
            end = min(begin + chunk_size, n)
            rows = x[begin:end] if index is None else x[index[begin:end]]
            distance, labels = self._assign_rows(rows, self._batch_centroids, centroid_norm)
            if captured_labels is not None:
                captured_labels[begin:end] = labels
            if self.spherical:
                objective += 2 * len(rows)
            if self._is_torch:
                objective += distance.sum(dtype=torch.float64)
                if operand is None:
                    if self._stats_operand is None:
                        with torch.inference_mode(self._batch_stats.is_inference()):
                            self._stats_operand = self._centroids.new_ones(
                                (min(chunk_size, self.recipe["batch_size"]), self.d + 1))
                    operand = (self._stats_operand if accumulation is self._batch_stats
                               else self._stats_operand.double())
                values = operand[:len(rows)]
                values[:, :-1].copy_(rows)
                accumulation.index_add_(0, labels, values)
            else:
                objective += distance.sum(dtype="float64")
                counts += np.bincount(labels, minlength=self.k)
                np.add.at(sums, labels, rows)
        self._batch_objective, self._batch_operand = objective, operand
        total_n = pending_n + n
        if total_n == target and accumulation is not self._batch_stats:
            if self._is_torch:
                self._batch_stats.copy_(accumulation)
            else:
                np.copyto(self._batch_stats, accumulation, casting="unsafe")
            sums, counts = self._batch_stats[:, :-1], self._batch_stats[:, -1]
        complete = total_n == target or total_n == 0
        stats = {"sums": sums if complete else None,
                 "counts": counts if complete else None,
                 "objective": objective, "n": total_n}
        if captured_labels is not None:
            stats["labels"] = captured_labels
        return stats

    def update(self, stats):
        """Consume one frozen-snapshot batch and perform EMA then revival.

        ``n`` must equal the prescribed logical batch size. Statistics use
        the same array type/device as state; values and row membership are
        caller obligations. Global CPU statistics may be combined externally.
        """
        self._require_running()
        if (self._pending_stats is not None
                and 0 < self._pending_stats["n"] < self._local_batch_size()):
            raise ValueError("complete the pending batch before update")
        if self._epoch_steps == self.recipe["epoch_steps"][self._epoch]:
            raise RuntimeError("call end_epoch before the next update")
        sums, counts, objective, n = (
            stats["sums"], stats["counts"], stats["objective"], stats["n"]
        )
        expected_n = self._local_batch_size()
        if not isinstance(n, (int, np.integer)) or n != expected_n:
            raise ValueError(
                "stats n must equal the prescribed batch size %d" % expected_n
            )
        for value, shape, expected_dtype in (
            (sums, (self.k, self.d), "float32"),
            (counts, (self.k,), "float32"),
            (objective, (), "float64"),
        ):
            if self._is_torch:
                import torch

                if not isinstance(value, torch.Tensor):
                    raise TypeError("statistics must be tensors")
                if (value.layout != torch.strided or value.shape != shape
                        or value.dtype != getattr(torch, expected_dtype)
                        or value.device != self._centroids.device):
                    raise ValueError(
                        "statistics have incompatible shape, dtype or device"
                    )
            elif (not isinstance(value, (np.ndarray, np.generic))
                    or value.shape != shape or value.dtype != np.dtype(expected_dtype)):
                raise ValueError("statistics have incompatible shape or dtype")
        if self._is_torch:
            sums, counts, objective = sums.detach(), counts.detach(), objective.detach()
        alpha = _scheduled_alpha(
            self.recipe["alpha0"], self._step, self.recipe["total_steps"]
        )
        try:
            self._apply_statistics(sums, counts, objective,
                                   alpha, 1.0 - alpha, alpha == 1)
            self._finish_step(expected_n)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise

    def _apply_statistics(self, sums, counts, objective,
                          alpha, decay, replace_usage=False):
        if self._is_torch:
            import torch

        # Capture this pass's own pre-update assignments before workspace reuse.
        self._pass_counts += counts
        self._objective += objective
        if self._pass_sums is not None:
            self._pass_sums += sums
        touched = counts > 0
        if self._is_torch:
            # Fixed-shape masks avoid dynamic row selection/host reads. Inactive
            # lanes divide zero by one; only touched lanes divide by statistics.
            safe_counts = torch.where(touched, counts, 1)
            mean = sums.clone()
            mean.masked_fill_(~touched[:, None], 0)
            mean.div_(safe_counts[:, None])
            retained = self._usage * decay
            mass = retained + alpha * counts
            weight = alpha * counts / torch.where(touched, mass, 1)
            position = mean - self._centroids
            position.mul_(weight[:, None]).add_(self._centroids)
            torch.where((weight == 1)[:, None], mean, position, out=position)
            torch.where(
                touched[:, None], position, self._centroids, out=self._centroids
            )
            self._usage.copy_(
                torch.where(touched, mass, self._usage) if replace_usage else mass
            )
        else:
            if not replace_usage:
                self._usage *= decay
            retained = (
                self._usage[touched] if not replace_usage
                else np.zeros_like(counts[touched])
            )
            mass = retained + alpha * counts[touched]
            mean = sums[touched] / counts[touched, None]
            weight = alpha * counts[touched] / mass
            position = self._centroids[touched] + weight[:, None] * (
                mean - self._centroids[touched]
            )
            position[weight == 1] = mean[weight == 1]
            self._centroids[touched] = position
            self._usage[touched] = mass
        self._invalidate_readout()

    def _finish_step(self, expected_n):
        if (self._step + 1) % _CHECK_UNUSED_EVERY == 0:
            self._revive_unused(self._split_rng)
        self._invalidate_readout()  # Also covers graph/raw kernel writes on replay.
        self._step += 1
        self._epoch_steps += 1
        self._epoch_rows += expected_n
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_norm = self._batch_operand = None

    def end_epoch(self):
        """Finish one complete pass and record its moving-pass objective."""
        self._require_running()
        rows = self.recipe["epoch_rows"][self._epoch]
        if (self._pending_stats is not None or self._epoch_rows != rows
                or self._epoch_steps != self.recipe["epoch_steps"][self._epoch]):
            raise RuntimeError("epoch is incomplete; consume all prescribed batches")
        self._failed = True
        if self._is_torch:
            import torch

            totals = torch.stack(
                (self._objective, torch.dot(self._pass_counts, self._pass_counts))
            )
            objective, count_square = totals.tolist()
        else:
            objective = float(self._objective)
            count_square = float(np.dot(self._pass_counts, self._pass_counts))
        stat = {
            "obj": objective,
            "time": time.perf_counter() - self._start_time,
            "imbalance_factor": self.k * count_square / (rows * rows),
            "nsplit": self.n_splits - self._splits_before,
        }
        self.iteration_stats.append(stat)
        if self.verbose:
            print(
                "e-means iteration %d: objective=%.6g imbalance=%.3f nsplit=%d"
                % (self._epoch + 1, objective,
                   stat["imbalance_factor"], stat["nsplit"])
            )
        self._epoch += 1
        if self._epoch < len(self.recipe["epoch_rows"]):
            self._start_epoch()
        self._failed = False

    def _spherical_closing(self):
        from .e_means_spherical import normalize_into

        result = (self._centroids.new_empty(self._centroids.shape) if self._is_torch
                  else np.empty_like(self._centroids))
        # Bound wide means separately from projection scratch; normalize before F32.
        rows = max(1, (1 << 25) // (8 * self.d))
        for begin in range(0, self.k, rows):
            end = min(begin + rows, self.k)
            sums, counts = self._pass_sums[begin:end], self._pass_counts[begin:end]
            raw = self._centroids[begin:end]
            if self._is_torch:
                import torch

                dtype = torch.float64 if sums.dtype == torch.float64 else torch.float32
                mean = sums.to(dtype) / counts.clamp_min(1).to(dtype)[:, None]
                mean = torch.where((counts > 0)[:, None], mean, raw)
            else:
                mean = sums / np.maximum(counts, 1)[:, None].astype(sums.dtype)
                mean = np.where((counts > 0)[:, None], mean, raw)
            normalize_into(mean, result[begin:end])
        return result

    def finalize(self):
        """Set and return FP32 closing means from the final pass's assignments.

        Never-assigned centroids retain their current positions. No new search
        is performed. All epochs must be complete; repeated finalization returns
        the same result.
        """
        if self._failed:
            raise RuntimeError("reset or begin a new run after execution failure")
        if self.centroids is not None:
            return self.centroids
        if self._centroids is None or self._epoch != len(self.recipe["epoch_rows"]):
            raise RuntimeError("complete all epochs before finalize")
        self._failed = True
        nonempty = self._pass_counts > 0
        centroids = self._centroids if self._is_torch or self.spherical else self._centroids.copy()
        objectives = [stat["obj"] for stat in self.iteration_stats]
        if self.spherical:
            self.centroids = self._spherical_closing()
            if self._is_torch:
                import torch

                self.obj = torch.tensor(objectives, dtype=torch.float64, device=centroids.device)
            else:
                self.obj = np.asarray(objectives, dtype="float64")
        elif self._is_torch:
            import torch

            mean = self._pass_sums / self._pass_counts.clamp_min(1).float()[:, None]
            self.centroids = torch.where(
                nonempty[:, None], mean, centroids
            ).contiguous()
            self.obj = torch.tensor(
                objectives, dtype=torch.float64, device=centroids.device
            )
        else:
            centroids[nonempty] = (
                self._pass_sums[nonempty]
                / self._pass_counts[nonempty, None].astype("float32")
            )
            self.centroids = np.ascontiguousarray(centroids, dtype="float32")
            self.obj = np.asarray(objectives, dtype="float64")
        self._centroids = self._usage = self._batch_stats = None
        self._unit_centroids = self._unit_version = None
        self._pass_sums = self._pass_counts = self._objective = None
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_norm = self._batch_operand = None
        if self._is_torch:
            self._stats_operand = None
        self._failed = False
        return self.centroids

    def _replay_eligible(self, x):
        # Caller-driven/custom runs keep their original eager implementation.
        if (type(self) is not Emeans or not self._is_torch or not x.is_cuda
                or self.batch_size is not None or self.alpha0 is not None
                or self.assignment_chunk_size is not None
                or self.centroid_chunk_size != 4096
                or self.recipe["batch_size"] > 2**24
                or self.recipe["steps_per_pass"] < 3):
            return False
        return x.is_contiguous()

    def _capture_step(self, x, index, rate, batches, epoch_rates, position, labels_out=None):
        import torch

        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        graph = torch.cuda.CUDAGraph()
        # The training call retains its pool across epochs until graph teardown.
        pool = torch.cuda.MemPool()
        captured = False
        stats = None
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                torch.cuda.current_blas_handle()
                # The context manager empties process caches; use primitives.
                graph.capture_begin(pool=pool.id, capture_error_mode="thread_local")
                try:
                    torch.index_select(batches, 0, position, out=index.unsqueeze(0))
                    torch.index_select(epoch_rates, 0, position, out=rate.unsqueeze(0))
                    # Explicit staging is recorded even before the final epoch.
                    stats = self._accumulate_core(x, index, labels_out)
                    self._apply_statistics(stats["sums"], stats["counts"],
                                           stats["objective"], rate[0], rate[1])
                    position.add_(1)
                finally:
                    graph.capture_end()
            captured = True
            current.wait_stream(stream)
        finally:
            if not captured:
                # No retry/rollback of failed capture or execution errors.
                stats = None
                graph.reset()
        return graph, pool

    def train(self, x, init_centroids=None):
        """Train a dense FP32 matrix using the same caller-driven numerical loop.

        Return centroids and last-pass labels; objective history remains in obj. Initialization and epoch shuffles
        use the existing local seeded policy. No whole logical batch is gathered.
        Explicit initial centroids must match x's array type and device.
        """
        # A failed new train preflight must not expose labels from an older run.
        self.assignments = None
        is_torch = _is_torch_tensor(x)
        if is_torch:
            import torch

        x = _as_float32_matrix(x, "x", is_torch)
        n, d = x.shape
        if d != self.d or n < self.k:
            raise ValueError("x must have dimension d and at least k rows")
        device = x.device if is_torch else None
        if init_centroids is None:
            init_centroids = (
                x if n == self.k
                else x[_randperm(n, self.seed, is_torch, device)[:self.k]]
            )
        elif _is_torch_tensor(init_centroids) != is_torch:
            raise TypeError("x and init_centroids must use the same array type")
        if is_torch and init_centroids.device != device:
            raise ValueError("x and init_centroids must be on the same device")
        self.begin(n, init_centroids)
        rates = None
        order = index = rate = epoch_rates = position = batches = None
        graph_has_sums = False
        graph = pool = None
        assignments = batch_labels = None
        try:
            if self._replay_eligible(x):
                alphas = [_scheduled_alpha(self.recipe["alpha0"], step,
                                           self.recipe["total_steps"])
                          for step in range(self.recipe["total_steps"])]
                # Independently round the host-computed alpha and decay to FP32.
                rates = torch.tensor([(a, 1.0 - a) for a in alphas],
                                     dtype=torch.float32, device=device)
                index = torch.empty(self.recipe["batch_size"], dtype=torch.int64,
                                    device=device)
                rate = torch.empty(2, dtype=torch.float32, device=device)
                epoch_rates = rate.new_empty((self.recipe["steps_per_pass"], 2))
                position = torch.zeros(1, dtype=torch.int64, device=device)
            if self.capture_assignments:
                assignments = (torch.empty(n, dtype=torch.int64, device=device)
                               if is_torch else np.empty(n, dtype="int64"))
                if self.recipe["epoch_rows"][-1] < n:
                    assignments.fill_(-1) if is_torch else assignments.fill(-1)
                if rates is not None:
                    batch_labels = torch.empty_like(index)
            for epoch, rows in enumerate(self.recipe["epoch_rows"]):
                shuffled = _randperm(n, _epoch_seed(self.seed, epoch), is_torch, device)
                if rates is None or order is None:
                    order = shuffled
                else:
                    # Capture reads this storage; eager primes/tails use its new values.
                    order.copy_(shuffled)
                shuffled = None
                primed = graph is not None
                if rates is not None:
                    # Stable storage, original epoch order and independently rounded rates.
                    batches = order[:n - n % len(index)].view(-1, len(index))
                    # Fixed full-N tables survive graph reuse; only the active
                    # prefix is indexed during a shorter epoch.
                    steps = self.recipe["epoch_steps"][epoch]
                    epoch_rates[:steps].copy_(rates[self._step:self._step + steps])
                    position.zero_()
                for begin in range(0, rows, self.recipe["batch_size"]):
                    batch = order[begin:min(begin + self.recipe["batch_size"], rows)]
                    replay = (rates is not None and len(batch) == len(index)
                              and alphas[self._step] != 1)
                    if replay and primed:
                        if graph is None:
                            graph_has_sums = self._pass_sums is not None
                            graph, pool = self._capture_step(
                                x, index, rate, batches, epoch_rates, position, batch_labels)
                        graph.replay()
                        if assignments is not None and epoch == len(self.recipe["epoch_rows"]) - 1:
                            # Consume captured row IDs before the next replay.
                            assignments[index] = batch_labels
                        if self._pass_sums is not None and not graph_has_sums:
                            # Same dense ATen closing add, before revival/next stats zero.
                            self._pass_sums.add_(self._batch_stats[:, :-1])
                        # Replay writes do not execute Python mutation notifications.
                        torch.autograd.graph.increment_version((self._centroids, self._usage))
                        self._finish_step(len(batch))
                    elif (type(self) is Emeans and rates is None
                          and assignments is not None and epoch == len(self.recipe["epoch_rows"]) - 1):
                        # Keep eager resident capture bounded by assignment chunks.
                        # All chunks still share one logical batch's fixed snapshot
                        # and one update; public caller labels remain caller-owned.
                        chunk_size = self.recipe["assignment_chunk_size"]
                        for offset in range(0, len(batch), chunk_size):
                            chunk = batch[offset:offset + chunk_size]
                            stats = self._accumulate(x, chunk)
                            # This internal pending receipt no longer needs labels.
                            labels = stats.pop("labels")
                            assignments[chunk] = labels
                            labels = None
                        self.update(stats)
                    else:
                        stats = self._accumulate(x, batch, batch_labels)
                        self.update(stats)
                        if rates is not None:
                            position.add_(1)
                        primed = primed or replay
                        if assignments is not None and epoch == len(self.recipe["epoch_rows"]) - 1:
                            assignments[batch] = stats["labels"]
                self.end_epoch()
                # Reporting drains work; retain the owned graph/pool for the next epoch.
        except BaseException:
            self._failed = True
            self.assignments = None
            if rates is not None:
                torch.cuda.current_stream(device).synchronize()
            raise
        finally:
            try:
                if graph is not None:
                    graph.reset()
            except BaseException:
                self._failed = True
                raise
            finally:
                graph = pool = None  # noqa: F841 - release pool after reset
                order = shuffled = batches = None
                index = rate = epoch_rates = position = None
                batch_labels = None
        self.finalize()
        self.assignments = assignments
        return self.centroids, self.assignments

    def assign(self, x):
        """Return squared L2 distances or spherical IP, and centroid indices.

        Spherical mode returns represented inner-product similarity instead:
        callers normalize input before conversion; rounded unit directions
        approximate cosine without repairing represented norm drift.
        """
        if self.centroids is None or self._failed:
            raise RuntimeError("train must be called before assign")
        is_torch = _is_torch_tensor(self.centroids)
        if _is_torch_tensor(x) != is_torch:
            raise TypeError("x and centroids must use the same array type")
        if is_torch:
            import torch

        x = _as_float32_matrix(x, "x", is_torch)
        centroids = self.centroids.detach() if is_torch else self.centroids
        assignment_chunk_size = self._get_assignment_chunk_size(
            is_torch, is_torch and self.centroids.is_cuda)
        if x.shape[1] != self.d:
            raise ValueError(
                "x has dimension %d, expected %d" % (x.shape[1], self.d)
            )
        if is_torch and x.device != centroids.device:
            raise ValueError("x and centroids must be on the same device")

        if is_torch:
            distance = torch.empty(
                len(x), dtype=torch.float32, device=x.device
            )
            labels = torch.empty(
                len(x), dtype=torch.int64, device=x.device
            )
            centroids, centroid_norm = self._prepare_torch_assignment(centroids)
        else:
            distance = np.empty(len(x), dtype="float32")
            labels = np.empty(len(x), dtype="int64")
            centroid_norm = None

        for begin in range(0, len(x), assignment_chunk_size):
            end = min(begin + assignment_chunk_size, len(x))
            distance[begin:end], labels[begin:end] = self._assign_rows(
                x[begin:end], centroids, centroid_norm
            )
        if self.spherical:
            distance *= -0.5
        return distance, labels
