# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Floating PyTorch E-means on CPU and CUDA."""

import time

import torch

from .torch_backend import (
    TorchBackend,
    TritonBackend,
    CuteBackend,
    select_backend,
    default_chunk_size,
    _separate_splits,
)

from .base import (
    _EmeansBase,
    _INIT_USAGE,
    _REMOVAL_THRESHOLD,
    _SPLIT_EPSILON,
    _epoch_rows,
    _epoch_seed,
    _resolve_recipe,
    _resolve_unused_threshold,
    _round_power_of_two as _round_power_of_two,
    _scheduled_alpha,
    _same_operation,
)


# Rows per step of the closing-pass scan, bounding its temporaries.
_HANDOVER_CHUNK = 1 << 22

__all__ = [
    "Emeans",
    "_separate_splits",
    "_epoch_rows",
    "_epoch_seed",
    "_resolve_recipe",
    "_round_power_of_two",
    "_scheduled_alpha",
]


def _torch_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _synchronize(device):
    if device.type == "cuda":

        torch.cuda.synchronize(device)


def _closing_centroid_values(sums, counts, current, moved):
    """FP32 closing values, with immutable inputs and bounded promotion scratch."""
    dtype = torch.float64 if sums.dtype in (torch.int64, torch.float64) else torch.float32
    result = torch.empty_like(current, dtype=torch.float32, memory_format=torch.contiguous_format)
    row_bytes = 32  # Clamp/cast/mask vectors, including the preceding block's last vector.
    if sums.device.type == "cpu":
        # CPU TensorIterator materializes common-dtype output and promoted inputs.
        row_bytes += (8 * sums.shape[1]) if dtype == torch.float64 else 0
        if sums.dtype != dtype:
            row_bytes += (8 if dtype == torch.float64 else 4) * sums.shape[1]
    block_rows = max(1, (32 << 20) // row_bytes)
    for begin in range(0, len(counts), block_rows):
        end = min(begin + block_rows, len(counts))
        count = counts[begin:end]
        output = result[begin:end]
        # The divisor selects the original common compute dtype; out only casts the result.
        torch.div(sums[begin:end], count.clamp_min(1).to(dtype)[:, None], out=output)
        closed = count > 0
        if moved is not None:
            closed = closed & ~moved[begin:end]
        torch.where(closed[:, None], output, current[begin:end], out=output)
    return result


class Emeans(_EmeansBase):
    """Floating tensors: CPU FP32 computation or CUDA FP32/FP16/BF16 operands.

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
    the median to split the donors with the longest axes, if a full pass of
    data follows or within the first pass: one rule at every budget and
    dimension. Both centroids of a split made during the last pass keep their
    running (EMA) positions: their uniform last-pass averages would mix in rows
    from before the split. With ``closing_handover`` (default) and captured
    assignments, ``train`` assigns again the last-pass rows such a revived
    centroid held before moving, like a training batch, at their share of one
    epoch's cost; the caller-driven ``begin``/``accumulate``/``update`` loop
    leaves labels to the caller. A value revives clusters below that
    fraction of the mean mass instead, splitting mass-sampled donors (line 9,
    Appendix J); 0 disables revival.

    ``finalize`` computes unweighted means from assignments collected while
    the centroids move during the last epoch (line 12). Unassigned centroids
    retain their positions. ``train`` returns centroids and captured assignments.
    For Euclidean clustering, ``obj`` reports moving-pass sums of squared
    distances; use ``assign`` to evaluate the final centroids.

    Implementation details: the initial mass is 2**-16 (the paper uses 1e-10),
    remainder batches are retained, and fractional epoch budgets are supported.
    ``spherical=True`` extends the algorithm to projected centroid directions.

    ``max_points_per_centroid`` caps the training set as in ``faiss.Kmeans``:
    when the input has more than ``max_points_per_centroid * k`` rows,
    ``train`` uses a random subset of that many rows (drawn from ``seed``) and
    returns label -1 for the other rows. The default None trains on all rows.

    Model coordinates and results are FP32. CUDA FP32 input defaults to BF16
    assignment and rounded-row statistics; CPU and native FP16/BF16 stay native.
    The first typed input, including empty input, selects the representation,
    not initial centroids. Use QuantizedEmeans(quantizer=None) for native CUDA
    FP32; explicit quantizers and encoded formats belong to QuantizedEmeans.
    """

    _backend = "torch"
    _automatic = True

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
        centroid_chunk_size=4096,
        *,
        assignment_backend="auto",
        capture_assignments=True,
        spherical=False,
        unused_threshold=None,
        max_points_per_centroid=None,
        closing_handover=True,
    ):
        self.d = int(d)
        self.k = int(k)
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
        self.centroid_chunk_size = int(centroid_chunk_size)
        self.capture_assignments = bool(capture_assignments)
        self.spherical = bool(spherical)
        self.assignment_backend = assignment_backend

        self.centroids = None
        self.assignments = None
        self.obj = None
        self.iteration_stats = None
        self.recipe = None
        self.n_splits = 0
        self._usage = None
        self._centroids = None
        self._removal_stats = self._axes = self._moved = None
        # Closing-pass reassignment of rows held by revived centroids (train() only).
        self._handover = None
        self.closing_handover_stats = None
        self._check_graph = self._check_pool = self._check_async = self._check_splits = None
        self._pending_stats = None
        self._removal_folded = False
        self._failed = False
        self._batch_accumulation = self._batch_objective = None
        self._batch_norm = None
        self._batch_centroids = None
        self._batch_stats = None
        self._pass_counts = self._pass_sums = self._objective = None
        self._split_rng = None
        self._input_dtype = None
        self._cast_dtype = None
        self._resolved_assignment_backend = None
        self.assignment_recipe = None
        # A caller may begin before its first typed input. Native state setup
        # requires no GPU software; the actual operation resolves on that input.
        self._execution_backend = TorchBackend(self.centroid_chunk_size)
        self._query_backend = None
        self._unit_centroids = self._unit_version = None
        self._unit_generation = 0

    def _finish_step(self, expected_n):
        super()._finish_step(expected_n)
        self._batch_norm = None

    def _readout(self, force=False):
        """Return effective FP32 centroids for assignment and inspection.

        "Readout" means converting the internal EMA positions into the
        coordinates used by assignment and ``current_centroids``. Euclidean
        clustering uses the stored positions directly. Spherical clustering
        uses their unit directions, keeping the EMA positions unchanged;
        zero vectors remain zero.

        Directions are cached using model updates and Torch's tensor version.
        ``force=True`` recomputes them even when the cache appears current,
        so CUDA graph capture records normalization for every replay. The
        returned tensor borrows model storage; clone it for a snapshot.
        """
        if not self.spherical:
            return self._centroids
        from .e_means_spherical import normalize_into

        version = None
        try:
            version = self._centroids._version
        except (
            RuntimeError
        ):  # Inference tensors use explicit mutation generation.
            pass
        token = (self._unit_generation, version)
        if self._unit_centroids is None:
            self._unit_centroids = self._centroids.new_empty(
                self._centroids.shape
            )
        if force or self._unit_version != token:
            normalize_into(self._centroids, self._unit_centroids)
            self._unit_version = token
        return self._unit_centroids

    def current_centroids(self):
        """Borrow the active FP32 model coordinates without copying data.

        Available after begin, including completed epochs, until finalize.
        Returns a detached alias on the model device. It is not immutable
        or safe for concurrent mutation.
        Do not write through the result. Finish readers before update,
        end_epoch or finalize, then reacquire. Old handles are not valid
        current-state reads after finalize.
        Copy/clone explicitly for an independent snapshot. CUDA callers must
        order producer, reader and subsequent mutation streams themselves;
        this method adds no synchronization or collective. These are FP32
        model coordinates, not rounded or encoded assignment operands. In
        spherical mode they are effective unit directions (zero stays zero),
        borrowed from one reusable FP32 cache with the same read window.
        """
        if self._failed:
            raise RuntimeError("create a new object after execution failure")
        centroids = self._readout()
        return centroids.detach()

    def _get_assignment_chunk_size(self):
        if self.assignment_chunk_size is not None:
            return self.assignment_chunk_size
        state = (
            self._centroids if self._centroids is not None else self.centroids
        )
        return self._execution_backend.chunk_size(state)

    @property
    def _assignment_backend(self):
        return self._execution_backend.name

    def _resolve_assignment_backend(self, x):
        if self._resolved_assignment_backend is not None:
            return
        try:
            cls = select_backend(x, self._input_dtype or x.dtype, self.k, self.d, self.assignment_backend)
            if not (isinstance(self._execution_backend, cls) and self._execution_backend.name == cls.name):
                query = self._query_backend
                self._execution_backend = (
                    query.fork_query() if isinstance(query, cls) and query.name == cls.name
                    else cls(self.centroid_chunk_size))
            # Prediction requires assignment, not update/revival software.
            self._execution_backend.require_assignment(x.device)
        except BaseException:
            if self.recipe is not None:
                self._failed = True
                self.assignments = None
            raise
        self._resolved_assignment_backend = cls.name

    def _bind_input_dtype(self, dtype, *, cast=False):
        self._input_dtype = dtype
        self._cast_dtype = dtype if cast else None

    def _resolve_input_dtype(self, x):
        if self._input_dtype is None:
            dtype = torch.float32 if x.device.type == "cpu" else x.dtype
            cast = self._automatic and x.device.type == "cuda" and x.dtype == torch.float32
            if cast:
                dtype = torch.bfloat16
            self._bind_input_dtype(dtype, cast=cast)

    def _prepare_input(self, x):
        self._resolve_input_dtype(x)
        self._resolve_assignment_backend(x)
        if self.recipe is not None:
            self.recipe.update(
                representation=str(self._input_dtype).removeprefix("torch."),
                assignment_backend=self._assignment_backend,
                assignment_backend_requested=self.assignment_backend,
                assignment_chunk_size=self._get_assignment_chunk_size(),
            )
        return x

    def _prepare_rows(self, rows):
        rows = rows.detach()
        if rows.device.type == "cpu":
            rows = rows.float()
        elif self._cast_dtype is not None and rows.dtype in (torch.float16, torch.float32):
            rows = rows.to(self._cast_dtype)
        return rows.contiguous()

    def _initial_centroid_values(self, rows):
        return self._prepare_rows(rows).float()

    def _prepare_assignment_query(self, centroids):
        return self._prepare_torch_assignment(centroids)

    def _prepare_torch_assignment(self, centroids):
        if self.spherical and centroids is self._centroids:
            centroids = self._readout()
        return self._execution_backend.prepare_centroids(
            centroids, self._input_dtype, self.spherical
        )

    def _select_statistics_dtype(self, dtype):
        pass

    def _choose_splits(self, rng):
        if self._unused_threshold is None:
            if self._frozen():
                return None, None
            if self._device_check():
                if self._handover is not None:
                    # Rows are counted on the host; the check reads them here.
                    self._handover["done"].fill_(self._handover["rows"])
                return self._removal_check_graph()
            stats = self._removal_stats
            source, target = self._choose_removal_splits(*torch.stack((
                stats[:, -1].double(), self._usage.double(),
                torch.linalg.vector_norm(stats[:, :-1], dim=1, dtype=torch.float64))).cpu().numpy())
            return (None, None) if source is None else tuple(
                torch.from_numpy(ids).to(self._usage.device) for ids in (source, target))
        # Sample donors without replacement in proportion to their usage.

        threshold = self._usage.sum() * self._unused_threshold / self.k
        unused = self._usage < threshold
        donors = (~unused) & (self._usage > 0)
        nunused, ndonors = torch.stack((unused.sum(), donors.sum())).tolist()
        nsplit = min(nunused, ndonors)
        if nsplit == 0:
            return None, None

        uniform = torch.rand(
            self.k,
            dtype=self._usage.dtype,
            device=self._usage.device,
            generator=rng,
        ).clamp_min(torch.finfo(self._usage.dtype).tiny)
        if self._usage.is_cuda:
            # Fixed shapes avoid compaction's host reads.
            keys = torch.where(
                donors,
                self._usage.log() - (-uniform.log()).log(),
                float("-inf"),
            )
            source = torch.topk(keys, nsplit).indices
            target = torch.sort(
                unused.byte(), descending=True, stable=True
            ).indices[:nsplit]
        else:
            keys = torch.full_like(self._usage, float("-inf"))
            keys[donors] = (
                self._usage[donors].log() - (-uniform[donors].log()).log()
            )
            source = torch.topk(keys, nsplit).indices
            target = unused.nonzero(as_tuple=False).flatten()[:nsplit]

        return source, target

    @property
    def n_splits(self):
        """Splits so far; device-side splits are read (synchronizing) on access."""
        return self._n_splits + (0 if self._check_splits is None else int(self._check_splits))

    @n_splits.setter
    def n_splits(self, value):
        self._n_splits = value
        if getattr(self, "_check_splits", None) is not None:
            self._check_splits.zero_()

    def _device_check(self):
        """Pair and split on the device when the Triton split kernels apply."""
        if self._check_async is None:
            self._check_async = (
                self._usage.is_cuda
                and getattr(self._execution_backend.split, "__func__", None) is TritonBackend.split
                and getattr(self._apply_splits, "__func__", None) is Emeans._apply_splits
                and self._input_dtype in (None, torch.float32, torch.bfloat16) and self.d <= 8192)
        return self._check_async

    def _removal_check(self):
        """``_choose_removal_splits`` then ``_apply_splits`` on the device, without host reads."""
        from .triton_assignment import _write_float_splits, check_pair, nearest_other, removal_split

        stats = self._removal_stats
        removal, ids = stats[:, -1].sort(stable=True)
        candidate = self._removal_candidates_device(removal.double())
        # Candidates by (R, index) first, then the rest; position = slot, or count if not one.
        count = candidate.sum()
        slot = torch.where(candidate, candidate.cumsum(0) - 1, count + (~candidate).cumsum(0) - 1)
        order = torch.empty_like(ids).scatter_(0, slot, ids)
        position = torch.empty_like(ids).scatter_(0, ids, torch.where(candidate, slot, count))
        centroids = self._readout(force=True)
        nearest = nearest_other(centroids, order, count)
        donor = self._usage > 0
        # Donors by (-|U|, index) first; the order of the rest does not matter.
        gain = torch.linalg.vector_norm(stats[:, :-1], dim=1, dtype=torch.float64).float()
        donors = torch.where(donor, -gain, float("inf")).sort(stable=True).indices
        victims, chosen, moves = check_pair(nearest, position, count, order, donors, donor, self._check_splits,
                                            centroids)
        _write_float_splits(self._centroids, self._usage, chosen, victims, self._input_dtype, _SPLIT_EPSILON,
                            count=moves)
        removal_split(stats, self._axes, chosen, victims, count=moves, moved=self._moved)
        if self._handover is not None:
            # O(k): when each revived centroid moved. Entries of victims past
            # the pair count are not written, so they are masked out.
            h = self._handover
            valid = torch.arange(len(victims), device=victims.device) < moves
            h["moved_at"].scatter_reduce_(0, torch.where(valid, victims, 0),
                                          torch.where(valid, h["done"], -1), "amax")

    def _removal_candidates_device(self, removal):
        median = (removal[(self.k - 1) // 2] + removal[self.k // 2]) / 2
        return (removal <= 0) | (removal < _REMOVAL_THRESHOLD * torch.where(removal[-1].isnan(), float("nan"), median))

    def _removal_check_graph(self):
        """Run ``_removal_check`` as one CUDA graph per check."""
        if self._check_graph is None:
            device = self._usage.device
            self._check_splits = torch.zeros((), dtype=torch.int64, device=device)
            self._removal_check()  # Also compiles the kernels before capture.
            current, stream = torch.cuda.current_stream(device), torch.cuda.Stream(device)
            graph, self._check_pool = torch.cuda.CUDAGraph(), torch.cuda.MemPool()  # Freed at finalize.
            stream.wait_stream(current)
            try:
                with torch.cuda.stream(stream):
                    graph.capture_begin(pool=self._check_pool.id, capture_error_mode="thread_local")
                    try:
                        self._removal_check()
                    finally:
                        graph.capture_end()
                self._check_graph = graph
            except RuntimeError:
                self._check_graph = False  # Not capturable here: replay eagerly.
            current.wait_stream(stream)
        elif self._check_graph:
            self._check_graph.replay()
        else:
            self._removal_check()
        self._invalidate_readout()
        return None, None

    def _nearest_other(self, ids):
        """Nearest other centroid by FP64 |q - c|^2 of the FP32 centroids, ties to the lower index."""
        centroids, nearest = self._readout(), []
        ids = torch.from_numpy(ids).to(centroids.device)
        if centroids.is_cuda and isinstance(self._execution_backend, TritonBackend):
            from .triton_assignment import nearest_other

            return nearest_other(centroids.contiguous(), ids, torch.tensor(len(ids), device=ids.device)).cpu().numpy()
        for chunk in ids.split(max(1, 2**25 // (self.k * self.d))):
            distance = (centroids[chunk, None].double() - centroids[None].double()).square().sum(-1)
            distance[torch.arange(len(chunk), device=chunk.device), chunk] = float("inf")
            nearest.append(torch.where(distance.isnan(), float("inf"), distance).argmin(dim=1))
        return torch.cat(nearest).cpu().numpy()

    def _apply_splits(self, source, target):
        self._execution_backend.split(
            self._centroids,
            self._usage,
            source,
            target,
            self._input_dtype,
            _SPLIT_EPSILON,
        )
        self._split_removal_statistics(source, target)
        self._n_splits += len(source)
        self._invalidate_readout()
        return len(source)

    def _split_removal_statistics(self, source, target):
        if self._removal_stats is not None and not (
                isinstance(self._execution_backend, TritonBackend) and self._execution_backend.removal_split(
                    self._removal_stats, self._axes, source, target, None, self._moved)):
            self._removal_stats[source] /= 2
            self._removal_stats[target] = self._removal_stats[source]
            self._axes[target] = self._axes[source]
            self._moved[source] = self._moved[target] = True
        if self._handover is not None:
            h = self._handover
            h["moved_at"][torch.as_tensor(target, device=h["done"].device)] = h["rows"]

    def _handover_start(self, labels, x):
        """Set up the closing pass's reassignment before training starts, so
        the device revival check records into it from its first capture.
        ``labels`` (n,) holds the captured assignments. A revival only records,
        per revived centroid, how many closing-pass rows were assigned when it
        moved (``moved_at``, O(k)); records from earlier passes are cleared when
        the closing pass starts, and the rows are sorted out once, at the end.
        Closing-pass rows are counted on the host (``rows``) and copied to the
        device (``done``) only when the device check runs."""
        device = labels.device
        self._handover = dict(
            labels=labels, x=x,
            moved_at=torch.full((self.k,), -1, dtype=torch.int64, device=device),
            done=torch.zeros((), dtype=torch.int64, device=device), rows=0)

    def _handover_finish(self, order, rows):
        """One pass over the ``rows`` closing-pass rows in visiting ``order``
        finds those assigned before their centroid moved; they are assigned
        again like a training batch (same backend, row count and outputs,
        against the current centroids), so this costs their share of one epoch,
        in whole chunks. They join their new centroid's closing sums; the
        centroids they left moved, keep their EMA positions, and their sums are
        not used."""
        h, self._handover = self._handover, None
        labels, x, moved_at = h["labels"], h["x"], h["moved_at"]
        if not bool((moved_at >= 0).any()):  # O(k): nothing moved in this pass.
            self.closing_handover_stats = dict(searched=0)
            return
        stale_at = torch.zeros(rows, dtype=torch.bool, device=order.device)  # by visiting rank
        for begin in range(0, rows, _HANDOVER_CHUNK):
            index = order[begin:min(begin + _HANDOVER_CHUNK, rows)]
            rank = torch.arange(begin, begin + len(index), device=index.device)
            stale_at[begin:begin + len(index)] = rank < moved_at[labels[index]]
        search = order[:rows][stale_at]
        if len(search):
            centroids, centroid_norm = self._prepare_torch_assignment(self._centroids)
            rows_per_chunk = min(self.recipe["assignment_chunk_size"], self.recipe["batch_size"])
            second_out = x.new_empty(rows_per_chunk, dtype=torch.float32)
            for begin in range(0, len(search), rows_per_chunk):
                chunk = search[begin:begin + rows_per_chunk]
                padded = torch.cat((chunk, chunk[-1:].expand(rows_per_chunk - len(chunk))))
                _, nearest = self._assign_rows(self._prepare_rows(x[padded]), centroids, centroid_norm,
                                               None, None, second_out)
                labels[chunk] = nearest[:len(chunk)]
            for chunk in search.split(_HANDOVER_CHUNK):
                rows_x = self._prepare_rows(x[chunk])
                self._pass_sums.index_add_(0, labels[chunk], rows_x.to(self._pass_sums.dtype))
                self._pass_counts.index_add_(0, labels[chunk], self._pass_counts.new_ones(len(chunk)))
        self.closing_handover_stats = dict(searched=len(search))

    def _assign_rows(
        self,
        x,
        centroids,
        centroid_norm=None,
        distance_out=None,
        labels_out=None,
        second_out=None,
    ):
        """Assign floating rows against represented centroids."""
        return self._execution_backend.assign(
            x,
            centroids,
            centroid_norm,
            self.spherical,
            distance_out,
            labels_out,
            second_out,
        )

    def begin(self, global_n, init_centroids):
        """Initialize a run with caller-owned batches and explicit FP32 state.

        Call once on a fresh object.
        ``init_centroids`` has shape (k, d); its device selects
        execution. It is copied. Each epoch must supply the resolved batch
        sizes for ``recipe["epoch_rows"][epoch]``, including the final
        remainder,
        then call :meth:`end_epoch`.
        Loading, ordering, transfers and prefetch are the caller's
        responsibility.
        """
        if self._failed or self.recipe is not None:
            raise RuntimeError("create a new object for another run or after failure")
        n = int(global_n)
        centroids = init_centroids.detach().float().contiguous()
        self._failed = True
        self._centroids = centroids.clone()
        if self.spherical:
            from .e_means_spherical import normalize_into

            normalize_into(self._centroids, self._centroids)
        self._usage = centroids.new_full((self.k,), _INIT_USAGE)
        self._batch_stats = centroids.new_empty((self.k, self.d + 1))
        self._split_rng = _torch_generator(centroids.device, self.seed)
        self._resolve_removal_revival()
        if self._unused_threshold is None:
            self._initialize_removal_statistics()
        self._resolve_run_recipe(n)
        self.recipe.update(
            assignment_backend=self._resolved_assignment_backend,
            assignment_backend_requested=self.assignment_backend,
            device=str(centroids.device),
            centroid_chunk_size=self.centroid_chunk_size,
        )
        self._execution_backend.clear_statistics()
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
        _synchronize(centroids.device)
        self._start_time = time.perf_counter()
        self._failed = False
        self._header_due, self._quiet = True, False

    def _initialize_removal_statistics(self):
        # EMAs [U | R] and batch sums [p y | d2 - d1].
        centroids = self._centroids
        self._removal_stats = centroids.new_zeros((self.k, self.d + 1))
        self._batch_removal = torch.zeros_like(self._removal_stats)
        axes = torch.randn(self.k, self.d, dtype=torch.float32, device=centroids.device,
                           generator=self._split_rng)
        self._axes = axes / axes.norm(dim=1, keepdim=True)
        self._moved = centroids.new_zeros(self.k, dtype=torch.bool)

    def _statistics_rows(self, rows):
        return rows

    def _start_epoch(self):
        self._epoch_steps = self._epoch_rows = 0
        if self._moved is not None:  # Only moves during the closing pass count.
            self._moved.zero_()
        self._splits_before = self.n_splits

        if self._pass_counts is None:
            self._pass_counts = self._usage.new_zeros(
                self.k, dtype=torch.float64
            )
            self._objective = self._usage.new_zeros((), dtype=torch.float64)
        else:
            self._pass_counts.zero_()
            self._objective.zero_()
        self._pass_sums = (
            self._centroids.new_zeros(
                (self.k, self.d), dtype=self._batch_stats.dtype
            )
            if self._epoch == len(self.recipe["epoch_rows"]) - 1
            else None
        )

    def accumulate(self, x):
        """Accumulate one chunk against a fixed centroid snapshot.

        Shapes, device, representation and row counts must match the run.
        Complete totals borrow workspace. Objective uses FP64; sums/counts
        follow the implementation's statistics dtype. CUDA rows must fit
        the representation. Preparation detaches and makes rows contiguous;
        CPU rows are converted to FP32. No value validation is performed.
        """
        self._require_running()
        x = self._prepare_input(x)
        return self._accumulate(x.detach())

    def _accumulate(self, x, index=None, labels_out=None):
        # Indexed matrix training gathers only assignment chunks, never B*d.
        n = len(x) if index is None else len(index)
        if n == 0 and self._pending_stats is not None:
            stats = dict(self._pending_stats)
            if (
                self.capture_assignments
                and self._epoch == len(self.recipe["epoch_rows"]) - 1
            ):
                stats["labels"] = x.new_empty(0, dtype=torch.int64)
            return stats
        try:
            stats = self._accumulate_core(x, index, labels_out=labels_out)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        self._pending_stats = stats
        return stats

    def _accumulate_core(self, x, index=None, raw=False, labels_out=None, ema=None):
        n = len(x) if index is None else len(index)
        pending_n = (
            0 if self._pending_stats is None else self._pending_stats["n"]
        )
        target = self._local_batch_size()
        if pending_n == 0:
            self._batch_accumulation = (
                self._batch_stats.new_zeros(
                    self._batch_stats.shape, dtype=torch.float64
                )
                if n > 0 and target > 2**24
                else self._batch_stats.zero_()
            )
            self._batch_objective = self._objective.new_zeros(())
            if self.spherical:
                self._readout(force=raw)
            self._batch_centroids, self._batch_norm = (
                self._prepare_torch_assignment(self._centroids)
                if n
                else (None, None)
            )
            self._execution_backend.begin_batch(
                self._batch_stats,
                min(
                    self.recipe["assignment_chunk_size"],
                    self.recipe["batch_size"],
                ),
            )
        accumulation = self._batch_accumulation
        objective, centroid_norm = self._batch_objective, self._batch_norm
        centroids = self._batch_centroids
        captured_labels = None if labels_out is None else labels_out[:n]
        if (
            captured_labels is None
            and self.capture_assignments
            and self._epoch == len(self.recipe["epoch_rows"]) - 1
        ):
            captured_labels = x.new_empty(n, dtype=torch.int64)
        chunk_size = self.recipe["assignment_chunk_size"]
        for begin in range(0, n, chunk_size):
            end = min(begin + chunk_size, n)
            rows = x[begin:end] if index is None else x[index[begin:end]]
            rows = self._prepare_rows(rows)
            # Nothing moves once frozen: no removal statistics.
            second = (None if self._removal_stats is None or self._frozen()
                      else rows.new_empty(len(rows), dtype=torch.float32))
            distance, labels = self._assign_rows(
                rows, centroids, centroid_norm, None, None, second
            )
            values = self._statistics_rows(rows)
            # A training step whose batch is one chunk may apply the removal EMA here.
            fold = ema is not None and pending_n == 0 and n == target <= chunk_size
            fused = second is not None and isinstance(self._execution_backend, TritonBackend) and (
                self._execution_backend.accumulate_removal(rows, labels, accumulation, distance, second,
                                                           self._axes, self._readout(), self._batch_removal,
                                                           **({"statistics_rows": values} if values is not rows else {}),
                                                           **({"ema": ema} if fold else {})))
            self._removal_folded = fused == "folded"
            if second is not None and not fused:
                # Oja's long axis: p y with y = x - c, and removal cost d2 - d1.
                y = values.float() - self._readout()[labels]
                p = (y * self._axes[labels]).sum(1, keepdim=True)
                self._batch_removal.index_add_(0, labels, torch.cat(
                    (p * y, (second - distance).nan_to_num(0, 0)[:, None]), dim=1))
            if captured_labels is not None:
                captured_labels[begin:end] = labels
            if self.spherical:
                objective += 2 * len(rows)
            objective += distance.sum(dtype=torch.float64)
            if not fused:
                self._execution_backend.accumulate(
                    values,
                    labels,
                    accumulation,
                    target,
                    indexed=index is not None and type(x) is torch.Tensor,
                )
        self._batch_objective = objective
        total_n = pending_n + n
        if total_n == target and accumulation is not self._batch_stats:
            self._batch_stats.copy_(accumulation)
            accumulation = self._batch_stats
        complete = total_n == target or total_n == 0
        stats = {
            "sums": accumulation[:, :-1] if complete else None,
            "counts": accumulation[:, -1] if complete else None,
            "objective": objective,
            "n": total_n,
        }
        if self._removal_stats is not None:
            stats["removal"] = self._batch_removal
        if captured_labels is not None:
            stats["labels"] = captured_labels
        return stats

    def _prepare_statistics(self, stats, expected_n):
        sums, counts, objective = (
            stats["sums"].detach(),
            stats["counts"].detach(),
            stats["objective"].detach(),
        )
        self._select_statistics_dtype(sums.dtype)
        return None, sums, counts, objective

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
        # Capture already-global statistics once before reuse/EMA/revival.
        self._objective += (
            objective  # Local contribution; reduced once per epoch.
        )
        if ids is None:
            self._pass_counts += counts
            if self._pass_sums is not None:
                self._pass_sums += sums
        else:
            self._pass_counts.index_add_(
                0, ids, counts.to(self._pass_counts.dtype)
            )
            if self._pass_sums is not None:
                self._pass_sums.index_add_(0, ids, sums)
        if self._removal_folded:
            self._removal_folded = False
        elif self._removal_stats is not None and not self._frozen() and not (
                isinstance(self._execution_backend, TritonBackend) and self._execution_backend.removal_update(
                    self._removal_stats, self._batch_removal, self._axes, alpha, decay)):
            self._removal_stats.mul_(decay).add_(alpha * self._batch_removal)
            axis = self._removal_stats[:, :-1].double()
            norm = axis.norm(dim=1, keepdim=True)  # FP64: |U|^2 can overflow FP32
            torch.where(norm > 0, (axis / norm).float(), self._axes, out=self._axes)
            self._batch_removal.zero_()

        self._execution_backend.update(
            self._centroids,
            self._usage,
            ids,
            sums,
            counts,
            alpha,
            decay,
            replace_usage,
        )
        self._invalidate_readout()

    def _epoch_totals(self):
        objective = self._epoch_objective()
        report_counts = self._pass_counts.double()
        return torch.stack(
            (objective, torch.dot(report_counts, report_counts))
        ).tolist()

    def _spherical_closing(self):
        from .e_means_spherical import normalize_into

        result = self._centroids.new_empty(self._centroids.shape)
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

            dtype = (
                torch.float64 if sums.dtype == torch.float64 else torch.float32
            )
            mean = sums.to(dtype) / counts.clamp_min(1).to(dtype)[:, None]
            closed = counts > 0 if self._moved is None else (counts > 0) & ~self._moved[begin:end]
            mean = torch.where(closed[:, None], mean, raw)
            normalize_into(mean, result[begin:end])
        return result

    def finalize(self):
        """Set and return FP32 closing means from the final pass's assignments.

        Never-assigned centroids retain their current positions, and so do
        centroids moved during that pass. With ``capture_assignments``,
        ``train`` first assigns again the rows held by a centroid revived during
        that pass. All epochs must be complete; repeated finalization returns
        the same result.
        """
        if self._failed:
            raise RuntimeError("create a new object after execution failure")
        if self.centroids is not None:
            return self.centroids
        self._failed = True
        # Centroids moved during the closing pass keep their EMA position.
        centroids = self._centroids
        objectives = [stat["obj"] for stat in self.iteration_stats]
        if self.spherical:
            self.centroids = self._spherical_closing()
        else:
            self.centroids = _closing_centroid_values(
                self._pass_sums, self._pass_counts, centroids, self._moved,
            )
        self.obj = torch.tensor(
            objectives, dtype=torch.float64, device=centroids.device
        )
        self._centroids = self._usage = self._batch_stats = None
        self._removal_stats = self._axes = self._batch_removal = self._moved = None
        self._n_splits = self.n_splits
        self._check_graph = self._check_pool = self._check_async = self._check_splits = None
        self._unit_centroids = self._unit_version = None
        self._pass_sums = self._pass_counts = self._objective = None
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_norm = None
        self._batch_centroids = None
        self._execution_backend.clear_statistics()
        self._failed = False
        return self.centroids

    def _replay_representation_eligible(self):
        return True

    def _capture_operations_match(self, owner=None):
        # Graph replay executes recorded GPU work without these Python calls.
        # Overrides use the ordinary eager call path.
        owner = Emeans if owner is None else owner
        return all(_same_operation(getattr(self, name), getattr(owner, name), self)
                   for name in ("_accumulate", "update", "_invalidate_readout",
                                "_prepare_statistics", "_select_statistics_dtype",
                                "_accumulate_core", "_apply_statistics", "_assign_rows",
                                "_prepare_rows", "_prepare_torch_assignment",
                                "_local_batch_size")) and (
            not self.spherical or _same_operation(self._readout, owner._readout, self))

    def _chunked_capture_matches(self, owner=None):
        # Chunked label capture calls accumulation for each assignment chunk,
        # removes its labels, then updates once for the logical batch.
        # Overrides keep the ordinary batch call and its statistics intact.
        owner = Emeans if owner is None else owner
        return all(_same_operation(getattr(self, name), getattr(owner, name), self)
                   for name in ("_accumulate", "_accumulate_core", "update", "_prepare_statistics"))

    def _replay_eligible(self, x):
        # Replay requires a fitted (batch, alpha) pair and the default effective
        # chunk sizes. Either fitted law qualifies, so an explicit recipe equal to
        # the throughput or the quality law keeps replay whatever batch_law
        # resolved to. Supporting other recipes is TODO.
        if (
            not self._capture_operations_match()
            or not x.is_cuda
            or (self.recipe["batch_size"] > 2**24)
            or (self.recipe["steps_per_pass"] < 3)
        ):
            return False
        fitted = [
            _resolve_recipe(self._n, self.k, self.nepochs, law, None, None)
            for law in ("throughput", "quality")
        ]
        batch, alpha = self.recipe["batch_size"], self.recipe["alpha0"]
        if (batch, alpha) not in fitted:
            return False
        backend = self._execution_backend
        # An installed backend with a different assign callback trains eagerly.
        if not any(_same_operation(backend.assign, assign, backend) for assign in (
                TorchBackend.assign, TritonBackend.assign, CuteBackend.assign)):
            return False
        if backend.name == "native" and min(self.k, backend.centroid_chunk_size) != min(self.k, 4096):
            return False
        if min(batch, self.recipe["assignment_chunk_size"]) != min(
                batch, default_chunk_size(backend.name, self._centroids)):
            return False
        if not self._replay_representation_eligible():
            return False
        return x.is_contiguous()

    def _capture_step(
        self, x, index, rate, batches, epoch_rates, position, labels_out=None
    ):
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        graph = torch.cuda.CUDAGraph()
        # Each capture, the frozen re-capture included, gets a fresh pool that lives until its graph is retired.
        pool = torch.cuda.MemPool()
        cached = self._execution_backend.capture_state()
        captured = False
        stats = None
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                self._execution_backend.initialize_capture()
                # The context manager empties process caches; use primitives.
                graph.capture_begin(
                    pool=pool.id, capture_error_mode="thread_local"
                )
                try:
                    torch.index_select(
                        batches, 0, position, out=index.unsqueeze(0)
                    )
                    torch.index_select(
                        epoch_rates, 0, position, out=rate.unsqueeze(0)
                    )
                    # Explicit staging is recorded even before the final epoch.
                    stats = self._accumulate_core(
                        x, index, raw=True, labels_out=labels_out,
                        **({} if self._removal_stats is None else {"ema": (self._removal_stats, rate[0], rate[1])}),
                    )
                    self._apply_statistics(
                        None,
                        stats["sums"],
                        stats["counts"],
                        stats["objective"],
                        rate[0],
                        rate[1],
                    )
                    position.add_(1)
                finally:
                    graph.capture_end()
            captured = True
            current.wait_stream(stream)
        finally:
            self._execution_backend.finish_capture(cached, captured)
            if not captured:
                # No retry/rollback of failed capture or execution errors.
                stats = None
                graph.reset()

        return graph, cached, pool, stats and stats["sums"]

    def train(self, x, init_centroids=None):
        """Train on x, or on a random subset when max_points_per_centroid binds.

        Return centroids and last-pass labels for every row of x (None with
        capture disabled); rows left out of the subset get label -1.
        """
        keep = self._training_subset(len(x))
        if keep is None:
            return self._train_rows(x, init_centroids)
        index = torch.as_tensor(keep, device=x.device)
        _, sampled = self._train_rows(x[index], init_centroids)
        if sampled is not None:
            try:
                labels = torch.full(
                    (len(x),), -1, dtype=sampled.dtype, device=sampled.device
                )
                labels[index.to(sampled.device)] = sampled
            except BaseException:
                self._failed = True
                self.assignments = None
                raise
            self.assignments = labels
        return self.centroids, self.assignments

    def _train_rows(self, x, init_centroids=None):
        """Train a dense supported matrix through the same caller-driven loop.

        Return centroids and last-pass labels (None with capture disabled).
        Objective history remains in obj. Initialization and epoch shuffles
        use the existing local seeded policy. No whole logical batch is
        gathered.
        Explicit initial centroids must be on x's device.
        """
        start, given_init = time.perf_counter(), init_centroids is not None
        x = x.detach()
        n, d = x.shape
        device = x.device
        self._resolve_input_dtype(x)
        self._resolve_assignment_backend(x)
        self._execution_backend.require_training(device)
        if init_centroids is None:
            init_centroids = (
                x
                if n == self.k
                else x[
                    torch.randperm(
                        n,
                        generator=_torch_generator(device, self.seed),
                        device=device,
                    )[: self.k]
                ]
            )
            init_centroids = self._initial_centroid_values(init_centroids)
        self.begin(n, init_centroids)
        x = self._prepare_input(x)
        # Cost-based revival can move a centroid during the closing pass;
        # captured rows it held are then assigned again at the end.
        self._handover_planned = bool(
            self.closing_handover and self.capture_assignments and self._removal_stats is not None
            and (self._input_dtype is None or self._input_dtype.is_floating_point))
        self.recipe["closing_handover"] = "search" if self._handover_planned else None
        self._report_header_once()
        if self._reports():
            self._report_preprocessing(start, given_init, n)
        rates = None
        order = index = rate = epoch_rates = position = batches = None
        graph_has_sums = False
        graph = cached = _pool = closing = None
        captured_bindings = ()
        assignments = batch_labels = None
        try:
            if self._replay_eligible(x):
                alphas = [
                    _scheduled_alpha(
                        self.recipe["alpha0"], step, self.recipe["total_steps"]
                    )
                    for step in range(self.recipe["total_steps"])
                ]
                # Independently round the host-computed alpha and decay to FP32.
                rates = torch.tensor(
                    [(a, 1.0 - a) for a in alphas],
                    dtype=torch.float32,
                    device=device,
                )
                index = torch.empty(
                    self.recipe["batch_size"], dtype=torch.int64, device=device
                )
                rate = torch.empty(2, dtype=torch.float32, device=device)
                epoch_rates = rate.new_empty(
                    (self.recipe["steps_per_pass"], 2)
                )
                position = torch.zeros(1, dtype=torch.int64, device=device)
            if self.capture_assignments:
                assignments = torch.empty(n, dtype=torch.int64, device=device)
                if self.recipe["epoch_rows"][-1] < n or self._handover_planned:
                    assignments.fill_(-1)
                if rates is not None:
                    batch_labels = torch.empty_like(index)
            if self._handover_planned:
                self._handover_start(assignments, x)
            for epoch, rows in enumerate(self.recipe["epoch_rows"]):
                closing_pass = epoch == len(self.recipe["epoch_rows"]) - 1
                handover = closing_pass and self._handover_planned
                if handover:
                    # Records from earlier passes do not count.
                    self._handover["moved_at"].fill_(-1)
                    self._handover["rows"] = 0
                shuffled = torch.randperm(
                    n,
                    generator=_torch_generator(
                        device, _epoch_seed(self.seed, epoch)
                    ),
                    device=device,
                )
                if rates is None or order is None:
                    order = shuffled
                else:
                    # Capture reads this storage; eager primes/tails use its new
                    # values.
                    order.copy_(shuffled)
                shuffled = None
                primed = graph is not None
                if rates is not None:
                    # Stable storage, original epoch order and independently
                    # rounded rates.
                    batches = order[: n - n % len(index)].view(-1, len(index))
                    # Fixed full-N tables survive graph reuse; only the active
                    # prefix is indexed during a shorter epoch.
                    steps = self.recipe["epoch_steps"][epoch]
                    epoch_rates[:steps].copy_(
                        rates[self._step : self._step + steps]
                    )
                    position.zero_()
                for begin in range(0, rows, self.recipe["batch_size"]):
                    batch = order[
                        begin : min(begin + self.recipe["batch_size"], rows)
                    ]
                    replay = (
                        rates is not None
                        and len(batch) == len(index)
                        and alphas[self._step] != 1
                    )
                    if replay and graph is not None and graph_stats and self._frozen() and (
                            self.recipe["epoch_rows"][-1] * self.d >= 2**28):
                        # Frozen: re-capture without removal statistics, after one eager batch
                        # (re-capturing costs ~1 ms; smaller final epochs keep the training graph).
                        graph.reset()
                        graph = cached = closing = captured_bindings = _pool = None
                        primed = False
                    if replay and primed:
                        self._require_running()
                        if graph is None:
                            graph_has_sums = self._pass_sums is not None
                            graph_stats = self._removal_stats is not None and not self._frozen()
                            graph, cached, _pool, closing = self._capture_step(
                                x,
                                index,
                                rate,
                                batches,
                                epoch_rates,
                                position,
                                batch_labels,
                            )
                            captured_bindings = (
                                self._execution_backend.capture_bindings(
                                    cached
                                )
                            )
                        self._execution_backend.validate_capture(cached)
                        graph.replay()
                        if (
                            assignments is not None
                            and epoch == len(self.recipe["epoch_rows"]) - 1
                        ):
                            # Consume captured row IDs before the next replay.
                            assignments[index] = batch_labels
                            if handover:
                                self._handover["rows"] += len(batch)
                        if self._pass_sums is not None and not graph_has_sums:
                            # Same dense ATen closing add, before revival/next
                            # stats zero.
                            self._pass_sums.add_(closing)
                        # Replay writes do not execute Python mutation
                        # notifications.
                        torch.autograd.graph.increment_version(
                            (self._centroids, self._usage)
                        )
                        self._finish_step(len(batch))
                    elif (
                        self._chunked_capture_matches()
                        and rates is None
                        and assignments is not None
                        and epoch == len(self.recipe["epoch_rows"]) - 1
                    ):
                        # Keep eager resident capture bounded by assignment
                        # chunks.
                        # All chunks still share one logical batch's fixed
                        # snapshot
                        # and one update; public caller labels remain caller-
                        # owned.
                        chunk_size = self.recipe["assignment_chunk_size"]
                        for offset in range(0, len(batch), chunk_size):
                            chunk = batch[offset : offset + chunk_size]
                            stats = self._accumulate(x, chunk)
                            # Move this chunk's labels from pending statistics
                            # to the public result.
                            labels = stats.pop("labels")
                            assignments[chunk] = labels
                            labels = None
                        if handover:
                            self._handover["rows"] += len(batch)
                        self.update(stats)
                    else:
                        stats = self._accumulate(x, batch, batch_labels)
                        if assignments is not None and closing_pass:
                            # Before update(): a revival there must see them.
                            assignments[batch] = stats["labels"]
                            if handover:
                                self._handover["rows"] += len(batch)
                        self.update(stats)
                        if rates is not None:
                            position.add_(1)
                        primed = primed or replay
                self.end_epoch()
                # Reporting drains work; retain the owned graph/pool for the
                # next epoch.
            if self._handover is not None:
                self._handover_finish(order, self.recipe["epoch_rows"][-1])
        except BaseException:
            self._failed = True
            self.assignments = None
            self._handover = None
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
                del captured_bindings, _pool
                graph = cached = closing = None
                order = shuffled = batches = None
                index = rate = epoch_rates = position = None
                batch_labels = None
        self.finalize()
        self.assignments = assignments
        return self.centroids, self.assignments

    def assign(self, x):
        """Assign in this call's representation and record assignment_recipe.

        Evaluation keeps the training recipe and captured labels unchanged.
        Automatic CUDA FP32 queries use BF16. QuantizedEmeans(quantizer=None)
        keeps the native query dtype; explicit codecs keep their representation.
        """
        import copy

        self.assignment_recipe = None
        evaluation = copy.copy(self)
        evaluation._centroids = None
        evaluation._input_dtype = None
        evaluation._cast_dtype = None
        evaluation._resolved_assignment_backend = None
        evaluation._execution_backend = (
            self._query_backend or self._execution_backend.fork_query())
        evaluation.recipe = None
        result = evaluation._assign(x)
        self._execution_backend.finish_query(evaluation._execution_backend)
        self._query_backend = evaluation._execution_backend
        self.assignment_recipe = dict(
            assignment_backend_requested=self.assignment_backend,
            assignment_backend=evaluation._assignment_backend,
            representation=str(evaluation._input_dtype).removeprefix("torch."),
            device=str(x.device),
            assignment_chunk_size=evaluation._get_assignment_chunk_size(),
        )
        return result

    def _assign(self, x):
        """Return squared L2 distances and centroid indices.

        Distances are reported in model coordinates, using the row and
        centroid representations selected by the implementation.
        Spherical mode returns represented inner-product similarity instead:
        callers normalize input before conversion; rounded unit directions
        approximate cosine without repairing represented norm drift.
        """
        if self._failed:
            raise RuntimeError("create a new object after execution failure")

        x = self._prepare_input(x)
        centroids = self._execution_backend.query_centroids(self.centroids)
        assignment_chunk_size = self._get_assignment_chunk_size()

        distance = torch.empty(len(x), dtype=torch.float32, device=x.device)
        labels = torch.empty(len(x), dtype=torch.int64, device=x.device)
        centroids, centroid_norm = self._prepare_assignment_query(centroids)

        for begin in range(0, len(x), assignment_chunk_size):
            end = min(begin + assignment_chunk_size, len(x))
            rows = self._prepare_rows(x[begin:end])
            if self._execution_backend.writes_outputs:
                self._assign_rows(
                    rows,
                    centroids,
                    centroid_norm,
                    distance[begin:end],
                    labels[begin:end],
                )
            else:
                distance[begin:end], labels[begin:end] = self._assign_rows(
                    rows, centroids, centroid_norm
                )
        if self.spherical:
            distance *= -0.5
        return distance, labels
