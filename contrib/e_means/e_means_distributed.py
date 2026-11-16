# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Replicated-centroid CPU/GPU E-means with caller-owned communication groups.

Sparse packing/length discovery and coordinated revival can synchronize with
the host. Stock synchronous collectives order consumption on the caller's
current stream.
"""

import copy

import numpy as np
import torch
import torch.distributed as dist

from .base import _divisibility_error
from .e_means_quantized import QuantizedEmeans
from .e_means import NumpyEmeans


def _check_group(group):
    if not dist.is_initialized():
        raise RuntimeError(
            "initialize the process group before DistributedEmeans"
        )
    if not isinstance(group, dist.ProcessGroup) or dist.get_rank(group) < 0:
        raise ValueError("group must contain the caller")
    return dist.get_world_size(group)


_AXIS_HOST_BYTES = 1 << 25


def _fill_initial_axes(axes, rng):
    """Host FP64 Gaussian directions, rounded once into required FP32 state."""
    k, d = axes.shape
    capacity = max(1, (_AXIS_HOST_BYTES - 8) // 24)  # Gaussian, norm and conversion scratch.
    numpy_state = isinstance(axes, np.ndarray)

    def write(destination, values):
        rounded = values.astype("float32")
        if numpy_state:
            np.copyto(destination, rounded)
        else:
            destination.copy_(torch.from_numpy(rounded))

    if d <= capacity:
        rows = max(1, _AXIS_HOST_BYTES // (24 * d + 8))
        for begin in range(0, k, rows):
            end = min(begin + rows, k)
            values = rng.standard_normal((end - begin, d))
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            write(axes[begin:end], values)
        return
    # Even an oversized row has bounded scratch: replay only random draws,
    # never caller data, after obtaining its host FP64 norm.
    numpy_rng = isinstance(rng, np.random.RandomState)
    for row in range(k):
        state = rng.get_state() if numpy_rng else copy.deepcopy(rng.bit_generator.state)
        square = np.float64(0)
        for begin in range(0, d, capacity):
            values = rng.standard_normal(min(capacity, d - begin))
            square += np.square(values).sum(dtype="float64")
        del values
        if numpy_rng:
            rng.set_state(state)
        else:
            rng.bit_generator.state = state
        norm = np.sqrt(square)
        for begin in range(0, d, capacity):
            end = min(begin + capacity, d)
            write(axes[row, begin:end], rng.standard_normal(end - begin) / norm)


class _ReplicatedCentroids:
    """E-means with equal row contributions and replicated FP32 centroids.

    ``group`` is explicit and caller-owned; ``transport`` is fixed at construction.
    All ranks must establish matching recipes/codecs and collective schedules. Auto does not add partial-accumulation communication. Begin
    takes same-shape/device tensors on each rank; group rank zero supplies their
    values. CPU state uses Gloo and FP32 statistics; CUDA retains
    its supported schemas. Use the caller-driven lifecycle, not the local trainer.

    Accumulate supplied chunks until each rank completes its equal local share
    of the prescribed global batch (including the final tail). Chunk sizes and
    call counts may differ across ranks; partial accumulation does not
    communicate. Update reduces each completed contribution and advances the
    global batch once. With capture_assignments enabled, final-pass labels
    belong to each supplied chunk; this class has no dataset-wide assignments.
    Floating inputs use the local cost rule with common CPU-initialized axes.
    Encoded inputs retain their existing mass rule; their cost vertical is unfinished.
    """

    def __init__(self, *args, group, transport="auto", **kwargs):
        ranks = _check_group(group)
        if kwargs.get("max_points_per_centroid") is not None:
            raise ValueError(
                "max_points_per_centroid applies to train(x); subsample the "
                "rows you pass to accumulate instead"
            )
        super().__init__(*args, **kwargs)
        self._group, self._group_size, self._transport = group, ranks, transport
        self._rank = dist.get_rank(group)
        self._axis_seed = None

    def _reports(self):
        return self.verbose and self._rank == 0

    def _describe_parallel(self):
        cuda = not isinstance(self._centroids, np.ndarray) and self._centroids.is_cuda
        return "Using %d %s with class %s" % (
            self._group_size, "GPUs" if cuda else "CPU processes", type(self).__name__)

    def begin(self, global_n, init_centroids):
        if self._failed or self.recipe is not None:
            raise RuntimeError("create a new object for another run or after failure")
        quantizer = getattr(self, "quantizer", None)
        if quantizer is not None and not self._automatic:
            from .e_means_quantized import check_codec_fit

            try:
                check_codec_fit(self._group, quantizer, init_centroids.device)
            except BaseException:
                self._failed = True
                self.assignments = None
                raise
        numpy_state = isinstance(self, NumpyEmeans)
        device = torch.device("cpu") if numpy_state else init_centroids.device
        if device.type == "cpu" and dist.get_backend(self._group) != "gloo":
            self._failed = True
            raise ValueError("CPU DistributedEmeans requires a Gloo group")
        if (self.unused_threshold is None and self.k > 1
                and getattr(self, "_input_dtype", None) in
                (None, torch.float32, torch.float16, torch.bfloat16)):
            try:
                # Preserve existing class seed ranges, without drawing from
                # either the model generator or the ambient RNG.
                if numpy_state:
                    np.random.RandomState(self.seed)
                    seed = self.seed
                else:
                    seed = torch.Generator().manual_seed(self.seed).initial_seed()
                packed = torch.tensor(list(seed.to_bytes(8, "little")),
                                      dtype=torch.uint8, device=device)
                dist.broadcast(packed, src=dist.get_global_rank(self._group, 0), group=self._group)
                self._axis_seed = int.from_bytes(bytes(packed.cpu().tolist()), "little")
            except BaseException:
                self._failed = True
                self.assignments = None
                raise
        super().begin(global_n, init_centroids)
        ranks, batch = self._group_size, self.recipe["batch_size"]
        if (
            min(batch, self._n) % ranks
            or self._n % ranks
            or any(rows % ranks for rows in self.recipe["epoch_rows"])
        ):
            self._failed = True
            raise _divisibility_error(self.recipe, self._n, batch, ranks)

    def _initial_centroid_state(self, centroids):
        # The superclass copy is the model's only initial-state allocation.
        # Root values must be present before any genuine derived-state hook.
        state = super()._initial_centroid_state(centroids)
        try:
            dist.broadcast(torch.from_numpy(state) if isinstance(state, np.ndarray) else state,
                           src=dist.get_global_rank(self._group, 0), group=self._group)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        return state

    def _local_batch_size(self):
        return super()._local_batch_size() // self._group_size

    def _complete_statistics(self, accumulation):
        numpy_state = isinstance(self._batch_stats, np.ndarray)
        floating = (np.issubdtype(self._batch_stats.dtype, np.floating)
                    if numpy_state else self._batch_stats.is_floating_point())
        double = np.dtype("float64") if numpy_state else torch.float64
        return accumulation if floating and accumulation.dtype == double else super()._complete_statistics(accumulation)

    def _accumulate(self, *args, **kwargs):
        stats = super()._accumulate(*args, **kwargs)
        if (isinstance(self._batch_stats, torch.Tensor) and stats is self._pending_stats and stats["sums"] is not None
                and stats["sums"].dtype != self._batch_stats.dtype):
            stats.setdefault("stat_schema", self._batch_stats.dtype)
        return stats

    def _initial_removal_axes(self):
        numpy_state = isinstance(self._centroids, np.ndarray)
        if numpy_state:
            # Keep NumPy's own Gaussian stream and consumption.
            self._split_rng.seed(self._axis_seed)
            rng = self._split_rng
            axes = np.empty((self.k, self.d), dtype="float32")
        else:
            # A new replica-only stream; CUDA launch geometry never defines it.
            rng = np.random.Generator(np.random.PCG64(self._axis_seed))
            axes = self._centroids.new_empty((self.k, self.d))
        _fill_initial_axes(axes, rng)
        return axes

    def _prepare_statistics(self, stats, expected_n):
        if (self.unused_threshold is None and self.k > 1
                and self._removal_stats is None and getattr(self, "_input_dtype", None) is None):
            raise ValueError("unused_threshold=None needs typed input statistics from this model's accumulate")
        if stats["n"] != expected_n // self._group_size:
            raise ValueError("stats n must match the prescribed batch size")
        _, sums, counts, objective = super()._prepare_statistics(
            stats, expected_n // self._group_size
        )
        try:
            removal = self._batch_removal if self._removal_stats is not None and not self._frozen() else None
            workspace = self._batch_stats
            numpy_state = isinstance(workspace, np.ndarray)
            double, integer = (np.dtype("float64"), np.dtype("int64")) if numpy_state else (torch.float64, torch.int64)
            floating = np.issubdtype(workspace.dtype, np.floating) if numpy_state else workspace.is_floating_point()
            if floating and workspace.dtype != double and (sums.dtype == double or counts.dtype in (double, integer)):
                workspace = self._batch_accumulation
                if workspace is None or workspace.dtype != double or workspace.shape != self._batch_stats.shape:
                    workspace = (np.empty(self._batch_stats.shape, dtype=double) if numpy_state
                                 else self._batch_stats.new_empty(self._batch_stats.shape, dtype=double))
                    self._batch_accumulation = workspace
            ids, sums, counts = _combine(sums, counts, workspace, self._group,
                                        self._transport, self.recipe["batch_size"], removal=removal)
            if sums.dtype != self._batch_stats.dtype:
                ready = self._batch_stats[:len(sums)]
                ready[:, :-1], ready[:, -1] = sums, counts
                sums, counts = ready[:, :-1], ready[:, -1]
        except BaseException:
            self._failed = True
            raise
        return ids, sums, counts, objective

    def update(self, stats):
        if (
            self._pending_stats is not None
            and 0 < self._pending_stats["n"] < self._local_batch_size()
        ):
            raise ValueError("complete the pending batch before update")
        super().update(stats)

    def finalize(self):
        result = super().finalize()
        self._axis_seed = None
        return result

    def _revive_unused(self, rng):
        return _revive(self, self._group)

    def _epoch_objective(self):
        return _epoch_report(self, self._group)

    def train(self, x, init_centroids=None):
        raise NotImplementedError(
            "DistributedEmeans uses begin/accumulate/update/end_epoch/finalize, not train(x)"
        )


class DistributedEmeans(_ReplicatedCentroids, QuantizedEmeans):
    """Replicated tensor E-means; use the caller-driven distributed lifecycle.

    Wide completed contributions expose authoritative FP64 S/N. Their
    ``stat_schema`` declares normal Float32/Float64 statistics storage;
    copies retain it and edits supply the values reduced. Without it,
    external S/N dtype selects storage as usual. An already-FP64 ``double()``
    is a no-op: set the declaration to Float64 to request Float64 storage.
    Ranks must supply matching raw precision and declared storage; differing
    declarations may leave different schemas even if the SUM succeeds.
    """


class DistributedNumpyEmeans(_ReplicatedCentroids, NumpyEmeans):
    """Replicated NumPy/Faiss E-means with CPU/Gloo communication.

    Numerical state remains NumPy; conversion occurs only at collective
    boundaries. Recipes, equal row shares and communication match
    DistributedEmeans. Use begin/accumulate/update/end_epoch/finalize.
    Wide contributions expose raw FP64 S/N; global completion narrows once
    into fixed FP32 storage. ``stat_schema`` is Tensor-only.
    """


def _select_transport(batch_size, k, d, ranks, element_size=4, *, record_bytes=None):
    # Conservative initial rule: amortize sparse packing on large dense
    # payloads and bound worst-case padded gather storage, including unequal
    # occupancy lengths, to twice the dense table. This is not a portable
    # crossover; callers can override it for their network/occupancy. No density probe.
    dense_bytes = k * (element_size * (d + 1) if record_bytes is None else record_bytes)
    bounded_records = ranks * min(batch_size, k)
    return (
        "sparse"
        if dense_bytes >= 1 << 30 and bounded_records <= 2 * k
        else "dense"
    )


def _combine(sums, counts, workspace, group, transport, batch_size, *, removal=None):
    # Reuse consumed local workspace; native accumulate's views copy to
    # themselves. External statistics are packed into the same allocation.
    numpy_workspace = isinstance(workspace, np.ndarray)
    if numpy_workspace:
        # Stage into owned storage; collectives never write caller statistics.
        np.copyto(workspace[:, :-1], sums)
        np.copyto(workspace[:, -1], counts)
        workspace = torch.from_numpy(workspace)
    else:
        workspace[:, :-1].copy_(sums)
        workspace[:, -1].copy_(counts)
    payloads = (workspace,) if removal is None else (
        workspace, torch.from_numpy(removal) if numpy_workspace else removal)
    ids, reduced = _combine_payloads(payloads, workspace[:, -1], group, transport, batch_size)
    if removal is not None and ids is not None:
        # The shared local removal EMA consumes a full raw table. Dense has
        # already reduced it in place; sparse scatters only the merged IDs.
        payloads[1].zero_().index_copy_(0, ids, reduced[1])
    merged = reduced[0]
    result = (ids, merged[:, :-1], merged[:, -1])
    return tuple(None if value is None else value.numpy() for value in result) if numpy_workspace else result


def _combine_payloads(payloads, counts, group, transport, batch_size):
    """Reduce known typed tables over one occupied-ID schedule."""
    workspace = payloads[0]
    ranks = dist.get_world_size(group)
    if transport == "auto":
        transport = _select_transport(
            batch_size,
            workspace.shape[0],
            workspace.shape[1] - 1,
            ranks,
            workspace.element_size(),
            record_bytes=sum(value.shape[1] * value.element_size() for value in payloads),
        )
    if transport == "dense":
        for value in payloads:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
        return None, payloads

    ids = (counts > 0).nonzero(as_tuple=True)[0]
    size = torch.tensor([len(ids)], dtype=torch.int64, device=workspace.device)
    sizes = [torch.empty_like(size) for _ in range(ranks)]
    dist.all_gather(sizes, size, group=group)
    lengths = torch.cat(sizes).tolist()
    capacity = max(max(lengths), 1)
    padded_ids = torch.zeros(
        capacity, dtype=torch.int64, device=workspace.device
    )
    padded_ids[: len(ids)] = ids
    gathered_ids = [torch.empty_like(padded_ids) for _ in lengths]
    dist.all_gather(gathered_ids, padded_ids, group=group)
    joined = torch.cat([part[:n] for part, n in zip(gathered_ids, lengths)])
    unique, inverse = joined.unique(sorted=True, return_inverse=True)
    reduced = []
    for value in payloads:
        padded = value.new_zeros((capacity, value.shape[1]))
        padded[:len(ids)] = value[ids]
        gathered = [torch.empty_like(padded) for _ in lengths]
        dist.all_gather(gathered, padded, group=group)
        merged = value.new_zeros((len(unique), value.shape[1]))
        offset = 0
        # Unique within rank; ordered additions avoid cross-rank atomic order.
        for part, n in zip(gathered, lengths):
            merged.index_add_(0, inverse[offset:offset + n], part[:n])
            offset += n
        reduced.append(merged)
    return unique, tuple(reduced)


def _revive(model, group):
    coordinator = dist.get_rank(group) == 0
    source, target = (
        model._choose_splits(model._split_rng) if coordinator else (None, None)
    )
    n = len(source) if source is not None else 0
    numpy_state = isinstance(model._centroids, np.ndarray)
    device = torch.device("cpu") if numpy_state else model._centroids.device
    size = torch.tensor([n], device=device, dtype=torch.int64)
    root = dist.get_global_rank(group, 0)
    dist.broadcast(size, src=root, group=group)
    n = int(size.item())
    if not n:
        return 0
    if not coordinator:
        source = torch.empty(n, device=size.device, dtype=torch.int64)
        target = torch.empty_like(source)
    elif numpy_state:
        source = torch.from_numpy(np.array(source, dtype=np.int64, copy=True, order="C"))
        target = torch.from_numpy(np.array(target, dtype=np.int64, copy=True, order="C"))
    dist.broadcast(source, src=root, group=group)
    dist.broadcast(target, src=root, group=group)
    if numpy_state:
        source, target = source.numpy(), target.numpy()
    return model._apply_splits(source, target)


def _epoch_report(model, group):
    # Only the local FP64 objective needs reduction. Pass counts/sums already
    # contain global statistics; row contributions were checked on the host.
    numpy_state = isinstance(model._centroids, np.ndarray)
    objective = torch.as_tensor(model._objective) if numpy_state else model._objective
    dist.all_reduce(objective, op=dist.ReduceOp.SUM, group=group)
    if numpy_state:
        model._objective = np.float64(objective.item())
    return model._objective
