# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Replicated-centroid CPU/GPU E-means with caller-owned communication groups.

Sparse packing/length discovery and coordinated revival can synchronize with
the host. Stock synchronous collectives order consumption on the caller's
current stream.
"""

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
        super().begin(global_n, init_centroids)
        numpy_state = isinstance(self._centroids, np.ndarray)
        if (numpy_state or self._centroids.device.type == "cpu") and dist.get_backend(self._group) != "gloo":
            self._failed = True
            raise ValueError("CPU DistributedEmeans requires a Gloo group")
        ranks, batch = self._group_size, self.recipe["batch_size"]
        if (
            min(batch, self._n) % ranks
            or self._n % ranks
            or any(rows % ranks for rows in self.recipe["epoch_rows"])
        ):
            self._failed = True
            raise _divisibility_error(self.recipe, self._n, batch, ranks)
        try:
            dist.broadcast(
                torch.from_numpy(self._centroids) if numpy_state else self._centroids,
                src=dist.get_global_rank(self._group, 0),
                group=self._group,
            )
        except BaseException:
            self._failed = True
            self.assignments = None
            raise

    def _local_batch_size(self):
        return super()._local_batch_size() // self._group_size

    def _prepare_statistics(self, stats, expected_n):
        if stats["n"] != expected_n // self._group_size:
            raise ValueError("stats n must match the prescribed batch size")
        _, sums, counts, objective = super()._prepare_statistics(
            stats, expected_n // self._group_size
        )
        try:
            ids, sums, counts = _combine(
                sums,
                counts,
                self._batch_stats,
                self._group,
                self._transport,
                self.recipe["batch_size"],
            )
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

    def _revive_unused(self, rng):
        return _revive(self, self._group)

    def _epoch_objective(self):
        return _epoch_report(self, self._group)

    def train(self, x, init_centroids=None):
        raise NotImplementedError(
            "DistributedEmeans uses begin/accumulate/update/end_epoch/finalize, not train(x)"
        )


class DistributedEmeans(_ReplicatedCentroids, QuantizedEmeans):
    """Replicated tensor E-means; use the caller-driven distributed lifecycle."""


class DistributedNumpyEmeans(_ReplicatedCentroids, NumpyEmeans):
    """Replicated NumPy/Faiss E-means with CPU/Gloo communication.

    Numerical state remains NumPy; conversion occurs only at collective
    boundaries. Recipes, equal row shares and communication match
    DistributedEmeans. Use begin/accumulate/update/end_epoch/finalize.
    """


def _select_transport(batch_size, k, d, ranks, element_size=4):
    # Conservative initial rule: amortize sparse packing on large dense
    # payloads and bound worst-case padded gather storage, including unequal
    # occupancy lengths, to twice the dense table. This is not a portable
    # crossover; callers can override it for their network/occupancy. No density probe.
    dense_bytes = element_size * k * (d + 1)
    bounded_records = ranks * min(batch_size, k)
    return (
        "sparse"
        if dense_bytes >= 1 << 30 and bounded_records <= 2 * k
        else "dense"
    )


def _combine(sums, counts, workspace, group, transport, batch_size):
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
    ranks = dist.get_world_size(group)
    if transport == "auto":
        transport = _select_transport(
            batch_size,
            workspace.shape[0],
            workspace.shape[1] - 1,
            ranks,
            workspace.element_size(),
        )
    if transport == "dense":
        dist.all_reduce(workspace, op=dist.ReduceOp.SUM, group=group)
        if numpy_workspace:
            return None, workspace[:, :-1].numpy(), workspace[:, -1].numpy()
        return None, workspace[:, :-1], workspace[:, -1]

    ids = (workspace[:, -1] > 0).nonzero(as_tuple=True)[0]
    values = workspace[ids]
    size = torch.tensor([len(ids)], dtype=torch.int64, device=workspace.device)
    sizes = [torch.empty_like(size) for _ in range(ranks)]
    dist.all_gather(sizes, size, group=group)
    lengths = torch.cat(sizes).tolist()
    capacity = max(max(lengths), 1)
    padded_ids = torch.zeros(
        capacity, dtype=torch.int64, device=workspace.device
    )
    padded_values = workspace.new_zeros((capacity, workspace.shape[1]))
    padded_ids[: len(ids)] = ids
    padded_values[: len(ids)] = values
    gathered_ids = [torch.empty_like(padded_ids) for _ in lengths]
    gathered_values = [torch.empty_like(padded_values) for _ in lengths]
    dist.all_gather(gathered_ids, padded_ids, group=group)
    dist.all_gather(gathered_values, padded_values, group=group)
    joined = torch.cat([part[:n] for part, n in zip(gathered_ids, lengths)])
    unique, inverse = joined.unique(sorted=True, return_inverse=True)
    merged = workspace.new_zeros((len(unique), workspace.shape[1]))
    offset = 0
    # IDs are unique within each rank. Sequential rank additions avoid a
    # duplicate-ID atomic scatter whose order could differ across replicas.
    for part, n in zip(gathered_values, lengths):
        merged.index_add_(0, inverse[offset : offset + n], part[:n])
        offset += n
    if numpy_workspace:
        return unique.numpy(), merged[:, :-1].numpy(), merged[:, -1].numpy()
    return unique, merged[:, :-1], merged[:, -1]


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
