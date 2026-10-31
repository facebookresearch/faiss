# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""E-means with caller-owned groups and contiguous centroid ownership."""

import math

import torch
import torch.distributed as dist

from .e_means import Emeans, _epoch_seed, _resolve_recipe, _torch_generator
from .e_means_distributed import _check_group

# Preserve the original star-import names and lazy access to the moved class.
__all__ = ["math", "torch", "dist", "Emeans", "EmeansShardedCentroids",  # noqa: F822 - lazy export supplied by __getattr__
           "EmeansShardedCentroidsIVF"]


def __getattr__(name):
    if name == "EmeansShardedCentroidsIVF":
        from .sharded_centroids_ivf import EmeansShardedCentroidsIVF
        return EmeansShardedCentroidsIVF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class EmeansShardedCentroids(Emeans):
    """Train resident rank-local rows against sharded CUDA centroids.

    ``k`` at construction is global; ``k`` on the instance and ``centroids``
    describe the local shard. ``global_k`` and ``centroid_begin`` locate it.
    ``begin`` takes only this rank's FP32 centroid shard. All ranks must use
    identical parameters and enter the lifecycle in the same order. Row
    contributions must have equal sizes; global batches, row counts and
    centroid counts must be divisible by the group size. Communication uses
    bounded row broadcasts;
    centroid tables and EMA statistics stay with their owners.

    Accumulate matching rank-local chunks until the prescribed local share is
    complete, then update once. Public statistics n counts global broadcast
    rows; sums/counts describe owned centroids. Last-pass labels describe only
    this caller's supplied rows, using global centroid IDs.
    """

    def __init__(self, d, k, *args, group, **kwargs):
        ranks = _check_group(group)
        if k < ranks or k % ranks:
            raise ValueError("k must be positive and divisible by group size")
        self.global_k = int(k)
        self._group, self._group_size = group, ranks
        self._rank = dist.get_rank(group)
        self.centroid_begin = (self.global_k // ranks) * self._rank
        end = self.centroid_begin + self.global_k // ranks
        super().__init__(d, end - self.centroid_begin, *args, **kwargs)
        if self.quantizer is not None:
            raise ValueError("sharded centroids currently require quantizer=None")

    def _get_assignment_chunk_size(self, is_torch, cuda=False):
        if self.assignment_backend == "auto" and cuda and self.assignment_chunk_size is None:
            # Keep the old native broadcast bound, independent of compute choice.
            width = max(self.d, min(self.k, self.centroid_chunk_size))
            return min(2**20, max(8192, 2**26 // width))
        return super()._get_assignment_chunk_size(is_torch, cuda)

    def begin(self, global_n, init_centroids):
        if global_n < self.global_k:
            raise ValueError("global_n must be at least global_k")
        if not isinstance(init_centroids, torch.Tensor) or not init_centroids.is_cuda:
            raise ValueError("sharded centroids require CUDA tensors")
        super().begin(global_n, init_centroids)
        batch, alpha = _resolve_recipe(
            self._n,
            self.global_k,
            self.niter,
            self.operating_point,
            self.batch_size,
            self.alpha0,
        )
        if (self._n % self._group_size or batch % self._group_size
                or any(rows % self._group_size for rows in self.recipe["epoch_rows"])):
            self.reset()
            raise ValueError("every epoch's global rows and batches must be divisible by group size")
        steps = math.ceil(self._n / batch)
        epoch_steps = [math.ceil(rows / batch) for rows in self.recipe["epoch_rows"]]
        self.recipe.update(
            batch_size=batch,
            alpha0=alpha,
            steps_per_pass=steps,
            epoch_steps=epoch_steps,
            total_steps=sum(epoch_steps),
            global_k=self.global_k,
            centroid_begin=self.centroid_begin,
            centroid_shards=self._group_size,
        )

    def _prepare_assignment(self):
        return self._prepare_torch_assignment(self._centroids)

    def _local_assignment(self, rows, prepared):
        distance_out = (torch.empty(len(rows), dtype=torch.int64, device=rows.device)
                        if rows.dtype in (torch.int8, torch.uint8) else None)
        return self._assign_rows(rows, *prepared, distance_out=distance_out)

    def _winning_labels(self, rows, prepared, return_score=False):
        distance, labels = self._local_assignment(rows, prepared)
        best = distance.clone()
        dist.all_reduce(best, op=dist.ReduceOp.MIN, group=self._group)
        sentinel = torch.iinfo(torch.int64).max
        winner = torch.where(
            (distance == best) & (labels >= 0),
            labels.long() + self.centroid_begin,
            sentinel,
        )
        dist.all_reduce(winner, op=dist.ReduceOp.MIN, group=self._group)
        return (winner, best) if return_score else winner

    def _row_chunks(self, x, index=None):
        chunk = self.recipe["assignment_chunk_size"]
        for source in range(self._group_size):
            length = len(x) if index is None else len(index)
            root = dist.get_global_rank(self._group, source)
            for begin in range(0, length, chunk):
                n = min(chunk, length - begin)
                rows = (
                    (
                        x[begin : begin + n]
                        if index is None
                        else x[index[begin : begin + n]]
                    ).contiguous()
                    if source == self._rank
                    else torch.empty((n, self.d), dtype=x.dtype, device=x.device)
                )
                dist.broadcast(rows, src=root, group=self._group)
                yield source, begin, rows

    def _add_owned(self, rows, winner, sums, counts, objective, score=None):
        owned = (winner >= self.centroid_begin) & (
            winner < self.centroid_begin + self.k
        )
        labels = winner[owned] - self.centroid_begin
        values = rows[owned].to(sums.dtype)
        sums.index_add_(0, labels, values)
        counts.index_add_(
            0,
            labels,
            torch.ones(len(labels), device=rows.device, dtype=counts.dtype),
        )
        if self.spherical:
            objective.add_(2 * len(labels) + score[owned].sum(dtype=torch.float64))
        else:
            objective.add_(
                (values.float() - self._centroids[labels]).square().sum(dtype=torch.float64)
            )

    def _local_batch_size(self):
        return super()._local_batch_size() // self._group_size

    def _accumulate(self, x, index=None, labels_out=None):
        stats = super()._accumulate(x, index, labels_out)
        # Checked pending receipts count source rows; public statistics cover all owners.
        return dict(stats, n=stats["n"] * self._group_size)

    def _accumulate_core(self, x, index=None, raw=False, labels_out=None):
        n = len(x) if index is None else len(index)
        pending_n = 0 if self._pending_stats is None else self._pending_stats["n"]
        target = self._local_batch_size()
        if pending_n == 0:
            global_target = target * self._group_size
            self._batch_accumulation = (
                self._batch_stats.new_zeros(
                    self._batch_stats.shape, dtype=torch.float64
                )
                if n
                and global_target > 2**24
                and self._batch_stats.dtype == torch.float32
                else self._batch_stats.zero_()
            )
            self._batch_objective = self._objective.new_zeros(())
            self._batch_centroids = self._prepare_assignment() if n else None
        accumulation = self._batch_accumulation
        sums, counts = accumulation[:, :-1], accumulation[:, -1]
        objective, prepared = self._batch_objective, self._batch_centroids
        labels = None if labels_out is None else labels_out[:n]
        if (
            labels is None
            and self.capture_assignments
            and self._epoch == len(self.recipe["epoch_rows"]) - 1
        ):
            labels = x.new_empty(n, dtype=torch.int64)
        for source, begin, rows in self._row_chunks(x, index):
            if self.spherical:
                winner, score = self._winning_labels(rows, prepared, return_score=True)
                self._add_owned(rows, winner, sums, counts, objective, score)
            else:
                winner = self._winning_labels(rows, prepared)
                self._add_owned(rows, winner, sums, counts, objective)
            if labels is not None and source == self._rank:
                labels[begin : begin + len(rows)] = winner
        total_n = pending_n + n
        if total_n == target and accumulation is not self._batch_stats:
            self._batch_stats.copy_(accumulation)
            sums, counts = self._batch_stats[:, :-1], self._batch_stats[:, -1]
        complete = total_n == target or total_n == 0
        stats = {
            "sums": sums if complete else None,
            "counts": counts if complete else None,
            "objective": objective,
            "n": total_n,
        }
        if labels is not None:
            stats["labels"] = labels
        return stats

    def _revive_unused(self, rng):
        # Only usage (O(k)) is gathered; selected donor rows are broadcast.
        usage = self._usage
        parts = [torch.empty_like(usage) for _ in range(self._group_size)]
        dist.all_gather(parts, usage, group=self._group)
        full = torch.cat(parts)
        source = target = None
        if self._rank == 0:
            local_k, local_usage = self.k, self._usage
            try:
                self.k, self._usage = self.global_k, full
                source, target = self._choose_splits(rng)
            finally:
                self.k, self._usage = local_k, local_usage
        root = dist.get_global_rank(self._group, 0)
        size = torch.tensor(
            [0 if source is None else len(source)],
            dtype=torch.int64,
            device=usage.device,
        )
        dist.broadcast(size, src=root, group=self._group)
        n = size.item()
        if not n:
            return 0
        if self._rank != 0:
            source = torch.empty(n, dtype=torch.int64, device=usage.device)
            target = torch.empty_like(source)
        dist.broadcast(source, src=root, group=self._group)
        dist.broadcast(target, src=root, group=self._group)
        donor = self._centroids.new_empty((n, self.d))
        for rank in range(self._group_size):
            begin = self.k * rank
            end = begin + self.k
            selected = (source >= begin) & (source < end)
            rows = (
                self._centroids[source[selected] - begin].contiguous()
                if rank == self._rank
                else donor.new_empty((int(selected.sum()), self.d))
            )
            dist.broadcast(
                rows,
                src=dist.get_global_rank(self._group, rank),
                group=self._group,
            )
            donor[selected] = rows
        split = Emeans(self.d, 2 * n)
        split._centroids = torch.cat((donor, donor.clone()))
        split._usage = torch.cat((full[source], full[source]))
        split._input_dtype = getattr(self, "_revival_dtype", self._input_dtype)
        ids = torch.arange(n, device=usage.device)
        split._apply_splits(ids, ids + n)
        for global_ids, values, mass in (
            (source, split._centroids[:n], split._usage[:n]),
            (target, split._centroids[n:], split._usage[n:]),
        ):
            owned = (global_ids >= self.centroid_begin) & (
                global_ids < self.centroid_begin + self.k
            )
            local = global_ids[owned] - self.centroid_begin
            self._centroids[local] = values[owned]
            self._usage[local] = mass[owned]
        self.n_splits += n
        self._invalidate_readout()
        return n

    def _epoch_objective(self):
        dist.all_reduce(self._objective, group=self._group)
        return self._objective

    def end_epoch(self):
        self._require_running()
        rows = self.recipe["epoch_rows"][self._epoch]
        if (
            self._pending_stats is not None
            or self._epoch_rows != rows
            or self._epoch_steps != self.recipe["epoch_steps"][self._epoch]
        ):
            raise RuntimeError("epoch is incomplete; consume all prescribed batches")
        verbose = self.verbose
        self.verbose = False
        try:
            square = self._pass_counts.double().square().sum()
            dist.all_reduce(square, group=self._group)
            super().end_epoch()
            stat = self.iteration_stats[-1]
            stat["imbalance_factor"] = self.global_k * square.item() / rows**2
            if verbose and self._rank == 0:
                print(f"e-means iteration {self._epoch}: {stat}", flush=True)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        finally:
            self.verbose = verbose

    def final_closing_mean(self, batches):
        """Scan fresh rows against one frozen final snapshot, then finalize.

        ``batches`` supplies equal rank-local CUDA contributions in any order.
        The caller can use a sequential SSD iterator. Every row must occur
        exactly once. All training epochs must be complete and ``finalize``
        must not yet have been called. Empty clusters retain their positions.
        Floating scans with global_n > 2**24 use FP64 sums (8 rather than
        4 bytes per entry); smaller scans keep FP32 and raw bytes keep INT64.
        Division precedes the final FP32 conversion. This does not guarantee
        stability of assignment distances or arbitrarily long reductions.
        Budget for the existing training tables until the scan completes,
        converted FP64 row chunks, FP64 counts/divisor and FP32 mean/output
        temporaries as well as the new sum table.
        """
        if (
            self._failed
            or self._centroids is None
            or self._epoch != len(self.recipe["epoch_rows"])
            or self.centroids is not None
        ):
            raise RuntimeError("complete epochs before the frozen closing scan")
        if self._pending_stats is not None:
            raise RuntimeError("consume pending statistics before closing")
        try:
            dtype = self._batch_stats.dtype
            if dtype == torch.float32 and self._n > 2**24:
                dtype = torch.float64
            sums = self._centroids.new_zeros(
                self._centroids.shape, dtype=dtype
            )
            counts = self._usage.new_zeros(
                self.k,
                dtype=torch.int64 if sums.dtype == torch.int64 else torch.float64,
            )
            objective = self._objective.new_zeros(())
            prepared = None
            n = 0
            for x in batches:
                x = self._prepare_input(x)
                n += len(x) * self._group_size
                if n > self._n:
                    raise ValueError("closing scan exceeds global_n")
                if prepared is None and len(x):
                    prepared = self._prepare_assignment()
                for _, _, rows in self._row_chunks(x):
                    if self.spherical:
                        winner, score = self._winning_labels(rows, prepared, return_score=True)
                        self._add_owned(rows, winner, sums, counts, objective, score)
                    else:
                        winner = self._winning_labels(rows, prepared)
                        self._add_owned(rows, winner, sums, counts, objective)
            if n != self._n:
                raise ValueError("closing scan must cover global_n rows")
            dist.all_reduce(objective, group=self._group)
            self.closing_stats = {
                "rows": n,
                "assignment_error"
                if self.spherical
                else "assignment_mse": objective.item() / n,
            }
            self._pass_sums, self._pass_counts = sums, counts
            return self.finalize()
        except BaseException:
            self._failed = True
            self.assignments = None
            raise

    def train(self, x, init_centroids=None):
        """Train equal row shards; return local centroids and last-pass labels."""
        self.assignments = None
        if not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim != 2:
            raise ValueError("train requires a resident CUDA matrix")
        n = len(x) * self._group_size
        if init_centroids is None:
            ids = torch.randperm(
                len(x),
                device=x.device,
                generator=_torch_generator(x.device, self.seed + self._rank),
            )[: self.k]
            init_centroids = x[ids].float()
        self.begin(n, init_centroids)
        x = self._prepare_input(x)
        batch = self.recipe["batch_size"]
        try:
            assignments = (
                x.new_empty(len(x), dtype=torch.int64)
                if self.capture_assignments
                else None
            )
            if assignments is not None and self.recipe["epoch_rows"][-1] < n:
                assignments.fill_(-1)
            for epoch, rows in enumerate(self.recipe["epoch_rows"]):
                order = torch.randperm(
                    len(x),
                    device=x.device,
                    generator=_torch_generator(x.device, _epoch_seed(self.seed, epoch)),
                )
                for start in range(0, rows, batch):
                    end = min(rows, start + batch)
                    a, b = start // self._group_size, end // self._group_size
                    stats = self._accumulate(x, order[a:b])
                    self.update(stats)
                    if assignments is not None and epoch == len(self.recipe["epoch_rows"]) - 1:
                        assignments[order[a:b]] = stats["labels"]
                self.end_epoch()
            result = self.finalize()
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        self.assignments = assignments
        return result, self.assignments
