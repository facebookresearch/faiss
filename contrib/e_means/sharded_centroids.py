# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""E-means with caller-owned groups and contiguous centroid ownership."""

import math
import time

import torch
import torch.distributed as dist

from .base import _REMOVAL_THRESHOLD, _divisibility_error, _epoch_seed, _protect_candidates, _resolve_recipe
from .e_means_quantized import QuantizedEmeans, check_codec_fit, statistics_values
from .e_means_torch import _closing_centroid_values, _torch_generator
from .e_means_distributed import _check_group

# Lazy access to the moved IVF class.
__all__ = [  # noqa: F822
    "math", "torch", "dist", "EmeansShardedCentroids",
    "EmeansShardedCentroidsIVF",
]


def __getattr__(name):
    if name == "EmeansShardedCentroidsIVF":
        from .sharded_centroids_ivf import EmeansShardedCentroidsIVF
        return EmeansShardedCentroidsIVF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _removal_scalar_stats(removal_stats, usage, *, chunk_rows=None):
    """Gather local [removal cost, usage, axis norm] in bounded FP64 chunks."""
    if chunk_rows is None:
        # vector_norm(dtype=FP64) can widen its strided input; cap that
        # temporary at 64 MiB rather than converting the full (k, d) table.
        chunk_rows = max(1, (64 << 20) // (8 * (removal_stats.shape[1] - 1)))
    if chunk_rows < 1:
        raise ValueError("norm chunk_rows must be positive")
    scalars = torch.empty(
        (len(usage), 3), dtype=torch.float64, device=usage.device
    )
    for begin in range(0, len(usage), chunk_rows):
        end = min(begin + chunk_rows, len(usage))
        block = removal_stats[begin:end]
        scalars[begin:end, 0].copy_(block[:, -1])
        scalars[begin:end, 1].copy_(usage[begin:end])
        scalars[begin:end, 2].copy_(torch.linalg.vector_norm(
            block[:, :-1], dim=1, dtype=torch.float64
        ))
    return scalars


def _removal_candidates(removal):
    """Threshold by the FP64 median, sorting only candidates by cost then ID."""
    n = len(removal)
    if not n:
        return torch.empty(0, dtype=torch.int64, device=removal.device)
    lower = torch.kthvalue(removal, (n + 1) // 2).values.double()
    upper = torch.kthvalue(removal, n // 2 + 1).values.double() if n % 2 == 0 else lower
    median = torch.where(removal.isnan().any(), torch.nan, (lower + upper) / 2)
    candidates = torch.nonzero(
        (removal <= 0) | (removal.double() < _REMOVAL_THRESHOLD * median)
    ).flatten()
    return candidates[torch.argsort(removal[candidates], stable=True)]


def _protect_removal_candidates(candidates, nearest, same):
    if candidates.is_cuda:
        from .triton_assignment import protect_candidates

        return protect_candidates(candidates, nearest, same)
    kept = _protect_candidates(candidates.tolist(), nearest.tolist(), lambda position: bool(same[position]))
    return torch.as_tensor(kept, dtype=torch.int64, device=candidates.device)


def _removal_version(stats):
    try:
        return id(stats), stats._version
    except RuntimeError:  # Inference tensors have no version counter.
        return id(stats), None


def _partial_removal_donors(kept, usage, axis_norm):
    """GPU partial donor selection, ordered by FP32 gain then global ID.

    Find the kth score without sorting the full table. Recover cutoff ties
    in ID order before sorting only the selected donors. NaNs sort last,
    matching the host stable argsort.
    """
    eligible = usage > 0
    kept = torch.as_tensor(kept, dtype=torch.int64, device=usage.device)
    eligible[kept] = False
    n = min(len(kept), int(eligible.sum()))
    if not n:
        return kept.new_empty(0)
    score = torch.where(eligible, -axis_norm.float(), torch.inf)
    score.nan_to_num_(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
    cutoff = torch.kthvalue(score, n).values
    better = torch.nonzero(eligible & (score < cutoff)).flatten()
    tied = torch.nonzero(eligible & (score == cutoff)).flatten()
    chosen = torch.cat((better, tied[:n - len(better)]))
    # nonzero gives ascending IDs, including within the cutoff tie. Stable
    # sorting preserves that order for equal gains above the cutoff too.
    return chosen[torch.argsort(score[chosen], stable=True)]


class EmeansShardedCentroids(QuantizedEmeans):
    """Train resident rank-local rows against sharded CUDA centroids.

    ``k`` at construction is global; ``k`` on the instance and ``centroids``
    describe the local shard. ``global_k`` and ``centroid_begin`` locate it.
    ``begin`` takes only this rank's FP32 centroid shard. All ranks must use
    identical parameters and enter the lifecycle in the same order. Row
    contributions must have equal sizes; global batches, row counts and
    centroid counts must be divisible by the group size. Communication uses
    one full-batch all-gather;
    centroid tables and EMA statistics stay with their owners.

    Accumulate matching rank-local chunks until the prescribed local share is
    complete, then update once. Public statistics n counts global contributed
    rows; sums/counts describe owned centroids. Last-pass labels describe only
    this caller's supplied rows, using global centroid IDs.
    Native None requires a common native dtype. Explicit codecs gather codes
    in their shared dtype and decode them for owner statistics. Ordinary obj
    is direct SSE from decoded rows to the FP32 working centroids.
    BF16 defaults to removal-cost revival with Oja-ranked donors. Top-2
    assignment adds one MIN reduction per batch; selection gathers scalar
    statistics and searches nearest neighbors across all centroid shards.
    """

    def _resolve_removal_revival(self):
        # Keep the existing BF16 owner integration until other represented
        # paths have their own complete global-statistics qualification.
        if self._input_dtype == torch.bfloat16:
            super()._resolve_removal_revival()

    def __init__(self, d, k, *args, group, **kwargs):
        ranks = _check_group(group)
        if kwargs.get("max_points_per_centroid") is not None:
            raise ValueError(
                "max_points_per_centroid is not supported with sharded "
                "centroids; subsample x_local before train"
            )
        if k < ranks or k % ranks:
            raise ValueError("k must be positive and divisible by group size")
        self.global_k = int(k)
        self._group, self._group_size = group, ranks
        self._rank = dist.get_rank(group)
        self.centroid_begin = (self.global_k // ranks) * self._rank
        end = self.centroid_begin + self.global_k // ranks
        super().__init__(d, end - self.centroid_begin, *args, **kwargs)

    def _resolve_first_input(self, x):
        """Agree the first automatic gathered row dtype before allocation.

        Every rank must allocate and interpret gathered rows using the same
        dtype. This agreement prevents peer rows being read in different units.
        Automatic FP32 and mixed FP32/BF16 first inputs select BF16 casting
        on every rank. Uniform other dtypes stay native. Incompatible mixtures
        fail together. Typed empty inputs also participate in this agreement.
        """
        if not self._automatic or self._input_dtype is not None:
            self._resolve_input_dtype(x)
            return
        try:
            types = (torch.float32, torch.float16, torch.bfloat16,
                     torch.float8_e4m3fn, torch.int8, torch.uint8)
            code = types.index(x.dtype) if x.dtype in types else -1
            peers = self._gather_local(torch.tensor([code], device=x.device)).tolist()
            if -1 in peers:
                raise ValueError("unsupported first automatic population representation")
            seen = {types[code] for code in peers}
            if torch.float32 in seen and seen <= {torch.float32, torch.bfloat16}:
                self._bind_input_dtype(torch.bfloat16, cast=True)
            elif len(seen) == 1:
                self._bind_input_dtype(types[peers[0]])
            else:
                raise ValueError("incompatible first automatic population representations")
        except BaseException:
            self._failed = True
            self.assignments = None
            raise

    def accumulate(self, x):
        self._require_running()
        self._resolve_first_input(x)
        x = self._prepare_input(x)
        return self._accumulate(x.detach())

    def _get_assignment_chunk_size(self):
        state = self._centroids if self._centroids is not None else self.centroids
        if self.assignment_backend == "auto" and state.is_cuda and self.assignment_chunk_size is None:
            # Keep the native compute bound, independent of compute choice.
            width = max(self.d, min(self.k, self.centroid_chunk_size))
            return min(2**20, max(8192, 2**26 // width))
        return super()._get_assignment_chunk_size()

    def begin(self, global_n, init_centroids):
        if self._failed or self.recipe is not None:
            raise RuntimeError("create a new object for another run or after failure")
        if global_n < self.global_k:
            raise ValueError("global_n must be at least global_k")
        if not isinstance(init_centroids, torch.Tensor) or not init_centroids.is_cuda:
            raise ValueError("sharded centroids require CUDA tensors")
        try:
            if not self._automatic and self.quantizer is not None:
                check_codec_fit(self._group, self.quantizer, init_centroids.device)
            super().begin(global_n, init_centroids)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        batch, alpha = _resolve_recipe(
            self._n,
            self.global_k,
            self.nepochs,
            self._batch_law_resolved,
            self.batch_size,
            self.alpha0,
        )
        if (self._n % self._group_size or batch % self._group_size
                or any(rows % self._group_size for rows in self.recipe["epoch_rows"])):
            self._failed = True
            raise _divisibility_error(self.recipe, self._n, batch, self._group_size)
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
            batch_transport="all_gather",
        )

    def _prepare_assignment(self):
        return self._prepare_torch_assignment(self._centroids)

    def _initialize_removal_statistics(self):
        super()._initialize_removal_statistics()
        self._axis_norms = self._usage.new_zeros(self.k, dtype=torch.float32)
        self._axis_norms_version = _removal_version(self._removal_stats)

    def _apply_statistics(self, *args, **kwargs):
        super()._apply_statistics(*args, **kwargs)
        if self._removal_stats is not None and not self._frozen():
            self._axis_norms_version = _removal_version(self._removal_stats)

    def _gather_removal_scalars(self):
        # Normal updates write the cached norm alongside the axes. Direct
        # state edits or splits invalidate it; retain bounded recomputation.
        if self._axis_norms_version != _removal_version(self._removal_stats):
            local = _removal_scalar_stats(self._removal_stats, self._usage).float()
            self._axis_norms.copy_(local[:, 2])
            self._axis_norms_version = _removal_version(self._removal_stats)
        else:
            local = torch.stack((self._removal_stats[:, -1], self._usage.float(), self._axis_norms), dim=1)
        full = local.new_empty((self.global_k if self._rank == 0 else 0, 3))
        # Every owner sends its FP32 scalars only to rank 0. Other ranks do
        # not allocate or receive a global statistics table.
        dist.all_to_all_single(full, local,
            input_split_sizes=[self.k] + [0] * (self._group_size - 1),
            output_split_sizes=[self.k if self._rank == 0 else 0] * self._group_size,
            group=self._group)
        return full

    def _choose_sharded_removal_splits(self):
        scalars = self._gather_removal_scalars()
        root = dist.get_global_rank(self._group, 0)
        candidates = _removal_candidates(scalars[:, 0]) if self._rank == 0 else None
        size = torch.tensor([len(candidates) if self._rank == 0 else 0],
                            dtype=torch.int64, device=self._usage.device)
        dist.broadcast(size, src=root, group=self._group)
        n = size.item()
        if not n:
            return None, None
        if self._rank != 0:
            candidates = torch.empty(n, dtype=torch.int64, device=self._usage.device)
        dist.broadcast(candidates, src=root, group=self._group)
        # All owners search, but selection and protection run only on rank 0.
        nearest = self._nearest_other(candidates, return_tensor=True)
        same = self._same_readout_pairs(candidates, nearest)
        if self._rank != 0:
            return None, None
        kept = _protect_removal_candidates(candidates, nearest, same)
        donors = self._choose_removal_donors(kept, scalars[:, 1], scalars[:, 2])
        n = min(len(kept), len(donors))
        return (donors[:n], kept[:n]) if n else (None, None)

    def _same_readout_pairs(self, candidates, nearest):
        """Exact current-row identity for root's neighbors, with bounded owner lookup."""
        readout = self._readout()
        same = candidates.new_empty(len(candidates), dtype=torch.uint8)
        root = dist.get_global_rank(self._group, 0)
        # Query rows, two indexed row copies, equality scratch and ID/mask vectors.
        chunk_rows = max(1, min(8192, (32 << 20) // (16 * self.d + 64)))
        for begin in range(0, len(candidates), chunk_rows):
            ids = candidates[begin:begin + chunk_rows]
            partner = torch.empty_like(ids)
            if self._rank == 0:
                partner.copy_(nearest[begin:begin + len(ids)])
            # Callback results on other ranks are intentionally ignored, as before.
            dist.broadcast(partner, src=root, group=self._group)
            owned = (ids >= self.centroid_begin) & (ids < self.centroid_begin + self.k)
            queries = readout.new_zeros((len(ids), self.d))
            queries[owned] = readout[ids[owned] - self.centroid_begin]
            dist.all_reduce(queries, group=self._group)
            valid = (partner >= 0) & (partner < self.global_k)
            owned = valid & (partner >= self.centroid_begin) & (partner < self.centroid_begin + self.k)
            equal = ids.new_zeros(len(ids), dtype=torch.uint8)
            equal[owned] = (readout[partner[owned] - self.centroid_begin] == queries[owned]).all(1).byte()
            dist.all_reduce(equal, op=dist.ReduceOp.MAX, group=self._group)
            same[begin:begin + len(ids)].copy_(equal)
        return same

    def _local_assignment(self, rows, prepared, second_out=None):
        distance_out = (torch.empty(len(rows), dtype=torch.int64, device=rows.device)
                        if self.quantizer is None and rows.dtype in (torch.int8, torch.uint8) else None)
        distance, labels = self._assign_rows(rows, *prepared, distance_out=distance_out, second_out=second_out)
        if second_out is not None and self._assignment_backend == "cute":
            self._execution_backend.complete_second(rows, labels, second_out)
        return distance, labels

    def _winning_labels(self, rows, prepared, return_score=False):
        distance, labels = self._local_assignment(rows, prepared)
        return self._reduce_winners(distance, labels, return_score)

    def _reduce_winners(self, distance, labels, return_score=False):
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

    def _reduce_top2(self, distance, labels, second):
        winner, best = self._reduce_winners(distance, labels, return_score=True)
        # The winning owner contributes its second candidate; every other
        # owner contributes its first. One more MIN gives the global second.
        other = torch.where(labels.long() + self.centroid_begin == winner, second, distance)
        dist.all_reduce(other, op=dist.ReduceOp.MIN, group=self._group)
        return winner, best, other

    def _gather_local(self, local):
        fp8 = local.dtype == torch.float8_e4m3fn
        if fp8:
            local = local.view(torch.uint8)
        rows = local.new_empty((len(local) * self._group_size, *local.shape[1:]))
        if len(local):
            gather = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
            gather(rows, local, group=self._group)
        return rows.view(torch.float8_e4m3fn) if fp8 else rows

    def _wire_rows(self, rows):
        rows = self._prepare_rows(rows)
        # CuTe FP8 alone defers local encoding until assignment.
        if self._assignment_backend == "cute" and self._fp8_quantizer and rows.dtype != self._input_dtype:
            rows = self.quantizer.compute_codes(rows)
        if rows.dtype != self._input_dtype:
            raise ValueError("source rows do not match the bound native representation")
        return rows.contiguous()

    def _nearest_other(self, ids, *, return_tensor=False):
        """Protect nearest neighbors globally without gathering centroid tables."""
        from .torch_backend import TritonBackend

        centroids, result = self._readout(), []
        ids = torch.as_tensor(ids, device=centroids.device)
        sentinel = torch.iinfo(torch.int64).max
        use_triton = centroids.is_cuda and (isinstance(self._execution_backend, TritonBackend) or hasattr(self, "_ivf"))
        if use_triton:
            from .triton_assignment import nearest_other, prepare_nearest_other

            prepared = prepare_nearest_other(centroids)
        for chunk in ids.split(max(1, min(8192, (32 << 20) // (4 * self.d)))):
            owned = (chunk >= self.centroid_begin) & (chunk < self.centroid_begin + self.k)
            queries = centroids.new_zeros((len(chunk), self.d))
            queries[owned] = centroids[chunk[owned] - self.centroid_begin]
            dist.all_reduce(queries, group=self._group)
            if use_triton:
                score, labels = nearest_other(
                    centroids, chunk, torch.tensor(len(chunk), device=chunk.device),
                    queries=queries, centroid_begin=self.centroid_begin, prepared=prepared,
                )
                labels = torch.where(labels == sentinel, sentinel, labels + self.centroid_begin)
            else:
                score = torch.full((len(chunk),), torch.inf, device=chunk.device, dtype=torch.float64)
                labels = torch.full_like(chunk, sentinel)
                block = max(1, (32 << 20) // (8 * len(chunk) * self.d))
                for begin in range(0, self.k, block):
                    c = centroids[begin:begin + block]
                    distance = (queries[:, None].double() - c[None].double()).square().sum(-1)
                    candidate = torch.arange(begin, begin + len(c), device=chunk.device) + self.centroid_begin
                    distance = torch.where(candidate[None] == chunk[:, None], torch.inf, distance)
                    distance = torch.where(distance.isnan(), torch.inf, distance)
                    best = distance.min(dim=1).values
                    nearest = torch.where(distance == best[:, None], candidate[None], sentinel).min(dim=1).values
                    nearest = torch.where(nearest == chunk, sentinel, nearest)
                    better = (best < score) | ((best == score) & (nearest < labels))
                    score, labels = torch.where(better, best, score), torch.where(better, nearest, labels)
            best = score.clone()
            dist.all_reduce(best, op=dist.ReduceOp.MIN, group=self._group)
            labels = torch.where(score == best, labels, sentinel)
            dist.all_reduce(labels, op=dist.ReduceOp.MIN, group=self._group)
            result.append(labels)
        nearest = torch.cat(result) if result else ids.new_empty(0)
        return nearest if return_tensor else nearest.cpu().numpy()

    def _gather_training_rows(self, x, index=None):
        if self.quantizer is None or x.dtype == self.quantizer.dtype:
            local = self._wire_rows(x if index is None else x[index])
        else:
            n = len(x) if index is None else len(index)
            local = x.new_empty((n, self.d), dtype=self._input_dtype)
            chunk = self.recipe["assignment_chunk_size"]
            for begin in range(0, n, chunk):
                end = min(begin + chunk, n)
                rows = x[begin:end] if index is None else x[index[begin:end]]
                local[begin:end].copy_(self._wire_rows(rows))
        return self._gather_local(local)

    def _gather_assignment_rows(self, x, prepared, index=None):
        return self._gather_training_rows(x, index), None

    def _local_assignment_batch(self, rows, prepared, routing=None, second_out=None):
        """Bound native assignment scratch while reducing winners once per batch."""
        distance = labels = None
        local_n = len(rows) // self._group_size
        chunk = self.recipe["assignment_chunk_size"]
        for source in range(self._group_size):
            for begin in range(0, local_n, chunk):
                part = rows[source * local_n + begin:
                            source * local_n + min(local_n, begin + chunk)]
                at = source * local_n + begin
                second = None if second_out is None else second_out[at:at + len(part)]
                score, ids = self._local_assignment(part, prepared, **({"second_out": second} if second is not None else {}))
                if distance is None:
                    distance = score.new_empty(len(rows))
                    labels = ids.new_empty(len(rows))
                distance[at:at + len(part)].copy_(score)
                labels[at:at + len(part)].copy_(ids)
        return distance, labels

    def _add_owned(self, rows, winner, sums, counts, objective, score=None, removal=None):
        owned = (winner >= self.centroid_begin) & (
            winner < self.centroid_begin + self.k
        )
        labels = winner[owned] - self.centroid_begin
        values = rows[owned].to(sums.dtype)
        if removal is not None:
            y = values.float() - self._readout()[labels]
            p = (y * self._axes[labels]).sum(1, keepdim=True)
            self._batch_removal.index_add_(0, labels, torch.cat(
                (p * y, removal[owned].nan_to_num(0, 0)[:, None]), dim=1))
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

    def _use_fused_owned_statistics(self):
        from .torch_backend import TritonBackend

        return isinstance(self._execution_backend, TritonBackend)

    def _accumulate_gathered(self, rows, prepared, sums, counts, objective, routing=None):
        second = (rows.new_empty(len(rows), dtype=torch.float32)
                  if self._removal_stats is not None
                  and self._epoch < len(self.recipe["epoch_rows"])
                  and not self._frozen() else None)
        distance, local_labels = self._local_assignment_batch(
            rows, prepared, routing, **({"second_out": second} if second is not None else {}))
        if second is None:
            winner, score = self._reduce_winners(distance, local_labels, return_score=True)
            removal = None
        else:
            winner, score, other = self._reduce_top2(distance, local_labels, second)
            removal = other - score
        from .quantizer import CastQuantizer

        # Use the packed training table; callers supplying separate closing
        # sums/counts or custom owner hooks retain the ordinary accumulation.
        packed = self._batch_accumulation
        if (rows.is_cuda and not self.spherical
                and self._use_fused_owned_statistics()
                and getattr(self._add_owned, "__func__", None) is EmeansShardedCentroids._add_owned
                and (self.quantizer is None or (type(self.quantizer) is CastQuantizer
                     and not self.quantizer.center and "decode" not in vars(self.quantizer)))
                and packed is not None and sums.data_ptr() == packed.data_ptr()
                and counts.data_ptr() == packed.data_ptr() + self.d * packed.element_size()):
            from .ivf_kernels import accumulate_owned

            if accumulate_owned(rows, winner, self._readout(), self._axes, packed,
                                self._batch_removal if removal is not None else None,
                                objective, removal, self.centroid_begin):
                return winner
        # Preserve source/chunk accumulation order and bound row scratch.
        local_n = len(rows) // self._group_size
        chunk = self.recipe["assignment_chunk_size"]
        for source in range(self._group_size):
            for begin in range(0, local_n, chunk):
                a = source * local_n + begin
                b = source * local_n + min(local_n, begin + chunk)
                values = statistics_values(rows[a:b], rows[a:b], self.quantizer)
                self._add_owned(
                    values, winner[a:b], sums, counts, objective,
                    *((score[a:b],) if self.spherical else ()),
                    **({"removal": removal[a:b]} if removal is not None else {}),
                )
        return winner

    def _local_batch_size(self):
        return super()._local_batch_size() // self._group_size

    def _accumulate(self, x, index=None, labels_out=None):
        stats = super()._accumulate(x, index, labels_out)
        # Pending statistics count local source rows; public n counts rows
        # contributed by every rank.
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
        rows, routing = self._gather_assignment_rows(x, prepared, index)
        if n:
            winner = self._accumulate_gathered(
                rows, prepared, sums, counts, objective, routing,
            )
            if labels is not None:
                labels.copy_(winner[self._rank * n:(self._rank + 1) * n])
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
        if self._removal_stats is not None:
            stats["removal"] = self._batch_removal
        if labels is not None:
            stats["labels"] = labels
        return stats

    def _choose_removal_donors(self, kept, usage, axis_norm):
        # Nearest-neighbor protection needs every owner; donor ranking does
        # not. The selected IDs are broadcast by _revive_unused afterward.
        if self._rank != 0:
            return usage.new_empty(0, dtype=torch.int64)
        return _partial_removal_donors(kept, usage, axis_norm)

    def _revive_unused(self, rng, *, chunk_rows=None):
        # Keep global donor selection, but stage at most 32 MiB of donor rows.
        # Sources are unique and disjoint from targets, so completing one
        # chunk cannot change a donor needed by a later chunk.
        if chunk_rows is None:
            chunk_rows = max(1, (32 << 20) // (self.d * self._centroids.element_size()))
        if chunk_rows < 1:
            raise ValueError("revival chunk_rows must be positive")
        if self._removal_stats is not None and self._frozen():
            return 0
        usage = self._usage
        source = target = None
        if self._removal_stats is not None:
            source, target = self._choose_sharded_removal_splits()
        else:
            parts = [torch.empty_like(usage) for _ in range(self._group_size)]
            dist.all_gather(parts, usage, group=self._group)
            full = torch.cat(parts)
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

        def exchange(values, send, recv):
            output = values.new_empty((sum(recv), *values.shape[1:]))
            dist.all_to_all_single(
                output, values.contiguous(), input_split_sizes=send,
                output_split_sizes=recv, group=self._group,
            )
            return output

        for begin in range(0, n, chunk_rows):
            src = source[begin:begin + chunk_rows]
            dst = target[begin:begin + chunk_rows]
            owned = (src >= self.centroid_begin) & (
                src < self.centroid_begin + self.k
            )
            src, dst = src[owned], dst[owned]
            local = src - self.centroid_begin
            m = len(local)
            split = QuantizedEmeans(self.d, 2 * m, quantizer=self.quantizer,
                                    spherical=self.spherical, assignment_backend=self._assignment_backend)
            # Split uses the same selected operation, but no assignment/cache mutation.
            split._execution_backend = self._execution_backend
            split._cast_fit_token = self._cast_fit_token
            # Only the source owner computes each pair. Retain the same split
            # arithmetic and representation separation as the bulk path.
            split._centroids = self._centroids.new_empty((2 * m, self.d))
            split._centroids[:m].copy_(self._centroids[local])
            split._centroids[m:].copy_(split._centroids[:m])
            mass = usage[local]
            split._usage = torch.cat((mass, mass))
            split._input_dtype = getattr(self, "_revival_dtype", self._input_dtype)
            if m:
                ids = torch.arange(m, device=usage.device)
                split._apply_splits(ids, ids + m)
                self._centroids[local] = split._centroids[:m]
                self._usage[local] = split._usage[:m]
                if self._removal_stats is not None:
                    self._removal_stats[local] /= 2
                    self._moved[local] = True

            # Send only children to their target owners. Empty senders and
            # receivers still enter every collective with matching splits.
            owner = dst // self.k
            order = torch.argsort(owner, stable=True)
            sent = torch.bincount(owner, minlength=self._group_size)
            received = torch.empty_like(sent)
            dist.all_to_all_single(received, sent, group=self._group)
            send, recv = sent.cpu().tolist(), received.cpu().tolist()
            targets = exchange(dst[order], send, recv) - self.centroid_begin
            payload = self._centroids.new_empty((m, self.d + 1))
            payload[:, :-1].copy_(split._centroids[m:])
            payload[:, -1].copy_(split._usage[m:])
            children = exchange(payload[order], send, recv)
            self._centroids[targets] = children[:, :-1]
            self._usage[targets] = children[:, -1]
            if self._removal_stats is not None:
                # Transfer the child's cost EMA and long axis from its donor.
                state = torch.cat((self._removal_stats[local], self._axes[local]), dim=1)
                children = exchange(state[order], send, recv)
                self._removal_stats[targets] = children[:, :self.d + 1]
                self._axes[targets] = children[:, self.d + 1:]
                self._moved[targets] = True
                # The next batch update refreshes all cached norms. A check
                # before another update must recompute the lengths.
                self._axis_norms_version = None
        self.n_splits += n
        self._invalidate_readout()
        return n

    def _epoch_objective(self):
        dist.all_reduce(self._objective, group=self._group)
        if self._input_dtype == torch.float8_e4m3fn:
            # _epoch_totals initializes the peak on every rank before reduction.
            dist.all_reduce(self._fp8_peak, op=dist.ReduceOp.MAX, group=self._group)
        return self._objective

    def _reports(self):
        return self.verbose and self._rank == 0

    def _describe_parallel(self):
        return "Using %d GPUs with class %s" % (self._group_size, type(self).__name__)

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
                self._report_epoch(stat, self._epoch - 1)
        except BaseException:
            self._failed = True
            self.assignments = None
            raise
        finally:
            self.verbose = verbose

    @torch.no_grad()
    def _fixed_closing_mean(self, batches, assignments, chunk_rows):
        """Rebuild saved owner statistics and score the returned FP32 centroids.

        Moved slots retain EMA positions. The error uses the retained statistics
        precision; sufficient-statistic cancellation can still lose small SSE.
        """
        if chunk_rows < 1:
            raise ValueError("closing chunk_rows must be positive")
        if (not isinstance(assignments, torch.Tensor)
                or assignments.device.type != "cpu"
                or assignments.dtype not in (torch.int32, torch.int64)
                or assignments.ndim != 1
                or len(assignments) != self._n // self._group_size):
            raise ValueError("closing assignments must be source-row CPU integer codes")
        if self.spherical:
            raise ValueError("saved-assignment closing currently requires spherical=False")
        started = time.perf_counter()
        expected = self._pass_counts.to(torch.int64)
        sums = self._pass_sums.zero_()
        counts = torch.zeros_like(expected)
        square = self._objective.new_zeros(())
        device = sums.device
        # Release assignment workspaces; input validation still uses batch dtype.
        self._batch_centroids = self._batch_accumulation = None
        self._batch_operand = self._batch_norm = self._stats_operand = None
        if hasattr(self, "_ivf"):
            self._ivf = None
        offset = 0

        def exchange(values, send, recv):
            output = values.new_empty((sum(recv), *values.shape[1:]))
            dist.all_to_all_single(
                output, values.contiguous(), input_split_sizes=send,
                output_split_sizes=recv, group=self._group,
            )
            return output

        for x in batches:
            x = self._prepare_input(x)
            if offset + len(x) > len(assignments):
                raise ValueError("closing scan exceeds global_n")
            for begin in range(0, len(x), chunk_rows):
                end = min(begin + chunk_rows, len(x))
                codes = assignments[offset + begin:offset + end].to(
                    device=device, dtype=torch.int64
                )
                torch._assert_async(
                    ((codes >= -1) & (codes < self.global_k)).all(),
                    "saved closing assignment is outside global centroid range",
                )
                valid = codes >= 0
                codes = codes[valid]
                encoded = self._wire_rows(x[begin:end])
                values = statistics_values(encoded, encoded, self.quantizer)
                rows = values[valid].to(sums.dtype)
                owner = codes // self.k
                order = torch.argsort(owner, stable=True)
                sent = torch.bincount(owner, minlength=self._group_size)
                received = torch.empty_like(sent)
                dist.all_to_all_single(received, sent, group=self._group)
                send, recv = sent.cpu().tolist(), received.cpu().tolist()
                rows = exchange(rows[order], send, recv)
                local = exchange(codes[order], send, recv) - self.centroid_begin
                # Empty receivers must still participate in all collectives.
                if len(local):
                    sums.index_add_(0, local, rows)
                    counts.index_add_(0, local, torch.ones_like(local))
                    square.add_((rows * rows).sum(dtype=torch.float64))
            offset += len(x)
        if offset != len(assignments):
            raise ValueError("closing scan must cover global_n rows")
        mismatch = torch.tensor(
            [not torch.equal(counts, expected)], dtype=torch.int64, device=device
        )
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=self._group)
        if mismatch.item():
            raise ValueError("closing counts differ from final-epoch assignments")
        explained = square.new_zeros(())
        correction = square.new_zeros(())
        # At most 48*d + 64 explicit temporary bytes per row, with a one-row floor.
        block_rows = max(1, (32 << 20) // (48 * self.d + 64))
        for begin in range(0, self.k, block_rows):
            end = min(begin + block_rows, self.k)
            block = sums[begin:end].double()
            count = counts[begin:end]
            divisor = count.clamp_min(1).double()
            explained.add_((block.square().sum(1) / divisor).sum())
            ideal = block / divisor[:, None]
            moved = None if self._moved is None else self._moved[begin:end]
            returned = _closing_centroid_values(
                sums[begin:end], count, self._centroids[begin:end], moved,
            )
            nonempty = count > 0
            delta = torch.where(nonempty[:, None], returned.double() - ideal, 0.0)
            correction.add_((delta.square().sum(1) * count.double()).sum())
            del block, count, divisor, ideal, moved, returned, nonempty, delta
        # Score before finalization releases the model-owned working tables.
        totals = torch.stack((square - explained + correction, counts.sum().double()))
        dist.all_reduce(totals, group=self._group)
        n = int(totals[1].item())
        self.closing_stats = {
            "rows": n, "assignment_mse": max(0.0, totals[0].item()) / max(1, n),
            "saved_assignments": True, "chunk_rows": chunk_rows,
        }
        self._pass_sums, self._pass_counts = sums, counts
        result = self.finalize()
        torch.cuda.synchronize(device)
        self.closing_stats["time_s"] = time.perf_counter() - started
        return result

    def final_closing_mean(self, batches, *, assignments=None, chunk_rows=65536):
        """Finalize means from saved labels or a fresh frozen-assignment scan.

        With ``assignments=None``, ``batches`` supplies equal rank-local CUDA
        contributions in any order, assigned against one frozen final snapshot.
        Every nonempty slot installs its fresh mean, even if moved during training.
        ``assignment_mse`` still scores decoded rows against the frozen FP32
        snapshot; spherical ``assignment_error`` retains the represented-IP
        ``2 + score`` units. Neither is a returned-centroid error in this mode.
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

        With ``assignments``, use final-epoch CPU int32/int64 global IDs in
        source-row order instead. ``batches`` must scan that same order. Rows
        are routed only to their saved centroid owner; no search is performed.
        Existing final-pass sum storage is reused at its training precision,
        and observed counts must match final-epoch assignments. ``-1`` marks rows
        omitted from a partial final epoch. In this saved mode,
        moved slots keep EMA positions, and ``assignment_mse`` is reconstructed
        against the actual returned FP32 centroids, with the existing SSE clamp.
        Routing uses at most ``chunk_rows``
        source rows per rank per exchange. This mode requires spherical=False.
        """
        if (
            self._failed
            or self._centroids is None
            or self._epoch != len(self.recipe["epoch_rows"])
            or self.centroids is not None
        ):
            raise RuntimeError("complete epochs before the closing scan")
        if self._pending_stats is not None:
            raise RuntimeError("consume pending statistics before closing")
        try:
            if assignments is not None:
                return self._fixed_closing_mean(batches, assignments, chunk_rows)
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
                rows, routing = self._gather_assignment_rows(x, prepared)
                if len(rows):
                    self._accumulate_gathered(rows, prepared, sums, counts, objective, routing)
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
            if self._moved is not None:
                self._moved.zero_()
            return self.finalize()
        except BaseException:
            self._failed = True
            self.assignments = None
            raise

    def train(self, x, init_centroids=None):
        """Train equal row shards; return local centroids and last-pass labels."""
        if self._failed or self.recipe is not None:
            raise RuntimeError("create a new object for another run or after failure")
        self.assignments = None
        if not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim != 2:
            raise ValueError("train requires a resident CUDA matrix")
        self._resolve_first_input(x)
        self._resolve_assignment_backend(x)
        self._execution_backend.require_training(x.device)
        n = len(x) * self._group_size
        if init_centroids is None:
            if not self._automatic and self.quantizer is not None:
                try:
                    check_codec_fit(self._group, self.quantizer, x.device)
                except BaseException:
                    self._failed = True
                    self.assignments = None
                    raise
            ids = torch.randperm(
                len(x),
                device=x.device,
                generator=_torch_generator(x.device, self.seed + self._rank),
            )[: self.k]
            init_centroids = self._initial_centroid_values(x[ids])
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
