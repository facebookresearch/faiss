# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Original IVF assignment over the current centroid-owner lifecycle.

Index precision follows ivf_flat, including original grouped output precision
and its limitations. Each reader routes its local rows before gathering them.
"""

import math

import torch
import torch.distributed as dist

from .e_means_torch import _torch_generator
from .sharded_centroids import EmeansShardedCentroids

__all__ = ["EmeansShardedCentroidsIVF"]


class EmeansShardedCentroidsIVF(EmeansShardedCentroids):
    """Sharded E-means using the reference Torch IVF-Flat assignment.

    All owners share a coarse quantizer trained once on a distributed sample
    of the initial centroids. Candidate lists are rebuilt for every frozen
    batch; query routing uses occupancy across all owners. Final means and
    batch SSE use original FP32 rows and winning FP32 centroids.
    Fine search and split separation use BF16. Default revival uses the
    removal margin and nearest-neighbor protection within the probed lists.
    """

    _revival_dtype = torch.bfloat16

    def _use_fused_owned_statistics(self):
        # IVF owns assignment; owner statistics can still use Triton kernels.
        return True

    def _update_removal_statistics(self, alpha, decay):
        if not self._removal_stats.is_cuda:
            return False
        from .triton_assignment import removal_update

        return removal_update(
            self._removal_stats, self._batch_removal, self._axes, alpha, decay,
            self._axis_norms)

    def _resolve_removal_revival(self):
        # Fine assignment is BF16; row statistics retain original FP32 units.
        if self.unused_threshold is None and self.global_k > 1:
            self._unused_threshold = None

    def __init__(
        self,
        *args,
        nlist=256,
        nprobe=4,
        coarse_niter=10,
        search_workspace_bytes=512 << 20,
        **kwargs,
    ):
        if kwargs.get("spherical", False):
            raise ValueError("IVF does not support spherical=True")
        kwargs.setdefault("quantizer", None)
        if kwargs["quantizer"] is not None:
            raise ValueError("IVF sharding requires quantizer=None")
        super().__init__(*args, **kwargs)
        if not 1 <= nprobe <= nlist <= self.global_k:
            raise ValueError("require 1 <= nprobe <= nlist <= global_k")
        if coarse_niter < 1 or search_workspace_bytes < 1:
            raise ValueError("coarse iterations and workspace must be positive")
        self.nlist, self.nprobe = int(nlist), int(nprobe)
        self.coarse_niter = int(coarse_niter)
        self.search_workspace_bytes = int(search_workspace_bytes)
        self._coarse_query_chunk = max(1, min(
            16384, self.search_workspace_bytes // (2 * self.d + 12 * self.nlist)))
        self._ivf = None

    def _resolve_assignment_backend(self, x):
        if self.assignment_backend == "auto":
            self._resolved_assignment_backend = "native"
        else:
            super()._resolve_assignment_backend(x)

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self.recipe.update(
            index_factory=f"IVF{self.nlist},Flat", nprobe=self.nprobe,
            coarse_routing="reader",
            fine_dtype="bf16",
            coarse_query_chunk=self._coarse_query_chunk,
            revival_protection="ivf",
        )

    def _train_coarse(self):
        from .torch_backend import _get_triton_assign_rows

        device = self._centroids.device
        selected = torch.randperm(
            self.global_k,
            device="cpu",
            generator=_torch_generator("cpu", self.seed),
        )[: self.nlist]
        coarse = self._centroids.new_empty((self.nlist, self.d))
        for rank in range(self._group_size):
            mask = (selected >= rank * self.k) & (
                selected < (rank + 1) * self.k
            )
            local = (selected[mask] - rank * self.k).to(device)
            rows = (
                self._centroids[local].contiguous()
                if rank == self._rank
                else coarse.new_empty((len(local), self.d))
            )
            dist.broadcast(
                rows,
                src=dist.get_global_rank(self._group, rank),
                group=self._group,
            )
            coarse[mask.to(device)] = rows
        sample_ids = torch.randperm(
            self.k,
            device=device,
            generator=_torch_generator(device, self.seed + self._rank),
        )[: min(self.k, math.ceil(20 * self.nlist / self._group_size))]
        sample = self._centroids[sample_ids]
        packed = coarse.new_zeros((self.nlist, self.d + 1))
        chunk = self.recipe["assignment_chunk_size"]
        for _ in range(self.coarse_niter):
            represented = coarse.half()
            norms = represented.float().square().sum(1)
            packed.zero_()
            for start in range(0, len(sample), chunk):
                rows = sample[start : start + chunk]
                labels = _get_triton_assign_rows()(
                    rows.half(), represented, norms
                )[1].long()
                packed[:, :-1].index_add_(0, labels, rows)
                packed[:, -1].index_add_(0, labels, rows.new_ones(len(rows)))
            dist.all_reduce(packed, group=self._group)
            counts = packed[:, -1]
            coarse = torch.where(
                (counts > 0)[:, None],
                packed[:, :-1] / counts.clamp_min(1)[:, None],
                coarse,
            )
        return coarse

    def _prepare_assignment(self):
        from .ivf_flat import IndexIVFFlat

        if self._ivf is None:
            self._ivf = IndexIVFFlat(
                self._train_coarse(),
                nprobe=self.nprobe,
                # Larger coarse-query blocks reduce Python/kernel launch
                # overhead; cap their represented rows and score temporaries.
                query_chunk=self._coarse_query_chunk,
                search_workspace_bytes=self.search_workspace_bytes,
            )
            self._ivf.dtype = "bf16"
        self._ivf.add(self._centroids)
        occupancy = self._ivf.list_sizes.long()
        dist.all_reduce(occupancy, group=self._group)
        self._ivf.set_routing_nonempty(occupancy > 0)
        return self._ivf

    def _nearest_other(self, ids, *, return_tensor=False):
        """Protect a different centroid in the IVF probes across all owners.

        Reuse the frozen training index: queries use updated positions, but
        indexed positions and list memberships precede the batch update.
        Two local results suffice to exclude self before reducing the best
        remaining score and ID. An isolated query protects no one
        (INT64_MAX); there is no exhaustive fallback.
        """
        centroids = self._centroids
        ids = torch.as_tensor(ids, dtype=torch.int64, device=centroids.device)
        if not len(ids):
            return ids if return_tensor else ids.cpu().numpy()
        prepared = self._ivf
        if prepared is None:
            prepared = self._prepare_assignment()
        result = []
        chunk_rows = max(1, min(8192, (32 << 20) // (4 * self.d)))
        for chunk in ids.split(chunk_rows):
            owned = (chunk >= self.centroid_begin) & (
                chunk < self.centroid_begin + self.k
            )
            queries = centroids.new_zeros((len(chunk), self.d))
            queries[owned] = centroids[chunk[owned] - self.centroid_begin]
            dist.all_reduce(queries, group=self._group)
            # Queries and the coarse quantizer are identical on every owner.
            # Route once, then share compact probe IDs instead of repeating it.
            if self._rank == 0:
                routing = prepared.route(queries).to(torch.int32)
            else:
                routing = torch.empty((len(chunk), prepared.nprobe),
                                      dtype=torch.int32, device=queries.device)
            dist.broadcast(routing, src=dist.get_global_rank(self._group, 0), group=self._group)
            scores, labels = prepared.search_2nn(queries, coarse_idx=routing)
            valid = (labels >= 0) & (
                labels.long() + self.centroid_begin != chunk[:, None]
            ) & torch.isfinite(scores)
            scores = torch.where(valid, scores, torch.inf)
            # search_2nn orders equal scores by local ID, so min keeps that tie.
            score, at = scores.min(dim=1)
            labels = labels.gather(1, at[:, None]).flatten()
            labels = torch.where(torch.isfinite(score), labels, -1)
            result.append(self._reduce_winners(score, labels))
        nearest = torch.cat(result)
        return nearest if return_tensor else nearest.cpu().numpy()

    def _local_assignment(self, rows, prepared, second_out=None):
        if second_out is None:
            return prepared.search_1nn(rows)
        scores, ids = prepared.search_2nn(rows)
        second_out.copy_(scores[:, 1])
        return scores[:, 0], ids[:, 0]

    def _gather_assignment_rows(self, x, prepared, index=None):
        local = (x if index is None else x[index]).contiguous()
        if not len(local):
            return self._gather_training_rows(local), None
        # Preparation installs global occupancy, so all readers route against
        # the same lists, including lists empty on their own centroid shard.
        dtype = torch.int32 if self.nlist <= torch.iinfo(torch.int32).max else torch.int64
        routing = prepared.route(local).to(dtype)
        rows = self._gather_training_rows(local)
        return rows, self._gather_local(routing)

    def _local_assignment_batch(self, rows, prepared, routing=None, second_out=None):
        # IVF bounds its own search workspace; plan all queries together.
        if second_out is None:
            return prepared.search_1nn(rows, coarse_idx=routing)
        scores, ids = prepared.search_2nn(rows, coarse_idx=routing)
        second_out.copy_(scores[:, 1])
        return scores[:, 0], ids[:, 0]
