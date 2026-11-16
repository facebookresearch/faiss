# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Original IVF assignment over the current centroid-owner lifecycle.

Index precision follows ivf_flat, including original grouped output precision
and its limitations. Each reader routes its local rows before gathering them.
"""

import torch
import torch.distributed as dist

from .ivf_assignment import _IVFAssignmentMixin
from .sharded_centroids import EmeansShardedCentroids

__all__ = ["EmeansShardedCentroidsIVF"]


class EmeansShardedCentroidsIVF(_IVFAssignmentMixin, EmeansShardedCentroids):
    """Sharded E-means using the reference Torch IVF-Flat assignment.

    All owners share a coarse quantizer trained once on a distributed sample
    of the initial centroids. Candidate lists are rebuilt for every frozen
    batch; query routing uses occupancy across all owners. Final means and
    batch SSE use original FP32 rows and winning FP32 centroids.
    CUDA fine search and split separation use BF16; CPU/Gloo uses FP32.
    Default revival uses the
    removal margin and nearest-neighbor protection within the probed lists.
    """

    _revival_dtype = torch.bfloat16

    def _use_fused_owned_statistics(self):
        # IVF owns assignment; owner statistics can still use Triton kernels.
        return True

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self.recipe["coarse_routing"] = "reader"

    def _train_coarse(self):
        return self._ivf_assignment.train_coarse(
            self._centroids, self.seed, self.recipe["assignment_chunk_size"],
            group=self._group)

    def _prepare_assignment(self):
        if self._ivf is None:
            self._build_ivf_index(self._centroids)
        else:
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
        return self._search_ivf(rows, prepared, second_out)

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
        return self._search_ivf(rows, prepared, second_out, coarse_idx=routing)
