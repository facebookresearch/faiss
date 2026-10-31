# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Original IVF assignment over the current centroid-owner lifecycle.

Index precision follows ivf_flat, including original grouped output precision
and its limitations. Original local routing is retained on every owner.
"""

import math

import torch
import torch.distributed as dist

from .e_means import _torch_generator
from .sharded_centroids import EmeansShardedCentroids

__all__ = ["EmeansShardedCentroidsIVF"]


class EmeansShardedCentroidsIVF(EmeansShardedCentroids):
    """Sharded E-means using the reference Torch IVF-Flat assignment.

    All owners share a coarse quantizer trained once on a distributed sample
    of the initial centroids. Candidate lists are rebuilt for every frozen
    batch; query routing uses occupancy across all owners. Final means and
    batch SSE use original FP32 rows and winning FP32 centroids.
    """

    _revival_dtype = torch.float16

    def __init__(
        self,
        *args,
        nlist=256,
        nprobe=4,
        coarse_niter=10,
        search_workspace_bytes=512 << 20,
        **kwargs,
    ):
        if kwargs.get("spherical", False) or (len(args) > 13 and args[13]):
            raise ValueError("IVF does not support spherical=True")
        super().__init__(*args, **kwargs)
        if not 1 <= nprobe <= nlist <= self.global_k:
            raise ValueError("require 1 <= nprobe <= nlist <= global_k")
        if coarse_niter < 1 or search_workspace_bytes < 1:
            raise ValueError("coarse iterations and workspace must be positive")
        self.nlist, self.nprobe = int(nlist), int(nprobe)
        self.coarse_niter = int(coarse_niter)
        self.search_workspace_bytes = int(search_workspace_bytes)
        self._ivf = None

    def _resolve_assignment_backend(self, x):
        if self.assignment_backend == "auto":
            self._resolved_assignment_backend = "native"
        else:
            super()._resolve_assignment_backend(x)

    def reset(self):
        super().reset()
        self._ivf = None

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self._ivf = None
        self.recipe.update(
            index_factory=f"IVF{self.nlist},Flat", nprobe=self.nprobe
        )

    def _train_coarse(self):
        from .e_means import _get_triton_assign_rows

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
                search_workspace_bytes=self.search_workspace_bytes,
            )
        self._ivf.add(self._centroids)
        occupancy = self._ivf.list_sizes.long()
        dist.all_reduce(occupancy, group=self._group)
        self._ivf.set_routing_nonempty(occupancy > 0)
        return self._ivf

    def _local_assignment(self, rows, prepared):
        return prepared.search_1nn(rows)
