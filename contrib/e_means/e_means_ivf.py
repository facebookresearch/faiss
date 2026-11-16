# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Local CPU/CUDA and replicated CPU/Gloo E-means with IVF assignment."""

import torch
import torch.distributed as dist

from .e_means_quantized import QuantizedEmeans
from .e_means_distributed import _ReplicatedCentroids
from .e_means_torch import Emeans
from .ivf_assignment import _IVFAssignmentMixin

__all__ = ["EmeansIVF", "DistributedEmeansIVF"]


class EmeansIVF(_IVFAssignmentMixin, QuantizedEmeans):
    """IVF E-means on one CPU or GPU, without a process group.

    Rows and statistical state are FP32. CUDA fine assignment uses BF16;
    CPU search uses FP32. ``ivf_engine='torch'`` has lowest-ID ties. To opt
    into native Faiss CPU scanning, also specify ``ivf_tie_policy='faiss'``.
    Spherical clustering and external quantizers are not supported.

    Both resident ``train`` and streaming ``begin/accumulate/update`` reuse
    the ordinary E-means lifecycle, schedules, revival and closing handover.
    IVF snapshots are rebuilt once per logical batch. Evaluation builds its
    own index, and revival queries the last frozen training index.
    """

    def _resolve_assignment_backend(self, x):
        if self.assignment_backend == "cute":
            raise ValueError("local IVF does not use the CuTe exact-assignment backend")
        super()._resolve_assignment_backend(x)

    def _resolve_input_dtype(self, x):
        if x.dtype != torch.float32:
            raise ValueError("local IVF training/query rows must be FP32")
        self._bind_input_dtype(torch.float32)

    def _train_coarse(self):
        return self._ivf_assignment.train_coarse(self._centroids, self.seed,
                                                 self.recipe["assignment_chunk_size"])

    def _prepare_torch_assignment(self, centroids):
        return self._build_ivf_index(centroids), None

    def _accumulate_core(self, x, index=None, raw=False, labels_out=None, ema=None):
        # IVF has FP32 floating statistics even when revival is disabled;
        # use the shared objective hook rather than encoded-row accumulation.
        return Emeans._accumulate_core(self, x, index, raw, labels_out, ema)

    def _prepare_assignment_query(self, centroids):
        # A shallow query fork must never rebuild the training index. Copy
        # only the fixed coarse table; candidate storage belongs to this call.
        coarse = self._ivf.coarse if self._ivf is not None else self._ivf_assignment.train_coarse(
            centroids, self.seed, self._get_assignment_chunk_size())
        index = self._ivf_assignment.create_index(coarse)
        index.add(centroids)
        return index, None

    def _assign_rows(self, rows, centroids, centroid_norm=None,
                     distance_out=None, labels_out=None, second_out=None):
        score, labels = self._search_ivf(rows, centroids, second_out)
        labels = labels.long()
        distance = score
        if distance_out is not None:
            distance_out.copy_(distance)
            distance = distance_out
        if labels_out is not None:
            labels_out.copy_(labels)
            labels = labels_out
        return distance, labels

    def _assignment_objective(self, rows, labels, distance):
        # Keep raw search margins unchanged; SSE is computed separately from
        # statistical rows and the FP32 working centroid table.
        residual = rows - self._readout()[labels.clamp_min(0)]
        return torch.where((labels >= 0)[:, None], residual.square(), torch.inf).sum(dtype=torch.float64)

    def _assign(self, x):
        x = x.detach()
        _, labels = super()._assign(x)
        distance = x.new_empty(len(x))
        for begin in range(0, len(x), self._get_assignment_chunk_size()):
            end = min(len(x), begin + self._get_assignment_chunk_size())
            ids = labels[begin:end]
            residual = x[begin:end] - self.centroids[ids.clamp_min(0)]
            distance[begin:end] = torch.where(ids >= 0, residual.square().sum(1), torch.inf)
        return distance, labels

    def _nearest_other(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.int64, device=self._centroids.device)
        if not len(ids):
            return ids.cpu().numpy()
        index = self._ivf if self._ivf is not None else self._build_ivf_index(self._centroids)
        result = []
        chunk_rows = max(1, min(8192, (32 << 20) // (4 * self.d)))
        for chunk in ids.split(chunk_rows):
            scores, labels = index.search_2nn(self._centroids[chunk])
            valid = (labels >= 0) & (labels != chunk[:, None]) & torch.isfinite(scores)
            score, at = torch.where(valid, scores, torch.inf).min(dim=1)
            nearest = labels.gather(1, at[:, None]).flatten().long()
            result.append(torch.where(torch.isfinite(score), nearest, torch.iinfo(torch.int64).max))
        return torch.cat(result).cpu().numpy()

    def _separate_split_values(self, donor_centroids, child_centroids):
        # Statistics stay FP32, but CUDA split separation must be visible
        # to the BF16 fine assignment, matching the existing sharded class.
        from .torch_backend import _separate_splits

        _separate_splits(donor_centroids, child_centroids, self._revival_dtype)

    def assign(self, x):
        # Ordinary assign forks the tensor backend. Its IVF preparation hook
        # creates independent candidate storage, even during a pending batch.
        self.assignment_recipe = None
        if self.centroids is not None:
            result = super().assign(x)
        else:
            import copy

            self._require_running()
            evaluation = copy.copy(self)
            evaluation.centroids = self.current_centroids().clone()
            result = Emeans.assign(evaluation, x)
            self.assignment_recipe = evaluation.assignment_recipe
        self.assignment_recipe.update(index_factory=f"IVF{self.nlist},Flat",
            nprobe=self.nprobe, ivf_engine=self.ivf_engine,
            ivf_tie_policy=self.ivf_tie_policy,
            fine_dtype="bf16" if x.is_cuda else "fp32", objective="fp32_sse")
        return result


class DistributedEmeansIVF(_ReplicatedCentroids, EmeansIVF):
    """Replicated FP32 CPU/Gloo IVF with globally reduced removal/Oja sums.

    Each rank supplies equal local shares through the streaming lifecycle.
    ``transport`` selects shared dense/sparse reduction of sums/counts and
    removal/Oja contributions before the global EMA.
    Root chooses revival splits and all replicas apply the same state changes.
    """

    _removal_revival = True

    def begin(self, global_n, init_centroids):
        if init_centroids.device.type != "cpu":
            raise ValueError("DistributedEmeansIVF currently requires CPU/Gloo")
        super().begin(global_n, init_centroids)
        self.recipe.update(centroid_replicas=self._group_size,
                           removal_transport=self._transport, coarse_routing="reader")

    def _train_coarse(self):
        if self._rank == 0:
            coarse = EmeansIVF._train_coarse(self)
        else:
            coarse = self._centroids.new_empty((self.nlist, self.d))
        dist.broadcast(coarse, src=dist.get_global_rank(self._group, 0), group=self._group)
        return coarse

