# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""IVF assignment engines shared by local, replicated and owner trainers.

Search scores omit query norms. Reporting SSE belongs to the trainer and
uses its FP32 statistical rows and working centroids. Index rebuilds copy
candidate positions; revival queries deliberately use the last frozen index.
"""

import math

import numpy as np
import torch
import torch.distributed as dist

from .e_means_torch import _torch_generator
from .ivf_flat import IndexIVFFlat


class TorchIVFEngine(IndexIVFFlat):
    """Existing padded-BMM IVF with BF16 CUDA and native FP32 CPU search."""

    def __init__(self, coarse, **kwargs):
        super().__init__(coarse, **kwargs)
        if coarse.is_cuda:
            self.dtype = "bf16"
        else:
            self.implem = 2

    @property
    def coarse(self):
        return self.centroids


class FaissIVFEngine:
    """CPU FP32 native IVF scan, explicitly opting into Faiss tie ordering.

    Faiss may omit a lower-ID equal-distance candidate from its heap. Sorting
    the returned pair cannot restore it. This engine therefore advertises
    scan-order ties rather than the Torch engine's lowest-ID contract.
    """

    def __init__(self, coarse, *, nprobe=4, query_chunk=16384,
                 search_workspace_bytes=512 << 20):
        import faiss

        if coarse.device.type != "cpu" or coarse.dtype != torch.float32:
            raise ValueError("Faiss IVF requires FP32 CPU tensors")
        self.coarse = coarse.detach().contiguous().clone()
        self.nlist, self.d = self.coarse.shape
        self.nprobe, self.query_chunk = int(nprobe), int(query_chunk)
        self.search_workspace_bytes = search_workspace_bytes
        self.quantizer = faiss.IndexFlatL2(self.d)
        self.quantizer.add(self.coarse.numpy())
        self.index = faiss.IndexIVFFlat(self.quantizer, self.d, self.nlist)
        self.index.is_trained = True
        self.index.max_codes = 0
        self.index.parallel_mode = 3  # search_preassigned does not split queries in mode 0.
        self.list_sizes = torch.zeros(self.nlist, dtype=torch.int64)
        self._routing_nonempty = None

    def _rows(self, rows):
        if rows.device.type != "cpu" or rows.dtype != torch.float32:
            raise ValueError("Faiss IVF requires FP32 CPU tensors")
        return rows.detach().contiguous().numpy()

    def add(self, centroids):
        import faiss

        vectors = self._rows(centroids)
        membership = np.ascontiguousarray(self.quantizer.search(vectors, 1)[1][:, 0])
        self.index.reset()
        self.index.add_core(len(vectors), faiss.swig_ptr(vectors), None, faiss.swig_ptr(membership))
        self.list_sizes = torch.from_numpy(np.bincount(membership, minlength=self.nlist))
        self.set_routing_nonempty(self.list_sizes > 0)

    def set_routing_nonempty(self, nonempty):
        self._routing_nonempty = nonempty.detach().cpu().bool().clone()

    def route(self, rows):
        # Request all coarse IDs only when empty lists must be masked. This
        # keeps routing bounded by query_chunk*nlist and applies global
        # occupancy for sharded indexes, just as the Torch engine does.
        x = self._rows(rows)
        result = np.full((len(x), self.nprobe), -1, dtype=np.int64)
        occupied = self._routing_nonempty.numpy()
        for begin in range(0, len(x), self.query_chunk):
            end = min(len(x), begin + self.query_chunk)
            width = self.nprobe if occupied.all() else self.nlist
            _, ids = self.quantizer.search(x[begin:end], width)
            if width == self.nprobe:
                result[begin:end] = ids
            else:
                valid = occupied[ids]
                order = np.argsort(~valid, axis=1, kind="stable")[:, :self.nprobe]
                selected = np.take_along_axis(ids, order, axis=1)
                result[begin:end] = np.where(np.take_along_axis(valid, order, axis=1), selected, -1)
        return torch.from_numpy(result)

    def _search(self, rows, topk, coarse_idx):
        x = self._rows(rows)
        probes = self.route(rows) if coarse_idx is None else coarse_idx
        probes = np.ascontiguousarray(probes.cpu().numpy(), dtype=np.int64)
        if probes.shape != (len(x), self.nprobe):
            raise ValueError("coarse_idx must have shape (n_queries, nprobe)")
        distances = np.full((len(x), topk), np.inf, dtype=np.float32)
        labels = np.full((len(x), topk), -1, dtype=np.int64)
        self.index.nprobe = self.nprobe
        for begin in range(0, len(x), self.query_chunk):
            end = min(len(x), begin + self.query_chunk)
            d, ids = self.index.search_preassigned(
                x[begin:end], topk, probes[begin:end],
                np.zeros((end - begin, self.nprobe), dtype=np.float32))
            # The IVF-Flat scanner does not consume coarse distances.
            d -= np.einsum("ij,ij->i", x[begin:end], x[begin:end])[:, None]
            d[ids < 0] = np.inf
            order = np.lexsort((ids, d), axis=1)
            distances[begin:end] = np.take_along_axis(d, order, axis=1)
            labels[begin:end] = np.take_along_axis(ids, order, axis=1)
        return torch.from_numpy(distances), torch.from_numpy(labels)

    def search_1nn(self, rows, coarse_idx=None):
        scores, ids = self._search(rows, 1, coarse_idx)
        return scores[:, 0], ids[:, 0]

    def search_2nn(self, rows, coarse_idx=None):
        return self._search(rows, 2, coarse_idx)


class IVFAssignment:
    """Coarse-training policy and engine factory; owns no update rules/group."""

    def __init__(self, d, k, *, nlist=256, nprobe=4, coarse_niter=10,
                 search_workspace_bytes=512 << 20, engine="torch", tie_policy="lowest_id"):
        if not 1 <= nprobe <= nlist <= k:
            raise ValueError("require 1 <= nprobe <= nlist <= k")
        if coarse_niter < 1 or search_workspace_bytes < 1:
            raise ValueError("coarse iterations and workspace must be positive")
        if engine not in ("torch", "faiss"):
            raise ValueError("ivf_engine must be 'torch' or 'faiss'")
        if tie_policy != ("lowest_id" if engine == "torch" else "faiss"):
            raise ValueError("Torch IVF requires tie_policy='lowest_id'; Faiss IVF requires explicit tie_policy='faiss'")
        self.d, self.k = int(d), int(k)
        self.nlist, self.nprobe, self.coarse_niter = int(nlist), int(nprobe), int(coarse_niter)
        self.search_workspace_bytes = int(search_workspace_bytes)
        self.engine, self.tie_policy = engine, tie_policy
        self.query_chunk = max(1, min(16384,
            self.search_workspace_bytes // (4 * self.d + 12 * self.nlist)))
        self.index = None
        self.version = 0

    def train_coarse(self, centroids, seed, chunk_size, *, group=None):
        """Seeded Lloyd sample; a group means disjoint centroid ownership.

        CUDA retains the existing FP16 Triton coarse assignment. CPU uses
        FP32 Torch assignment. A local call needs no process group.
        """
        from .torch_backend import TorchBackend, _get_triton_assign_rows

        ranks = dist.get_world_size(group) if group is not None else 1
        rank = dist.get_rank(group) if group is not None else 0
        local_k, device = len(centroids), centroids.device
        selected = torch.randperm(local_k * ranks, device="cpu",
            generator=_torch_generator("cpu", seed))[:self.nlist]
        coarse = centroids.new_empty((self.nlist, self.d))
        for owner in range(ranks):
            mask = (selected >= owner * local_k) & (selected < (owner + 1) * local_k)
            local = (selected[mask] - owner * local_k).to(device)
            rows = centroids[local].contiguous() if owner == rank else coarse.new_empty((len(local), self.d))
            if group is not None:
                dist.broadcast(rows, src=dist.get_global_rank(group, owner), group=group)
            coarse[mask.to(device)] = rows
        sample_ids = torch.randperm(local_k, device=device,
            generator=_torch_generator(device, seed + rank))[:min(local_k, math.ceil(20 * self.nlist / ranks))]
        sample = centroids[sample_ids]
        packed = coarse.new_zeros((self.nlist, self.d + 1))
        backend = TorchBackend(self.nlist)
        for _ in range(self.coarse_niter):
            represented = coarse.half() if centroids.is_cuda else coarse
            norms = represented.float().square().sum(1)
            packed.zero_()
            for begin in range(0, len(sample), chunk_size):
                rows = sample[begin:begin + chunk_size]
                if centroids.is_cuda:
                    labels = _get_triton_assign_rows()(rows.half(), represented, norms)[1].long()
                else:
                    labels = backend.assign(rows, represented, norms)[1]
                packed[:, :-1].index_add_(0, labels, rows)
                packed[:, -1].index_add_(0, labels, rows.new_ones(len(rows)))
            if group is not None:
                dist.all_reduce(packed, group=group)
            counts = packed[:, -1]
            coarse = torch.where((counts > 0)[:, None], packed[:, :-1] / counts.clamp_min(1)[:, None], coarse)
        return coarse

    def create_index(self, coarse):
        engine = TorchIVFEngine if self.engine == "torch" else FaissIVFEngine
        return engine(coarse, nprobe=self.nprobe, query_chunk=self.query_chunk,
                      search_workspace_bytes=self.search_workspace_bytes)

    def index_build(self, centroids, *, coarse=None):
        if self.index is None:
            if coarse is None:
                raise ValueError("supply the trained coarse centroids for the first build")
            self.index = self.create_index(coarse)
        self.index.add(centroids)
        self.index.centroid_version = self.version
        return self.index

    def invalidate(self):
        # Preserve the last frozen snapshot for revival protection.
        self.version += 1


class _IVFAssignmentMixin:
    """Configuration and search delegation shared by local and owner IVF."""

    def __init__(self, *args, nlist=256, nprobe=4, coarse_niter=10,
                 search_workspace_bytes=512 << 20, ivf_engine="torch",
                 ivf_tie_policy="lowest_id", **kwargs):
        if kwargs.get("spherical", False):
            raise ValueError("IVF does not support spherical=True")
        kwargs.setdefault("quantizer", None)
        if kwargs["quantizer"] is not None:
            raise ValueError("IVF requires quantizer=None")
        super().__init__(*args, **kwargs)
        self._ivf_assignment = IVFAssignment(self.d, getattr(self, "global_k", self.k),
            nlist=nlist, nprobe=nprobe, coarse_niter=coarse_niter,
            search_workspace_bytes=search_workspace_bytes, engine=ivf_engine,
            tie_policy=ivf_tie_policy)
        self.nlist, self.nprobe, self.coarse_niter = int(nlist), int(nprobe), int(coarse_niter)
        self.search_workspace_bytes = int(search_workspace_bytes)
        self.ivf_engine, self.ivf_tie_policy = ivf_engine, ivf_tie_policy
        self._coarse_query_chunk = self._ivf_assignment.query_chunk
        self._ivf = None

    def _resolve_removal_revival(self):
        if self.unused_threshold is None and getattr(self, "global_k", self.k) > 1:
            self._unused_threshold = None

    def _resolve_assignment_backend(self, x):
        if self.assignment_backend == "auto":
            self._resolved_assignment_backend = "native"
        else:
            super()._resolve_assignment_backend(x)

    def begin(self, global_n, init_centroids):
        if self.ivf_engine == "faiss" and init_centroids.is_cuda:
            raise ValueError("Faiss IVF requires CPU tensors")
        self._revival_dtype = torch.bfloat16 if init_centroids.is_cuda else torch.float32
        super().begin(global_n, init_centroids)
        self.recipe.update(index_factory=f"IVF{self.nlist},Flat", nprobe=self.nprobe,
            fine_dtype="bf16" if init_centroids.is_cuda else "fp32",
            coarse_query_chunk=self._coarse_query_chunk, revival_protection="ivf",
            ivf_engine=self.ivf_engine, ivf_tie_policy=self.ivf_tie_policy,
            ivf_capture=False)

    def _build_ivf_index(self, centroids):
        if self._ivf is None:
            self._ivf = self._ivf_assignment.index_build(centroids, coarse=self._train_coarse())
        else:
            self._ivf_assignment.index = self._ivf
            self._ivf_assignment.index_build(centroids)
        return self._ivf

    def _search_ivf(self, rows, prepared, second_out=None, coarse_idx=None):
        if second_out is None:
            return prepared.search_1nn(rows, coarse_idx=coarse_idx)
        scores, ids = prepared.search_2nn(rows, coarse_idx=coarse_idx)
        second_out.copy_(scores[:, 1])
        return scores[:, 0], ids[:, 0]

    def _finish_step(self, expected_n):
        super()._finish_step(expected_n)
        self._ivf_assignment.invalidate()

    def _update_removal_statistics(self, alpha, decay):
        if not self._removal_stats.is_cuda:
            return False
        from .triton_assignment import removal_update

        return removal_update(self._removal_stats, self._batch_removal, self._axes,
                              alpha, decay, self._axis_norms)

    def _replay_eligible(self, x):
        return False

    def _device_check(self):
        # IVF protection queries the frozen index eagerly.
        return False
