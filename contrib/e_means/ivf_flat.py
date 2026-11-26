# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Torch IVF-Flat assignment, adapted from the reference anns_indexing.py.

Coarse assignment uses the existing e-means Triton kernel on CUDA. Routing and
the search inside the probed lists run in ivf_fine.
"""

import contextlib

import numpy as np
import torch

from .torch_backend import _get_triton_assign_rows

def _dot_fp32(x, y):
    """Keep represented operands; avoid reduced-precision dot-output rounding."""
    operation = torch.bmm if x.ndim == 3 else torch.mm
    if x.dtype in (torch.float16, torch.bfloat16):
        if x.is_cuda:
            return operation(x, y, out_dtype=torch.float32)
        return operation(x.float(), y.float())
    return operation(x, y).float()


def _check_int32_count(n, name):
    if n > torch.iinfo(torch.int32).max:
        raise ValueError(f"{name} must fit signed INT32 offsets")


def _assign_body(xq_lp, xb_t_lp, xb_norms):
    dis = xb_norms - 2 * _dot_fp32(xq_lp, xb_t_lp)
    return dis.min(dim=1)


def _topk_body(xq_lp, xb_t_lp, xb_norms, k):
    dis = xb_norms - 2 * _dot_fp32(xq_lp, xb_t_lp)
    return torch.topk(dis, k, dim=1, largest=False)


def _bucket_sort(idx, n):
    assert idx.ndim == 1
    _check_int32_count(idx.numel(), "bucket rows")
    counts = torch.bincount(idx, minlength=n)
    lims = torch.empty(n + 1, dtype=torch.int32, device=idx.device)
    lims[0] = 0
    torch.cumsum(counts, dim=0, out=lims[1:])
    buckets = torch.argsort(idx, stable=True)
    return buckets, lims


class IndexIVFFlat:
    """Original native/FP16/BF16 IVF-Flat assignment and list planning.

    Coarse centroids are fixed. Call ``add`` after each EMA update to rebuild
    the local lists. A sharded caller supplies global occupancy before routing.
    Query norms are omitted from returned scores; scores are comparable across
    owners searching the same row. Accumulate SSE against FP32 winning rows.
    MM/BMM reduced operands produce FP32 dots. Grouped MM retains original
    output precision, including FP16 overflow and BF16 rounding limitations.
    """

    _IMPLEMS = frozenset(range(1, 6))
    _DTYPES = frozenset(("fp16", "bf16"))
    _LOWP_SCOPES = frozenset(("all", "fine"))

    @property
    def implem(self):
        return self._implem

    @implem.setter
    def implem(self, value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value not in self._IMPLEMS
        ):
            raise ValueError("implem must be one of 1, 2, 3, 4, or 5")
        self._set_precision_selector("_implem", int(value))

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if type(value) is not str or value not in self._DTYPES:
            raise ValueError("dtype must be 'fp16' or 'bf16'; implem < 3 selects native")
        self._set_precision_selector("_dtype", value)

    @property
    def lowp_scope(self):
        return self._lowp_scope

    @lowp_scope.setter
    def lowp_scope(self, value):
        if type(value) is not str or value not in self._LOWP_SCOPES:
            raise ValueError("lowp_scope must be 'all' or 'fine'")
        self._set_precision_selector("_lowp_scope", value)

    def _set_precision_selector(self, name, value):
        setattr(self, name, value)

    def __init__(
        self,
        centroids,
        nprobe=4,
        *,
        query_chunk=2048,
    ):
        if centroids.ndim != 2 or not len(centroids):
            raise ValueError("centroids must be a nonempty matrix")
        if not 1 <= nprobe <= len(centroids):
            raise ValueError("nprobe must be between 1 and nlist")
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.centroids = centroids.float().contiguous()
        self.nlist, self.d = centroids.shape
        self.nprobe = nprobe
        self.query_chunk = query_chunk
        self._implem, self._dtype, self._lowp_scope = 4, "fp16", "all"
        # Retain the original visible tensor as the fixed FP16 CUDA-add norms.
        self.centroids_norms = self._sq_norms(self.centroids, "coarse")
        self._query_norms_key = self._query_centroids_norms = None
        self._routing_norms_key = self.routing_norms = None
        self._routing_nonempty = None
        self._fine_storage_key = None
        self.ndis = self.coarse_ndis = 0
        self.ids = None

    def _timed(self, name):
        return contextlib.nullcontext()

    def _chunk_body(self, kind):
        return _assign_body if kind == "assign" else _topk_body

    def _assign_to_coarse(self, x):
        if x.is_cuda:
            return _get_triton_assign_rows()(
                x.half(), self.centroids.half(), self.centroids_norms
            )[1].long()
        return self._exact_1nn(
            x,
            self.centroids,
            xb_norms=self._current_query_norms(),
            stage="coarse",
        )[1]

    def search_1nn(self, x, coarse_idx=None):
        """Return the scores and IDs of the nearest candidates, shaped (n,)."""
        scores, ids = self.search_2nn(x, coarse_idx)
        return scores[:, 0].contiguous(), ids[:, 0].contiguous()

    def search_2nn(self, x, coarse_idx=None):
        """Return scores and IDs of two distinct candidates, shaped (n, 2).

        Search the routed lists once (ivf_fine), merging both results across
        probes. Ties use the lower candidate ID. Missing entries have score inf
        and ID -1; query norms are omitted.
        """
        if self.ids is None:
            raise RuntimeError("add candidates before search")
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError("queries must have shape (n, d)")
        self._check_fine_storage()
        _check_int32_count(x.shape[0] * self.nprobe, "routed query rows")
        idx = self.route(x) if coarse_idx is None else coarse_idx
        if idx.shape != (x.shape[0], self.nprobe):
            raise ValueError("coarse_idx must have shape (len(x), nprobe)")
        idx = idx.to(device=x.device, dtype=torch.int64)
        self.ndis += self.list_sizes[idx.ravel()].sum()
        from .ivf_fine import search_2nn

        return search_2nn(self, x, idx)

    def _mm_dtype(self, stage):
        assert stage in ("coarse", "fine")
        if self.implem < 3:
            return None
        if stage == "coarse" and self.lowp_scope != "all":
            return None
        return torch.bfloat16 if self.dtype == "bf16" else torch.float16

    def _policy_key(self, stage):
        dtype = self._mm_dtype(stage)
        return "native" if dtype is None else self.dtype

    def _current_query_norms(self):
        key = self._policy_key("coarse")
        if self._query_norms_key != key:
            self._query_centroids_norms = self._sq_norms(
                self.centroids, "coarse"
            )
            self._query_norms_key = key
        return self._query_centroids_norms

    def _current_routing_norms(self):
        if self._routing_nonempty is None:
            raise RuntimeError("add candidates or set routing occupancy before route")
        key = self._policy_key("coarse")
        if self._routing_norms_key != key:
            norms = self._current_query_norms()
            self.routing_norms = torch.where(
                self._routing_nonempty,
                norms.float(),
                torch.full_like(norms, torch.inf, dtype=torch.float32),
            )
            self._routing_norms_key = key
        return self.routing_norms

    def _check_fine_storage(self):
        current = self._policy_key("fine")
        if self._fine_storage_key != current:
            raise RuntimeError(
                "fine precision changed after add; call add again to rebuild "
                f"candidate storage ({self._fine_storage_key} -> {current})"
            )

    def _as_computed(self, x, stage):
        """Round ``x`` to the precision the selected matmul will see."""
        dt = self._mm_dtype(stage)
        return x if dt is None else x.to(dt).float()

    def _sq_norms(self, x, stage):
        dtype = torch.float32 if self._mm_dtype(stage) is not None else x.dtype
        item_bytes = torch.empty((), dtype=dtype).element_size()
        chunk_rows = max(1, (32 * 1024**2) // max(1, x.shape[1] * item_bytes))
        norms = torch.empty(x.shape[0], dtype=dtype, device=x.device)
        for begin in range(0, x.shape[0], chunk_rows):
            end = min(begin + chunk_rows, x.shape[0])
            norms[begin:end] = (
                self._as_computed(x[begin:end], stage) ** 2
            ).sum(1)
        return norms

    def _lowp_operands(self, xb, stage):
        """Cast and transpose centroids once rather than once per chunk."""
        dt = self._mm_dtype(stage)
        xb_lp = xb if dt is None else xb.to(dt)
        return dt, xb_lp.T.contiguous()

    def _exact_1nn(self, xq, xb, xb_norms=None, stage="fine"):
        nq, d = xq.shape
        nb, d2 = xb.shape
        assert d2 == d
        if xb_norms is None:
            xb_norms = self._sq_norms(xb, stage)
        dt, xb_t = self._lowp_operands(xb, stage)
        body = self._chunk_body("assign") if stage == "coarse" else _assign_body
        chunk_size = self.query_chunk if self.implem >= 2 else nq
        if chunk_size >= nq:
            return body(xq if dt is None else xq.to(dt), xb_t, xb_norms)
        out_dis = torch.empty(nq, dtype=torch.float32, device=xq.device)
        out_idx = torch.empty(nq, dtype=torch.int64, device=xq.device)
        for i0 in range(0, nq, chunk_size):
            i1 = min(i0 + chunk_size, nq)
            chunk = xq[i0:i1] if dt is None else xq[i0:i1].to(dt)
            out_dis[i0:i1], out_idx[i0:i1] = body(chunk, xb_t, xb_norms)
        return out_dis, out_idx

    def _exact_topk(self, xq, xb, k, xb_norms=None, stage="fine"):
        nq, d = xq.shape
        nb, d2 = xb.shape
        assert d2 == d
        if xb_norms is None:
            xb_norms = self._sq_norms(xb, stage)
        dt, xb_t = self._lowp_operands(xb, stage)
        body = self._chunk_body("topk") if stage == "coarse" else _topk_body
        chunk_size = self.query_chunk if self.implem >= 2 else nq
        if chunk_size >= nq:
            return body(xq if dt is None else xq.to(dt), xb_t, xb_norms, k)
        out_dis = torch.empty((nq, k), dtype=torch.float32, device=xq.device)
        out_idx = torch.empty((nq, k), dtype=torch.int64, device=xq.device)
        for i0 in range(0, nq, chunk_size):
            i1 = min(i0 + chunk_size, nq)
            chunk = xq[i0:i1] if dt is None else xq[i0:i1].to(dt)
            out_dis[i0:i1], out_idx[i0:i1] = body(chunk, xb_t, xb_norms, k)
        return out_dis, out_idx

    def add(self, x):
        with self._timed("add"):
            self._add(x)

    def _add(self, x):
        n, d = x.shape
        assert d == self.d
        _check_int32_count(n, "candidate rows")

        self.coarse_ndis += int(n) * self.nlist
        list_nos = self._assign_to_coarse(x)
        order, lims = _bucket_sort(list_nos, self.nlist)
        self.list_sizes = lims[1:] - lims[:-1]
        self.set_routing_nonempty(self.list_sizes > 0)
        self.xb_lims = lims.long()
        self.ids = order.int()
        dt = self._mm_dtype("fine")
        if self.implem >= 4 and dt is not None:
            self.vectors = None
            self.vectors_lowp = torch.empty_like(x, dtype=dt)
            self.vectors_norms = torch.empty(
                n, device=x.device, dtype=torch.float32
            )
            rows = max(1, (32 << 20) // max(1, d * x.element_size()))
            for begin in range(0, n, rows):
                end = min(begin + rows, n)
                block = x[order[begin:end]].to(dt)
                self.vectors_lowp[begin:end].copy_(block)
                self.vectors_norms[begin:end] = self._sq_norms(block, "fine")
        else:
            self.vectors = x[order]
            self.vectors_norms = self._sq_norms(self.vectors, "fine")
            self.vectors_lowp = None if dt is None else self.vectors.to(dt)
        self._fine_storage_key = self._policy_key("fine")

    def set_routing_nonempty(self, nonempty):
        """Probe only lists flagged non-empty (others get +inf distance).

        ``add`` sets the local occupancy. A centroid-sharded caller that
        routes rows on one rank for all shards replaces it with the GLOBAL
        occupancy (non-empty on any shard), so every shard probes the same
        lists and a row still finds a candidate somewhere.
        """
        nonempty = nonempty.to(device=self.centroids.device, dtype=torch.bool)
        if nonempty.shape != (self.nlist,):
            raise ValueError("nonempty must have shape (nlist,)")
        self._routing_nonempty = nonempty.clone()
        self._routing_norms_key = None
        self.routing_norms = None
        self._current_routing_norms()

    def route(self, x):
        """Return nearest routable lists, shaped ``(n, nprobe)``."""
        self.coarse_ndis += int(x.shape[0]) * self.nlist
        with self._timed("coarse"):
            if type(self)._chunk_body is IndexIVFFlat._chunk_body:
                from . import ivf_fine

                if ivf_fine.route_eligible(self, x):
                    return ivf_fine.route(self, x)
            _, idx = self._exact_topk(
                x,
                self.centroids,
                self.nprobe,
                xb_norms=self._current_routing_norms(),
                stage="coarse",
            )
        return idx
