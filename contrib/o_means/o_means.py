# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""o-means: e-means training with a two-level nearest-centroid assignment."""

import torch

from faiss.contrib.e_means.base import (
    _CHECK_UNUSED_EVERY,
    _INIT_USAGE,
    _SPLIT_EPSILON,
    _epoch_seed,
)

from .schedule import make_schedule
from .triton_kernels import (
    DenseWorkspace,
    accumulate,
    apply_splits,
    assign_dense,
    cast_pad_rows,
    dense_layout,
    ema_update,
    refresh_table,
    scale_exponent,
)
from .triton_two_level import (
    TwoLevelWorkspace,
    build_cells,
    permute_rows,
    record_splits,
    two_level_assign,
)

__all__ = ["Omeans", "train_omeans"]

# Lloyd iterations on the centroids when the cells are built: the first
# build starts from random centroids and needs more. Rebuilds during the
# first pass only re-sort the centroids into the existing cells, which is
# enough to place the children of the many splits made early in training.
_FIRST_BUILD_ITERS = 4
_FIRST_PASS_REBUILD_ITERS = 0
_REBUILD_ITERS = 2

_ASSIGN_CHUNK_ROWS = 65536

# Share of the GPU memory that training plans to use, data included.
_MEMORY_FRACTION = 0.875


class Omeans:
    """GPU k-means trained with e-means' updates and a two-level assignment.

    The training loop is e-means' (batch and step laws, cosine schedule,
    per-epoch shuffles, EMA update, split revival), with two changes:

    - Each batch row is compared only with the centroids of its ``nprobe``
      nearest coarse cells, out of ``nlist`` cells built over the centroids
      by k-means on the centroids themselves. The cells are rebuilt after
      every revival check during the first pass, then every half pass.
    - Below 2 epochs, the final centroids are a mean over the whole run in
      which each batch is weighted by exp(-(rows seen after it) / n). From
      2 epochs up they are the mean of the last pass, as in e-means.

    ``nlist`` and ``nprobe`` default to the rule o-means was tuned with.
    ``batch_law="auto"`` uses the throughput law up to 2M rows and the
    quality law above. Training data must be a float32 CUDA tensor; the
    assignment runs on FP16 copies of the rows and centroids.

    Each object represents one run. With
    ``torch.use_deterministic_algorithms(True)`` a run is bitwise
    reproducible for a given input, seed and GPU. Otherwise the float32
    atomics of ``index_add_`` in the cell build can change the last bits of
    the cell centres, and through them occasionally a cell, between runs.
    """

    def __init__(
        self,
        d,
        k,
        nepochs=5,
        batch_law="auto",
        seed=1234,
        *,
        nlist=None,
        nprobe=None,
        capture_assignments=True,
    ):
        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_law = batch_law
        self.seed = int(seed)
        self.nlist = nlist
        self.nprobe = nprobe
        self.capture_assignments = bool(capture_assignments)
        self.centroids = None
        self.assignments = None
        self.recipe = None

    def train(self, x, init_centroids=None):
        """Train on x and return (centroids, labels).

        x is a float32 CUDA tensor of shape (n, d). Without init_centroids,
        the starting centroids are the k rows of x that Emeans would draw
        with the same seed. The labels are each row's assignment in the last
        pass (-1 for rows the last pass did not visit, when nepochs is
        fractional), or None with capture_assignments=False. Like e-means'
        labels, they come from centroids that were still moving, and here
        from an approximate search; call ``assign`` for labels from a full
        search.
        """
        if self.centroids is not None:
            raise RuntimeError("create a new Omeans object for another run")
        _check_rows(x, self.d)
        n = x.shape[0]
        sched = make_schedule(n, self.d, self.k, self.nepochs, self.batch_law,
                              self.nlist, self.nprobe)
        if init_centroids is None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(self.seed)
            init_centroids = x[torch.randperm(n, generator=generator, device=x.device)[:self.k]]
        elif tuple(init_centroids.shape) != (self.k, self.d):
            raise ValueError("init_centroids must have shape (k, d)")
        elif init_centroids.device != x.device:
            raise ValueError("init_centroids must be on the device of x")
        labels = None
        if self.capture_assignments:
            labels = torch.empty(n, dtype=torch.int64, device=x.device)
            if sched.epoch_rows[-1] < n:
                labels.fill_(-1)
        run = _Training(x, init_centroids, sched, self.seed)
        self.centroids = run.train(labels)
        self.assignments = labels
        self.recipe = dict(
            batch_law=sched.batch_law,
            batch_size=sched.batch_size,
            alpha0=sched.alpha0,
            nepochs=sched.nepochs,
            epoch_rows=list(sched.epoch_rows),
            total_steps=len(sched.steps),
            unused_threshold=sched.unused_threshold,
            nlist=sched.nlist,
            nprobe=sched.nprobe,
            rebuild_every=sched.rebuild_every,
            closing_mean="recency_weighted" if sched.recency_close else "last_pass",
            resident_fp16_copy=run.x16 is not None,
            scale_exponent=run.e,
        )
        return self.centroids, self.assignments

    def assign(self, x):
        """Return (squared distances, labels) of x's nearest centroids.

        Every row is compared with all k centroids, in FP16 as in training,
        so a near-tie can resolve to the second nearest. The distance to the
        chosen centroid is computed in float32.
        """
        if self.centroids is None:
            raise RuntimeError("train the model before calling assign")
        _check_rows(x, self.d)
        n = x.shape[0]
        layout = dense_layout(self.d, self.k)
        ws = DenseWorkspace(min(n, _ASSIGN_CHUNK_ROWS), layout.parts, x.device)
        c16 = torch.zeros((layout.kp, layout.dp), dtype=torch.float16, device=x.device)
        cn = torch.full((layout.kp,), float("inf"), dtype=torch.float32, device=x.device)
        table_e = None
        distances = torch.empty(n, dtype=torch.float32, device=x.device)
        labels = torch.empty(n, dtype=torch.int64, device=x.device)
        for b0 in range(0, n, _ASSIGN_CHUNK_ROWS):
            rows = x[b0:b0 + _ASSIGN_CHUNK_ROWS].contiguous()
            x16, e = cast_pad_rows(rows, layout.dp)
            if e != table_e:
                # Rows and centroids must share the power-of-two scale.
                refresh_table(self.centroids * 2.0**e, c16, cn, self.k, self.d)
                table_e = e
            lab = assign_dense(x16, c16, cn, layout, ws, rows=len(rows)).long()
            labels[b0:b0 + len(rows)] = lab
            distances[b0:b0 + len(rows)] = (rows - self.centroids[lab]).square().sum(1)
        return distances, labels


def train_omeans(x, k, nepochs=5, init_centroids=None, seed=1234, **kwargs):
    """Train o-means on the rows of x and return the (k, d) centroids."""
    model = Omeans(x.shape[1], k, nepochs=nepochs, seed=seed, capture_assignments=False,
                   **kwargs)
    return model.train(x, init_centroids)[0]


def _check_rows(x, d):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise TypeError("o-means needs a CUDA tensor; move the data to the GPU first")
    if x.dtype != torch.float32:
        raise TypeError("o-means needs float32 data, got %s" % x.dtype)
    if x.dim() != 2 or x.shape[1] != d:
        raise ValueError("expected a tensor of shape (n, %d), got %s" % (d, tuple(x.shape)))


def _fits_fp16_copy(n, d, layout, sched, device):
    """Whether an FP16 copy of the data fits next to the training state.

    Planned from sizes and the card's total memory, including the caller's
    float32 data. Querying the free memory instead costs as much as a few
    training steps on small problems.
    """
    data = 4 * n * d + 2 * n * layout.dp
    state = 22 * layout.kp * layout.dp
    batch = sched.batch_size
    workspace = 4 * sched.nlist * batch + 8 * sched.nprobe * batch + 8 * n
    total = torch.cuda.get_device_properties(device).total_memory
    return data + state + workspace + 2**29 <= _MEMORY_FRACTION * total


class _Training:
    """State of one run.

    The centroid table and every per-centroid tensor are kept sorted by
    coarse cell, and are permuted at every cell rebuild. ``ids`` maps each
    table position back to the centroid's index in ``init_centroids``, which
    is the order of the returned centroids and labels.
    """

    def __init__(self, x, init, sched, seed):
        n, d = x.shape
        k = sched.k
        dev = x.device
        self.sched = sched
        self.seed = seed
        self.x = x.detach().contiguous()
        layout = dense_layout(d, k)
        dp, kp, batch = layout.dp, layout.kp, sched.batch_size

        if _fits_fp16_copy(n, d, layout, sched, dev):
            self.x16, self.e = cast_pad_rows(self.x, dp)
        else:
            # Without room for an FP16 copy of the data, each batch is
            # gathered and cast on the fly, with the same scale and results.
            self.x16 = None
            lo, hi = torch.aminmax(self.x)
            self.e = scale_exponent(max(abs(float(lo)), abs(float(hi))))
            self.gather16 = torch.zeros((batch, dp), dtype=torch.float16, device=dev)
            self.gather32 = torch.empty((batch, d), dtype=torch.float32, device=dev)

        self.C = torch.zeros((kp, dp), dtype=torch.float32, device=dev)
        self.C[:k, :d] = init
        if self.e:
            self.C[:k, :d] *= 2.0**self.e
        self.c16 = torch.zeros((kp, dp), dtype=torch.float16, device=dev)
        self.cn = torch.full((kp,), float("inf"), dtype=torch.float32, device=dev)
        refresh_table(self.C, self.c16, self.cn, k, d)
        self.usage = torch.zeros(kp, dtype=torch.float32, device=dev)
        self.usage[:k] = _INIT_USAGE
        self.sums = torch.zeros((kp, dp), dtype=torch.int64, device=dev)
        self.cnt = torch.zeros(kp, dtype=torch.int32, device=dev)
        # Closing-mean accumulators: weighted float32 for the recency mean,
        # exact fixed point (like the batch sums) for the last-pass mean.
        if sched.recency_close:
            self.close_sums = torch.zeros((kp, dp), dtype=torch.float32, device=dev)
            self.close_cnt = torch.zeros(kp, dtype=torch.float32, device=dev)
        else:
            self.close_sums = torch.zeros((kp, dp), dtype=torch.int64, device=dev)
            self.close_cnt = torch.zeros(kp, dtype=torch.int32, device=dev)
        self.ids = torch.arange(k, device=dev)

        self.split_rng = torch.Generator(device=dev)
        self.split_rng.manual_seed(seed)
        self.uniform = torch.empty(k, dtype=torch.float32, device=dev)

        self.cells = None
        self.cell_rng = torch.Generator(device=dev)
        self.cell_rng.manual_seed(seed * 7919 + 17)
        self.ws = TwoLevelWorkspace(batch, sched.nlist, sched.nprobe, dev)

    def batch_rows(self, perm, b0, b1):
        """(source, index or None, row count) of rows perm[b0:b1]."""
        idx = perm[b0:b1]
        if self.x16 is not None:
            return self.x16, idx, b1 - b0
        R = b1 - b0
        torch.index_select(self.x, 0, idx, out=self.gather32[:R])
        if self.e:
            self.gather32[:R].mul_(2.0**self.e)
        self.gather16[:R, :self.sched.d].copy_(self.gather32[:R])
        return self.gather16, None, R

    def rebuild_cells(self, first_pass):
        if self.cells is None:
            iters = _FIRST_BUILD_ITERS
        else:
            iters = _FIRST_PASS_REBUILD_ITERS if first_pass else _REBUILD_ITERS
        order, self.cells = build_cells(self.C, self.sched.k, self.sched.nlist, iters,
                                        prev=self.cells, generator=self.cell_rng)
        permute_rows(order, self.sched.k, [self.C, self.c16, self.cn, self.usage,
                                           self.close_sums, self.close_cnt])
        self.ids = self.ids[order]

    def revive(self):
        """e-means' split revival, without reading anything back to the host.

        Centroids whose usage is below tau times the mean are dead; donors
        are drawn among the others in proportion to their usage (Gumbel
        top-m), and the m = min(#dead, #donors) lowest-index dead centroids
        become their children. Uniforms are drawn at every check, so the
        random stream does not depend on m.
        """
        k = self.sched.k
        u = self.usage[:k]
        thr = u.sum() * (self.sched.unused_threshold / k)
        dead = u < thr
        donors = (~dead) & (u > 0)
        torch.rand(k, generator=self.split_rng, out=self.uniform)
        r = self.uniform.clamp_min(torch.finfo(torch.float32).tiny)
        keys = torch.where(donors, u.log() - (-r.log()).log(), float("-inf"))
        src = torch.argsort(keys, descending=True)
        tgt = torch.sort(dead.to(torch.uint8), descending=True, stable=True).indices
        m = torch.minimum(dead.sum(), donors.sum())
        recency = self.sched.recency_close
        apply_splits(self.C, self.c16, self.cn, self.usage, src, tgt, m, self.sched.d,
                     _SPLIT_EPSILON, max_pairs=k // 2 + 1,
                     wsum=self.close_sums if recency else None,
                     wcnt=self.close_cnt if recency else None)
        record_splits(self.cells.child, src, tgt, m, k // 2 + 1)

    def closing_mean(self):
        """Final centroids in the order of init_centroids, in data units.

        A centroid with no closing mass, or a negligible weighted mass
        (below 1e-3 rows), keeps its EMA position.
        """
        k, d = self.sched.k, self.sched.d
        sums, cnt = self.close_sums[:k, :d], self.close_cnt[:k]
        if self.sched.recency_close:
            cnt = cnt.double()
            ok = cnt > 1e-3
            mean = (sums.double() / torch.where(ok, cnt, 1.0)[:, None]).float()
        else:
            ok = cnt > 0
            mean = (sums.double() * 2.0**-24 / cnt.clamp_min(1).double()[:, None]).float()
        out = torch.where(ok[:, None], mean, self.C[:k, :d])
        if self.e:
            out = out * 2.0**-self.e
        result = torch.empty_like(out)
        result[self.ids] = out
        return result

    def train(self, labels_out=None):
        sched = self.sched
        n, k, d = sched.n, sched.k, sched.d
        last_epoch = len(sched.epoch_rows) - 1
        perm, epoch = None, -1
        rows_done, since_rebuild = 0, 0
        for step in sched.steps:
            if step.epoch != epoch:
                epoch = step.epoch
                generator = torch.Generator(device=self.x.device)
                generator.manual_seed(_epoch_seed(self.seed, epoch))
                perm = torch.randperm(n, generator=generator, device=self.x.device)
            src, idx, R = self.batch_rows(perm, step.b0, step.b1)
            first_pass = rows_done < n
            # Most splits happen during the first pass: rebuild after each
            # revival check there, so that new children land in their cells.
            every = _CHECK_UNUSED_EVERY if first_pass else sched.rebuild_every
            if self.cells is None or since_rebuild >= every:
                self.rebuild_cells(first_pass)
                since_rebuild = 0
            since_rebuild += 1
            labels = two_level_assign(src, self.c16, self.cn, self.cells, self.ws,
                                      sched.nprobe, idx=idx, rows=R)
            if labels_out is not None and epoch == last_epoch:
                labels_out[perm[step.b0:step.b1]] = self.ids[labels]
            rows_done += R
            accumulate(src, labels, self.sums, self.cnt, d, idx=idx)
            close = step.omega is not None
            ema_update(self.C, self.c16, self.cn, self.usage, self.sums, self.cnt, k, d,
                       step.alpha, self.close_sums if close else None,
                       self.close_cnt if close else None,
                       omega=step.omega if close else 1.0)
            if step.check:
                self.revive()
        return self.closing_mean()
