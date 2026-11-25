# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib.util
import unittest
from unittest import mock

import torch

TRITON_CUDA = (torch.cuda.is_available() and not torch.version.hip
               and torch.cuda.get_device_capability() >= (8, 0)
               and importlib.util.find_spec("triton") is not None)

if TRITON_CUDA:
    from faiss.contrib.e_means import Emeans
    from faiss.contrib.e_means.base import _epoch_seed
    from faiss.contrib.o_means import Omeans
    from faiss.contrib.o_means import o_means as om
    from faiss.contrib.o_means import triton_two_level as tl2
    from faiss.contrib.o_means.schedule import make_schedule
    from faiss.contrib.o_means.triton_kernels import (
        DenseWorkspace,
        apply_splits,
        assign_dense,
        cast_pad_rows,
        dense_layout,
        ema_update,
        refresh_table,
    )


@contextlib.contextmanager
def deterministic():
    # The cell build sums centroids with index_add_, whose CUDA atomics make
    # the last bits, and occasionally a cell, depend on scheduling.
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def mixture(n, k, d, generator, spread=0.3, modes=None):
    """Rows and centroids around shared modes, so that cells are meaningful."""
    modes = modes or max(4, k // 64)
    centres = 3.0 * torch.randn(modes, d, device="cuda", generator=generator)
    c = centres[torch.randint(0, modes, (k,), device="cuda", generator=generator)]
    c = c + spread * torch.randn(k, d, device="cuda", generator=generator)
    x = c[torch.randint(0, k, (n,), device="cuda", generator=generator)]
    x = x + spread * torch.randn(n, d, device="cuda", generator=generator)
    return x, c


def cell_sorted_table(x, c, G, seed=0):
    """FP16 rows, and cells built over c with the table sorted by cell."""
    k, d = c.shape
    layout = dense_layout(d, k)
    x16, e = cast_pad_rows(x.contiguous(), layout.dp)
    assert e == 0
    C = torch.zeros((layout.kp, layout.dp), device="cuda")
    C[:k, :d] = c
    generator = torch.Generator(device="cuda").manual_seed(seed)
    order, cells = tl2.build_cells(C, k, G, 4, generator=generator)
    tl2.permute_rows(order, k, [C])
    c16 = torch.zeros((layout.kp, layout.dp), dtype=torch.float16, device="cuda")
    cn = torch.full((layout.kp,), float("inf"), device="cuda")
    refresh_table(C, c16, cn, k, d)
    return layout, x16, c16, cn, cells


def scores(xs, cs):
    x, c = xs.double(), cs.double()
    return (c * c).sum(1)[None, :] - 2.0 * x @ c.t()


def reference_two_level(xs, c16, cn, cells, p, child=None):
    """Brute-force float64 two-level assignment on the same FP16 operands."""
    G, d = cells.G, xs.shape[1]
    coarse = cells.gn[:G].double()[None, :] - 2.0 * xs.double() @ cells.g16[:G, :d].double().t()
    probed = torch.sort(coarse, dim=1, stable=True).indices[:, :p]
    full = scores(xs, c16[:, :d])
    full = torch.where(cn.isinf()[None, :], float("inf"), full)
    pos = torch.arange(c16.shape[0], device="cuda")
    off = cells.off.long()
    inside = torch.zeros_like(full, dtype=torch.bool)
    for s in range(p):
        lo, hi = off[probed[:, s]], off[probed[:, s] + 1]
        inside |= (pos[None, :] >= lo[:, None]) & (pos[None, :] < hi[:, None])
    best, label = torch.where(inside, full, float("inf")).min(1)
    if child is not None:
        ch = child.long()
        c1 = torch.where(ch[label] >= 0, ch[label], label)
        c2 = torch.where(ch[c1] >= 0, ch[c1], c1)
        for prev, nxt in ((label, c1), (c1, c2)):
            s = full.gather(1, nxt[:, None])[:, 0]
            take = (nxt != prev) & (s < best)
            best = torch.where(take, s, best)
            label = torch.where(take, nxt, label)
    return label


def relative_gap(xs, cs, labels, best):
    """Score of the chosen centroid minus the best score, relative to the norms."""
    s = scores(xs, cs)
    picked = s.gather(1, labels[:, None])[:, 0]
    scale = xs.double().square().sum(1) + cs.double().square().sum(1).max()
    return (picked - best) / scale


@unittest.skipUnless(TRITON_CUDA, "qualified Triton CUDA required")
class TestTwoLevelAssignment(unittest.TestCase):
    def test_matches_brute_force(self):
        g = torch.Generator(device="cuda").manual_seed(3)
        # G = 32 and 128 use the per-cell-atomic coarse kernel, G = 512 the per-pair one.
        for n, k, d, G, p in [(4097, 4096, 96, 32, 8), (2000, 16384, 128, 128, 8),
                              (1500, 8192, 100, 512, 16), (777, 2304, 768, 16, 4)]:
            for with_index in (False, True):
                with self.subTest(n=n, G=G, p=p, with_index=with_index):
                    x, c = mixture(n, k, d, g)
                    layout, x16, c16, cn, cells = cell_sorted_table(x, c, G)
                    ws = tl2.TwoLevelWorkspace(n, G, p, "cuda")
                    if with_index:
                        idx = torch.randperm(n, device="cuda", generator=g)
                        lab = tl2.two_level_assign(x16, c16, cn, cells, ws, p, idx=idx)
                        xs = x16[idx, :d]
                    else:
                        lab = tl2.two_level_assign(x16, c16, cn, cells, ws, p, rows=n)
                        xs = x16[:, :d]
                    lab = lab.long()
                    self.assertEqual(int(ws.cnt.abs().sum()), 0, "counters reset for next batch")
                    ref = reference_two_level(xs, c16, cn, cells, p)
                    # FP32 accumulation can flip a near-tie of coarse scores and
                    # change which cells a row probes.
                    self.assertGreaterEqual(float((lab == ref).double().mean()), 0.995)
                    best = scores(xs, c16[:, :d]).gather(1, ref[:, None])[:, 0]
                    gap = relative_gap(xs, c16[:, :d], lab, best)
                    self.assertLessEqual(float(gap.quantile(0.99)), 2e-6)

    def test_probing_every_cell_is_exact(self):
        g = torch.Generator(device="cuda").manual_seed(4)
        for n, k, d, G in [(3000, 4096, 96, 32), (1000, 2304, 768, 16)]:
            with self.subTest(n=n, G=G):
                x, c = mixture(n, k, d, g)
                layout, x16, c16, cn, cells = cell_sorted_table(x, c, G)
                nonempty = int((cells.size > 0).sum())
                ws = tl2.TwoLevelWorkspace(n, G, nonempty, "cuda")
                lab = tl2.two_level_assign(x16, c16, cn, cells, ws, nonempty, rows=n).long()
                dense = assign_dense(x16, c16, cn, layout, DenseWorkspace(n, layout.parts, "cuda"),
                                     rows=n).long()
                self.assertGreaterEqual(float((lab == dense).double().mean()), 0.999)
                xs = x16[:, :d]
                best = scores(xs, c16[:k, :d]).min(1).values
                self.assertLessEqual(float(relative_gap(xs, c16[:k, :d], lab, best).max()), 2e-6)

    def test_split_children_are_reached(self):
        g = torch.Generator(device="cuda").manual_seed(5)
        n, k, d, G, p = 9000, 4096, 96, 32, 4
        x, c = mixture(n, k, d, g)
        layout, x16, c16, cn, cells = cell_sorted_table(x, c, G)
        # Give some centroids a child placed next to them in the data but
        # elsewhere in the table, as a split since the last rebuild would.
        parent = torch.randperm(k, device="cuda", generator=g)[:300]
        child = torch.randperm(k, device="cuda", generator=g)[:300]
        keep = parent != child
        parent, child = parent[keep], child[keep]
        cells.child[parent] = child.int()
        cells.child[child] = -1
        c16[child] = c16[parent] + 0.05 * torch.randn_like(c16[parent].float()).half()
        cn[child] = c16[child].float().square().sum(1)
        ws = tl2.TwoLevelWorkspace(4000, G, p, "cuda")
        perm = torch.randperm(n, device="cuda", generator=g)
        lab = torch.cat([tl2.two_level_assign(x16, c16, cn, cells, ws, p,
                                              idx=perm[b0:b0 + 4000]).long().clone()
                         for b0 in range(0, n, 4000)])
        ref = reference_two_level(x16[perm, :d], c16, cn, cells, p, child=cells.child)
        self.assertGreaterEqual(float((lab == ref).double().mean()), 0.995)
        self.assertGreater(int(torch.isin(lab, child).sum()), 0)

    def test_record_splits(self):
        child = torch.full((11,), -1, dtype=torch.int32, device="cuda")
        child[2] = 7
        src = torch.tensor([2, 5, 9, 1], device="cuda")
        tgt = torch.tensor([4, 6, 0, 3], device="cuda")
        tl2.record_splits(child, src, tgt, torch.tensor(2, device="cuda"), 4)
        # 2 -> 4 -> 7 (4 takes over 2's earlier child), 5 -> 6; pairs past m ignored.
        self.assertEqual(child.tolist(), [-1, -1, 4, -1, 7, 6, -1, -1, -1, -1, -1])

    def test_empty_cells_never_win(self):
        g = torch.Generator(device="cuda").manual_seed(8)
        n, k, d, G, p = 4000, 512, 64, 64, 12
        x, c = mixture(n, k, d, g, modes=4)
        layout, x16, c16, cn, cells = cell_sorted_table(x, c, G)
        # All centroids in cells 0-2; the empty cells' centres sit on rows,
        # where they would win the coarse ranking if they were not marked.
        size = torch.zeros(G, dtype=torch.int64, device="cuda")
        size[:3] = torch.tensor([200, 200, 112], device="cuda")
        cells.off[1:] = torch.cumsum(size, 0).int()
        cells.size = size
        cells.g16[3:G] = x16[:G - 3]
        cells.gn[:G] = cells.g16[:G].float().square().sum(1)
        cells.gn[3:G] = tl2.EMPTY_NORM
        ws = tl2.TwoLevelWorkspace(n, G, p, "cuda")
        lab = tl2.two_level_assign(x16, c16, cn, cells, ws, p, rows=n).long()
        xs = x16[:, :d]
        best = scores(xs, c16[:k, :d]).min(1).values
        self.assertTrue(bool((lab < k).all()))
        self.assertLessEqual(float(relative_gap(xs, c16[:k, :d], lab, best).max()), 2e-6)


@unittest.skipUnless(TRITON_CUDA, "qualified Triton CUDA required")
class TestClosingMean(unittest.TestCase):
    def fp16_rows(self, x, e):
        return (x * 2.0**e).half().double() * 2.0**-e

    def test_last_pass_mean(self):
        # From 2 epochs up, each centroid is the mean of the rows it received
        # in the last pass, which are the returned labels.
        g = torch.Generator(device="cuda").manual_seed(0)
        x, _ = mixture(60_000, 512, 48, g, spread=1.0)
        model = Omeans(48, 512, nepochs=2, seed=3)
        centroids, labels = model.train(x)
        self.assertEqual(model.recipe["closing_mean"], "last_pass")
        rows = self.fp16_rows(x, model.recipe["scale_exponent"])
        sums = torch.zeros(512, 48, dtype=torch.float64, device="cuda").index_add_(0, labels, rows)
        counts = torch.bincount(labels, minlength=512)
        hit = counts > 0
        ref = (sums[hit] / counts[hit, None]).float()
        torch.testing.assert_close(centroids[hit], ref, rtol=1e-6, atol=1e-6)

    def test_recency_weighted_mean(self):
        # Below 2 epochs every batch counts with weight exp(-(rows after it) / n).
        # In 2-D revival is off, so no split moves mass between centroids.
        g = torch.Generator(device="cuda").manual_seed(1)
        n, k, seed = 50_000, 64, 5
        x = torch.rand(n, 2, device="cuda", generator=g) * 100
        model = Omeans(2, k, nepochs=1, seed=seed)
        centroids, labels = model.train(x)
        sched = make_schedule(n, 2, k, 1)
        self.assertEqual(model.recipe["closing_mean"], "recency_weighted")
        generator = torch.Generator(device="cuda").manual_seed(_epoch_seed(seed, 0))
        perm = torch.randperm(n, generator=generator, device="cuda")
        weight = torch.empty(n, dtype=torch.float64, device="cuda")
        for step in sched.steps:
            weight[perm[step.b0:step.b1]] = step.omega
        rows = self.fp16_rows(x, model.recipe["scale_exponent"])
        sums = torch.zeros(k, 2, dtype=torch.float64, device="cuda")
        sums.index_add_(0, labels, rows * weight[:, None])
        mass = torch.zeros(k, dtype=torch.float64, device="cuda").index_add_(0, labels, weight)
        hit = mass > 1e-3
        torch.testing.assert_close(centroids[hit], (sums[hit] / mass[hit, None]).float(),
                                   rtol=1e-5, atol=1e-4)

    def test_weighted_accumulation_and_split(self):
        g = torch.Generator(device="cuda").manual_seed(2)
        d, k = 100, 700
        layout = dense_layout(d, k)
        kp, dp = layout.kp, layout.dp
        C = torch.zeros((kp, dp), device="cuda")
        C[:k, :d] = torch.randn(k, d, device="cuda", generator=g)
        c16 = torch.zeros((kp, dp), dtype=torch.float16, device="cuda")
        cn = torch.full((kp,), float("inf"), device="cuda")
        refresh_table(C, c16, cn, k, d)
        usage = torch.rand(kp, device="cuda", generator=g)
        cnt = torch.randint(0, 3, (kp,), device="cuda", generator=g).int()
        cnt[k:] = 0
        sums = torch.zeros((kp, dp), dtype=torch.int64, device="cuda")
        sums[:, :d] = (torch.randn(kp, d, device="cuda", generator=g).half().double()
                       * cnt.double()[:, None] * 2**24).long()
        wsum = torch.rand((kp, dp), device="cuda", generator=g)
        wcnt = torch.rand(kp, device="cuda", generator=g)
        want_sum, want_cnt = wsum.clone(), wcnt.clone()
        t = cnt > 0
        want_sum[t, :d] += 0.37 * (sums[t, :d].double() * 2.0**-24).float()
        want_cnt[t] += 0.37 * cnt[t].float()
        ema_update(C, c16, cn, usage, sums, cnt, k, d, 0.2, wsum, wcnt, omega=0.37)
        torch.testing.assert_close(wsum, want_sum, rtol=0, atol=1e-5)
        torch.testing.assert_close(wcnt, want_cnt, rtol=0, atol=1e-5)
        self.assertEqual(int(sums.abs().sum()) + int(cnt.abs().sum()), 0)

        before_sum, before_cnt = wsum.clone(), wcnt.clone()
        src = torch.tensor([3, 9, 0], device="cuda")
        tgt = torch.tensor([200, 201, 1], device="cuda")
        apply_splits(C, c16, cn, usage, src, tgt, torch.tensor(2, device="cuda"), d, 1e-4, 3,
                     wsum=wsum, wcnt=wcnt)
        for s, t in ((3, 200), (9, 201)):
            for r in (s, t):
                self.assertTrue(torch.equal(wsum[r, :d], before_sum[s, :d] * 0.5))
                self.assertEqual(float(wcnt[r]), float(before_cnt[s]) * 0.5)
        untouched = torch.ones(kp, dtype=torch.bool, device="cuda")
        untouched[[3, 200, 9, 201]] = False
        self.assertTrue(torch.equal(wsum[untouched], before_sum[untouched]))


@unittest.skipUnless(TRITON_CUDA, "qualified Triton CUDA required")
class TestOmeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = torch.Generator(device="cuda").manual_seed(0)
        cls.n, cls.d, cls.k = 200_000, 64, 1024
        modes = 2.0 * torch.randn(4096, cls.d, device="cuda", generator=g)
        cls.x = modes[torch.randint(0, 4096, (cls.n,), device="cuda", generator=g)]
        cls.x += torch.randn(cls.n, cls.d, device="cuda", generator=g)
        cls.init = cls.x[torch.randperm(cls.n, device="cuda", generator=g)[:cls.k]].clone()

    def mse(self, centroids):
        c = centroids.double()
        total = 0.0
        for b0 in range(0, self.n, 50_000):
            xb = self.x[b0:b0 + 50_000].double()
            total += float(torch.cdist(xb, c).min(1).values.square().sum())
        return total / self.n

    def test_mse_close_to_e_means(self):
        for T in (1, 3):
            with self.subTest(nepochs=T):
                em = Emeans(self.d, self.k, nepochs=T, seed=1, assignment_backend="triton")
                ce, _ = em.train(self.x, init_centroids=self.init)
                co, labels = Omeans(self.d, self.k, nepochs=T, seed=1).train(self.x, self.init)
                self.assertEqual(co.shape, (self.k, self.d))
                self.assertEqual(co.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(co).all()))
                self.assertEqual(labels.shape, (self.n,))
                self.assertTrue(bool(((labels >= 0) & (labels < self.k)).all()))
                ratio = self.mse(co) / self.mse(ce)
                self.assertLess(abs(ratio - 1), 0.01, "o-means / e-means MSE %.4f" % ratio)

    def test_assign(self):
        model = Omeans(self.d, self.k, nepochs=1, seed=2)
        centroids, _ = model.train(self.x)
        xs = self.x[:20_000]
        distances, labels = model.assign(xs)
        exact = torch.cdist(xs.double(), centroids.double()).square()
        best = exact.min(1).values
        picked = exact.gather(1, labels[:, None])[:, 0]
        # FP16 operands may pick a near-tie.
        self.assertLessEqual(float(((picked - best) / best.clamp_min(1e-6)).quantile(0.999)), 1e-2)
        self.assertGreaterEqual(float((labels == exact.argmin(1)).double().mean()), 0.99)
        torch.testing.assert_close(distances.double(), picked, rtol=1e-4, atol=1e-4)

    def test_deterministic_and_seeded(self):
        with deterministic():
            a, la = Omeans(self.d, self.k, nepochs=1.6, seed=4).train(self.x, self.init)
            b, lb = Omeans(self.d, self.k, nepochs=1.6, seed=4).train(self.x, self.init)
            c, _ = Omeans(self.d, self.k, nepochs=1.6, seed=5).train(self.x, self.init)
        self.assertTrue(torch.equal(a, b) and torch.equal(la, lb))
        self.assertFalse(torch.equal(a, c))

    def test_power_of_two_scale_is_exact(self):
        # Data far outside the FP16 range is rescaled by a power of two, so
        # these two inputs reach the kernels as the same FP16 rows.
        with deterministic():
            small = Omeans(self.d, self.k, nepochs=1, seed=6)
            a, _ = small.train(self.x * 2.0**20, self.init * 2.0**20)
            large = Omeans(self.d, self.k, nepochs=1, seed=6)
            b, _ = large.train(self.x * 2.0**24, self.init * 2.0**24)
        self.assertEqual(large.recipe["scale_exponent"], small.recipe["scale_exponent"] - 4)
        self.assertNotEqual(small.recipe["scale_exponent"], 0)
        self.assertTrue(torch.equal(a * 16, b))

    def test_without_fp16_copy(self):
        for T in (0.8, 2):
            with self.subTest(nepochs=T), deterministic():
                a, la = Omeans(self.d, self.k, nepochs=T, seed=7).train(self.x, self.init)
                with mock.patch.object(om, "_MEMORY_FRACTION", 0.0):
                    model = Omeans(self.d, self.k, nepochs=T, seed=7)
                    b, lb = model.train(self.x, self.init)
                self.assertFalse(model.recipe["resident_fp16_copy"])
                self.assertTrue(torch.equal(a, b) and torch.equal(la, lb))

    def test_fractional_epochs(self):
        model = Omeans(self.d, self.k, nepochs=0.5, seed=8)
        _, labels = model.train(self.x, self.init)
        self.assertEqual(int((labels == -1).sum()), self.n - model.recipe["epoch_rows"][-1])
        self.assertIsNone(Omeans(self.d, self.k, nepochs=0.5, capture_assignments=False)
                          .train(self.x, self.init)[1])

    def test_function_entry(self):
        from faiss.contrib.o_means import train_omeans
        with deterministic():
            a = train_omeans(self.x, self.k, nepochs=1, init_centroids=self.init, seed=9)
            b, _ = Omeans(self.d, self.k, nepochs=1, seed=9).train(self.x, self.init)
        self.assertTrue(torch.equal(a, b))

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            Omeans(self.d, self.k).train(self.x.cpu())
        with self.assertRaises(TypeError):
            Omeans(self.d, self.k).train(self.x.half())
        with self.assertRaises(ValueError):
            Omeans(self.d + 1, self.k).train(self.x)
        with self.assertRaises(ValueError):
            Omeans(self.d, self.k).train(self.x, self.init[:10])
        model = Omeans(self.d, self.k, nepochs=0.2)
        with self.assertRaises(RuntimeError):
            model.assign(self.x)
        model.train(self.x)
        with self.assertRaises(RuntimeError):
            model.train(self.x)


if __name__ == "__main__":
    unittest.main()
