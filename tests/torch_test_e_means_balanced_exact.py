# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Exact balancing on tensors: the auction, its kernels, and the EmeansBalanced method."""

import itertools
import unittest

import numpy as np
import torch

from faiss.contrib.e_means import e_means_balanced_exact_numpy
from faiss.contrib.e_means.e_means_balanced import EmeansBalanced
from faiss.contrib.e_means.e_means_balanced_exact import (
    _distances,
    _move_list,
    auction_balance,
    balance_candidates,
    exact_balance,
    exact_capacities,
)


def _blobs(n, d, groups, seed, spread=1.0):
    """Overlapping Gaussian groups with Zipf-like masses."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(groups, d, generator=g) * spread
    mass = 1.0 / torch.arange(1, groups + 1, dtype=torch.float64)
    labels = torch.multinomial(mass / mass.sum(), n, replacement=True, generator=g)
    return centers[labels] + torch.randn(n, d, generator=g)


def _brute_force(cost, cap):
    n, k = cost.shape
    best = None
    for codes in itertools.product(range(k), repeat=n):
        if all(codes.count(j) == cap[j] for j in range(k)):
            total = sum(cost[i, c] for i, c in enumerate(codes))
            best = total if best is None or total < best else best
    return best


class TestExactBalanceCPU(unittest.TestCase):

    def test_capacities(self):
        cap = exact_capacities(torch.tensor([5, 9, 1, 9, 0, 3]), 28)
        self.assertEqual(cap.tolist(), [5, 5, 4, 5, 4, 5])

    def test_tiny_problems_reach_the_brute_force_optimum(self):
        for seed, n, k in ((0, 8, 4), (1, 9, 4), (2, 7, 3)):
            g = torch.Generator().manual_seed(seed)
            x = torch.randint(-5, 6, (n, 3), generator=g).float()
            c = torch.randint(-5, 6, (k, 3), generator=g).float()
            b = torch.randn(k, generator=g)
            dist, labels, stats = exact_balance(x, c, b, m=k, graph=k, tau=1e9, eps=1e-9,
                                                stall_progress=0, min_progress=0)
            cap = torch.bincount(labels, minlength=k)
            self.assertLessEqual(int(cap.max() - cap.min()), 1)
            self.assertEqual(stats["n_greedy"], 0)
            cost = ((x[:, None] - c[None]) ** 2).sum(-1).double()
            self.assertAlmostEqual(float(cost[torch.arange(n), labels].sum()),
                                   float(_brute_force(cost, cap.tolist())), places=4)

    def test_exact_counts_and_true_distances(self):
        x = _blobs(6007, 8, 16, seed=0)
        k = 50
        dist, labels, stats = exact_balance(x, x[:k].clone(), torch.zeros(k))
        counts = torch.bincount(labels, minlength=k)
        self.assertEqual(sorted(set(counts.tolist())), [6007 // k, 6007 // k + 1])
        self.assertEqual(int((counts == 6007 // k + 1).sum()), 6007 % k)
        direct = ((x - x[:k][labels]) ** 2).sum(1)
        torch.testing.assert_close(dist, direct, rtol=1e-4, atol=1e-4)

    def test_matches_the_numpy_module(self):
        x = _blobs(4001, 8, 16, seed=1)
        k = 40
        em = EmeansBalanced(8, k, nepochs=6, polish_epochs=2, seed=0)
        em.train(x)
        c, b = em.centroids, em.bias
        dist, labels, stats = exact_balance(x, c, b)
        dist_np, labels_np, stats_np = e_means_balanced_exact_numpy.exact_balance(
            x.numpy(), c.numpy(), b.numpy())
        np.testing.assert_array_equal(np.bincount(labels.numpy(), minlength=k),
                                      np.bincount(labels_np, minlength=k))
        # Same steps; the candidate order and the sweep order (Faiss vs PyTorch) differ on
        # near-ties, so the partitions are equally balanced and close, not identical.
        self.assertGreater((labels.numpy() == labels_np).mean(), 0.9)
        self.assertAlmostEqual(float(dist.double().mean()), float(dist_np.mean()),
                               delta=1e-2 * float(dist.double().mean()))

    def test_exit_phase_keeps_the_counts_exact(self):
        # Stop the first batches early so the exit candidates and the greedy step both run.
        x = _blobs(6007, 8, 16, seed=8)
        k = 60
        dist, labels, stats = exact_balance(x, x[:k].clone(), torch.zeros(k), min_progress=0.9)
        self.assertGreater(stats["exit_batches"], 0)
        counts = torch.bincount(labels, minlength=k)
        self.assertEqual(sorted(set(counts.tolist())), [6007 // k, 6007 // k + 1])
        torch.testing.assert_close(dist, ((x - x[:k][labels]) ** 2).sum(1), rtol=1e-4, atol=1e-4)

    def test_prices_only_rise_and_slackness_holds(self):
        x = _blobs(3000, 8, 8, seed=2)
        k = 30
        c = x[:k].clone()
        b = torch.zeros(k)
        cost, labels = balance_candidates(x, c, b, m=6, graph=k)
        prices = torch.zeros(k, dtype=torch.float64)
        cap = exact_capacities(torch.bincount(labels[:, 0].long(), minlength=k), len(x))
        slot, counts, stats = auction_balance(cost, labels, prices, cap, eps=1e-6,
                                              stall_progress=0, min_progress=0)
        self.assertTrue(bool((prices >= 0).all()))
        net = cost.double() + prices[labels.long()]
        own = net.gather(1, slot[:, None])[:, 0]
        slack = own - net.min(1).values
        self.assertLessEqual(float(slack.max()), stats["eps"] + 1e-9)

    def test_distances_are_shortest_chains(self):
        x = _blobs(2000, 4, 8, seed=3)
        k = 20
        c = x[:k].clone()
        cost, labels = balance_candidates(x, c, torch.zeros(k), m=4, graph=k)
        slot = torch.zeros(len(x), dtype=torch.int64)
        assigned = labels[:, 0].long()
        prices = torch.zeros(k, dtype=torch.float64)
        row, col, src, dst, w = _move_list(cost, labels, slot, assigned, prices, 1e30)
        holes = torch.zeros(k, dtype=torch.bool)
        holes[:3] = True
        dist, hops, _ = _distances(src, dst.long(), w, holes, k)
        # Floyd-Warshall on the cluster graph (cheapest move between two clusters).
        big = float("inf")
        edge = torch.full((k, k), big, dtype=torch.float64)
        for a, bb, ww in zip(src.tolist(), dst.tolist(), w.tolist()):
            edge[a, bb] = min(edge[a, bb], ww)
        edge.fill_diagonal_(0)
        for j in range(k):
            edge = torch.minimum(edge, edge[:, j:j + 1] + edge[j:j + 1, :])
        want = edge[:, :3].min(1).values
        torch.testing.assert_close(dist.double(), want, rtol=1e-5, atol=1e-5)

    def test_method_on_a_trained_model(self):
        x = _blobs(8000, 8, 16, seed=4)
        em = EmeansBalanced(8, 32, nepochs=6, polish_epochs=2, seed=0)
        em.train(x)
        dist, labels = em.exact_balance(x)
        counts = torch.bincount(labels, minlength=32)
        self.assertEqual(int(counts.max() - counts.min()), 0)
        self.assertIn("n_greedy", em.exact_balance_stats)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestExactBalanceGPU(unittest.TestCase):

    def setUp(self):
        from faiss.contrib.e_means import balanced_exact_triton
        self.kernels = balanced_exact_triton

    def test_top_m_matches_dense_top_m(self):
        for n, k, d, m in ((5000, 1000, 32, 8), (3000, 300, 200, 4), (2048, 4096, 96, 16)):
            x = torch.randn(n, d, device="cuda").bfloat16()
            c = torch.randn(k, d, device="cuda")
            norms = (c.double() ** 2).sum(1).float() + torch.rand(k, device="cuda")
            lab, score = self.kernels.top_m(x, c.bfloat16(), norms, m)
            dense = norms[None] - 2 * (x.float() @ c.bfloat16().float().t())
            want = torch.topk(dense, m, dim=1, largest=False).values
            got = torch.sort(score, dim=1).values
            torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-2)
            torch.testing.assert_close(dense.gather(1, lab.long()), score, rtol=1e-3, atol=1e-2)

    def test_exact_costs(self):
        x = torch.randn(3000, 130, device="cuda").bfloat16()
        c = torch.randn(500, 130, device="cuda")
        lab = torch.randint(0, 500, (3000, 8), device="cuda", dtype=torch.int32)
        got = self.kernels.exact_costs(x, c, lab)
        want = ((x.float()[:, None] - c[lab.long()]) ** 2).sum(-1)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)

    def test_csr_distances_match_the_cpu_sweeps(self):
        x = _blobs(20000, 8, 32, seed=5)
        k = 200
        c = x[:k].clone()
        cost, labels = balance_candidates(x, c, torch.zeros(k), m=6, graph=32)
        slot = torch.zeros(len(x), dtype=torch.int64)
        assigned = labels[:, 0].long()
        prices = torch.zeros(k, dtype=torch.float64)
        row, col, src, dst, w = _move_list(cost, labels, slot, assigned, prices, 1e30)
        holes = torch.rand(k, generator=torch.Generator().manual_seed(0)) < 0.1
        want, _, _ = _distances(src, dst.long(), w, holes, k)
        got, hops, _ = _distances(src.cuda(), dst.cuda(), w.cuda(), holes.cuda(), k)
        torch.testing.assert_close(got.cpu(), want, rtol=1e-5, atol=1e-5)
        self.assertTrue(bool((hops[holes.cuda()] == 0).all()))

    def test_gpu_matches_cpu(self):
        x = _blobs(30011, 16, 64, seed=6)
        k = 300
        em = EmeansBalanced(16, k, nepochs=6, polish_epochs=2, seed=0)
        em.train(x)
        c, b = em.centroids, em.bias
        dist, labels, stats = exact_balance(x, c, b)
        dist_g, labels_g, stats_g = exact_balance(x.cuda(), c.cuda(), b.cuda())
        for lab in (labels, labels_g.cpu()):
            counts = torch.bincount(lab, minlength=k)
            self.assertEqual(sorted(set(counts.tolist())), [len(x) // k, len(x) // k + 1])
            self.assertEqual(int((counts == len(x) // k + 1).sum()), len(x) % k)
        # In-place sweeps on the GPU, Jacobi sweeps on the CPU: close, not identical.
        self.assertAlmostEqual(float(dist_g.double().mean()), float(dist.double().mean()),
                               delta=1e-2 * float(dist.double().mean()))

    def test_method_with_16_bit_rows(self):
        x = _blobs(50000, 32, 64, seed=7).cuda().bfloat16()
        em = EmeansBalanced(32, 128, nepochs=6, polish_epochs=2, seed=0)
        em.train(x)
        dist, labels = em.exact_balance(x)
        counts = torch.bincount(labels, minlength=128)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertEqual(int(counts.sum()), 50000)
        biased = em.assign(x)[0].double().mean()
        self.assertLess(float(dist.double().mean()), 1.05 * float(biased))


if __name__ == "__main__":
    unittest.main()
