# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import unittest

import numpy as np

from faiss.contrib.e_means.e_means_balanced_numpy import NumpyEmeansBalanced
from faiss.contrib.e_means.e_means_balanced_exact_numpy import (
    auction_balance,
    balance_candidates,
    exact_balance,
    exact_capacities,
)


def skewed_data(n=20000, d=16, groups=32, seed=0, spread=3.0):
    """Gaussian groups with Zipf-like masses, so plain k-means is uneven."""
    rs = np.random.RandomState(seed)
    centers = rs.randn(groups, d).astype("float32") * spread
    mass = 1.0 / np.arange(1, groups + 1)
    labels = rs.choice(groups, n, p=mass / mass.sum())
    return (centers[labels] + rs.randn(n, d)).astype("float32")


def sq_dist(x, centroids):
    diff = x[:, None, :].astype("float64") - centroids[None].astype("float64")
    return (diff ** 2).sum(-1)                                                 # (n, k)


def brute_force(cost, cap):
    """Smallest total cost over all assignments with exactly cap[j] rows in cluster j."""
    n, k = cost.shape
    codes = np.indices((k,) * n).reshape(n, -1).T                              # (k^n, n)
    counts = np.stack([(codes == j).sum(1) for j in range(k)], axis=1)
    ok = (counts == cap[None]).all(1)
    totals = cost[np.arange(n)[None, :], codes[ok]].sum(1)
    return totals.min()


class TestCapacities(unittest.TestCase):
    def test_capacities_sum_to_n_and_favour_the_fullest(self):
        counts = np.array([5, 9, 1, 9, 0, 3])
        cap = exact_capacities(counts, 28)
        self.assertEqual(cap.sum(), 28)
        np.testing.assert_array_equal(cap, [5, 5, 4, 5, 4, 5])   # 28 = 4 * 6 + 4
        np.testing.assert_array_equal(exact_capacities(counts, 24), [4] * 6)


class TestExactBalance(unittest.TestCase):
    def test_tiny_problems_reach_the_brute_force_optimum(self):
        for seed, n, k in ((0, 8, 4), (1, 9, 4), (2, 8, 3), (3, 9, 3)):
            rs = np.random.RandomState(seed)
            x = rs.randint(-5, 6, size=(n, 3)).astype("float32")
            centroids = rs.randint(-5, 6, size=(k, 3)).astype("float32")
            bias = rs.randn(k).astype("float32")
            dist, labels, stats = exact_balance(
                x, centroids, bias, m=k, graph=k, tau=1e9, eps=1e-9, stall_progress=0)
            cap = np.bincount(labels, minlength=k)
            self.assertEqual(set(np.unique(cap)) - {n // k, n // k + 1}, set())
            self.assertEqual(stats["n_greedy"], 0)
            cost = sq_dist(x, centroids)
            self.assertAlmostEqual(cost[np.arange(n), labels].sum(), brute_force(cost, cap),
                                   places=4)

    def test_counts_are_exact_and_distances_true(self):
        n, k = 20003, 64
        x = skewed_data(n=n, spread=1.0)
        km = NumpyEmeansBalanced(16, k, seed=0)
        km.train(x)
        dist, labels = km.exact_balance(x)
        counts = np.bincount(labels, minlength=k)
        self.assertEqual(counts.sum(), n)
        self.assertEqual(set(np.unique(counts)), {n // k, n // k + 1})
        self.assertEqual((counts == n // k + 1).sum(), n % k)
        direct = sq_dist(x, km.centroids)[np.arange(len(x)), labels]
        np.testing.assert_allclose(dist, direct, rtol=1e-4, atol=1e-4)
        stats = km.exact_balance_stats
        self.assertLessEqual(stats["n_greedy"], 0.01 * n)
        # Balancing costs distortion, but little: within a few percent of the biased routing.
        biased = km.assign(x)[0].mean()
        self.assertLess(dist.mean(), 1.05 * biased)

    def test_exit_phase_keeps_the_counts_exact(self):
        # Stop the first batches early so the exit candidates and the greedy step both run.
        x = skewed_data(n=6007, spread=1.0, seed=3)
        k = 60
        rs = np.random.RandomState(0)
        centroids = x[rs.choice(len(x), k, replace=False)]
        dist, labels, stats = exact_balance(x, centroids, np.zeros(k, dtype="float32"),
                                            min_progress=0.9)
        self.assertGreater(stats["exit_batches"], 0)
        counts = np.bincount(labels, minlength=k)
        self.assertEqual(set(np.unique(counts)), {6007 // k, 6007 // k + 1})
        np.testing.assert_allclose(dist, sq_dist(x, centroids)[np.arange(len(x)), labels],
                                   rtol=1e-4, atol=1e-3)

    def test_rows_move_only_between_their_candidates_before_the_greedy_step(self):
        x = skewed_data(n=4000, seed=1)
        rs = np.random.RandomState(0)
        centroids = x[rs.choice(len(x), 50, replace=False)]
        bias = np.zeros(50, dtype="float32")
        cost, labels = balance_candidates(x, centroids, bias, m=4, graph=12)
        prices = -bias.astype(np.float64)
        cap = exact_capacities(np.bincount(labels[:, 0], minlength=50), len(x))
        slot, counts, stats = auction_balance(cost, labels, prices, cap)
        self.assertTrue((slot >= 0).all() and (slot < 4).all())
        np.testing.assert_array_equal(
            counts, np.bincount(labels[np.arange(len(x)), slot], minlength=50))
        # Prices only rise from the starting bias.
        self.assertTrue((prices >= -bias - 1e-12).all())
        self.assertLessEqual(stats["surplus_history"][-1], stats["surplus_history"][0])

    def test_candidates_are_the_biased_nearest(self):
        x = skewed_data(n=2000, seed=2)
        rs = np.random.RandomState(1)
        centroids = rs.randn(40, 16).astype("float32") * 3
        bias = rs.randn(40).astype("float32")
        cost, labels = balance_candidates(x, centroids, bias, m=5, graph=40)
        full = sq_dist(x, centroids)
        biased = full - bias[None]
        want = np.argsort(biased, axis=1, kind="stable")[:, :5]
        np.testing.assert_array_equal(labels, want)
        np.testing.assert_allclose(cost, full[np.arange(len(x))[:, None], labels],
                                   rtol=1e-4, atol=1e-3)


class TestExactBalanceImports(unittest.TestCase):
    def test_numpy_module_does_not_import_torch(self):
        code = ("import sys; import faiss.contrib.e_means.e_means_balanced_exact_numpy; "
                "sys.exit(int('torch' in sys.modules))")
        self.assertEqual(subprocess.call([sys.executable, "-c", code]), 0)


if __name__ == "__main__":
    unittest.main()
