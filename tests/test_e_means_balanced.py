# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import subprocess
import sys
import unittest
import unittest.mock

import numpy as np
import faiss

from faiss.contrib.e_means.e_means_balanced_numpy import (
    NumpyEmeansBalanced,
    cap_first_pass,
    cap_round_evict,
    split_epochs,
)


def skewed_data(n=20000, d=16, groups=32, seed=0):
    """Gaussian groups with Zipf-like masses, so plain k-means is uneven."""
    rs = np.random.RandomState(seed)
    centers = rs.randn(groups, d).astype("float32") * 3
    mass = 1.0 / np.arange(1, groups + 1)
    labels = rs.choice(groups, n, p=mass / mass.sum())
    return (centers[labels] + rs.randn(n, d)).astype("float32")


def integer_problem(n=400, d=6, k=12, seed=0, crowd=3):
    """Small-integer rows, centroids and biases: every FP32 score is exact.

    Most rows sit near the first ``crowd`` centroids, so the cap binds and
    the runner-ups pile up.
    """
    rs = np.random.RandomState(seed)
    centroids = rs.randint(-6, 7, size=(k, d)).astype("float32")
    near = rs.randint(0, crowd, size=n)
    far = rs.randint(0, k, size=n)
    pick = np.where(rs.rand(n) < 0.8, near, far)
    x = (centroids[pick] + rs.randint(-2, 3, size=(n, d))).astype("float32")
    bias = rs.randint(-3, 4, size=k).astype("float32")
    return x, centroids, bias


def imbalance(labels, k):
    counts = np.bincount(labels, minlength=k).astype("float64")
    return k * (counts ** 2).sum() / counts.sum() ** 2


def biased_scores(x, centroids, bias):
    diff = x[:, None, :].astype("float64") - centroids[None].astype("float64")
    return (diff ** 2).sum(-1) - bias[None].astype("float64")


def rounds_reference(scores, row_norms, cap, rounds):
    """Slow sequential definition of the rounds cap (lower score is better).

    Every row takes its best cluster (ties to the lower index). A cluster
    over the cap keeps its rows of largest margin (runner-up minus best, ties
    by lower position), the others move to their runner-up. Then, up to
    ``rounds`` times while some cluster is over the cap: in every cluster,
    own first-choice rows first, then arrivals by lowest score with the row
    norm dropped (ties by position); rows past the cap are evicted and each
    takes its best cluster among those holding fewer than ``cap`` rows.
    """
    n, k = scores.shape
    order = np.argsort(scores, axis=1, kind="stable")
    best, second = order[:, 0], order[:, 1]
    rows = np.arange(n)
    margin = (scores[rows, second] - scores[rows, best]).astype("float32")
    codes = best.copy()
    score = (scores[rows, second] - row_norms).astype("float32")
    arrived = np.zeros(n, dtype=bool)
    for j in range(k):
        mine = [i for i in range(n) if best[i] == j]
        mine.sort(key=lambda i: (-float(margin[i]), i))
        for i in mine[cap:]:
            codes[i] = second[i]
            arrived[i] = True
    for _ in range(rounds):
        count = np.bincount(codes, minlength=k)
        if count.max() <= cap:
            break
        evicted = []
        for j in range(k):
            mine = [i for i in range(n) if codes[i] == j]
            mine.sort(key=lambda i: (bool(arrived[i]), float(score[i]), i))
            evicted += mine[cap:]
        full = count >= cap
        for i in evicted:
            j = int(np.argmin(np.where(full, np.inf, scores[i])))
            codes[i] = j
            arrived[i] = True
            score[i] = scores[i, j] - row_norms[i]
    return codes


def run_cap(model, x, centroids, bias, cap, lazy=False):
    """The model's assignment and cap on one batch, outside training.

    ``lazy``: top-1 search, then runner-ups for the rows of over-full
    clusters only (the rounds cap's path after a step where few rows were in
    an over-full cluster).
    """
    model.bias[:] = bias
    norms, shift = model._biased_norms(centroids, model.bias)
    if lazy:
        top_d, top_l = model._assign_batch(x, centroids, norms, shift, model.bias, 1)
        top_d, top_l = model._runner_up(x, top_d[:, 0], top_l[:, 0], cap, centroids, norms)
        return model._cap_rounds_step(x, top_d, top_l, cap, centroids, norms)
    top_d, top_l = model._assign_batch(x, centroids, norms, shift, model.bias, 2)
    return model._cap_rounds_step(x, top_d, top_l, cap, centroids, norms)


class TestBalancedCap(unittest.TestCase):
    def _model(self, x, centroids, **kwargs):
        model = NumpyEmeansBalanced(x.shape[1], len(centroids), nepochs=1,
                                    polish_epochs=0, batch_size=len(x), **kwargs)
        model.begin(len(x), centroids)
        return model

    def test_rounds_cap_matches_sequential_reference(self):
        for seed in range(6):
            x, centroids, bias = integer_problem(seed=seed)
            scores = biased_scores(x, centroids, bias)
            row_norms = (x.astype("float64") ** 2).sum(1)
            for (cap, rounds), lazy in itertools.product(
                    ((40, 2), (34, 2), (34, 0), (34, 1), (40, 5), (80, 2)), (False, True)):
                model = self._model(x, centroids, cap_rounds=rounds)
                codes, dist = run_cap(model, x, centroids, bias, cap, lazy)
                ref = rounds_reference(scores, row_norms, cap, rounds)
                np.testing.assert_array_equal(codes, ref, err_msg=str((seed, cap, rounds, lazy)))
                # The distances are the biased scores of the final codes (+ shift).
                np.testing.assert_allclose(
                    dist - bias.max(), scores[np.arange(len(x)), codes], atol=1e-3)

    def test_masked_search_on_the_open_clusters_only(self):
        # The repair searches may run on a copy of the open clusters instead
        # of masking the norms: same codes.
        from faiss.contrib.e_means import e_means_balanced_numpy as module
        for ratio in (0, 10 ** 9):
            with unittest.mock.patch.object(module, "_COMPACT_RATIO", ratio):
                for seed in range(3):
                    x, centroids, bias = integer_problem(seed=seed)
                    scores = biased_scores(x, centroids, bias)
                    row_norms = (x.astype("float64") ** 2).sum(1)
                    model = self._model(x, centroids, cap_rounds=3)
                    codes, _ = run_cap(model, x, centroids, bias, 34)
                    np.testing.assert_array_equal(
                        codes, rounds_reference(scores, row_norms, 34, 3))

    def test_rounds_leave_no_cluster_over_the_cap_when_rounds_suffice(self):
        x, centroids, bias = integer_problem(seed=1)
        model = self._model(x, centroids, cap_rounds=50)
        codes, _ = run_cap(model, x, centroids, bias, 40)
        self.assertLessEqual(np.bincount(codes, minlength=len(centroids)).max(), 40)

    def test_rounds_cap_matches_the_reference_on_random_shapes(self):
        for seed in range(20):
            rs = np.random.RandomState(100 + seed)
            n = int(rs.choice([200, 400, 800]))
            k = int(rs.choice([8, 12, 20, 40]))
            x, centroids, bias = integer_problem(n=n, d=5, k=k, seed=seed,
                                                 crowd=int(rs.choice([1, 2, 3, 5])))
            scores = biased_scores(x, centroids, bias)
            row_norms = (x.astype("float64") ** 2).sum(1)
            cap = int(np.ceil(rs.choice([1.2, 1.6, 2.5]) * n / k))
            rounds = int(rs.choice([1, 2, 4]))
            model = self._model(x, centroids, cap_rounds=rounds)
            codes, _ = run_cap(model, x, centroids, bias, cap)
            np.testing.assert_array_equal(codes, rounds_reference(scores, row_norms, cap, rounds))

    def test_first_pass_ties_keep_the_lower_position(self):
        codes = np.zeros(6, dtype="int64")
        margin = np.array([1, 2, 2, 0, 2, 1], dtype="float32")
        second = np.arange(1, 7)
        new, moved = cap_first_pass(codes, margin, second, 3, 8)
        # Margins 2, 2, 2 at positions 1, 2, 4 stay; the rest move.
        np.testing.assert_array_equal(moved, [1, 0, 0, 1, 0, 1])
        np.testing.assert_array_equal(new, [1, 0, 0, 4, 0, 6])

    def test_matches_the_tensor_functions(self):
        try:
            import torch
            from faiss.contrib.e_means import e_means_balanced as tensor
        except ImportError:
            self.skipTest("PyTorch or the tensor class is not available")
        rs = np.random.RandomState(0)
        n, k = 5000, 64
        codes = rs.randint(0, k, n)
        # Coarse values, so ties are frequent, and both signs of zero.
        margin = (rs.randint(0, 5, n) * 0.5).astype("float32")
        score = (rs.randint(-4, 5, n) * 0.25).astype("float32")
        score[:10] = -0.0
        second = rs.randint(0, k, n)
        arrived = rs.rand(n) < 0.5
        for cap in (40, 78, 100):
            ref = tensor.cap_first_pass(torch.from_numpy(codes), torch.from_numpy(margin),
                                        torch.from_numpy(second), cap, k)
            got = cap_first_pass(codes, margin, second, cap, k)
            np.testing.assert_array_equal(got[0], ref[0].numpy())
            np.testing.assert_array_equal(got[1], ref[1].numpy())
            ref = tensor.cap_round_evict(torch.from_numpy(codes), torch.from_numpy(arrived),
                                         torch.from_numpy(score), cap, k)
            got = cap_round_evict(codes, arrived, score, cap, k)
            np.testing.assert_array_equal(got[0], ref[0].numpy())
            np.testing.assert_array_equal(got[1], ref[1].numpy())


class TestBalancedAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = skewed_data(n=6000)
        cls.km = NumpyEmeansBalanced(16, 64, nepochs=3, polish_epochs=1, seed=1,
                                     u=0.05, batch_size=1024)
        cls.km.train(cls.x)

    def test_biased_assignment_is_exact(self):
        x, km = self.x, self.km
        self.assertTrue(np.any(km.bias != 0))
        brute = biased_scores(x, km.centroids, np.zeros(64, dtype="float32"))
        score = brute - km.bias
        rows = np.arange(len(x))
        distance, labels = km.assign(x)
        best = score.min(1)
        scale = np.abs(best) + float(np.abs(km.bias).max())
        gap = score[rows, labels] - best
        self.assertTrue(np.all(gap <= 1e-5 * scale), gap.max())
        self.assertGreater((labels == score.argmin(1)).mean(), 0.99)
        np.testing.assert_allclose(distance, brute[rows, labels], rtol=1e-4, atol=1e-3)
        # Small inputs are where Faiss would pick its sequential search, which
        # ignores precomputed norms; they must stay biased.
        cvar = faiss.cvar
        saved = cvar.distance_compute_blas_threshold, cvar.distance_compute_blas_database_bs
        for small in (1, 7, 19, 1000):
            _, few = km.assign(x[:small])
            gap = score[rows[:small], few] - best[:small]
            self.assertTrue(np.all(gap <= 1e-5 * scale[:small]), (small, gap.max()))
        self.assertEqual(
            (cvar.distance_compute_blas_threshold, cvar.distance_compute_blas_database_bs), saved)

    def test_unbiased_assignment_ignores_the_bias(self):
        _, labels = self.km.assign(self.x, biased=False)
        _, ref = faiss.knn(self.x, self.km.centroids, 1)
        np.testing.assert_array_equal(labels, ref.ravel())

    def test_ignored_norms_raise(self):
        real = faiss.knn_L2sqr

        def drop_norms(xp, yp, d, nx, ny, k, dp, ip, y_norm2):
            return real(xp, yp, d, nx, ny, k, dp, ip)

        with unittest.mock.patch.object(faiss, "knn_L2sqr", drop_norms):
            with self.assertRaises(RuntimeError):
                self.km.assign(self.x)


class TestBalancedTraining(unittest.TestCase):
    def test_defaults_are_the_recipe(self):
        # Only d and k: 25 passes, 15 training and 10 polish, and the recipe's settings.
        km = NumpyEmeansBalanced(16, 64)
        self.assertEqual((km.nepochs, km.train_epochs, km.polish_epochs), (25, 15, 10))
        self.assertEqual((km.u, km.polish_u_cap, km.cap_mode, km.cap_factor, km.cap_rounds),
                         (0.02, 0.02, "rounds", 2.0, 2))
        self.assertEqual((km.inplace_quantile, km.donor_rule), (0.5, "proportional"))
        # A budget is split 60 / 40 between training and polish unless polish_epochs is given.
        self.assertEqual(split_epochs(10), (6, 4))
        self.assertEqual(split_epochs(1), (1, 0))
        self.assertEqual(split_epochs(25, 0), (25, 0))
        for nepochs, polish in ((0, None), (3, 3), (2.5, None)):
            with self.assertRaises(ValueError):
                split_epochs(nepochs, polish)

    def test_balancing_evens_the_clusters(self):
        x = skewed_data()
        # The quality batch law gives 128 rows here (Q = 2): enough steps for
        # the bias to settle at u = 0.02.
        # The unbalanced run (u = 0, no cap) revives dead clusters with the
        # same donor rule; splitting the largest clusters already evens it.
        for rule, margin in (("largest", 0.6), ("proportional", 0.5)):
            plain = NumpyEmeansBalanced(16, 64, nepochs=8, polish_epochs=4, seed=0,
                                        u=0, cap_mode=None, donor_rule=rule)
            plain.train(x)
            _, lp = plain.assign(x)
            for mode in ("rounds", None):
                balanced = NumpyEmeansBalanced(16, 64, nepochs=8, polish_epochs=4, seed=0,
                                               cap_mode=mode, donor_rule=rule)
                balanced.train(x)
                _, lb = balanced.assign(x)
                self.assertLess(imbalance(lb, 64) - 1, margin * (imbalance(lp, 64) - 1),
                                (rule, mode))

    def test_runs_are_deterministic(self):
        x = skewed_data(n=8000)
        runs = []
        for seed in (3, 3, 4):
            km = NumpyEmeansBalanced(16, 64, nepochs=5, polish_epochs=2, seed=seed,
                                     batch_size=512)
            centroids, labels = km.train(x)
            runs.append((centroids, km.bias, labels, km.n_splits, km.n_inplace))
        for a, b in zip(runs[0], runs[1]):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(runs[0][0], runs[2][0]))

    def test_tail_is_dropped_and_labelled_by_the_final_routing(self):
        x = skewed_data(n=5000)
        km = NumpyEmeansBalanced(16, 32, nepochs=3, polish_epochs=1, seed=0,
                                 batch_size=1024)
        steps = []
        real = km._train_step
        km._train_step = lambda xx, index, step: steps.append(len(index)) or real(xx, index, step)
        _, labels = km.train(x)
        recipe = km.recipe
        self.assertEqual(recipe["epoch_rows"], [4096] * 3)
        self.assertEqual(recipe["train_steps"], 8)
        self.assertEqual(recipe["polish_steps"], 4)
        self.assertEqual(steps, [1024] * 12)
        self.assertTrue((labels >= 0).all())
        # The 904 tail rows of the last pass take the final biased routing.
        from faiss.contrib.e_means.base import _epoch_seed
        order = np.random.RandomState(_epoch_seed(0, 2)).permutation(len(x))
        tail = order[4096:]
        np.testing.assert_array_equal(labels[tail], km.assign(x[tail])[1])

    def test_max_points_per_centroid_trains_on_a_subset(self):
        x = skewed_data(n=6000)
        km = NumpyEmeansBalanced(16, 32, nepochs=3, polish_epochs=1, max_points_per_centroid=100)
        _, labels = km.train(x)
        self.assertEqual(km.recipe["epoch_rows"][0] // km.recipe["batch_size"],
                         3200 // km.recipe["batch_size"])
        self.assertEqual(int((labels < 0).sum()), 2800)

    def test_polish_freezes_centroids_and_caps_the_rate(self):
        x = skewed_data(n=6000)
        km = NumpyEmeansBalanced(16, 32, nepochs=4, polish_epochs=2, seed=0,
                                 batch_size=1000, u=0.1)
        snapshots = []
        real = km.end_epoch

        def end_epoch():
            if km._epoch == km.train_epochs - 1:
                snapshots.append(km._centroids.copy())
            real()

        km.end_epoch = end_epoch
        km.train(x)
        np.testing.assert_array_equal(km.centroids, snapshots[0])
        self.assertEqual(km.recipe["polish_u"], 0.02)
        self.assertEqual(km.recipe["total_steps"], 24)

    def test_ema_step_is_numpy_emeans_update(self):
        rs = np.random.RandomState(0)
        for alpha in (0.3, 1.0):
            km = NumpyEmeansBalanced(8, 50, nepochs=1, polish_epochs=0, batch_size=100)
            km.begin(1000, rs.randn(50, 8).astype("float32"))
            km._usage[:] = rs.rand(50).astype("float32") * 3
            counts = rs.randint(0, 4, 50).astype("float32")
            sums = (rs.randn(50, 8) * counts[:, None]).astype("float32")
            ref_c, ref_u = km._centroids.copy(), km._usage.copy()
            km._ema_step(sums.copy(), counts, alpha)
            km._centroids, ref_c = ref_c, km._centroids
            km._usage, ref_u = ref_u, km._usage
            km._apply_statistics(None, sums, counts, 0.0, alpha, 1 - alpha, alpha == 1)
            np.testing.assert_allclose(ref_c, km._centroids, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(ref_u, km._usage, rtol=1e-6)

    def test_bias_step_is_the_exact_band(self):
        km = NumpyEmeansBalanced(4, 4, nepochs=2, polish_epochs=1, u=0.5, batch_size=100)
        km.begin(400, np.eye(4, dtype="float32"))
        counts = np.array([40, 0, 30, 30], dtype="float32")
        km._bias_step(counts, 2.0, 1.0, True)
        # eta = u * D * s = 0.5 * 2.0 * 1, r = (m - 25) / 25
        r = (counts - 25) / 25
        np.testing.assert_allclose(km._lam, np.maximum(r, 0))
        np.testing.assert_allclose(km._mu, np.maximum(-r, 0))
        np.testing.assert_allclose(km.bias, km._mu - km._lam)
        # D <- 0.9 D + 0.1 mean; the polish rate is min(u, 0.02).
        km._bias_step(counts, 12.0, 0.5, False)
        eta = 0.02 * 3.0 * 0.5
        np.testing.assert_allclose(km._lam, np.maximum(r + eta * r, 0), rtol=1e-6)


class TestBalancedRevival(unittest.TestCase):
    def _state(self, quantile=0.5):
        km = NumpyEmeansBalanced(2, 6, nepochs=1, polish_epochs=0, seed=0,
                                 batch_size=600, inplace_quantile=quantile)
        centroids = np.arange(1, 13, dtype="float32").reshape(6, 2)
        km.begin(600, centroids)
        km._usage[:] = [10, 10, 10, 10, 0.01, 0.01]
        # Cluster 4 is dead under a low bias, cluster 5 under a high one.
        km.bias[:] = [0.0, 1.0, 2.0, 3.0, -5.0, 9.0]
        km._lam[:] = np.maximum(-km.bias, 0)
        km._mu[:] = np.maximum(km.bias, 0)
        return km, centroids

    def test_low_bias_dead_clusters_are_revived_in_place(self):
        km, centroids = self._state()
        km._revive()
        self.assertEqual(km.n_inplace, 1)
        self.assertEqual(km.n_splits, 1)
        # In place: same position, the live (lower) median bias, the mean usage.
        np.testing.assert_array_equal(km._centroids[4], centroids[4])
        self.assertEqual(km.bias[4], 1.0)
        self.assertEqual((km._lam[4], km._mu[4]), (0.0, 1.0))
        self.assertAlmostEqual(km._usage[4], (40 + 0.02) / 6, places=4)
        # Split: cluster 5 is a faiss multiplicative split of its donor and
        # copies the donor's bias and multipliers; both keep half its usage.
        donor = int(np.flatnonzero(
            np.abs(km._centroids[:5] - km._centroids[5]).max(1) < 1e-2)[0])
        source = centroids[donor]
        sign = np.array([1, -1], dtype="float32")
        np.testing.assert_array_equal(km._centroids[5], source * (1 + np.float32(1e-4) * sign))
        np.testing.assert_array_equal(km._centroids[donor],
                                      source * (1 - np.float32(1e-4) * sign))
        for state in (km.bias, km._lam, km._mu):
            self.assertEqual(state[5], state[donor])
        self.assertEqual(km._usage[5], km._usage[donor])

    def test_without_inplace_revival_every_dead_cluster_splits(self):
        km, _ = self._state(quantile=None)
        km._revive()
        self.assertEqual(km.n_inplace, 0)
        self.assertEqual(km.n_splits, 2)

    def _donor_state(self, rule):
        # Clusters 2, 5 and 7 are dead; without in-place revival each takes a split.
        km = NumpyEmeansBalanced(2, 10, nepochs=1, polish_epochs=0, seed=0, batch_size=1000,
                                 inplace_quantile=None, donor_rule=rule)
        centroids = np.arange(20, dtype="float32").reshape(10, 2)
        km.begin(1000, centroids)
        km._usage[:] = [3, 8, 0.01, 8, 5, 0.01, 9, 0.01, 1, 2]
        return km, centroids

    def _check_splits(self, km, centroids, usage, source):
        target = [2, 5, 7]
        sign = np.array([1, -1], dtype="float32")
        eps = np.float32(1e-4)
        np.testing.assert_array_equal(km._centroids[target], centroids[source] * (1 + eps * sign))
        np.testing.assert_array_equal(km._centroids[source], centroids[source] * (1 - eps * sign))
        np.testing.assert_array_equal(km._usage[target], usage[source] / 2)
        self.assertEqual(km.n_splits, 3)

    def test_largest_donor_rule_splits_the_largest_live_clusters(self):
        km, centroids = self._donor_state("largest")
        self.assertEqual(km.recipe["donor_rule"], "largest")
        usage = km._usage.copy()
        km._revive()
        # Usage 9, 8, 8: the tie between clusters 1 and 3 goes to the lower index.
        self._check_splits(km, centroids, usage, [6, 1, 3])

    def test_proportional_donor_rule_is_the_usage_draw(self):
        km, centroids = self._donor_state("proportional")
        usage = km._usage.copy()
        rng = np.random.RandomState()
        rng.set_state(km._split_rng.get_state())
        receivers = usage < 0.1 * usage.mean()
        donors = ~receivers & (usage > 0)
        uniform = np.maximum(rng.random_sample(10), np.finfo("float64").tiny)
        keys = np.full(10, -np.inf)
        keys[donors] = np.log(usage[donors]) - np.log(-np.log(uniform[donors]))
        source = np.argsort(keys, kind="stable")[-3:][::-1]
        km._revive()
        self._check_splits(km, centroids, usage, source)

    def test_largest_donors_skip_dead_clusters_and_match_the_tensor_functions(self):
        from faiss.contrib.e_means.e_means_balanced_numpy import _largest_donors
        weights = np.array([3, 8, 0.01, 8, 5, 0.01, 9, 0.01, 1, 2], dtype="float32")
        receivers = np.zeros(10, dtype=bool)
        receivers[[2, 5, 7]] = True
        dead = receivers.copy()
        dead[6] = True  # revived in place: not a donor
        source, target = _largest_donors(weights, receivers, dead)
        np.testing.assert_array_equal(source, [1, 3, 4])
        np.testing.assert_array_equal(target, [2, 5, 7])
        try:
            import torch
            from faiss.contrib.e_means import e_means_balanced as tensor
        except ImportError:
            self.skipTest("PyTorch or the tensor class is not available")
        rs = np.random.RandomState(0)
        for _ in range(100):
            k = rs.randint(2, 50)
            weights = rs.randint(0, 5, k).astype("float32")
            receivers = rs.rand(k) < 0.3
            dead = receivers | (rs.rand(k) < 0.1)
            got = _largest_donors(weights, receivers, dead)
            ref = tensor.largest_donors(torch.from_numpy(weights), torch.from_numpy(receivers),
                                        torch.from_numpy(dead))
            if ref[0] is None:
                self.assertIsNone(got[0])
                continue
            np.testing.assert_array_equal(got[0], ref[0].numpy())
            np.testing.assert_array_equal(got[1], ref[1].numpy())

    def test_dead_test_reads_the_usage(self):
        km, _ = self._state()
        km._usage[:] = 10
        km._revive()
        self.assertEqual(km.n_inplace + km.n_splits, 0)


class TestBalancedImports(unittest.TestCase):
    def test_numpy_class_does_not_import_torch(self):
        code = ("import sys; import faiss.contrib.e_means.e_means_balanced_numpy; "
                "sys.exit(int('torch' in sys.modules))")
        self.assertEqual(subprocess.call([sys.executable, "-c", code]), 0)


if __name__ == "__main__":
    unittest.main()
