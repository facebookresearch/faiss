# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset


class SuperKMeansTest(unittest.TestCase):
    def test_train_rejects_too_few_points(self):
        sc = faiss.SuperKMeans(128, 16)
        x = SyntheticDataset(128, 10, 0, 0).get_train()  # n=10 < k=16
        with self.assertRaises(RuntimeError):
            sc.train(x)

    def test_objective_decreases_monotonically(self):
        d, k, n = 32, 8, 1000
        x = SyntheticDataset(d, n, 0, 0).get_train()

        p = faiss.SuperKMeansParameters()
        p.seed = 42
        p.niter = 10
        sc = faiss.SuperKMeans(d, k, p)
        sc.train(x)

        stats = sc.iteration_stats
        objs = [stats.at(i).obj for i in range(stats.size())]
        for i in range(1, len(objs)):
            self.assertLessEqual(objs[i], objs[i - 1] * 1.001)

    def test_pruned_iter_assignments_close_to_vanilla(self):
        d, k, n = 64, 16, 2000
        x = SyntheticDataset(d, n, 0, 0).get_train()

        p = faiss.SuperKMeansParameters()
        p.seed = 42
        p.niter = 10
        sc = faiss.SuperKMeans(d, k, p)
        sc.train(x)

        quant = faiss.IndexFlatL2(d)
        vanilla = faiss.Clustering(d, k)
        vanilla.seed = 42
        vanilla.niter = 10
        vanilla.train(x, quant)

        sc_final = sc.iteration_stats.at(sc.iteration_stats.size() - 1).obj
        v_stats = vanilla.iteration_stats
        v_final = v_stats.at(v_stats.size() - 1).obj
        self.assertLess(abs(sc_final - v_final) / v_final, 0.05)

    def test_pruning_rate_in_expected_range(self):
        # The 10-d intrinsic manifold of SyntheticDataset gives ADSampling
        # the assigned-vs-other distance gap it needs to prune effectively.
        d, k, n = 64, 64, 5000
        x = SyntheticDataset(d, n, 0, 0).get_train()

        p = faiss.SuperKMeansParameters()
        p.seed = 42
        p.niter = 5
        sc = faiss.SuperKMeans(d, k, p)
        sc.train(x)

        rates = faiss.vector_to_array(sc.gemm_pruning_rates)
        self.assertEqual(rates.shape, (5,))
        self.assertEqual(rates[0], 0.0)  # iter 0: no pruning
        # Lower bound 0.30: looser than the 0.93 design target because
        # synthetic d=64 data lacks the cluster gap real embeddings have at
        # d=768; the assertion still confirms meaningful pruning occurs.
        for i in range(2, 5):
            self.assertGreaterEqual(rates[i], 0.30)

    def test_pdx_offset_correct_for_unaligned_y_batch(self):
        # PDX block layout is sized for the full k centroids, not per y-batch
        # tile. The y-batch tile boundary must be invisible to PDX-offset
        # arithmetic. k=70 with y_batch=32 forces a 6-wide tail tile; assert
        # the resulting centroids are identical to running with y_batch=70
        # (single tile, no tail).
        d, k, n = 64, 70, 5000
        x = SyntheticDataset(d, n, 0, 0).get_train()

        def train_with_y_batch(y_batch):
            p = faiss.SuperKMeansParameters()
            p.seed = 1234
            p.niter = 5
            p.y_batch = y_batch
            sc = faiss.SuperKMeans(d, k, p)
            sc.train(x)
            return faiss.vector_to_array(sc.centroids)

        tiled = train_with_y_batch(32)  # k % y_batch = 6
        single = train_with_y_batch(k)  # k % y_batch = 0
        np.testing.assert_array_equal(tiled, single)

    def test_determinism(self):
        # Bit-exact reproducibility requires MKL_CBWR=COMPATIBLE and a fixed
        # OMP thread count. The BUCK target sets these in `env`; for OSS
        # ctest the user must export them before running.
        d, k, n = 128, 64, 5000
        x = SyntheticDataset(d, n, 0, 0).get_train()

        p = faiss.SuperKMeansParameters()
        p.niter = 5
        p.seed = 1234

        sc1 = faiss.SuperKMeans(d, k, p)
        sc1.train(x)
        sc2 = faiss.SuperKMeans(d, k, p)
        sc2.train(x)

        np.testing.assert_array_equal(
            faiss.vector_to_array(sc1.centroids),
            faiss.vector_to_array(sc2.centroids),
        )
        np.testing.assert_array_equal(
            faiss.vector_to_array(sc1.gemm_pruning_rates),
            faiss.vector_to_array(sc2.gemm_pruning_rates),
        )

    def test_rejects_nan_input(self):
        # check_input_data_for_NaNs defaults to True.
        sc = faiss.SuperKMeans(128, 64)
        x = SyntheticDataset(128, 5000, 0, 0).get_train()
        x.flat[42] = float("nan")
        with self.assertRaises(RuntimeError):
            sc.train(x)

    def test_cluster_splitting_with_many_duplicates(self):
        # 80% duplicates force empty clusters after the first assignment;
        # split_clusters must fire and produce distinct centroids.
        d, k, n = 64, 16, 1000
        n_dup = int(n * 0.8)

        ds = SyntheticDataset(d, (n - n_dup) + 1, 0, 0).get_train()
        x = np.empty((n, d), dtype="float32")
        x[:n_dup] = ds[0]
        x[n_dup:] = ds[1:]

        p = faiss.SuperKMeansParameters()
        p.seed = 42
        p.niter = 5
        sc = faiss.SuperKMeans(d, k, p)
        sc.train(x)

        stats = sc.iteration_stats
        total_splits = sum(stats.at(i).nsplit for i in range(stats.size()))
        self.assertGreater(total_splits, 0)

        centroids = faiss.vector_to_array(sc.centroids).reshape(k, d)
        unique_centroids = {centroids[j].tobytes() for j in range(k)}
        self.assertEqual(len(unique_centroids), k)


class SuperKmeansWrapperTest(unittest.TestCase):
    def test_train_populates_kmeans_surface(self):
        d, k, n = 64, 16, 2000
        x = SyntheticDataset(d, n, 0, 0).get_train()

        km = faiss.SuperKmeans(d, k, niter=5, seed=42)
        final_obj = km.train(x)

        self.assertEqual(km.centroids.shape, (k, d))
        self.assertEqual(km.obj.shape, (5,))
        self.assertEqual(len(km.iteration_stats), 5)
        self.assertEqual(final_obj, km.obj[-1])
        self.assertEqual(km.gemm_pruning_rates.shape, (5,))

        D, I = km.assign(x)
        self.assertEqual(I.shape, (n,))
        self.assertEqual(D.shape, (n,))
        self.assertTrue((I >= 0).all() and (I < k).all())

    def test_final_obj_matches_vanilla_kmeans(self):
        d, k, n = 64, 16, 2000
        x = SyntheticDataset(d, n, 0, 0).get_train()

        vanilla = faiss.Kmeans(d, k, niter=10, seed=42)
        vanilla.train(x)

        sk = faiss.SuperKmeans(d, k, niter=10, seed=42)
        sk.train(x)

        rel_diff = abs(sk.obj[-1] - vanilla.obj[-1]) / vanilla.obj[-1]
        self.assertLess(rel_diff, 0.05)
