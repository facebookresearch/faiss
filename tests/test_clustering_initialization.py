# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss
from faiss.contrib import datasets


def get_method_name(method):
    """Get human-readable name for initialization method."""
    method_names = {
        faiss.ClusteringInitMethod_RANDOM: "RANDOM",
        faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS: "KMEANS++",
        faiss.ClusteringInitMethod_AFK_MC2: "AFK-MC2",
    }
    return method_names.get(method, str(method))


def verify_centroids_from_dataset(test_case, centroids, dataset):
    """Verify centroids are from the dataset."""
    D, _ = faiss.knn(centroids, dataset, 1)
    threshold = 1e-3 * centroids.shape[1]
    test_case.assertTrue(
        np.all(D[:, 0] < threshold), "Some centroids not from dataset"
    )


def verify_centroids_unique(test_case, centroids):
    """Verify all centroids are unique (no duplicates)."""
    D, _ = faiss.knn(centroids, centroids, 2)
    test_case.assertTrue(np.all(D[:, 1] > 1e-4), "Duplicate centroids found")


def generate_clustered_data(n, d, n_clusters, seed=42):
    """Generate well-separated clustered data for testing."""
    np.random.seed(seed)
    centers = np.random.randn(n_clusters, d).astype(np.float32) * 20
    labels = np.random.randint(0, n_clusters, n)
    noise = np.random.randn(n, d).astype(np.float32) * 0.5
    xt = centers[labels] + noise
    return np.ascontiguousarray(xt, dtype=np.float32)


def run_init(method, xt, d, k, seed=42, chain_length=None):
    """Run clustering initialization and return centroids."""
    init = faiss.ClusteringInitialization(d, k)
    init.method = method
    init.seed = seed
    if chain_length is not None:
        init.afkmc2_chain_length = chain_length
    centroids = np.zeros((k, d), dtype=np.float32)
    init.init_centroids(len(xt), faiss.swig_ptr(xt), faiss.swig_ptr(centroids))
    return centroids


class TestClusteringInitialization(unittest.TestCase):
    """Test ClusteringInitialization standalone API and algorithm behavior."""

    def test_initialization_methods(self):
        """Test all init methods produce valid, unique centroids."""
        ds = datasets.SyntheticDataset(128, 100000, 0, 0)
        xt = ds.get_train()
        k = 64

        for method in [
            faiss.ClusteringInitMethod_RANDOM,
            faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS,
            faiss.ClusteringInitMethod_AFK_MC2,
        ]:
            with self.subTest(method=get_method_name(method)):
                centroids = run_init(method, xt, ds.d, k)
                verify_centroids_from_dataset(self, centroids, xt)
                verify_centroids_unique(self, centroids)

    def test_determinism(self):
        """Test that initialization is deterministic with the same seed."""
        ds = datasets.SyntheticDataset(64, 100000, 0, 0)
        xt = ds.get_train()
        k = 32

        for method in [
            faiss.ClusteringInitMethod_RANDOM,
            faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS,
            faiss.ClusteringInitMethod_AFK_MC2,
        ]:
            with self.subTest(method=get_method_name(method)):
                centroids1 = run_init(method, xt, ds.d, k, seed=123)
                centroids2 = run_init(method, xt, ds.d, k, seed=123)
                np.testing.assert_array_equal(centroids1, centroids2)

    def test_afkmc2_chain_length_effect(self):
        """Test that longer AFK-MC² chain lengths produce better init."""
        d, k = 32, 64
        xt = generate_clustered_data(10000, d, k)

        chain_lengths = [1, 10, 50, 100, 200]
        n_trials = 5
        results = {}

        for chain_length in chain_lengths:
            total_obj = 0.0
            for trial_seed in range(n_trials):
                centroids = run_init(
                    faiss.ClusteringInitMethod_AFK_MC2, xt, d, k,
                    seed=42 + trial_seed, chain_length=chain_length
                )
                D, _ = faiss.knn(xt, centroids, 1)
                total_obj += np.sum(D)
            results[chain_length] = total_obj / n_trials

        # Verify monotonic improvement: each longer chain should not be worse
        for i in range(1, len(chain_lengths)):
            prev, curr = chain_lengths[i - 1], chain_lengths[i]
            self.assertLessEqual(
                results[curr], results[prev] * 1.05,
                f"Chain {curr} should not be worse than chain {prev}"
            )

    def test_with_existing_centroids(self):
        """Test that considering existing centroids improves initialization."""
        n, d, k = 10000, 32, 64
        n_existing = 10
        xt = generate_clustered_data(n, d, k)

        existing_indices = np.random.choice(n, n_existing, replace=False)
        existing_centroids = xt[existing_indices].copy()

        for init_method in [
            faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS,
            faiss.ClusteringInitMethod_AFK_MC2,
        ]:
            with self.subTest(init_method=get_method_name(init_method)):
                k_remaining = k - n_existing
                init = faiss.ClusteringInitialization(d, k_remaining)
                init.method = init_method
                init.seed = 42

                new_centroids = np.zeros((k_remaining, d), dtype=np.float32)
                init.init_centroids(
                    len(xt),
                    faiss.swig_ptr(xt),
                    faiss.swig_ptr(new_centroids),
                    n_existing,
                    faiss.swig_ptr(existing_centroids),
                )

                all_centroids = np.vstack([existing_centroids, new_centroids])
                D_with, _ = faiss.knn(xt, all_centroids, 1)
                distortion_with = np.sum(D_with)

                # Baseline: initialize without knowing about existing centroids
                baseline_centroids = run_init(init_method, xt, d, k_remaining)
                all_baseline = np.vstack(
                    [existing_centroids, baseline_centroids]
                )
                D_without, _ = faiss.knn(xt, all_baseline, 1)
                distortion_without = np.sum(D_without)

                self.assertLessEqual(
                    distortion_with, distortion_without * 1.05
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
