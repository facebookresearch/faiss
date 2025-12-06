# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for IndexHNSWFlatPanorama.

IndexHNSWFlatPanorama combines the hierarchical navigable small world (HNSW)
graph structure with Panorama's level-oriented storage and progressive
filtering for efficient approximate nearest neighbor search.

Paper: https://www.arxiv.org/pdf/2510.00566
"""

import unittest
import tempfile
import os
import numpy as np

import faiss
from faiss.contrib.datasets import SyntheticDataset


class TestIndexHNSWFlatPanorama(unittest.TestCase):
    """Test Suite for IndexHNSWFlatPanorama."""

    # Helper methods for index creation and data generation

    def generate_data(self, d, nt, nb, nq, seed=1234):
        """Generate synthetic data using SyntheticDataset."""
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def compute_ground_truth(self, xb, xq, k):
        """Compute ground truth using brute force search."""
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        D, I = index.search(xq, k)
        return D, I

    def compute_recall(self, gt_I, test_I):
        """Compute recall@k - fraction of ground truth results found in test results."""
        nq, k = gt_I.shape
        recalls = [
            np.isin(gt_I[i], test_I[i]).sum() 
            for i in range(nq)
        ]
        return sum(recalls) / (nq * k)

    # Core functionality tests

    def test_basic_construction(self):
        """Test basic index construction and properties."""
        d = 128
        M = 16
        num_levels = 8

        index = faiss.IndexHNSWFlatPanorama(d, M, num_levels)

        self.assertEqual(index.d, d)
        self.assertEqual(index.ntotal, 0)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.hnsw.nb_neighbors(0), 2 * M)  # Level 0 has 2*M neighbors
        self.assertEqual(index.num_panorama_levels, num_levels)
        self.assertEqual(index.metric_type, faiss.METRIC_L2)

    def test_add_and_search(self):
        """Test basic add and search operations."""
        d = 128
        nb = 1000
        nt = 1500
        nq = 10
        k = 10

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=1234)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        self.assertEqual(index.ntotal, nb)
        self.assertEqual(index.cum_sums.size(), nb * (8 + 1))

        # Search
        D, I = index.search(xq, k)

        # Check that results are valid
        self.assertTrue(np.all(I >= -1))
        self.assertTrue(np.all(I < nb))
        self.assertTrue(np.all(D >= 0.0))

    def test_recall(self):
        """Test search recall quality."""
        d = 128
        nb = 1000
        nt = 1500
        nq = 100
        k = 10

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=42)

        # Create index
        index = faiss.IndexHNSWFlatPanorama(d, 32, 8)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 64

        # Add to index
        index.add(xb)

        # Search
        D, I = index.search(xq, k)

        # Compute ground truth
        gt_D, gt_I = self.compute_ground_truth(xb, xq, k)

        # Check recall
        recall = self.compute_recall(gt_I, I)
        print(f"Recall@{k}: {recall}")

        # With efSearch=64, we should get reasonably good recall
        # The threshold is lower than vanilla HNSW because of approximate distances
        self.assertGreaterEqual(recall, 0.85)

    def test_different_panorama_levels(self):
        """Test with different numbers of panorama levels."""
        d = 256
        nb = 500
        nt = 700
        nq = 50
        k = 5

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        # Compute ground truth
        gt_D, gt_I = self.compute_ground_truth(xb, xq, k)

        # Test different number of panorama levels
        for nlevels in [4, 8, 16]:
            with self.subTest(nlevels=nlevels):
                index = faiss.IndexHNSWFlatPanorama(d, 32, nlevels)
                index.hnsw.efSearch = 64

                index.add(xb)

                D, I = index.search(xq, k)

                recall = self.compute_recall(gt_I, I)
                print(f"Recall@{k} with {nlevels} levels: {recall}")

                # More levels should still maintain reasonable recall
                self.assertGreaterEqual(recall, 0.80)

    def test_consistency(self):
        """Test that search results are consistent across multiple searches."""
        d = 64
        nb = 500
        nt = 700
        nq = 10
        k = 5

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=123)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        # Search twice and compare results
        D1, I1 = index.search(xq, k)
        D2, I2 = index.search(xq, k)

        # Results should be identical
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_array_equal(D1, D2)

    def test_io(self):
        """Test serialization and deserialization."""
        d = 64
        nb = 500
        nt = 700
        nq = 10
        k = 5

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        # Get search results before saving
        D_before, I_before = index.search(xq, k)

        # Save and load using temporary file
        fname = tempfile.mktemp(suffix='.index')
        try:
            faiss.write_index(index, fname)
            loaded_index = faiss.read_index(fname)

            self.assertIsInstance(loaded_index, faiss.IndexHNSWFlatPanorama)
            self.assertEqual(loaded_index.d, d)
            self.assertEqual(loaded_index.ntotal, nb)
            self.assertEqual(loaded_index.num_panorama_levels, 8)

            # Search after loading
            D_after, I_after = loaded_index.search(xq, k)

            # Results should be identical after serialization
            np.testing.assert_array_equal(I_before, I_after)
            np.testing.assert_array_equal(D_before, D_after)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_clone(self):
        """Test index cloning."""
        d = 64
        nb = 500
        nt = 700
        nq = 10
        k = 5

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2025)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        # Get search results before cloning
        D_before, I_before = index.search(xq, k)

        # Clone
        cloned_index = faiss.clone_index(index)

        self.assertIsInstance(cloned_index, faiss.IndexHNSWFlatPanorama)
        self.assertEqual(cloned_index.d, d)
        self.assertEqual(cloned_index.ntotal, nb)
        self.assertEqual(cloned_index.num_panorama_levels, 8)

        # Search after cloning
        D_after, I_after = cloned_index.search(xq, k)

        # Results should be identical after cloning
        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_array_equal(D_before, D_after)

    def test_factory(self):
        """Test factory string creation."""
        d = 64
        nb = 500
        nt = 700

        # Test factory creation with default levels
        index1 = faiss.index_factory(d, "HNSW32_FlatPanorama")
        self.assertIsInstance(index1, faiss.IndexHNSWFlatPanorama)
        self.assertEqual(index1.d, d)
        self.assertEqual(index1.hnsw.nb_neighbors(0), 2 * 32)  # Level 0 has 2*M neighbors
        self.assertEqual(index1.num_panorama_levels, 8)  # default

        # Test factory creation with explicit levels
        index2 = faiss.index_factory(d, "HNSW16_FlatPanorama12")
        self.assertIsInstance(index2, faiss.IndexHNSWFlatPanorama)
        self.assertEqual(index2.d, d)
        self.assertEqual(index2.hnsw.nb_neighbors(0), 2 * 16)  # Level 0 has 2*M neighbors
        self.assertEqual(index2.num_panorama_levels, 12)

        # Test that it works
        _, xb, _ = self.generate_data(d, nt, nb, 10, seed=888)
        index2.add(xb)
        self.assertEqual(index2.ntotal, nb)

    def test_reset(self):
        """Test index reset."""
        d = 64
        nb = 500
        nt = 700

        # Generate data
        _, xb, _ = self.generate_data(d, nt, nb, 10, seed=999)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        self.assertEqual(index.ntotal, nb)
        self.assertGreater(index.cum_sums.size(), 0)

        # Reset
        index.reset()

        self.assertEqual(index.ntotal, 0)
        self.assertEqual(index.cum_sums.size(), 0)

    def test_high_dimensional(self):
        """Test with high-dimensional data (Panorama is designed for this)."""
        d = 512
        nb = 500
        nt = 700
        nq = 50
        k = 10

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

        index = faiss.IndexHNSWFlatPanorama(d, 32, 16)
        index.hnsw.efSearch = 128
        index.add(xb)

        D, I = index.search(xq, k)

        # Compute ground truth
        gt_D, gt_I = self.compute_ground_truth(xb, xq, k)

        recall = self.compute_recall(gt_I, I)
        print(f"High-dimensional (d={d}) Recall@{k}: {recall}")

        # Should work well for high-dimensional data
        self.assertGreaterEqual(recall, 0.85)

    def test_add_after_search(self):
        """Test adding vectors after initial search."""
        d = 64
        nb_initial = 300
        nt_initial = 500
        nb_add = 200
        nt_add = 300
        nq = 10
        k = 5

        # Generate data
        _, xb_initial, xq = self.generate_data(d, nt_initial, nb_initial, nq, seed=1111)
        _, xb_add, _ = self.generate_data(d, nt_add, nb_add, 1, seed=2222)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)

        # Add initial vectors
        index.add(xb_initial)

        # Search on initial index
        D_before, I_before = index.search(xq, k)

        # Add more vectors after search
        index.add(xb_add)

        self.assertEqual(index.ntotal, nb_initial + nb_add)
        self.assertEqual(
            index.cum_sums.size(),
            (nb_initial + nb_add) * (index.num_panorama_levels + 1)
        )

        # Search again - should work correctly with all vectors
        D_after, I_after = index.search(xq, k)

        # Compute ground truth on all vectors
        xb_all = np.vstack([xb_initial, xb_add])

        gt_D, gt_I = self.compute_ground_truth(xb_all, xq, k)

        recall = self.compute_recall(gt_I, I_after)
        print(f"Recall after adding more vectors: {recall}")

        # Recall might be slightly lower than single-batch due to HNSW graph structure
        self.assertGreaterEqual(recall, 0.80)

        # Verify that previously found neighbors can still be found
        # (they might not be in exact same positions due to new neighbors)
        found_count = 0
        for i in range(nq):
            for j in range(k):
                if I_before[i, j] >= 0:
                    # Check if this neighbor is still in the result set
                    if I_before[i, j] in I_after[i, :]:
                        found_count += 1

        retention = float(found_count) / (nq * k)
        print(f"Retention of previous neighbors: {retention}")
        # Should retain a reasonable number of previous neighbors (new ones might push some out)
        # The threshold is lower to account for the approximate nature of HNSW
        self.assertGreaterEqual(retention, 0.5)

    def test_permute_entries(self):
        """Test permuting entries in the index."""
        d = 64
        nb = 500
        nt = 700
        nq = 10
        k = 5

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=3333)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        # Search before permutation
        D_before, I_before = index.search(xq, k)

        # Create a permutation (reverse order for simplicity)
        perm = np.arange(nb - 1, -1, -1, dtype=np.int64)

        # Store cum_sums before permutation for verification
        cum_sums_before = faiss.vector_to_array(index.cum_sums).copy()

        # Apply permutation
        index.permute_entries(perm)

        # Verify cum_sums were permuted correctly
        cum_sums_after = faiss.vector_to_array(index.cum_sums)
        for i in range(nb):
            src = perm[i]
            for j in range(index.num_panorama_levels + 1):
                expected = cum_sums_before[src * (index.num_panorama_levels + 1) + j]
                actual = cum_sums_after[i * (index.num_panorama_levels + 1) + j]
                self.assertEqual(
                    actual, expected,
                    f"cum_sums not permuted correctly at i={i}, j={j}"
                )

        # Search after permutation
        D_after, I_after = index.search(xq, k)

        # Results should be identical (with permuted IDs)
        np.testing.assert_allclose(D_before, D_after,
                                   err_msg="Distance changed after permutation")

        # The ID should be the permuted version
        # If before we found vector j, after permutation we should find
        # the new position of vector j
        for i in range(nq):
            for j in range(k):
                if I_before[i, j] >= 0:
                    # Find where I_before[i, j] moved to
                    old_id = I_before[i, j]
                    new_id = np.where(perm == old_id)[0][0]
                    self.assertEqual(
                        I_after[i, j], new_id,
                        f"Permuted ID mismatch at position ({i}, {j})"
                    )

        # Verify overall recall is maintained
        gt_D, gt_I = self.compute_ground_truth(xb, xq, k)

        # Map ground truth IDs through the permutation
        gt_I_permuted = np.copy(gt_I)
        for i in range(nq):
            for j in range(k):
                if gt_I[i, j] >= 0:
                    # Find where gt_I[i, j] moved to
                    gt_I_permuted[i, j] = np.where(perm == gt_I[i, j])[0][0]

        recall = self.compute_recall(gt_I_permuted, I_after)
        print(f"Recall after permutation: {recall}")
        self.assertGreaterEqual(recall, 0.85)

    def test_id_selector_range(self):
        """Test ID filtering with range selector."""
        d = 128
        nb = 1000
        nt = 1500
        nq = 10
        k = 10

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        index = faiss.IndexHNSWFlatPanorama(d, 32, 8)
        index.hnsw.efSearch = 64
        index.add(xb)

        selector = faiss.IDSelectorRange(200, 600)
        params = faiss.SearchParametersHNSW()
        params.sel = selector

        D, I = index.search(xq, k, params=params)

        # Verify all returned IDs are in range (HNSW may return < k)
        count = 0
        for i in range(nq):
            for j in range(k):
                if I[i, j] != -1:
                    self.assertGreaterEqual(I[i, j], 200)
                    self.assertLess(I[i, j], 600)
                    count += 1

        print(f"IDSelectorRange test: {count} results")

    def test_id_selector_batch(self):
        """Test ID filtering with batch selector."""
        d = 128
        nb = 1000
        nt = 1500
        nq = 10
        k = 5

        # Generate data
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=654)

        index = faiss.IndexHNSWFlatPanorama(d, 32, 8)
        index.hnsw.efSearch = 64
        index.add(xb)

        allowed_ids = np.arange(0, nb, 10, dtype=np.int64)

        selector = faiss.IDSelectorBatch(allowed_ids)
        params = faiss.SearchParametersHNSW()
        params.sel = selector

        D, I = index.search(xq, k, params=params)

        # Verify all returned IDs are in allowed set (HNSW may return < k)
        allowed_set = set(allowed_ids)
        count = 0
        for i in range(nq):
            for j in range(k):
                if I[i, j] != -1:
                    self.assertIn(
                        I[i, j], allowed_set,
                        f"ID {I[i, j]} not in allowed set"
                    )
                    count += 1

        print(f"IDSelectorBatch test: {count} results")
