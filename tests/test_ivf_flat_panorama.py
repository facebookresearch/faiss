# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for IndexIVFFlatPanorama.

Panorama is an adaptation of IndexIVFFlat that uses level-oriented storage
and progressive filtering with Cauchy-Schwarz bounds to achieve significant
speedups (up to 20x when combined with PCA or Cayley transforms) with zero
loss in accuracy.

Paper: https://www.arxiv.org/pdf/2510.00566
"""

import unittest

import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset


class TestIndexIVFFlatPanorama(unittest.TestCase):
    """Test Suite for IndexIVFFlatPanorama."""

    # Helper methods for index creation and data generation

    def generate_data(self, d, nt, nb, nq, seed=42):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def create_ivf_flat(
        self, d, nlist, xt, xb=None, nprobe=None, make_direct_map=False
    ):
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(xt)
        if make_direct_map:
            index.make_direct_map()
        if xb is not None:
            index.add(xb)
        if nprobe is not None:
            index.nprobe = nprobe
        return index

    def create_panorama(
        self, d, nlist, nlevels, xt, xb=None, nprobe=None, make_direct_map=False
    ):
        """Create and initialize IndexIVFFlatPanorama."""
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlatPanorama(quantizer, d, nlist, nlevels)
        index.train(xt)
        if make_direct_map:
            index.make_direct_map()
        if xb is not None:
            index.add(xb)
        if nprobe is not None:
            index.nprobe = nprobe
        return index

    def assert_search_results_equal(
        self,
        D_regular,
        I_regular,
        D_panorama,
        I_panorama,
        rtol=1e-5,
        atol=1e-7,
        otol=1e-3,
    ):
        # Allow small tolerance in overlap rate to account for floating-point errors
        # in distance computations that can affect ordering when distances are nearly equal.
        # Faiss: (a - b) * (a - b) vs. Panorama: a * a + b * b - 2(a * b)
        overlap_rate = np.mean(I_regular == I_panorama)

        self.assertGreater(
            overlap_rate,
            1 - otol,
            f"Overlap rate {overlap_rate:.6f} is not > {1-otol:.3f}. ",
        )
        np.testing.assert_allclose(
            D_regular, D_panorama, rtol=rtol, atol=atol, err_msg="Distances mismatch"
        )

    def assert_range_results_equal(
        self,
        lims_regular,
        D_regular,
        I_regular,
        lims_panorama,
        D_panorama,
        I_panorama,
        nq,
    ):
        np.testing.assert_array_equal(
            lims_regular, lims_panorama, err_msg="Different result counts"
        )

        for i in range(nq):
            n_results = lims_regular[i + 1] - lims_regular[i]
            if n_results > 0:
                ids_reg = I_regular[lims_regular[i] : lims_regular[i + 1]]
                dist_reg = D_regular[lims_regular[i] : lims_regular[i + 1]]
                ids_pan = I_panorama[lims_panorama[i] : lims_panorama[i + 1]]
                dist_pan = D_panorama[lims_panorama[i] : lims_panorama[i + 1]]

                sort_reg, sort_pan = np.argsort(ids_reg), np.argsort(ids_pan)
                np.testing.assert_array_equal(ids_reg[sort_reg], ids_pan[sort_pan])
                np.testing.assert_allclose(
                    dist_reg[sort_reg], dist_pan[sort_pan], rtol=1e-5
                )

    # Core functionality tests

    def test_exact_match_with_ivf_flat(self):
        """Core test: Panorama must return identical results to IndexIVFFlat"""
        d, nb, nt, nq, nlist, nlevels, k = 128, 100000, 150000, 1000, 512, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=42)

        for nprobe in [1, 16, 64, 256]:
            with self.subTest(nprobe=nprobe):
                index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe)
                index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe)

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_exact_match_with_ivf_flat_medium(self):
        """Core test: Medium scale version"""
        d, nb, nt, nq, nlist, nlevels, k = 64, 10000, 15000, 200, 128, 4, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=42)

        for nprobe in [1, 4, 8, nlist]:
            with self.subTest(nprobe=nprobe):
                index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe)
                index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe)

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_range_search(self):
        """Test range search returns correct results within radius"""
        d, nb, nt, nq, nlist, nlevels = 128, 50000, 75000, 500, 256, 8
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=123)

        index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=64)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=64)

        for radius in [0.5, 1.0, 2.0, 5.0]:
            with self.subTest(radius=radius):
                lims_reg, D_reg, I_reg = index_regular.range_search(xq, radius)
                lims_pan, D_pan, I_pan = index_panorama.range_search(xq, radius)

                self.assertTrue(
                    np.all(D_pan <= radius), f"Some distances exceed radius={radius}"
                )
                self.assert_range_results_equal(
                    lims_reg, D_reg, I_reg, lims_pan, D_pan, I_pan, nq
                )

    # Parameter variation tests

    def test_different_n_levels(self):
        """Test correctness with various n_levels parameter values"""
        d, nb, nt, nq, nlist, k = 128, 25000, 40000, 300, 128, 15
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        index_base = self.create_ivf_flat(d, nlist, xt, xb, nprobe=32)
        D_base, I_base = index_base.search(xq, k)

        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        prev_ratio_dims_scanned = float('inf')
        for nlevels in [1, 2, 4, 8, 16, 32]:
            with self.subTest(nlevels=nlevels):
                faiss.cvar.indexPanorama_stats.reset()
                index = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=32)
                D, I = index.search(xq, k)
                self.assert_search_results_equal(D_base, I_base, D, I)

                # Verify ratio_dims_scanned decreases as n_levels increases
                ratio_dims_scanned = (
                    faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                )
                self.assertLess(ratio_dims_scanned, prev_ratio_dims_scanned)
                prev_ratio_dims_scanned = ratio_dims_scanned

        faiss.omp_set_num_threads(nt)

    def test_uneven_dimension_division(self):
        """Test when n_levels doesn't evenly divide dimension"""
        test_cases = [(65, 4), (63, 8), (100, 7)]

        # TODO(aknayar): Test functions like get_single_code().

        for d, nlevels in test_cases:
            with self.subTest(d=d, nlevels=nlevels):
                nb, nt, nq, nlist, k = 5000, 700, 20, 16, 5
                xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

                index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=4)
                index_panorama = self.create_panorama(
                    d, nlist, nlevels, xt, xb, nprobe=4
                )

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_single_level(self):
        """Test edge case with n_levels=1"""
        d, nb, nt, nq, nlist, nlevels, k = 32, 500, 700, 20, 16, 1, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=333)

        index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=4)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=4)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    def test_multiple_levels_small_dimension(self):
        """Test edge case: more levels than dimension naturally supports"""
        d, nb, nt, nq, nlist, nlevels, k = 16, 200, 300, 10, 4, 8, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1212)

        index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=4)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=4)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    # ID selector tests

    def test_id_selector_range(self):
        """Test ID filtering with range selector"""
        d, nb, nt, nq, nlist, nlevels, k = 128, 80000, 120000, 500, 512, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=64)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=64)

        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(10000, 50000)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        self.assertTrue(np.all(I_panorama >= 10000))
        self.assertTrue(np.all(I_panorama < 50000))

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5
        )

    def test_id_selector_batch(self):
        """Test ID filtering with batch selector"""
        d, nb, nt, nq, nlist, nlevels, k = 128, 60000, 90000, 400, 256, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=654)

        index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=32)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=32)

        allowed_ids = np.array([i * 100 for i in range(500)], dtype=np.int64)
        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorBatch(allowed_ids)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        allowed_set = set(allowed_ids) | set([-1])
        for id_val in I_panorama.flatten():
            self.assertIn(int(id_val), allowed_set)

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_selector_with_small_dataset(self):
        """Test ID selectors with dataset smaller than batch size"""
        d, nb, nt, nq, nlist, nlevels, k = 32, 100, 200, 10, 4, 4, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=888)

        index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=nlist)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=nlist)

        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(20, 60)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        self.assertTrue(np.all(I_panorama >= 20))
        self.assertTrue(np.all(I_panorama < 60))

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_selector_excludes_all(self):
        """Test selector that excludes all results"""
        d, nb, nt, nq, nlist, nlevels, k = 32, 300, 400, 5, 8, 4, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=999)

        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=nlist)

        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(nb + 100, nb + 200)

        D, I = index_panorama.search(xq, k, params=params)
        self.assertTrue(np.all(I == -1))

    # Batch size and edge case tests

    def test_batch_boundaries(self):
        """Test correctness at various batch size boundaries (kBatchSize=256)"""
        d, nlist, nlevels, nt, nq, k = 128, 64, 8, 10000, 200, 15
        xt = np.random.rand(nt, d).astype("float32")
        xq = np.random.rand(nq, d).astype("float32")

        for nb in [5000, 10000, 20000, 50000, 75000]:
            with self.subTest(nb=nb):
                np.random.seed(987)
                xb = np.random.rand(nb, d).astype("float32")

                index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=16)
                index_panorama = self.create_panorama(
                    d, nlist, nlevels, xt, xb, nprobe=16
                )

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_empty_result_handling(self):
        """Test handling of empty search results"""
        d, nb, nt, nq, nlist, nlevels, k = 32, 100, 200, 5, 8, 4, 10
        xt, xb, _ = self.generate_data(d, nt, nb, nq, seed=111)
        xq = np.random.rand(nq, d).astype("float32") + 10.0  # Queries far from database

        index = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=1)
        D, I = index.search(xq, k)

        self.assertEqual(D.shape, (nq, k))
        self.assertEqual(I.shape, (nq, k))

    def test_very_small_dataset(self):
        """Test with dataset smaller than batch size (< 128 vectors)"""
        test_cases = [10, 50, 100]

        for nb in test_cases:
            with self.subTest(nb=nb):
                d, nt, nlist, nlevels, nq = 32, max(nb, 100), 4, 4, 5
                k = min(3, nb)
                xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=666 + nb)

                index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=nlist)
                index_panorama = self.create_panorama(
                    d, nlist, nlevels, xt, xb, nprobe=nlist
                )

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_single_vector_per_cluster(self):
        """Test extreme case where clusters have very few vectors"""
        d, nb, nt, nq, nlist, nlevels, k = 32, 20, 100, 5, 16, 4, 3
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1313)

        quantizer1, quantizer2 = faiss.IndexFlatL2(d), faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)

        index_regular.cp.min_points_per_centroid = 1
        index_panorama.cp.min_points_per_centroid = 1

        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = nlist

        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = nlist

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    def test_range_search_edge_cases(self):
        """Test range search with extreme radius values"""
        d, nb, nt, nq, nlist, nlevels = 32, 500, 700, 10, 8, 4
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=777)

        index_regular = self.create_ivf_flat(d, nlist, xt, xb, nprobe=nlist)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=nlist)

        for radius in [0.01, 0.1, 100.0, 1000.0]:
            with self.subTest(radius=radius):
                lims_reg, D_reg, I_reg = index_regular.range_search(xq, radius)
                lims_pan, D_pan, I_pan = index_panorama.range_search(xq, radius)

                np.testing.assert_array_equal(lims_reg, lims_pan)

                if len(I_reg) > 0:
                    self.assert_range_results_equal(
                        lims_reg, D_reg, I_reg, lims_pan, D_pan, I_pan, nq
                    )

    # Dynamic operations tests

    def test_incremental_add(self):
        """Test adding vectors incrementally in multiple batches"""
        d, nt, nlist, nlevels, k = 128, 20000, 256, 8, 15
        xt = np.random.rand(nt, d).astype("float32")

        index_regular = self.create_ivf_flat(d, nlist, xt, nprobe=32)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt, nprobe=32)

        batch_sizes = [10000, 15000, 20000, 5000]
        for batch_size in batch_sizes:
            xb_batch = np.random.rand(batch_size, d).astype("float32")
            index_regular.add(xb_batch)
            index_panorama.add(xb_batch)

        nq = 300
        xq = np.random.rand(nq, d).astype("float32")

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    def test_add_search_add_search(self):
        """Test interleaved add and search operations"""
        d, nt, nlist, nlevels, k = 32, 500, 8, 4, 5
        np.random.seed(555)
        xt = np.random.rand(nt, d).astype("float32")

        index_regular = self.create_ivf_flat(d, nlist, xt)
        index_panorama = self.create_panorama(d, nlist, nlevels, xt)

        # First add and search
        xb1 = np.random.rand(200, d).astype("float32")
        index_regular.add(xb1)
        index_panorama.add(xb1)

        xq1 = np.random.rand(10, d).astype("float32")
        index_regular.nprobe = 4
        index_panorama.nprobe = 4

        D_reg_1, I_reg_1 = index_regular.search(xq1, k)
        D_pan_1, I_pan_1 = index_panorama.search(xq1, k)
        self.assert_search_results_equal(D_reg_1, I_reg_1, D_pan_1, I_pan_1)

        # Second add and search
        xb2 = np.random.rand(300, d).astype("float32")
        index_regular.add(xb2)
        index_panorama.add(xb2)

        xq2 = np.random.rand(10, d).astype("float32")
        D_reg_2, I_reg_2 = index_regular.search(xq2, k)
        D_pan_2, I_pan_2 = index_panorama.search(xq2, k)
        self.assert_search_results_equal(D_reg_2, I_reg_2, D_pan_2, I_pan_2)

    def test_update_vectors(self):
        """Test update operations (single, batch, and interleaved with search)"""
        d, nb, nt, nq, nlist, nlevels, k = 128, 40000, 60000, 400, 256, 8, 15
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1414)

        index_regular = self.create_ivf_flat(
            d, nlist, xt, xb, nprobe=32, make_direct_map=True
        )
        index_panorama = self.create_panorama(
            d, nlist, nlevels, xt, xb, nprobe=32, make_direct_map=True
        )

        # Test batch update
        update_ids = np.array([100, 1000, 5000, 10000, 20000, 30000], dtype=np.int64)
        xb_new = np.random.rand(len(update_ids), d).astype("float32")
        index_regular.update_vectors(update_ids, xb_new)
        index_panorama.update_vectors(update_ids, xb_new)

        D_reg, I_reg = index_regular.search(xq, k)
        D_pan, I_pan = index_panorama.search(xq, k)
        self.assert_search_results_equal(D_reg, I_reg, D_pan, I_pan)

        # Test interleaved update/search
        update_ids_2 = np.array([50, 500, 2500, 15000], dtype=np.int64)
        xb_new_2 = np.random.rand(len(update_ids_2), d).astype("float32")
        index_regular.update_vectors(update_ids_2, xb_new_2)
        index_panorama.update_vectors(update_ids_2, xb_new_2)

        D_reg_2, I_reg_2 = index_regular.search(xq, k)
        D_pan_2, I_pan_2 = index_panorama.search(xq, k)
        self.assert_search_results_equal(D_reg_2, I_reg_2, D_pan_2, I_pan_2)

    def test_serialization(self):
        """Test that writing and reading Panorama indexes preserves search results"""
        d, nb, nt, nq, nlist, nlevels, k = 128, 10000, 15000, 100, 128, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        index = self.create_panorama(d, nlist, nlevels, xt, xb, nprobe=32)

        D_before, I_before = index.search(xq, k)
        index_after = faiss.deserialize_index(faiss.serialize_index(index))
        D_after, I_after = index_after.search(xq, k)

        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_array_equal(D_before, D_after)

    def test_ratio_dims_scanned(self):
        """Test the correctness of the ratio of dimensions scanned"""
        d, nb, nq, nlist, k = 128, 500000, 1, 1, 1

        # Setup: All vectors in the dataset are [1, 1, ..., 1], except for
        # one, which is [0, 0, ..., 0]. The query is also [0, 0, ..., 0]. This
        # ensures that pruning happens as early as it can. As a result, the
        # ratio of dimensions scanned should be approximately (1 / nlevels).
        xb = np.ones((nb, d)).astype("float32")
        xb[-1] = 0
        xt = xb
        xq = np.zeros((nq, d)).astype("float32")

        index_base = self.create_ivf_flat(d, nlist, xt, xb, nprobe=1)
        D_base, I_base = index_base.search(xq, k)

        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        ratios = []
        nlevels_list = [1, 2, 4, 8, 16, 32]
        for nlevels in nlevels_list:
            with self.subTest(nlevels=nlevels):
                faiss.cvar.indexPanorama_stats.reset()
                index = self.create_panorama(
                    d, nlist, nlevels, xt, xb, nprobe=1
                )
                D, I = index.search(xq, k)
                self.assert_search_results_equal(D_base, I_base, D, I)

                ratios.append(
                    faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                )

        expected_ratios = [1 / nlevels for nlevels in nlevels_list]
        np.testing.assert_allclose(ratios, expected_ratios, atol=1e-3)

        faiss.omp_set_num_threads(nt)
