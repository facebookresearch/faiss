# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test comparing IndexRefineFlat to IndexRefinePanorama with full-corpus refinement.

This test constructs two refine indexes over the same IVF-PQ base:
- IndexRefineFlat
- IndexRefinePanorama (with n_levels=8)

We set k=10 and k_factor such that k_base = k * k_factor = nb = 500,000, and
set nprobe = nlist on the base indexes so the base stage covers all vectors.
Then we check that both refined results return identical ids and (nearly)
identical distances.
"""

import unittest
import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset


class TestIndexRefinePanorama(unittest.TestCase):

    # Helper methods for index creation and data generation

    def generate_data(self, d, nt, nb, nq, seed=42):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def create_flat(self, d, base_factory, xt=None, xb=None):
        """Create and initialize IndexRefineFlat."""
        base_index = faiss.index_factory(d, base_factory)
        index = faiss.IndexRefineFlat(base_index)
        if xt is not None:
            index.train(xt)
        if xb is not None:
            index.add(xb)
        return index

    def create_panorama(self, d, base_factory, nlevels, xt=None, xb=None):
        """Create and initialize IndexRefinePanorama."""
        base_index = faiss.index_factory(d, base_factory)
        refine_index = faiss.index_factory(d, f"FlatL2Panorama{nlevels}_1")
        index = faiss.IndexRefinePanorama(base_index, refine_index)
        if xt is not None:
            index.train(xt)
        if xb is not None:
            index.add(xb)
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

    def test_refine_panorama_correctness(self):
        d = 128
        nt = 1000
        nb = 5000
        nq = 100
        k = 10
        n_levels = 8

        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2025)

        # Build two identical refine indexes
        base_cfg = "Flat"
        index_flat = self.create_flat(d, base_cfg, xt=xt, xb=xb)
        index_pan = self.create_panorama(d, base_cfg, n_levels, xt=xt, xb=xb)

        for k_factor in [1, 8, 64, 256, 1024, 50000]:
            params = faiss.IndexRefineSearchParameters(k_factor=float(k_factor))

            D_flat, I_flat = index_flat.search(xq, k, params=params)
            D_pano, I_pano = index_pan.search(xq, k, params=params)

            # Ids must match exactly; distances can be allclose
            np.testing.assert_array_equal(I_flat, I_pano)
            np.testing.assert_allclose(D_flat, D_pano, rtol=1e-5)

    def test_different_n_levels(self):
        """Test correctness with various n_levels parameter values"""
        d, nb, nt, nq, k = 128, 25000, 40000, 10, 15
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        base_cfg = "Flat"
        index_base = self.create_flat(d, base_cfg, xt=xt, xb=xb)
        D_base, I_base = index_base.search(xq, k)

        params = faiss.IndexRefineSearchParameters(k_factor=1000)

        nt_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        prev_ratio_dims_scanned = float("inf")
        for nlevels in [1, 2, 4, 8, 16, 32]:
            with self.subTest(nlevels=nlevels):
                faiss.cvar.indexPanorama_stats.reset()
                index = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)
                D, I = index.search(xq, k, params=params)
                self.assert_search_results_equal(D_base, I_base, D, I)

                ratio_dims_scanned = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                self.assertLess(ratio_dims_scanned, prev_ratio_dims_scanned)
                prev_ratio_dims_scanned = ratio_dims_scanned

        faiss.omp_set_num_threads(nt_threads)

    def test_uneven_dimension_division(self):
        """Test when n_levels doesn't evenly divide dimension"""
        test_cases = [(65, 4), (63, 8), (100, 7)]

        for d, nlevels in test_cases:
            with self.subTest(d=d, nlevels=nlevels):
                nb, nt, nq, k = 5000, 7000, 10, 5
                xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

                base_cfg = "Flat"
                index_regular = self.create_flat(d, base_cfg, xt=xt, xb=xb)
                index_panorama = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_single_level(self):
        """Test edge case with n_levels=1"""
        d, nb, nt, nq, nlevels, k = 32, 500, 700, 10, 1, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=333)

        base_cfg = "Flat"
        index_regular = self.create_flat(d, base_cfg, xt=xt, xb=xb)
        index_panorama = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    def test_multiple_levels_small_dimension(self):
        """Test edge case: more levels than dimension naturally supports"""
        d, nb, nt, nq, nlevels, k = 16, 200, 300, 10, 8, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1212)

        base_cfg = "Flat"
        index_regular = self.create_flat(d, base_cfg, xt=xt, xb=xb)
        index_panorama = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    def test_id_selector_range(self):
        """Test ID filtering with range selector"""
        d, nb, nt, nq, nlevels, k = 128, 80000, 120000, 500, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        base_cfg = "Flat"
        index_regular = self.create_flat(d, base_cfg, xt=xt, xb=xb)
        index_panorama = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)

        base_params = faiss.SearchParameters()
        base_params.sel = faiss.IDSelectorRange(10000, 50000)
        params = faiss.IndexRefineSearchParameters(k_factor=1, base_index_params=base_params)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        self.assertTrue(np.all(I_panorama >= 10000))
        self.assertTrue(np.all(I_panorama < 50000))

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_id_selector_batch(self):
        """Test ID filtering with batch selector"""
        d, nb, nt, nq, nlevels, k = 128, 60000, 90000, 400, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=522)

        base_cfg = "Flat"
        index_regular = self.create_flat(d, base_cfg, xt=xt, xb=xb)
        index_panorama = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)

        allowed_ids = np.array([i * 100 for i in range(500)], dtype=np.int64)
        base_params = faiss.SearchParameters()
        base_params.sel = faiss.IDSelectorBatch(allowed_ids)
        params = faiss.IndexRefineSearchParameters(k_factor=1, base_index_params=base_params)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        allowed_set = set(allowed_ids)
        for id_val in I_panorama.flatten():
            self.assertIn(int(id_val), allowed_set)

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_selector_with_small_dataset(self):
        """Test ID selectors with dataset smaller than batch size"""
        d, nb, nt, nq, nlevels, k = 32, 100, 200, 10, 4, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=888)

        base_cfg = "Flat"
        index_regular = self.create_flat(d, base_cfg, xt=xt, xb=xb)
        index_panorama = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)

        base_params = faiss.SearchParameters()
        base_params.sel = faiss.IDSelectorRange(20, 60)
        params = faiss.IndexRefineSearchParameters(k_factor=1, base_index_params=base_params)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        self.assertTrue(np.all(I_panorama >= 20))
        self.assertTrue(np.all(I_panorama < 60))

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_empty_result_handling(self):
        """Test handling of empty search results (shapes only)"""
        d, nb, nt, nq, nlevels, k = 32, 100, 200, 10, 4, 10
        xt, xb, _ = self.generate_data(d, nt, nb, nq, seed=111)
        xq = np.random.rand(nq, d).astype("float32") + 10.0  # Queries far from database

        base_cfg = "Flat"
        index = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)
        D, I = index.search(xq, k)

        self.assertEqual(D.shape, (nq, k))
        self.assertEqual(I.shape, (nq, k))

    def test_incremental_add(self):
        """Test adding vectors incrementally in multiple batches"""
        d, nlevels, k = 128, 8, 15

        base_cfg = "Flat"
        index_regular = self.create_flat(d, base_cfg)
        index_panorama = self.create_panorama(d, base_cfg, nlevels)

        # Keep total nb under 100k
        batch_sizes = [5000, 10000, 15000, 20000]
        for batch_size in batch_sizes:
            xb_batch = np.random.rand(batch_size, d).astype("float32")
            index_regular.add(xb_batch)
            index_panorama.add(xb_batch)

        nq = 10
        xq = np.random.rand(nq, d).astype("float32")

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    def test_add_search_add_search(self):
        """Test interleaved add and search operations"""
        d, nlevels, k = 32, 4, 5
        np.random.seed(555)

        base_cfg = "Flat"
        index_regular = self.create_flat(d, base_cfg)
        index_panorama = self.create_panorama(d, base_cfg, nlevels)

        # First add and search
        xb1 = np.random.rand(200, d).astype("float32")
        index_regular.add(xb1)
        index_panorama.add(xb1)

        xq1 = np.random.rand(10, d).astype("float32")

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

    def test_serialization(self):
        """Test that writing and reading Panorama indexes preserves search results"""
        d, nb, nt, nq, nlevels, k = 128, 10000, 15000, 100, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        base_cfg = "Flat"
        index = self.create_panorama(d, base_cfg, nlevels, xt=xt, xb=xb)

        D_before, I_before = index.search(xq, k)
        index_after = faiss.deserialize_index(faiss.serialize_index(index))
        D_after, I_after = index_after.search(xq, k)

        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_array_equal(D_before, D_after)

    def test_ratio_dims_scanned(self):
        """Test the correctness of the ratio of dimensions scanned"""
        d, nb, nq, k = 128, 500000, 10, 1

        # Setup: All vectors in the dataset are [1, 1, ..., 1], except for
        # one, which is [0, 0, ..., 0]. The queries are [0, 0, ..., 0]. This
        # ensures that pruning happens as early as it can. As a result, the
        # ratio of dimensions scanned should be approximately (1 / nlevels).
        xb = np.ones((nb, d)).astype("float32")
        xb[-1] = 0
        xq = np.zeros((nq, d)).astype("float32")

        # Create base flat index for comparison
        index_base = faiss.IndexFlatL2(d)
        index_base.add(xb)
        D_base, I_base = index_base.search(xq, k)

        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        # Force k_base = nb
        k_factor = nb // k  # 500000 // 10 = 50000
        params = faiss.IndexRefineSearchParameters(k_factor=float(k_factor))

        ratios = []
        nlevels_list = [1, 2, 4, 8, 16, 32]
        for nlevels in nlevels_list:
            with self.subTest(nlevels=nlevels):
                refine_index = faiss.index_factory(d, f"FlatL2Panorama{nlevels}_1")
                refine_index.add(xb)
                index = faiss.IndexRefinePanorama(index_base, refine_index)

                faiss.cvar.indexPanorama_stats.reset()
                D, I = index.search(xq, k, params=params)
                self.assert_search_results_equal(D_base, I_base, D, I)

                ratios.append(faiss.cvar.indexPanorama_stats.ratio_dims_scanned)

        expected_ratios = [1 / nlevels for nlevels in nlevels_list]

        # Extra low toleance for point-wise Panorama refinement
        np.testing.assert_allclose(ratios, expected_ratios, atol=1e-5)

        faiss.omp_set_num_threads(nt)
