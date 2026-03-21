# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for IndexIVFPQPanorama.

Panorama is an adaptation of IndexIVFPQ that uses level-oriented storage
and progressive filtering with Cauchy-Schwarz bounds to achieve significant
speedups when combined with PCA or Cayley transforms, with zero loss in
accuracy.

Paper: https://www.arxiv.org/pdf/2510.00566

Constraints specific to IndexIVFPQPanorama:
  - Only L2 metric is supported.
  - Only 8-bit PQ codes (nbits == 8).
  - M must be divisible by n_levels.
  - batch_size must be a multiple of 64.
  - use_precomputed_table must be 1.
"""

import unittest

import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset


class TestIndexIVFPQPanorama(unittest.TestCase):
    """Test Suite for IndexIVFPQPanorama."""

    # Helper methods for index creation and data generation

    def generate_data(self, d, nt, nb, nq, seed=42):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def create_ivfpq(self, d, nlist, M, nbits, xt, xb=None, nprobe=None):
        """Create and train a standard IndexIVFPQ (L2 only)."""
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        index.train(xt)
        if xb is not None:
            index.add(xb)
        if nprobe is not None:
            index.nprobe = nprobe
        return index

    def create_panorama(
        self, d, nlist, M, nbits, n_levels, xt, xb=None,
        nprobe=None, batch_size=128,
    ):
        """Create IndexIVFPQPanorama from a freshly trained IVFPQ.

        Trains a temporary IndexIVFPQ, copies PQ centroids and quantizer
        into the Panorama index, then sets up precomputed tables.
        """
        quantizer = faiss.IndexFlatL2(d)
        trained = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        trained.train(xt)

        trained.own_fields = False
        pano = faiss.IndexIVFPQPanorama(
            quantizer, d, nlist, M, nbits, n_levels, batch_size,
        )
        centroids = faiss.vector_to_array(trained.pq.centroids)
        faiss.copy_array_to_vector(centroids, pano.pq.centroids)
        pano.is_trained = True
        pano.use_precomputed_table = 1
        pano.precompute_table()

        if xb is not None:
            pano.add(xb)
        if nprobe is not None:
            pano.nprobe = nprobe
        return pano

    def create_pair(
        self, d, nlist, M, nbits, n_levels, xt, xb=None,
        nprobe=None, batch_size=128,
    ):
        """Create an IVFPQ and an IVFPQPanorama sharing the same training.

        Both indexes use the same quantizer centroids and PQ codebook,
        so search results should be identical.
        """
        quantizer = faiss.IndexFlatL2(d)
        trained = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        trained.train(xt)

        # Build the IVFPQ baseline from the trained state.
        ivfpq = faiss.clone_index(trained)

        # Build the Panorama from the same trained state.
        trained.own_fields = False
        pano = faiss.IndexIVFPQPanorama(
            quantizer, d, nlist, M, nbits, n_levels, batch_size,
        )
        centroids = faiss.vector_to_array(trained.pq.centroids)
        faiss.copy_array_to_vector(centroids, pano.pq.centroids)
        pano.is_trained = True
        pano.use_precomputed_table = 1
        pano.precompute_table()

        if xb is not None:
            ivfpq.add(xb)
            pano.add(xb)
        if nprobe is not None:
            ivfpq.nprobe = nprobe
            pano.nprobe = nprobe
        return ivfpq, pano

    def assert_search_results_equal(
        self,
        D_regular,
        I_regular,
        D_panorama,
        I_panorama,
        rtol=1e-4,
        atol=1e-6,
        otol=1e-3,
    ):
        overlap_rate = np.mean(I_regular == I_panorama)

        self.assertGreater(
            overlap_rate,
            1 - otol,
            f"Overlap rate {overlap_rate:.6f} is not > {1 - otol:.3f}. ",
        )
        np.testing.assert_allclose(
            D_regular,
            D_panorama,
            rtol=rtol,
            atol=atol,
            err_msg="Distances mismatch",
        )

    # Core functionality tests

    def test_exact_match_with_ivfpq(self):
        """Core test: Panorama must return identical results to IndexIVFPQ"""
        d, nb, nt, nq = 64, 50000, 60000, 500
        nlist, M, nbits, n_levels, k = 64, 16, 8, 4, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=42)

        for nprobe in [1, 4, 16, 64]:
            with self.subTest(nprobe=nprobe):
                ivfpq, pano = self.create_pair(
                    d, nlist, M, nbits, n_levels, xt, xb, nprobe,
                )
                D_regular, I_regular = ivfpq.search(xq, k)
                D_panorama, I_panorama = pano.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_exact_match_with_ivfpq_medium(self):
        """Core test: Medium scale version"""
        d, nb, nt, nq = 32, 10000, 15000, 200
        nlist, M, nbits, n_levels, k = 32, 8, 8, 4, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=42)

        for nprobe in [1, 4, 8, nlist]:
            with self.subTest(nprobe=nprobe):
                ivfpq, pano = self.create_pair(
                    d, nlist, M, nbits, n_levels, xt, xb, nprobe,
                )
                D_regular, I_regular = ivfpq.search(xq, k)
                D_panorama, I_panorama = pano.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    # Parameter variation tests

    def test_different_n_levels(self):
        """Test correctness with various n_levels parameter values"""
        d, nb, nt, nq = 64, 25000, 40000, 200
        nlist, M, nbits, k = 64, 16, 8, 15
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        # Train IVFPQ once for the baseline.
        ivfpq = self.create_ivfpq(d, nlist, M, nbits, xt, xb, nprobe=16)
        D_base, I_base = ivfpq.search(xq, k)

        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        prev_ratio = float("inf")
        # n_levels must divide M=16.
        for n_levels in [1, 2, 4, 8, 16]:
            with self.subTest(n_levels=n_levels):
                faiss.cvar.indexPanorama_stats.reset()

                pano = self.create_panorama(
                    d, nlist, M, nbits, n_levels, xt, xb, nprobe=16,
                )
                D, I = pano.search(xq, k)
                self.assert_search_results_equal(D_base, I_base, D, I)

                ratio = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                self.assertLess(ratio, prev_ratio)
                prev_ratio = ratio

        faiss.omp_set_num_threads(nt)

    def test_different_M_and_n_levels(self):
        """Test various M / n_levels combinations"""
        test_cases = [
            (32, 8, 2),   # M=8,  n_levels=2, chunk=4
            (64, 16, 4),  # M=16, n_levels=4, chunk=4
            (64, 32, 8),  # M=32, n_levels=8, chunk=4
        ]
        for d, M, n_levels in test_cases:
            with self.subTest(d=d, M=M, n_levels=n_levels):
                nb, nt, nq, nlist, nbits, k = 10000, 15000, 100, 32, 8, 10
                xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

                ivfpq, pano = self.create_pair(
                    d, nlist, M, nbits, n_levels, xt, xb, nprobe=8,
                )
                D_regular, I_regular = ivfpq.search(xq, k)
                D_panorama, I_panorama = pano.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_single_level(self):
        """Test edge case with n_levels=1 (no pruning, equivalent to IVFPQ)"""
        d, nb, nt, nq = 32, 5000, 7000, 50
        nlist, M, nbits, n_levels, k = 16, 8, 8, 1, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=333)

        ivfpq, pano = self.create_pair(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=4,
        )
        D_regular, I_regular = ivfpq.search(xq, k)
        D_panorama, I_panorama = pano.search(xq, k)

        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    def test_max_levels(self):
        """Test edge case with n_levels=M (each level is one subquantizer)"""
        d, nb, nt, nq = 64, 5000, 7000, 50
        nlist, M, nbits, n_levels, k = 16, 16, 8, 16, 5
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=444)

        ivfpq, pano = self.create_pair(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=4,
        )
        D_regular, I_regular = ivfpq.search(xq, k)
        D_panorama, I_panorama = pano.search(xq, k)

        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    # ID selector tests

    def test_id_selector_range(self):
        """Test ID filtering with range selector"""
        d, nb, nt, nq = 64, 50000, 60000, 300
        nlist, M, nbits, n_levels, k = 64, 16, 8, 4, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        ivfpq, pano = self.create_pair(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=16,
        )

        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(10000, 30000)

        D_regular, I_regular = ivfpq.search(xq, k, params=params)
        D_panorama, I_panorama = pano.search(xq, k, params=params)

        valid = I_panorama[I_panorama >= 0]
        self.assertTrue(np.all(valid >= 10000))
        self.assertTrue(np.all(valid < 30000))

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-4)

    def test_id_selector_batch(self):
        """Test ID filtering with batch selector"""
        d, nb, nt, nq = 64, 30000, 45000, 200
        nlist, M, nbits, n_levels, k = 64, 16, 8, 4, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=654)

        ivfpq, pano = self.create_pair(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=16,
        )

        allowed_ids = np.array([i * 50 for i in range(500)], dtype=np.int64)
        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorBatch(allowed_ids)

        D_regular, I_regular = ivfpq.search(xq, k, params=params)
        D_panorama, I_panorama = pano.search(xq, k, params=params)

        allowed_set = set(allowed_ids) | {-1}
        for id_val in I_panorama.flatten():
            self.assertIn(int(id_val), allowed_set)

        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-4)

    def test_selector_excludes_all(self):
        """Test selector that excludes all results"""
        d, nb, nt, nq = 32, 3000, 5000, 5
        nlist, M, nbits, n_levels, k = 8, 8, 8, 4, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=999)

        pano = self.create_panorama(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=nlist,
        )

        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(nb + 100, nb + 200)

        D, I = pano.search(xq, k, params=params)
        self.assertTrue(np.all(I == -1))

    # Batch size and edge case tests

    def test_batch_boundaries(self):
        """Test correctness at various database sizes relative to batch_size"""
        d, nq = 64, 50
        nlist, M, nbits, n_levels, k = 16, 16, 8, 4, 10
        xq = np.random.rand(nq, d).astype("float32")

        batch_size = 128
        test_sizes = [
            batch_size - 1,
            batch_size,
            batch_size + 1,
            batch_size * 2,
            batch_size * 3 - 1,
        ]
        for nb in test_sizes:
            with self.subTest(nb=nb):
                nt = max(nb, 500)
                np.random.seed(987)
                xt = np.random.rand(nt, d).astype("float32")
                xb = np.random.rand(nb, d).astype("float32")

                ivfpq, pano = self.create_pair(
                    d, nlist, M, nbits, n_levels, xt, xb,
                    nprobe=nlist, batch_size=batch_size,
                )
                D_regular, I_regular = ivfpq.search(xq, k)
                D_panorama, I_panorama = pano.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_different_batch_sizes(self):
        """Test correctness across different internal batch sizes"""
        d, nb, nt, nq = 64, 10000, 15000, 50
        nlist, M, nbits, n_levels, k = 32, 16, 8, 4, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=4242)

        ivfpq = self.create_ivfpq(d, nlist, M, nbits, xt, xb, nprobe=8)
        D_base, I_base = ivfpq.search(xq, k)

        for bs in [64, 128, 256, 512, 1024]:
            with self.subTest(batch_size=bs):
                pano = self.create_panorama(
                    d, nlist, M, nbits, n_levels, xt, xb,
                    nprobe=8, batch_size=bs,
                )
                D, I = pano.search(xq, k)
                self.assert_search_results_equal(D_base, I_base, D, I)

    def test_very_small_dataset(self):
        """Test with dataset smaller than batch size"""
        test_cases = [10, 50, 100]

        for nb in test_cases:
            with self.subTest(nb=nb):
                d, nlist, M, nbits, n_levels = 32, 4, 4, 8, 2
                nt, nq = max(nb, 1500), 5
                k = min(3, nb)
                xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=666 + nb)

                ivfpq, pano = self.create_pair(
                    d, nlist, M, nbits, n_levels, xt, xb, nprobe=nlist,
                )
                D_regular, I_regular = ivfpq.search(xq, k)
                D_panorama, I_panorama = pano.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_single_vector_per_cluster(self):
        """Test extreme case where clusters have very few vectors"""
        d, nb, nt, nq = 32, 20, 3000, 5
        nlist, M, nbits, n_levels, k = 16, 4, 8, 2, 3
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1313)

        ivfpq, pano = self.create_pair(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=nlist,
        )
        D_regular, I_regular = ivfpq.search(xq, k)
        D_panorama, I_panorama = pano.search(xq, k)

        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    def test_empty_result_handling(self):
        """Test handling of empty search results (shapes only)"""
        d, nb, nt, nq = 32, 100, 3000, 10
        nlist, M, nbits, n_levels, k = 8, 4, 8, 2, 10
        xt, xb, _ = self.generate_data(d, nt, nb, nq, seed=111)
        xq = np.random.rand(nq, d).astype("float32") + 10.0

        pano = self.create_panorama(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=1,
        )
        D, I = pano.search(xq, k)

        self.assertEqual(D.shape, (nq, k))
        self.assertEqual(I.shape, (nq, k))

    # Dynamic operations tests

    def test_incremental_add(self):
        """Test adding vectors incrementally in multiple batches"""
        d, nt = 64, 20000
        nlist, M, nbits, n_levels, k = 64, 16, 8, 4, 15
        xt = np.random.rand(nt, d).astype("float32")

        ivfpq, pano = self.create_pair(
            d, nlist, M, nbits, n_levels, xt, nprobe=16,
        )

        for batch_nb in [5000, 10000, 15000]:
            xb_batch = np.random.rand(batch_nb, d).astype("float32")
            ivfpq.add(xb_batch)
            pano.add(xb_batch)

        nq = 100
        xq = np.random.rand(nq, d).astype("float32")

        D_regular, I_regular = ivfpq.search(xq, k)
        D_panorama, I_panorama = pano.search(xq, k)

        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    def test_add_search_add_search(self):
        """Test interleaved add and search operations"""
        d, nt = 32, 500
        nlist, M, nbits, n_levels, k = 8, 8, 8, 4, 5
        np.random.seed(555)
        xt = np.random.rand(nt, d).astype("float32")

        ivfpq, pano = self.create_pair(
            d, nlist, M, nbits, n_levels, xt, nprobe=4,
        )

        xb1 = np.random.rand(200, d).astype("float32")
        ivfpq.add(xb1)
        pano.add(xb1)

        xq1 = np.random.rand(10, d).astype("float32")
        D_reg_1, I_reg_1 = ivfpq.search(xq1, k)
        D_pan_1, I_pan_1 = pano.search(xq1, k)
        self.assert_search_results_equal(D_reg_1, I_reg_1, D_pan_1, I_pan_1)

        xb2 = np.random.rand(300, d).astype("float32")
        ivfpq.add(xb2)
        pano.add(xb2)

        xq2 = np.random.rand(10, d).astype("float32")
        D_reg_2, I_reg_2 = ivfpq.search(xq2, k)
        D_pan_2, I_pan_2 = pano.search(xq2, k)
        self.assert_search_results_equal(D_reg_2, I_reg_2, D_pan_2, I_pan_2)

    # Serialization tests

    def test_serialization(self):
        """Test write/read preserves search results"""
        d, nb, nt, nq = 64, 10000, 15000, 100
        nlist, M, nbits, n_levels, k = 32, 16, 8, 4, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        pano = self.create_panorama(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=8,
        )

        D_before, I_before = pano.search(xq, k)
        pano_after = faiss.deserialize_index(faiss.serialize_index(pano))
        D_after, I_after = pano_after.search(xq, k)

        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-5)

    def test_serialization_preserves_params(self):
        """Test serialization preserves n_levels and batch_size correctly"""
        d, nb, nt, nq = 64, 10000, 15000, 50
        nlist, M, nbits, n_levels, k = 32, 16, 8, 4, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2025)

        pano = self.create_panorama(
            d, nlist, M, nbits, n_levels, xt, xb, nprobe=4,
        )
        D_before, I_before = pano.search(xq, k)

        pano_after = faiss.deserialize_index(
            faiss.serialize_index(pano)
        )
        self.assertEqual(pano_after.batch_size, 128)
        self.assertEqual(pano_after.n_levels, n_levels)

        D_after, I_after = pano_after.search(xq, k)
        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-5)

    # Statistics tests

    def test_ratio_dims_scanned(self):
        """Test that ratio_dims_scanned is 1.0 at n_levels=1 and strictly
        less for higher n_levels.

        Unlike IndexFlatPanorama, PQ quantization error prevents achieving
        the ideal 1/n_levels ratio even on synthetic data. We verify that
        n_levels=1 gives ratio=1.0 (exhaustive) and that multi-level
        pruning is effective (ratio well below 1.0).
        """
        d, nb, nt, nq = 64, 25000, 40000, 10
        nlist, M, nbits, k = 32, 16, 8, 1
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=5678)

        nt_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        faiss.cvar.indexPanorama_stats.reset()
        pano_1 = self.create_panorama(
            d, nlist, M, nbits, 1, xt, xb, nprobe=8,
        )
        pano_1.search(xq, k)
        ratio_1 = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
        self.assertAlmostEqual(ratio_1, 1.0, places=3)

        faiss.cvar.indexPanorama_stats.reset()
        pano_16 = self.create_panorama(
            d, nlist, M, nbits, 16, xt, xb, nprobe=8,
        )
        pano_16.search(xq, k)
        ratio_16 = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
        self.assertLess(ratio_16, 0.55)

        faiss.omp_set_num_threads(nt_threads)

    def test_pruning_improves_with_n_levels(self):
        """Test that increasing n_levels reduces the fraction scanned"""
        d, nb, nt, nq = 64, 25000, 40000, 50
        nlist, M, nbits, k = 32, 16, 8, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1234)

        nt_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        prev_ratio = float("inf")
        for n_levels in [1, 2, 4, 8, 16]:
            with self.subTest(n_levels=n_levels):
                faiss.cvar.indexPanorama_stats.reset()
                pano = self.create_panorama(
                    d, nlist, M, nbits, n_levels, xt, xb, nprobe=8,
                )
                pano.search(xq, k)
                ratio = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                self.assertLessEqual(ratio, prev_ratio)
                prev_ratio = ratio

        faiss.omp_set_num_threads(nt_threads)

    # Constraint validation tests

    def test_rejects_non_l2_metric(self):
        """Verify that non-L2 metrics are rejected"""
        d, nlist, M, nbits, n_levels = 32, 8, 8, 8, 4
        quantizer = faiss.IndexFlatIP(d)
        with self.assertRaises(RuntimeError):
            faiss.IndexIVFPQPanorama(
                quantizer, d, nlist, M, nbits, n_levels, 128,
                faiss.METRIC_INNER_PRODUCT,
            )

    def test_rejects_invalid_batch_size(self):
        """Verify that non-multiple-of-64 batch_size is rejected"""
        d, nlist, M, nbits, n_levels = 32, 8, 8, 8, 4
        quantizer = faiss.IndexFlatL2(d)
        with self.assertRaises(RuntimeError):
            faiss.IndexIVFPQPanorama(
                quantizer, d, nlist, M, nbits, n_levels, 100,
            )

    def test_rejects_m_not_divisible_by_n_levels(self):
        """Verify that M not divisible by n_levels is rejected"""
        d, nlist, M, nbits, n_levels = 32, 8, 8, 8, 3
        quantizer = faiss.IndexFlatL2(d)
        with self.assertRaises(RuntimeError):
            faiss.IndexIVFPQPanorama(
                quantizer, d, nlist, M, nbits, n_levels, 128,
            )
