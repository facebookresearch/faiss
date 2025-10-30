"""
Comprehensive test suite for IndexFlatL2Panorama.

Panorama is an adaptation of IndexFlatL2 that uses level-oriented storage
and progressive filtering with Cauchy-Schwarz bounds to achieve significant
speedups (up to 45x when combined with PCA or Cayley transforms) with zero
loss in accuracy.

Paper: https://www.arxiv.org/pdf/2510.00566
"""

import unittest
import tempfile
import os

import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset


# TODO(aknayar): Add parallel tests and batchSize = 1 tests.
class TestIndexFlatL2Panorama(unittest.TestCase):
    """Test Suite for IndexFlatL2Panorama."""

    # Helper methods for index creation and data generation

    def generate_data(self, d, nt, nb, nq, seed=42):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def create_flat(self, d, xb=None):
        index = faiss.IndexFlatL2(d)
        if xb is not None:
            index.add(xb)
        return index

    def create_panorama(self, d, nlevels, xb=None):
        """Create and initialize IndexFlatL2Panorama."""
        index = faiss.IndexFlatL2Panorama(d, nlevels)
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
        """Core test: Panorama must return identical results to IndexFlatL2"""
        d, nb, nt, nq, nlevels, k = 128, 50000, 60000, 10, 8, 20
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=42)

        index_regular = self.create_flat(d, xb)
        index_panorama = self.create_panorama(d, nlevels, xb)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    def test_exact_match_with_ivf_flat_medium(self):
        """Core test: Medium scale version"""
        d, nb, nt, nq, nlevels, k = 64, 10000, 15000, 10, 4, 10
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=42)

        index_regular = self.create_flat(d, xb)
        index_panorama = self.create_panorama(d, nlevels, xb)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    @unittest.skip("Range search disabled for now")
    def test_range_search(self):
        """Test range search returns correct results within radius"""
        d, nb, nt, nq, nlevels = 128, 50000, 75000, 10, 8
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=123)

        index_regular = self.create_flat(d, xb)
        index_panorama = self.create_panorama(d, nlevels, xb)

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
        d, nb, nt, nq, k = 128, 25000, 40000, 10, 15
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        index_base = self.create_flat(d, xb)
        D_base, I_base = index_base.search(xq, k)

        prev_ratio_dims_scanned = float("inf")
        for nlevels in [1, 2, 4, 8, 16, 32]:
            with self.subTest(nlevels=nlevels):
                faiss.cvar.indexPanorama_stats.reset()
                index = self.create_panorama(d, nlevels, xb)
                D, I = index.search(xq, k)
                self.assert_search_results_equal(D_base, I_base, D, I)

                ratio_dims_scanned = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                self.assertLess(ratio_dims_scanned, prev_ratio_dims_scanned)
                prev_ratio_dims_scanned = ratio_dims_scanned

    # def test_uneven_dimension_division(self):
    #     """Test when n_levels doesn't evenly divide dimension"""
    #     test_cases = [(65, 4), (63, 8), (100, 7)]

    #     for d, nlevels in test_cases:
    #         with self.subTest(d=d, nlevels=nlevels):
    #             nb, nt, nq, k = 5000, 7000, 10, 5
    #             _, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

    #             index_regular = self.create_flat(d, xb)
    #             index_panorama = self.create_panorama(d, nlevels, xb)

    #             D_regular, I_regular = index_regular.search(xq, k)
    #             D_panorama, I_panorama = index_panorama.search(xq, k)

    #             self.assert_search_results_equal(
    #                 D_regular, I_regular, D_panorama, I_panorama
    #             )

    def test_single_level(self):
        """Test edge case with n_levels=1"""
        d, nb, nt, nq, nlevels, k = 32, 500, 700, 10, 1, 5
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=333)

        index_regular = self.create_flat(d, xb)
        index_panorama = self.create_panorama(d, nlevels, xb)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    def test_multiple_levels_small_dimension(self):
        """Test edge case: more levels than dimension naturally supports"""
        d, nb, nt, nq, nlevels, k = 16, 200, 300, 10, 8, 5
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=1212)

        index_regular = self.create_flat(d, xb)
        index_panorama = self.create_panorama(d, nlevels, xb)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(D_regular, I_regular, D_panorama, I_panorama)

    # ID selector tests (disabled)

    @unittest.skip("ID selector (range) disabled for now")
    def test_id_selector_range(self):
        pass

    @unittest.skip("ID selector (batch) disabled for now")
    def test_id_selector_batch(self):
        pass

    @unittest.skip("Selector with small dataset disabled for now")
    def test_selector_with_small_dataset(self):
        pass

    @unittest.skip("Selector excludes all disabled for now")
    def test_selector_excludes_all(self):
        pass

    # Batch size and edge case tests

    def test_batch_boundaries(self):
        """Test correctness at various batch size boundaries"""
        d, nt, nq, k = 128, 10000, 10, 15
        # random train not needed for Flat indices
        xq = np.random.rand(nq, d).astype("float32")

        # Use index's batch_size
        probe_index = self.create_panorama(d, nlevels=8)
        batch_size = getattr(probe_index, "batch_size")

        for nb in [batch_size - 1, batch_size, batch_size * 2, batch_size * 3, batch_size * 5 - 1]:
            with self.subTest(nb=nb):
                np.random.seed(987)
                xb = np.random.rand(nb, d).astype("float32")

                index_regular = self.create_flat(d, xb)
                index_panorama = self.create_panorama(d, nlevels=8, xb=xb)

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_empty_result_handling(self):
        """Test handling of empty search results (shapes only)"""
        d, nb, nt, nq, nlevels, k = 32, 100, 200, 10, 4, 10
        _, xb, _ = self.generate_data(d, nt, nb, nq, seed=111)
        xq = np.random.rand(nq, d).astype("float32") + 10.0  # Queries far from database

        index = self.create_panorama(d, nlevels, xb)
        D, I = index.search(xq, k)

        self.assertEqual(D.shape, (nq, k))
        self.assertEqual(I.shape, (nq, k))

    def test_very_small_dataset(self):
        """Test with dataset smaller than batch size (< 128 vectors)"""
        test_cases = [10, 50, 100]

        for nb in test_cases:
            with self.subTest(nb=nb):
                d, nt, nlevels, nq = 32, max(nb, 100), 4, 10
                k = min(3, nb)
                _, xb, xq = self.generate_data(d, nt, nb, nq, seed=666 + nb)

                index_regular = self.create_flat(d, xb)
                index_panorama = self.create_panorama(d, nlevels, xb)

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    @unittest.skip("Range search edge cases disabled for now")
    def test_range_search_edge_cases(self):
        pass

    # Dynamic operations tests

    def test_incremental_add(self):
        """Test adding vectors incrementally in multiple batches"""
        d, nlevels, k = 128, 8, 15

        index_regular = self.create_flat(d)
        index_panorama = self.create_panorama(d, nlevels)

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

        index_regular = self.create_flat(d)
        index_panorama = self.create_panorama(d, nlevels)

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

    @unittest.skip("update_vectors not supported for IndexFlatL2Panorama; disabled")
    def test_update_vectors(self):
        pass

    # def test_serialization(self):
    #     """Test that writing and reading Panorama indexes preserves search results"""
    #     d, nb, nt, nq, nlevels, k = 128, 10000, 15000, 100, 8, 20
    #     _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

    #     index = self.create_panorama(d, nlevels, xb)

    #     D_before, I_before = index.search(xq, k)

    #     fd, fname = tempfile.mkstemp()
    #     os.close(fd)
    #     try:
    #         faiss.write_index(index, fname)
    #         index_after = faiss.read_index(fname)
    #     finally:
    #         if os.path.exists(fname):
    #             os.unlink(fname)

    #     D_after, I_after = index_after.search(xq, k)

    #     np.testing.assert_array_equal(I_before, I_after)
    #     np.testing.assert_array_equal(D_before, D_after)

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

        index_base = self.create_flat(d, xb)
        D_base, I_base = index_base.search(xq, k)

        ratios = []
        nlevels_list = [1, 2, 4, 8, 16, 32]
        for nlevels in nlevels_list:
            with self.subTest(nlevels=nlevels):
                faiss.cvar.indexPanorama_stats.reset()
                index = self.create_panorama(d, nlevels, xb)
                D, I = index.search(xq, k)
                self.assert_search_results_equal(D_base, I_base, D, I)

                ratios.append(faiss.cvar.indexPanorama_stats.ratio_dims_scanned)

        expected_ratios = [1 / nlevels for nlevels in nlevels_list]
        np.testing.assert_allclose(ratios, expected_ratios, atol=1e-3)


if __name__ == "__main__":
    unittest.main()


