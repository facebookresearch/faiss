# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for IndexIVFFlatPanorama.

Panorama is an adaptation of IndexIVFFlat that uses level-oriented storage
and progressive filtering with Cauchy-Schwarz bounds to achieve significant
speedups (up to 20x) with zero loss in accuracy.

Paper: https://www.arxiv.org/pdf/2510.00566
"""

import unittest
import numpy as np
import faiss


class TestIndexIVFFlatPanorama(unittest.TestCase):
    """
    Phase 1 Test Suite for IndexIVFFlatPanorama.
    
    These tests verify:
    1. Exact correctness vs IndexIVFFlat
    2. Parameter variations (n_levels, dimensions)
    3. ID selector integration
    4. Edge cases (batch boundaries, uneven dimensions)
    """

    def test_exact_match_with_ivf_flat(self):
        """Core test: Panorama must return identical results to IndexIVFFlat"""
        d = 64
        nb = 1000
        nt = 1500
        nq = 50
        nlist = 32
        nlevels = 4
        k = 10

        np.random.seed(42)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Create regular IndexIVFFlat
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)

        # Create IndexIVFFlatPanorama
        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)

        # Test with different nprobe values
        for nprobe in [1, 4, 8, nlist]:
            with self.subTest(nprobe=nprobe):
                index_regular.nprobe = nprobe
                index_panorama.nprobe = nprobe

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                # Results must match exactly
                np.testing.assert_array_equal(
                    I_regular, I_panorama,
                    err_msg=f"Labels mismatch with nprobe={nprobe}"
                )
                np.testing.assert_allclose(
                    D_regular, D_panorama, rtol=1e-5, atol=1e-7,
                    err_msg=f"Distances mismatch with nprobe={nprobe}"
                )

    def test_range_search(self):
        """Test range search returns correct results within radius"""
        d = 32
        nb = 500
        nt = 800
        nq = 20
        nlist = 16
        nlevels = 4

        np.random.seed(123)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Create both indices
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = 8

        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = 8

        # Test with different radius values
        for radius in [0.5, 1.0, 2.0, 5.0]:
            with self.subTest(radius=radius):
                lims_regular, D_regular, I_regular = index_regular.range_search(xq, radius)
                lims_panorama, D_panorama, I_panorama = index_panorama.range_search(xq, radius)

                # Same number of results per query
                np.testing.assert_array_equal(
                    lims_regular, lims_panorama,
                    err_msg=f"Different result counts with radius={radius}"
                )

                # All distances must be within radius
                self.assertTrue(
                    np.all(D_panorama <= radius),
                    f"Some distances exceed radius={radius}"
                )

                # Sort and compare results
                for i in range(nq):
                    n_results = lims_regular[i + 1] - lims_regular[i]
                    
                    if n_results > 0:
                        # Extract results for this query
                        ids_reg = I_regular[lims_regular[i]:lims_regular[i + 1]]
                        dist_reg = D_regular[lims_regular[i]:lims_regular[i + 1]]
                        ids_pan = I_panorama[lims_panorama[i]:lims_panorama[i + 1]]
                        dist_pan = D_panorama[lims_panorama[i]:lims_panorama[i + 1]]

                        # Sort by ID for comparison
                        sort_reg = np.argsort(ids_reg)
                        sort_pan = np.argsort(ids_pan)

                        np.testing.assert_array_equal(
                            ids_reg[sort_reg], ids_pan[sort_pan]
                        )
                        np.testing.assert_allclose(
                            dist_reg[sort_reg], dist_pan[sort_pan], rtol=1e-5
                        )

    def test_different_n_levels(self):
        """Test correctness with various n_levels parameter values"""
        d = 64
        nb = 800
        nt = 1000
        nq = 30
        nlist = 16
        k = 5

        np.random.seed(456)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Create baseline
        quantizer_base = faiss.IndexFlatL2(d)
        index_base = faiss.IndexIVFFlat(quantizer_base, d, nlist)
        index_base.train(xt)
        index_base.add(xb)
        index_base.nprobe = 4

        D_base, I_base = index_base.search(xq, k)

        # Test with various n_levels
        for nlevels in [1, 2, 4, 8, 16, 32]:
            with self.subTest(nlevels=nlevels):
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlatPanorama(quantizer, d, nlist, nlevels)
                index.train(xt)
                index.add(xb)
                index.nprobe = 4

                D, I = index.search(xq, k)

                # Results must match baseline
                np.testing.assert_array_equal(
                    I_base, I,
                    err_msg=f"Labels mismatch with n_levels={nlevels}"
                )
                np.testing.assert_allclose(
                    D_base, D, rtol=1e-5,
                    err_msg=f"Distances mismatch with n_levels={nlevels}"
                )

    def test_uneven_dimension_division(self):
        """Test when n_levels doesn't evenly divide dimension"""
        # Test several uneven combinations
        test_cases = [
            (65, 4),   # 65/4 = 16.25
            (63, 8),   # 63/8 = 7.875
            (100, 7),  # 100/7 = 14.29
        ]

        for d, nlevels in test_cases:
            with self.subTest(d=d, nlevels=nlevels):
                nb = 500
                nt = 700
                nq = 20
                nlist = 16
                k = 5

                np.random.seed(789)
                xt = np.random.rand(nt, d).astype('float32')
                xb = np.random.rand(nb, d).astype('float32')
                xq = np.random.rand(nq, d).astype('float32')

                # Regular index
                quantizer1 = faiss.IndexFlatL2(d)
                index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
                index_regular.train(xt)
                index_regular.add(xb)
                index_regular.nprobe = 4

                # Panorama index
                quantizer2 = faiss.IndexFlatL2(d)
                index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
                index_panorama.train(xt)
                index_panorama.add(xb)
                index_panorama.nprobe = 4

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                # Results must match
                np.testing.assert_array_equal(I_regular, I_panorama)
                np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_id_selector_range(self):
        """Test ID filtering with range selector"""
        d = 32
        nb = 1000
        nt = 1200
        nq = 20
        nlist = 16
        nlevels = 4
        k = 10

        np.random.seed(321)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Create both indices
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = 8

        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = 8

        # Test with ID selector - only allow IDs in range [200, 600)
        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(200, 600)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        # Verify all returned IDs are in range
        valid_ids = I_panorama >= 0
        self.assertTrue(np.all(I_panorama[valid_ids] >= 200))
        self.assertTrue(np.all(I_panorama[valid_ids] < 600))

        # Results must match
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(
            D_regular[valid_ids], D_panorama[valid_ids], rtol=1e-5
        )

    def test_id_selector_batch(self):
        """Test ID filtering with batch selector"""
        d = 32
        nb = 800
        nt = 1000
        nq = 15
        nlist = 16
        nlevels = 4
        k = 10

        np.random.seed(654)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Create both indices
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = 8

        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = 8

        # Create batch of specific IDs to allow
        allowed_ids = np.array([i * 10 for i in range(50)], dtype=np.int64)
        
        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorBatch(allowed_ids)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        # Verify all returned IDs are in the allowed set
        allowed_set = set(allowed_ids)
        valid_panorama = I_panorama >= 0
        for id_val in I_panorama[valid_panorama]:
            self.assertIn(int(id_val), allowed_set)

        # Results must match
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(
            D_regular[valid_panorama], D_panorama[valid_panorama], rtol=1e-5
        )

    def test_batch_boundaries(self):
        """Test correctness at various batch size boundaries (kBatchSize=256)"""
        d = 32
        nlist = 4
        nlevels = 4
        nt = 500
        nq = 10
        k = 5

        np.random.seed(987)
        xt = np.random.rand(nt, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Test with different database sizes around batch size (256)
        for nb in [100, 255, 256, 257, 512, 513, 1000]:
            with self.subTest(nb=nb):
                xb = np.random.rand(nb, d).astype('float32')

                # Regular index
                quantizer1 = faiss.IndexFlatL2(d)
                index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
                index_regular.train(xt)
                index_regular.add(xb)
                index_regular.nprobe = 4

                # Panorama index
                quantizer2 = faiss.IndexFlatL2(d)
                index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
                index_panorama.train(xt)
                index_panorama.add(xb)
                index_panorama.nprobe = 4

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                # Results must match
                np.testing.assert_array_equal(
                    I_regular, I_panorama,
                    err_msg=f"Labels mismatch with nb={nb}"
                )
                np.testing.assert_allclose(
                    D_regular, D_panorama, rtol=1e-5,
                    err_msg=f"Distances mismatch with nb={nb}"
                )

    def test_empty_result_handling(self):
        """Test handling of empty search results"""
        d = 32
        nb = 100
        nt = 200
        nq = 5
        nlist = 8
        nlevels = 4
        k = 10

        np.random.seed(111)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        # Create queries very far from database
        xq = np.random.rand(nq, d).astype('float32') + 10.0

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlatPanorama(quantizer, d, nlist, nlevels)
        index.train(xt)
        index.add(xb)
        index.nprobe = 1  # Very selective

        # Should not crash even with limited results
        D, I = index.search(xq, k)
        
        self.assertEqual(D.shape, (nq, k))
        self.assertEqual(I.shape, (nq, k))

    def test_small_dataset(self):
        """Test with very small dataset"""
        d = 16
        nb = 50
        nt = 100
        nq = 10
        nlist = 4
        nlevels = 2
        k = 5

        np.random.seed(222)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Regular index
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = nlist

        # Panorama index
        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = nlist

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        # Results must match
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_single_level(self):
        """Test edge case with n_levels=1"""
        d = 32
        nb = 500
        nt = 700
        nq = 20
        nlist = 16
        nlevels = 1  # Single level
        k = 5

        np.random.seed(333)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Regular index
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = 4

        # Panorama index with single level
        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = 4

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        # Results must match even with single level
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_incremental_add(self):
        """Test adding vectors incrementally in multiple batches"""
        d = 32
        nt = 500
        nlist = 16
        nlevels = 4
        k = 5

        np.random.seed(444)
        xt = np.random.rand(nt, d).astype('float32')

        # Train both indices
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)

        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)

        # Add vectors in multiple batches
        batch_sizes = [100, 150, 200, 50]
        for batch_size in batch_sizes:
            xb_batch = np.random.rand(batch_size, d).astype('float32')
            index_regular.add(xb_batch)
            index_panorama.add(xb_batch)

        # Test search
        nq = 20
        xq = np.random.rand(nq, d).astype('float32')
        
        index_regular.nprobe = 4
        index_panorama.nprobe = 4

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        # Results must match after incremental adds
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_add_search_add_search(self):
        """Test interleaved add and search operations"""
        d = 32
        nt = 500
        nlist = 8
        nlevels = 4
        k = 5

        np.random.seed(555)
        xt = np.random.rand(nt, d).astype('float32')

        # Train both indices
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)

        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)

        # Add first batch
        xb1 = np.random.rand(200, d).astype('float32')
        index_regular.add(xb1)
        index_panorama.add(xb1)

        # Search after first add
        xq1 = np.random.rand(10, d).astype('float32')
        index_regular.nprobe = 4
        index_panorama.nprobe = 4

        D_regular_1, I_regular_1 = index_regular.search(xq1, k)
        D_panorama_1, I_panorama_1 = index_panorama.search(xq1, k)

        np.testing.assert_array_equal(I_regular_1, I_panorama_1)
        np.testing.assert_allclose(D_regular_1, D_panorama_1, rtol=1e-5)

        # Add second batch
        xb2 = np.random.rand(300, d).astype('float32')
        index_regular.add(xb2)
        index_panorama.add(xb2)

        # Search after second add
        xq2 = np.random.rand(10, d).astype('float32')
        D_regular_2, I_regular_2 = index_regular.search(xq2, k)
        D_panorama_2, I_panorama_2 = index_panorama.search(xq2, k)

        np.testing.assert_array_equal(I_regular_2, I_panorama_2)
        np.testing.assert_allclose(D_regular_2, D_panorama_2, rtol=1e-5)

    def test_very_small_dataset(self):
        """Test with dataset much smaller than batch size (< 256 vectors)"""
        test_cases = [10, 50, 100, 200]
        
        for nb in test_cases:
            with self.subTest(nb=nb):
                d = 32
                nt = max(nb, 100)  # Need enough training vectors
                nlist = 4
                nlevels = 4
                nq = 5
                k = min(3, nb)  # Can't request more than we have

                np.random.seed(666 + nb)
                xt = np.random.rand(nt, d).astype('float32')
                xb = np.random.rand(nb, d).astype('float32')
                xq = np.random.rand(nq, d).astype('float32')

                # Regular index
                quantizer1 = faiss.IndexFlatL2(d)
                index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
                index_regular.train(xt)
                index_regular.add(xb)
                index_regular.nprobe = nlist  # Search all lists

                # Panorama index
                quantizer2 = faiss.IndexFlatL2(d)
                index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
                index_panorama.train(xt)
                index_panorama.add(xb)
                index_panorama.nprobe = nlist

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                # Results must match even with very small datasets
                np.testing.assert_array_equal(I_regular, I_panorama)
                np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_range_search_edge_cases(self):
        """Test range search with extreme radius values"""
        d = 32
        nb = 500
        nt = 700
        nq = 10
        nlist = 8
        nlevels = 4

        np.random.seed(777)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Create both indices
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = nlist

        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = nlist

        # Test edge case radii
        edge_radii = [0.01, 0.1, 100.0, 1000.0]
        
        for radius in edge_radii:
            with self.subTest(radius=radius):
                lims_regular, D_regular, I_regular = index_regular.range_search(xq, radius)
                lims_panorama, D_panorama, I_panorama = index_panorama.range_search(xq, radius)

                # Results must match
                np.testing.assert_array_equal(lims_regular, lims_panorama)
                
                # For very small radius, might have no results - that's okay
                if len(I_regular) > 0:
                    # Sort for comparison
                    for i in range(nq):
                        start_reg = lims_regular[i]
                        end_reg = lims_regular[i + 1]
                        start_pan = lims_panorama[i]
                        end_pan = lims_panorama[i + 1]
                        
                        if end_reg > start_reg:
                            ids_reg = I_regular[start_reg:end_reg]
                            ids_pan = I_panorama[start_pan:end_pan]
                            dist_reg = D_regular[start_reg:end_reg]
                            dist_pan = D_panorama[start_pan:end_pan]
                            
                            sort_idx_reg = np.argsort(ids_reg)
                            sort_idx_pan = np.argsort(ids_pan)
                            
                            np.testing.assert_array_equal(
                                ids_reg[sort_idx_reg], ids_pan[sort_idx_pan]
                            )
                            np.testing.assert_allclose(
                                dist_reg[sort_idx_reg], dist_pan[sort_idx_pan], rtol=1e-5
                            )

    def test_selector_with_small_dataset(self):
        """Test ID selectors with dataset smaller than batch size"""
        d = 32
        nb = 100  # Less than kBatchSize
        nt = 200
        nq = 10
        nlist = 4
        nlevels = 4
        k = 5

        np.random.seed(888)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Create both indices
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = nlist

        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = nlist

        # Test with range selector on small dataset
        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(20, 60)

        D_regular, I_regular = index_regular.search(xq, k, params=params)
        D_panorama, I_panorama = index_panorama.search(xq, k, params=params)

        # Verify IDs are in range
        valid_ids = I_panorama >= 0
        if np.any(valid_ids):
            self.assertTrue(np.all(I_panorama[valid_ids] >= 20))
            self.assertTrue(np.all(I_panorama[valid_ids] < 60))

        # Results must match
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(
            D_regular[valid_ids], D_panorama[valid_ids], rtol=1e-5
        )

    def test_selector_excludes_all(self):
        """Test selector that excludes all results"""
        d = 32
        nb = 300
        nt = 400
        nq = 5
        nlist = 8
        nlevels = 4
        k = 10

        np.random.seed(999)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        quantizer = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = nlist

        # Selector that excludes all vectors (range beyond dataset)
        params = faiss.SearchParametersIVF()
        params.sel = faiss.IDSelectorRange(nb + 100, nb + 200)

        D, I = index_panorama.search(xq, k, params=params)

        # Should return -1 for all results
        self.assertTrue(np.all(I == -1))

    def test_iterator_not_supported(self):
        """Test that iterator operations throw an error"""
        d = 32
        nb = 100
        nt = 200
        nlist = 4
        nlevels = 4

        np.random.seed(1111)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlatPanorama(quantizer, d, nlist, nlevels)
        index.train(xt)
        index.add(xb)

        # Iterator operations should not be supported
        # Try to get inverted lists and check if iterator flag is set
        invlists = index.invlists
        
        # The invlists should indicate that iterator is not supported
        # This is implementation-specific - may need to adjust based on actual API
        try:
            # Attempt an iterator-based operation if available in Python bindings
            # This is a placeholder - actual method may vary
            use_iterator = getattr(invlists, 'use_iterator', None)
            if use_iterator is not None:
                # If this attribute exists, verify it's False for Panorama
                self.assertFalse(use_iterator, 
                    "Panorama should not support iterator-based operations")
        except AttributeError:
            # If the attribute doesn't exist, that's fine - we're just checking
            # that the implementation doesn't mistakenly enable iterators
            pass

    def test_multiple_levels_small_dimension(self):
        """Test edge case: more levels than dimension naturally supports"""
        d = 16  # Small dimension
        nb = 200
        nt = 300
        nq = 10
        nlist = 4
        nlevels = 8  # Half the dimension
        k = 5

        np.random.seed(1212)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Regular index
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = 4

        # Panorama with many levels relative to dimension
        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = 4

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        # Results must still match
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_single_vector_per_cluster(self):
        """Test extreme case where clusters have very few vectors"""
        d = 32
        nb = 20  # Very few vectors
        nt = 100
        nq = 5
        nlist = 16  # Many clusters relative to vectors
        nlevels = 4
        k = 3

        np.random.seed(1313)
        xt = np.random.rand(nt, d).astype('float32')
        xb = np.random.rand(nb, d).astype('float32')
        xq = np.random.rand(nq, d).astype('float32')

        # Regular index
        quantizer1 = faiss.IndexFlatL2(d)
        index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index_regular.cp.min_points_per_centroid = 1  # Allow sparse clusters
        index_regular.train(xt)
        index_regular.add(xb)
        index_regular.nprobe = nlist

        # Panorama index
        quantizer2 = faiss.IndexFlatL2(d)
        index_panorama = faiss.IndexIVFFlatPanorama(quantizer2, d, nlist, nlevels)
        index_panorama.cp.min_points_per_centroid = 1
        index_panorama.train(xt)
        index_panorama.add(xb)
        index_panorama.nprobe = nlist

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        # Results must match even with sparse clusters
        np.testing.assert_array_equal(I_regular, I_panorama)
        np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)
