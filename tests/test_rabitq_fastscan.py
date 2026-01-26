# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss
from faiss.contrib import datasets


def compute_expected_code_size(d, nb_bits):
    """Helper: Compute expected code size based on formula."""
    ex_bits = nb_bits - 1
    # For 1-bit: use SignBitFactors (8 bytes) for non-IVF
    # For multi-bit: use SignBitFactorsWithError (12 bytes)
    base_size = (d + 7) // 8 + (8 if ex_bits == 0 else 12)
    if ex_bits > 0:
        # ex-bit codes + ExtraBitsFactors
        ex_size = (d * ex_bits + 7) // 8 + 8
        return base_size + ex_size
    return base_size


class TestRaBitQFastScan(unittest.TestCase):
    """Unified tests for IndexRaBitQFastScan and IndexIVFRaBitQFastScan."""

    NLIST = 16
    NPROBE = 4

    def _create_index(self, d, metric, use_ivf=False, nlist=None, bbs=32):
        """Create FastScan index (IVF or non-IVF)."""
        if use_ivf:
            nlist = nlist or self.NLIST
            quantizer = faiss.IndexFlat(d, metric)
            index = faiss.IndexIVFRaBitQFastScan(
                quantizer, d, nlist, metric, bbs
            )
            index.nprobe = self.NPROBE
        else:
            index = faiss.IndexRaBitQFastScan(d, metric, bbs, 1)
        return index

    def _create_baseline(self, d, metric, use_ivf=False, nlist=None):
        """Create baseline RaBitQ index (IVF or non-IVF)."""
        if use_ivf:
            nlist = nlist or self.NLIST
            quantizer = faiss.IndexFlat(d, metric)
            index = faiss.IndexIVFRaBitQ(quantizer, d, nlist, metric)
            index.nprobe = self.NPROBE
        else:
            index = faiss.IndexRaBitQ(d, metric)
        return index

    # ==================== Comparison Tests ====================

    def test_comparison_vs_baseline(self):
        """Test FastScan produces similar results to baseline RaBitQ."""
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)
        k = 10

        for use_ivf in [False, True]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(use_ivf=use_ivf, metric=metric):
                    index_flat = faiss.IndexFlat(ds.d, metric)
                    index_flat.add(ds.get_database())
                    _, I_gt = index_flat.search(ds.get_queries(), k)

                    # Baseline
                    index_base = self._create_baseline(ds.d, metric, use_ivf)
                    index_base.train(ds.get_train())
                    index_base.add(ds.get_database())
                    _, I_base = index_base.search(ds.get_queries(), k)

                    # FastScan
                    index_fs = self._create_index(ds.d, metric, use_ivf)
                    index_fs.train(ds.get_train())
                    index_fs.add(ds.get_database())
                    _, I_fs = index_fs.search(ds.get_queries(), k)

                    eval_base = faiss.eval_intersection(
                        I_base[:, :k], I_gt[:, :k]
                    ) / (ds.nq * k)
                    eval_fs = faiss.eval_intersection(
                        I_fs[:, :k], I_gt[:, :k]
                    ) / (ds.nq * k)

                    np.testing.assert_(abs(eval_base - eval_fs) < 0.05)

    # ==================== Encode/Decode Tests ====================

    def test_encode_decode_consistency(self):
        """Test encoding/decoding operations are consistent."""
        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                nlist = 32
                ds = datasets.SyntheticDataset(64, 1000, 1000, 0)
                test_vectors = ds.get_database()[:100]

                if use_ivf:
                    # IVF: use add + reconstruct_n
                    index_fs = self._create_index(
                        ds.d, faiss.METRIC_L2, True, nlist
                    )
                    index_fs.train(ds.get_train())
                    index_fs.add(test_vectors)
                    decoded_fs = index_fs.reconstruct_n(0, len(test_vectors))

                    index_base = self._create_baseline(
                        ds.d, faiss.METRIC_L2, True, nlist
                    )
                    index_base.train(ds.get_train())
                    index_base.add(test_vectors)
                    decoded_base = index_base.reconstruct_n(
                        0, len(test_vectors)
                    )
                else:
                    # Non-IVF: use sa_encode + sa_decode
                    index_fs = faiss.IndexRaBitQFastScan(
                        ds.d, faiss.METRIC_L2
                    )
                    index_fs.train(ds.get_train())

                    codes_fs = np.empty(
                        (len(test_vectors), index_fs.code_size), dtype=np.uint8
                    )
                    index_fs.compute_codes(
                        faiss.swig_ptr(codes_fs),
                        len(test_vectors),
                        faiss.swig_ptr(test_vectors)
                    )
                    decoded_fs = index_fs.sa_decode(codes_fs)

                    index_base = faiss.IndexRaBitQ(ds.d, faiss.METRIC_L2)
                    index_base.train(ds.get_train())
                    codes_base = index_base.sa_encode(test_vectors)
                    decoded_base = index_base.sa_decode(codes_base)

                # Compare reconstruction errors
                err_fs = np.mean(
                    np.sum((test_vectors - decoded_fs) ** 2, axis=1)
                )
                err_base = np.mean(
                    np.sum((test_vectors - decoded_base) ** 2, axis=1)
                )

                np.testing.assert_(abs(err_fs - err_base) < 0.01)

    # ==================== Serialization Tests ====================

    def test_serialization(self):
        """Test serialize/deserialize preserves search results."""
        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                ds = datasets.SyntheticDataset(64, 1000, 100, 20)

                index = self._create_index(ds.d, faiss.METRIC_L2, use_ivf)
                index.train(ds.get_train())
                index.add(ds.get_database())

                Dref, Iref = index.search(ds.get_queries(), 10)

                b = faiss.serialize_index(index)
                index2 = faiss.deserialize_index(b)
                if use_ivf:
                    index2.nprobe = self.NPROBE

                Dnew, Inew = index2.search(ds.get_queries(), 10)

                np.testing.assert_array_equal(Dref, Dnew)
                np.testing.assert_array_equal(Iref, Inew)

    # ==================== Memory Management Tests ====================

    def test_memory_management(self):
        """Test memory is managed correctly during chunked add."""
        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                ds = datasets.SyntheticDataset(128, 2000, 2000, 50)

                index = self._create_index(ds.d, faiss.METRIC_L2, use_ivf)
                index.train(ds.get_train())

                chunk_size = 500
                for i in range(0, ds.nb, chunk_size):
                    end_idx = min(i + chunk_size, ds.nb)
                    index.add(ds.get_database()[i:end_idx])

                np.testing.assert_equal(index.ntotal, ds.nb)

                _, I = index.search(ds.get_queries(), 5)
                np.testing.assert_equal(I.shape, (ds.nq, 5))

                I_gt = ds.get_groundtruth(5)
                recall = faiss.eval_intersection(I[:, :5], I_gt[:, :5])
                recall /= (ds.nq * 5)
                np.testing.assert_(recall > 0.1)

    # ==================== Thread Safety Tests ====================

    def test_thread_safety(self):
        """Test parallel operations work correctly via OpenMP."""
        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                nq = 300 if use_ivf else 500
                ds = datasets.SyntheticDataset(64, 2000, 2000, nq)

                index = self._create_index(ds.d, faiss.METRIC_L2, use_ivf)
                index.train(ds.get_train())
                index.add(ds.get_database())

                k = 10
                distances, labels = index.search(ds.get_queries(), k)

                np.testing.assert_equal(distances.shape, (ds.nq, k))
                np.testing.assert_equal(labels.shape, (ds.nq, k))
                np.testing.assert_(np.all(distances >= 0))
                np.testing.assert_(np.all(labels >= 0))
                np.testing.assert_(np.all(labels < ds.nb))

    # ==================== Factory Tests ====================

    def test_factory_construction(self):
        """Test factory construction for both IVF and non-IVF."""
        ds = datasets.SyntheticDataset(64, 500, 500, 20)
        nlist = 16

        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                factory_str = f"IVF{nlist},RaBitQfs" if use_ivf else "RaBitQfs"
                expected_type = (
                    faiss.IndexIVFRaBitQFastScan if use_ivf
                    else faiss.IndexRaBitQFastScan
                )

                index = faiss.index_factory(ds.d, factory_str)
                self.assertIsInstance(index, expected_type)

                if use_ivf:
                    index.nprobe = 4
                index.train(ds.get_train())
                index.add(ds.get_database())
                _, I = index.search(ds.get_queries(), 5)

                np.testing.assert_equal(I.shape, (ds.nq, 5))

        # Test with custom batch size
        for use_ivf in [False, True]:
            factory_str = (
                f"IVF{nlist},RaBitQfs_64" if use_ivf else "RaBitQfs_64"
            )
            index = faiss.index_factory(ds.d, factory_str)
            np.testing.assert_equal(index.bbs, 64)

    # ==================== Non-IVF Specific Tests ====================

    def test_query_quantization_bits(self):
        """Test different query quantization bit settings (non-IVF only)."""
        ds = datasets.SyntheticDataset(64, 2000, 2000, 50)
        k = 10

        index = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index.train(ds.get_train())
        index.add(ds.get_database())

        I_gt = ds.get_groundtruth(k)

        for qb in [4, 6, 8]:
            index.qb = qb
            _, I = index.search(ds.get_queries(), k)
            recall = faiss.eval_intersection(I[:, :k], I_gt[:, :k])
            recall /= ds.nq * k
            np.testing.assert_(recall > 0.4)

    def test_small_dataset(self):
        """Test on small dataset for basic functionality (non-IVF only)."""
        d, n, nq = 32, 100, 10

        rs = np.random.RandomState(123)
        xb = rs.rand(n, d).astype(np.float32)
        xq = rs.rand(nq, d).astype(np.float32)

        index = faiss.IndexRaBitQFastScan(d, faiss.METRIC_L2)
        index.train(xb)
        index.add(xb)

        k = 5
        D, I = index.search(xq, k)

        np.testing.assert_equal(D.shape, (nq, k))
        np.testing.assert_equal(I.shape, (nq, k))
        np.testing.assert_(np.all(I >= 0))
        np.testing.assert_(np.all(I < n))
        np.testing.assert_(np.all(D >= 0))

    def test_comparison_vs_pq_fastscan(self):
        """Compare RaBitQFastScan to PQFastScan (non-IVF only)."""
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)
        k = 10

        index_pq = faiss.IndexPQFastScan(ds.d, 16, 4, faiss.METRIC_L2)
        index_pq.train(ds.get_train())
        index_pq.add(ds.get_database())
        _, I_pq = index_pq.search(ds.get_queries(), k)

        index_rbq = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index_rbq.train(ds.get_train())
        index_rbq.add(ds.get_database())
        _, I_rbq = index_rbq.search(ds.get_queries(), k)

        I_gt = ds.get_groundtruth(k)
        eval_rbq = faiss.eval_intersection(I_rbq[:, :k], I_gt[:, :k])
        eval_rbq /= ds.nq * k

        np.testing.assert_(eval_rbq > 0.55)

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters (non-IVF only)."""
        with np.testing.assert_raises(Exception):
            faiss.IndexRaBitQFastScan(0, faiss.METRIC_L2)

        try:
            faiss.IndexRaBitQFastScan(64, faiss.METRIC_Lp)
            self.fail("Should have raised exception for invalid metric")
        except RuntimeError:
            pass

    # ==================== IVF Specific Tests ====================

    def test_nprobe_variations(self):
        """Test different nprobe values (IVF only)."""
        nlist = 32
        ds = datasets.SyntheticDataset(64, 1000, 1000, 50)
        k = 10
        I_gt = ds.get_groundtruth(k)

        for nprobe in [1, 4, 8, 16]:
            with self.subTest(nprobe=nprobe):
                # Baseline
                quantizer1 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
                index_base = faiss.IndexIVFRaBitQ(
                    quantizer1, ds.d, nlist, faiss.METRIC_L2
                )
                index_base.nprobe = nprobe
                index_base.train(ds.get_train())
                index_base.add(ds.get_database())

                params_base = faiss.IVFRaBitQSearchParameters()
                params_base.nprobe = nprobe
                params_base.qb = 8
                params_base.centered = False
                _, I_base = index_base.search(
                    ds.get_queries(), k, params=params_base
                )

                # FastScan
                quantizer2 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
                index_fs = faiss.IndexIVFRaBitQFastScan(
                    quantizer2, ds.d, nlist, faiss.METRIC_L2, 32
                )
                index_fs.qb = 8
                index_fs.centered = False
                index_fs.nprobe = nprobe
                index_fs.train(ds.get_train())
                index_fs.add(ds.get_database())

                params_fs = faiss.IVFSearchParameters()
                params_fs.nprobe = nprobe
                _, I_fs = index_fs.search(
                    ds.get_queries(), k, params=params_fs
                )

                eval_base = faiss.eval_intersection(
                    I_base[:, :k], I_gt[:, :k]
                ) / (ds.nq * k)
                eval_fs = faiss.eval_intersection(
                    I_fs[:, :k], I_gt[:, :k]
                ) / (ds.nq * k)

                np.testing.assert_(abs(eval_base - eval_fs) < 0.01)

    def _do_test_search_implementation(self, impl):
        """Helper to test a specific search implementation (IVF only)."""
        nlist, nprobe = 32, 8
        ds = datasets.SyntheticDataset(128, 2048, 2048, 100)
        k = 10

        I_gt = ds.get_groundtruth(k)

        # Baseline
        quantizer1 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_base = faiss.IndexIVFRaBitQ(
            quantizer1, ds.d, nlist, faiss.METRIC_L2
        )
        index_base.qb = 8
        index_base.nprobe = nprobe
        index_base.train(ds.get_train())
        index_base.add(ds.get_database())

        params_base = faiss.IVFRaBitQSearchParameters()
        params_base.nprobe = nprobe
        params_base.qb = 8
        params_base.centered = False
        _, I_base = index_base.search(ds.get_queries(), k, params=params_base)

        eval_base = faiss.eval_intersection(
            I_base[:, :k], I_gt[:, :k]
        ) / (ds.nq * k)

        # FastScan with specific implementation
        quantizer2 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_fs = faiss.IndexIVFRaBitQFastScan(
            quantizer2, ds.d, nlist, faiss.METRIC_L2, 32
        )
        index_fs.qb = 8
        index_fs.centered = False
        index_fs.nprobe = nprobe
        index_fs.implem = impl
        index_fs.train(ds.get_train())
        index_fs.add(ds.get_database())

        params_fs = faiss.IVFSearchParameters()
        params_fs.nprobe = nprobe
        _, I_fs = index_fs.search(ds.get_queries(), k, params=params_fs)

        eval_fs = faiss.eval_intersection(
            I_fs[:, :k], I_gt[:, :k]
        ) / (ds.nq * k)

        np.testing.assert_(abs(eval_base - eval_fs) < 0.05)

    def test_search_implem_10(self):
        self._do_test_search_implementation(impl=10)

    def test_search_implem_12(self):
        self._do_test_search_implementation(impl=12)

    def test_search_implem_14(self):
        self._do_test_search_implementation(impl=14)

    def test_search_with_parameters(self):
        """Test search_with_parameters code path (IVF only)."""
        nlist, nprobe, nq = 64, 8, 500
        ds = datasets.SyntheticDataset(128, 2048, 2048, nq)
        k = 10

        quantizer = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index = faiss.IndexIVFRaBitQFastScan(
            quantizer, ds.d, nlist, faiss.METRIC_L2, 32
        )
        index.qb = 8
        index.centered = False
        index.nprobe = nprobe
        index.train(ds.get_train())
        index.add(ds.get_database())

        params = faiss.IVFSearchParameters()
        params.nprobe = nprobe

        D, I = faiss.search_with_parameters(
            index, ds.get_queries(), k, params
        )

        self.assertEqual(D.shape, (nq, k))
        self.assertEqual(I.shape, (nq, k))
        self.assertGreater(np.sum(I >= 0), 0)

        I_gt = ds.get_groundtruth(k)
        recall = faiss.eval_intersection(I, I_gt) / (nq * k)
        self.assertGreater(recall, 0.4)


class TestMultiBitRaBitQFastScan(unittest.TestCase):
    """Consolidated tests for multi-bit RaBitQ FastScan.

    Tests IndexRaBitQFastScan and IndexIVFRaBitQFastScan for construction,
    basic operations, recall, serialization, and factory construction.
    """

    # ==================== Construction Tests ====================

    def test_valid_nb_bits_range(self):
        """Test that nb_bits 1-9 are valid for IndexRaBitQFastScan."""
        d = 128
        for nb_bits in range(1, 10):
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                index = faiss.IndexRaBitQFastScan(d, metric, 32, nb_bits)
                self.assertEqual(index.d, d)
                self.assertEqual(index.metric_type, metric)
                self.assertEqual(index.rabitq.nb_bits, nb_bits)

    def test_invalid_nb_bits(self):
        """Test that invalid nb_bits values raise errors."""
        d = 128
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQFastScan(d, faiss.METRIC_L2, 32, 0)
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQFastScan(d, faiss.METRIC_L2, 10, 32)

    def test_code_size_formula(self):
        """Test that code sizes match expected formula for all nb_bits."""
        d = 128
        for nb_bits in range(1, 10):
            index = faiss.IndexRaBitQFastScan(d, faiss.METRIC_L2, 32, nb_bits)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected_size)

    def test_ivf_construction(self):
        """Test IndexIVFRaBitQFastScan construction with valid/invalid nb_bits.
        """
        d, nlist = 128, 16
        # Valid nb_bits
        for nb_bits in [1, 2, 4, 8]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                quantizer = faiss.IndexFlat(d, metric)
                index = faiss.IndexIVFRaBitQFastScan(
                    quantizer, d, nlist, metric, 32, True, nb_bits
                )
                self.assertEqual(index.d, d)
                self.assertEqual(index.rabitq.nb_bits, nb_bits)
                expected = compute_expected_code_size(d, nb_bits)
                self.assertEqual(index.code_size, expected)

        # Invalid nb_bits
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
        with self.assertRaises(RuntimeError):
            faiss.IndexIVFRaBitQFastScan(
                quantizer, d, nlist, faiss.METRIC_L2, 32, True, 0
            )
        with self.assertRaises(RuntimeError):
            faiss.IndexIVFRaBitQFastScan(
                quantizer, d, nlist, faiss.METRIC_L2, 32, True, 10
            )

    # ==================== Basic Operations Tests ====================

    def test_basic_operations(self):
        """Test train/add/search pipeline for various configurations."""
        ds = datasets.SyntheticDataset(128, 300, 500, 20)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 4, 8]:
                for qb in [1, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index = faiss.IndexRaBitQFastScan(
                            ds.d, metric, 32, nb_bits
                        )
                        index.qb = max(1, qb)
                        index.train(ds.get_train())
                        index.add(ds.get_database())
                        D, I = index.search(ds.get_queries(), 10)

                        self.assertTrue(index.is_trained)
                        self.assertEqual(index.ntotal, ds.nb)
                        self.assertEqual(D.shape, (ds.nq, 10))
                        self.assertTrue(np.all(I >= 0))
                        self.assertTrue(np.all(np.isfinite(D)))

    def test_ivf_basic_operations(self):
        """Test IVF train/add/search pipeline."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 1000, 1000, 20, metric=metric_str
            )
            for nb_bits in [2, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    quantizer = faiss.IndexFlat(ds.d, metric)
                    index = faiss.IndexIVFRaBitQFastScan(
                        quantizer, ds.d, 16, metric, 32, True, nb_bits
                    )
                    index.nprobe = 4
                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    D, I = index.search(ds.get_queries(), 10)

                    self.assertTrue(index.is_trained)
                    self.assertEqual(index.ntotal, ds.nb)
                    self.assertEqual(D.shape, (ds.nq, 10))
                    self.assertTrue(np.all(I >= 0))

    # ==================== Recall Tests ====================

    def test_recall_monotonic_improvement(self):
        """Test that recall improves with more bits."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = 'L2' if metric == faiss.METRIC_L2 else 'IP'
            ds = datasets.SyntheticDataset(
                128, 500, 1000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for qb in [1, 4, 8]:
                with self.subTest(metric=metric, qb=qb):
                    recalls = {}
                    for nb_bits in [1, 2, 4, 8]:
                        index = faiss.IndexRaBitQFastScan(
                            ds.d, metric, 32, nb_bits
                        )
                        index.qb = qb
                        index.train(ds.get_train())
                        index.add(ds.get_database())
                        _, I = index.search(ds.get_queries(), 10)
                        recalls[nb_bits] = faiss.eval_intersection(
                            I, I_gt
                        ) / (ds.nq * 10)

                    tolerance = 0.03
                    self.assertGreaterEqual(recalls[2], recalls[1] - tolerance)
                    self.assertGreaterEqual(recalls[4], recalls[2] - tolerance)
                    self.assertGreaterEqual(recalls[8], recalls[4] - tolerance)
                    self.assertGreater(recalls[8], 0.75)

    def test_ivf_recall_improves_with_bits(self):
        """Test that recall improves monotonically with more bits for IVF."""
        d, nlist, nprobe = 128, 16, 8
        ds = datasets.SyntheticDataset(d, 1000, 1000, 50)
        I_gt = ds.get_groundtruth(10)

        recalls = {}
        for nb_bits in [1, 2, 4, 8]:
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
            index = faiss.IndexIVFRaBitQFastScan(
                quantizer, d, nlist, faiss.METRIC_L2, 32, True, nb_bits
            )
            index.nprobe = nprobe
            index.train(ds.get_train())
            index.add(ds.get_database())
            _, I = index.search(ds.get_queries(), 10)
            recalls[nb_bits] = faiss.eval_intersection(
                I, I_gt
            ) / (ds.nq * 10)

        self.assertGreater(recalls[2], recalls[1])
        self.assertGreater(recalls[4], recalls[2])
        self.assertGreater(recalls[8], recalls[4])

    def test_comparison_vs_rabitq(self):
        """Test multi-bit FastScan produces similar results to RaBitQ."""
        ds = datasets.SyntheticDataset(128, 500, 1000, 50)
        k = 10

        index_flat_l2 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_flat_l2.add(ds.get_database())
        _, I_f_l2 = index_flat_l2.search(ds.get_queries(), k)

        index_flat_ip = faiss.IndexFlat(ds.d, faiss.METRIC_INNER_PRODUCT)
        index_flat_ip.add(ds.get_database())
        _, I_f_ip = index_flat_ip.search(ds.get_queries(), k)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            I_f = I_f_l2 if metric == faiss.METRIC_L2 else I_f_ip
            for nb_bits in [2, 4, 8]:
                for qb in [1, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index_rbq = faiss.IndexRaBitQ(ds.d, metric, nb_bits)
                        index_rbq.qb = qb
                        index_rbq.train(ds.get_train())
                        index_rbq.add(ds.get_database())
                        _, I_rbq = index_rbq.search(ds.get_queries(), k)

                        index_fs = faiss.IndexRaBitQFastScan(
                            ds.d, metric, 32, nb_bits
                        )
                        index_fs.qb = qb
                        index_fs.train(ds.get_train())
                        index_fs.add(ds.get_database())
                        _, I_fs = index_fs.search(ds.get_queries(), k)

                        eval_rbq = faiss.eval_intersection(
                            I_rbq[:, :k], I_f[:, :k]
                        ) / (ds.nq * k)
                        eval_fs = faiss.eval_intersection(
                            I_fs[:, :k], I_f[:, :k]
                        ) / (ds.nq * k)

                        np.testing.assert_(abs(eval_rbq - eval_fs) < 0.05)

    def test_ivf_vs_ivfrabitq_equivalence(self):
        """Test that multi-bit IVF FastScan matches IndexIVFRaBitQ."""
        d, nlist, nprobe, k = 128, 16, 4, 10
        ds = datasets.SyntheticDataset(d, 1000, 1000, 50)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    quantizer1 = faiss.IndexFlat(d, metric)
                    index_ref = faiss.IndexIVFRaBitQ(
                        quantizer1, d, nlist, metric, True, nb_bits
                    )
                    index_ref.nprobe = nprobe
                    index_ref.train(ds.get_train())
                    index_ref.add(ds.get_database())
                    _, I_ref = index_ref.search(ds.get_queries(), k)

                    quantizer2 = faiss.IndexFlat(d, metric)
                    index_test = faiss.IndexIVFRaBitQFastScan(
                        quantizer2, d, nlist, metric, 32, True, nb_bits
                    )
                    index_test.nprobe = nprobe
                    index_test.train(ds.get_train())
                    index_test.add(ds.get_database())
                    _, I_test = index_test.search(ds.get_queries(), k)

                    index_flat = faiss.IndexFlat(d, metric)
                    index_flat.add(ds.get_database())
                    _, I_gt = index_flat.search(ds.get_queries(), k)

                    recall_ref = faiss.eval_intersection(
                        I_ref[:, :k], I_gt[:, :k]
                    ) / (ds.nq * k)
                    recall_test = faiss.eval_intersection(
                        I_test[:, :k], I_gt[:, :k]
                    ) / (ds.nq * k)

                    self.assertLess(abs(recall_ref - recall_test), 0.05)

    # ==================== Serialization Tests ====================

    def test_serialization(self):
        """Test serialize/deserialize preserves search results."""
        ds = datasets.SyntheticDataset(64, 150, 200, 10)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 4, 8, 9]:
                for qb in [1, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index1 = faiss.IndexRaBitQFastScan(
                            ds.d, metric, 32, nb_bits
                        )
                        index1.qb = qb
                        index1.train(ds.get_train())
                        index1.add(ds.get_database())
                        D1, I1 = index1.search(ds.get_queries(), 5)

                        index_bytes = faiss.serialize_index(index1)
                        index2 = faiss.deserialize_index(index_bytes)
                        index2.qb = qb
                        D2, I2 = index2.search(ds.get_queries(), 5)

                        self.assertEqual(index2.rabitq.nb_bits, nb_bits)
                        np.testing.assert_array_equal(I1, I2)
                        np.testing.assert_allclose(D1, D2, rtol=1e-5)

    def test_ivf_serialization(self):
        """Test IVF serialization preserves results."""
        ds = datasets.SyntheticDataset(64, 1000, 500, 20)
        nlist = 16

        for nb_bits in [2, 4, 8]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(nb_bits=nb_bits, metric=metric):
                    quantizer = faiss.IndexFlat(ds.d, metric)
                    index1 = faiss.IndexIVFRaBitQFastScan(
                        quantizer, ds.d, nlist, metric, 32, True, nb_bits
                    )
                    index1.nprobe = 4
                    index1.train(ds.get_train())
                    index1.add(ds.get_database())
                    D1, I1 = index1.search(ds.get_queries(), 10)

                    index_bytes = faiss.serialize_index(index1)
                    index2 = faiss.deserialize_index(index_bytes)
                    D2, I2 = index2.search(ds.get_queries(), 10)

                    self.assertEqual(index2.rabitq.nb_bits, nb_bits)
                    np.testing.assert_array_equal(I1, I2)
                    np.testing.assert_allclose(D1, D2, rtol=1e-5)

    # ==================== Reconstruction Tests ====================

    def test_ivf_reconstruction(self):
        """Test that reconstruct_n works for multi-bit IVF indices."""
        d, nlist = 64, 8
        ds = datasets.SyntheticDataset(d, 500, 100, 0)

        for nb_bits in [1, 2, 4, 8]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(nb_bits=nb_bits, metric=metric):
                    quantizer = faiss.IndexFlat(d, metric)
                    index = faiss.IndexIVFRaBitQFastScan(
                        quantizer, d, nlist, metric, 32, True, nb_bits
                    )
                    index.train(ds.get_train())
                    test_vectors = ds.get_database()
                    index.add(test_vectors)

                    reconstructed = index.reconstruct_n(0, len(test_vectors))
                    errors = np.sum(
                        (test_vectors - reconstructed) ** 2, axis=1
                    )
                    avg_error = np.mean(errors)

                    self.assertTrue(np.all(np.isfinite(reconstructed)))
                    self.assertLess(avg_error, 15.0)

    def test_encode_decode_roundtrip(self):
        """Test encode/decode round-trip produces consistent results."""
        d = 64

        for nb_bits in [1, 2, 4]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(nb_bits=nb_bits, metric=metric):
                    metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
                    ds = datasets.SyntheticDataset(
                        d, 500, 100, 10, metric=metric_str
                    )

                    index = faiss.IndexRaBitQFastScan(d, metric, 32, nb_bits)
                    index.train(ds.get_train())
                    index.add(ds.get_database())

                    xb = ds.get_database()
                    _, I = index.search(xb, 1)

                    self_retrieval_count = sum(
                        1 for i in range(len(xb)) if I[i, 0] == i
                    )
                    self_retrieval_rate = self_retrieval_count / len(xb)
                    self.assertGreater(self_retrieval_rate, 0.5)

    def test_ivf_encode_decode_roundtrip(self):
        """Test IVF encode/decode round-trip produces consistent results."""
        d, nlist = 64, 8

        for nb_bits in [1, 2, 4]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(nb_bits=nb_bits, metric=metric):
                    metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
                    ds = datasets.SyntheticDataset(
                        d, 500, 100, 10, metric=metric_str
                    )

                    quantizer = faiss.IndexFlat(d, metric)
                    index = faiss.IndexIVFRaBitQFastScan(
                        quantizer, d, nlist, metric, 32, True, nb_bits
                    )
                    index.nprobe = nlist
                    index.train(ds.get_train())
                    index.add(ds.get_database())

                    xb = ds.get_database()
                    _, I = index.search(xb, 1)

                    self_retrieval_count = sum(
                        1 for i in range(len(xb)) if I[i, 0] == i
                    )
                    self_retrieval_rate = self_retrieval_count / len(xb)
                    self.assertGreater(self_retrieval_rate, 0.5)

    def test_ivf_encoding_format_consistency(self):
        """Verify IVF FastScan encoding matches non-IVF FastScan pattern."""
        d, nb, nlist = 64, 50, 4

        np.random.seed(123)
        xb = np.random.randn(nb, d).astype(np.float32)
        xt = np.random.randn(500, d).astype(np.float32)

        for nb_bits in [1, 2, 4]:
            with self.subTest(nb_bits=nb_bits):
                index_flat = faiss.IndexRaBitQFastScan(
                    d, faiss.METRIC_L2, 32, nb_bits
                )
                index_flat.train(xt)
                index_flat.add(xb)

                quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
                index_ivf = faiss.IndexIVFRaBitQFastScan(
                    quantizer, d, nlist, faiss.METRIC_L2, 32, True, nb_bits
                )
                index_ivf.nprobe = nlist
                index_ivf.train(xt)
                index_ivf.add(xb)

                xq = xb[:10]
                _, I_ivf = index_ivf.search(xq, 5)

                for i in range(len(xq)):
                    self.assertIn(i, I_ivf[i])

    # ==================== Factory Tests ====================

    def test_factory_construction(self):
        """Test multi-bit RaBitQFastScan can be constructed via factory."""
        ds = datasets.SyntheticDataset(64, 150, 200, 10)

        for nb_bits in [2, 4, 8]:
            factory_str = f"RaBitQfs{nb_bits}"
            index = faiss.index_factory(ds.d, factory_str)
            self.assertIsInstance(index, faiss.IndexRaBitQFastScan)
            self.assertEqual(index.rabitq.nb_bits, nb_bits)

            index.train(ds.get_train())
            index.add(ds.get_database())
            D, I = index.search(ds.get_queries(), 5)
            self.assertEqual(D.shape, (ds.nq, 5))
            self.assertTrue(np.all(I >= 0))

    def test_factory_with_batch_size(self):
        """Test factory construction with both nb_bits and batch size."""
        ds = datasets.SyntheticDataset(64, 150, 200, 10)

        factory_str = "RaBitQfs4_64"
        index = faiss.index_factory(ds.d, factory_str)
        self.assertIsInstance(index, faiss.IndexRaBitQFastScan)
        self.assertEqual(index.rabitq.nb_bits, 4)
        self.assertEqual(index.bbs, 64)

        index.train(ds.get_train())
        index.add(ds.get_database())
        D, I = index.search(ds.get_queries(), 5)
        self.assertEqual(D.shape, (ds.nq, 5))

    def test_ivf_factory_construction(self):
        """Test that multi-bit IVF index can be constructed via factory."""
        nlist = 16
        ds = datasets.SyntheticDataset(64, 1000, 500, 20)

        for nb_bits in [2, 4, 8]:
            factory_str = f"IVF{nlist},RaBitQfs{nb_bits}"
            index = faiss.index_factory(ds.d, factory_str)
            self.assertIsInstance(index, faiss.IndexIVFRaBitQFastScan)
            self.assertEqual(index.rabitq.nb_bits, nb_bits)

            index.nprobe = 4
            index.train(ds.get_train())
            index.add(ds.get_database())
            D, I = index.search(ds.get_queries(), 10)
            self.assertEqual(D.shape, (ds.nq, 10))

    def test_ivf_factory_with_batch_size(self):
        """Test IVF factory construction with both nb_bits and batch size."""
        nlist = 16
        ds = datasets.SyntheticDataset(64, 1000, 200, 10)

        factory_str = f"IVF{nlist},RaBitQfs4_64"
        index = faiss.index_factory(ds.d, factory_str)
        self.assertIsInstance(index, faiss.IndexIVFRaBitQFastScan)
        self.assertEqual(index.rabitq.nb_bits, 4)
        self.assertEqual(index.bbs, 64)

        index.nprobe = 4
        index.train(ds.get_train())
        index.add(ds.get_database())
        D, I = index.search(ds.get_queries(), 5)
        self.assertEqual(D.shape, (ds.nq, 5))


class TestRaBitQStatsFastScan(unittest.TestCase):
    """Test RaBitQStats tracking for multi-bit two-stage search in FastScan."""

    INDEX_TYPES = [
        "IndexRaBitQFastScan",
        "IndexIVFRaBitQFastScan",
    ]

    @classmethod
    def setUpClass(cls):
        cls.stats_available = hasattr(faiss, 'cvar') and hasattr(
            faiss.cvar, 'rabitq_stats'
        )
        if cls.stats_available:
            cls.rabitq_stats = faiss.cvar.rabitq_stats

    def test_stats_reset_and_skip_percentage(self):
        """Test that stats can be reset and skip_percentage works."""
        if not self.stats_available:
            self.skipTest("rabitq_stats not available in Python bindings")
        self.rabitq_stats.reset()
        self.assertEqual(self.rabitq_stats.n_1bit_evaluations, 0)
        self.assertEqual(self.rabitq_stats.n_multibit_evaluations, 0)
        self.assertEqual(self.rabitq_stats.skip_percentage(), 0.0)

    def test_stats_collected_multibit_all_index_types(self):
        """Test stats are collected for all multi-bit FastScan index types."""
        if not self.stats_available:
            self.skipTest("rabitq_stats not available in Python bindings")
        ds = datasets.SyntheticDataset(384, 50000, 50000, 10)
        nlist = 16

        for index_type in self.INDEX_TYPES:
            for nb_bits in [2, 4]:
                with self.subTest(index_type=index_type, nb_bits=nb_bits):
                    self.rabitq_stats.reset()

                    if index_type == "IndexRaBitQFastScan":
                        index = faiss.IndexRaBitQFastScan(
                            ds.d, faiss.METRIC_L2, 32, nb_bits
                        )
                    elif index_type == "IndexIVFRaBitQFastScan":
                        quantizer = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
                        index = faiss.IndexIVFRaBitQFastScan(
                            quantizer, ds.d, nlist, faiss.METRIC_L2,
                            32, True, nb_bits
                        )
                        index.nprobe = 4
                    else:
                        raise ValueError(f"Unknown index type: {index_type}")

                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    index.search(ds.get_queries(), 10)

                    self.assertGreater(self.rabitq_stats.n_1bit_evaluations, 0)
                    self.assertGreater(self.rabitq_stats.n_multibit_evaluations, 0)
                    # For multi-bit, filtering should skip some candidates
                    self.assertLess(
                        self.rabitq_stats.n_multibit_evaluations, self.rabitq_stats.n_1bit_evaluations
                    )
                    skip_pct = self.rabitq_stats.skip_percentage()
                    self.assertGreater(skip_pct, 0.0)
                    self.assertLessEqual(skip_pct, 100.0)

                    print(
                        f"{index_type} nb_bits={nb_bits}: "
                        f"total={self.rabitq_stats.n_1bit_evaluations}, "
                        f"refined={self.rabitq_stats.n_multibit_evaluations}, "
                        f"skip={skip_pct:.1f}%"
                    )

if __name__ == "__main__":
    unittest.main()
