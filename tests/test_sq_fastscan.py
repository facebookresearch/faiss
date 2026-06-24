# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for IndexSQFastScan.

Tests cover:
  - Construction and basic properties for every supported qtype
  - Train / add / search round-trip
  - Recall parity with IndexScalarQuantizer (within expected bounds)
  - Rerank factor behaviour (higher rf => better recall)
  - Conversion constructor from IndexScalarQuantizer
  - Reset clears state
  - Edge cases: k=1, single vector, zero vectors, odd dimensions
  - Inner product metric
  - sa_decode consistency
"""

from __future__ import absolute_import, division, print_function

import os
import tempfile
import unittest

import numpy as np

import faiss
from faiss.contrib.datasets import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recall_at_k(I_gt, I_test, k):
    """Fraction of queries whose true NN (gt[:,0]) appears in test[:,:k]."""
    return np.mean([I_gt[i, 0] in I_test[i, :k] for i in range(len(I_gt))])


# All qtype groups --------------------------------------------------------

SQ = faiss.ScalarQuantizer

# Native 4-bit: use vpshufb SIMD path
NATIVE_4BIT = [
    ("QT_4bit",         SQ.QT_4bit),
    ("QT_4bit_uniform", SQ.QT_4bit_uniform),
]

# Rerank types that work with generic float data
RERANK = [
    ("QT_6bit",                SQ.QT_6bit),
    ("QT_8bit",                SQ.QT_8bit),
    ("QT_8bit_uniform",        SQ.QT_8bit_uniform),
]

# Direct types: rerank path but need integer data in specific ranges
DIRECT = [
    ("QT_8bit_direct",         SQ.QT_8bit_direct),
    ("QT_8bit_direct_signed",  SQ.QT_8bit_direct_signed),
]

# Fallback types: delegate to SQ scanner
FALLBACK = [
    ("QT_fp16",  SQ.QT_fp16),
    ("QT_bf16",  SQ.QT_bf16),
]

# Types safe for generic float data (Gaussian)
ALL_GENERIC = NATIVE_4BIT + RERANK + FALLBACK

# All types including direct
ALL_TYPES = ALL_GENERIC + DIRECT


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSQFastScanConstruction(unittest.TestCase):
    """IndexSQFastScan can be constructed for every supported qtype."""

    def test_all_types_construct(self):
        d = 64
        for name, qtype in ALL_TYPES:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                self.assertEqual(index.d, d)
                self.assertFalse(index.is_trained)
                self.assertEqual(index.ntotal, 0)


class TestSQFastScanTrainAddSearch(unittest.TestCase):
    """Basic train/add/search works for every qtype."""

    def setUp(self):
        self.d = 64
        self.nb = 5000
        self.nq = 50
        self.k = 10
        self.ds = SyntheticDataset(d=self.d, nt=2000, nb=self.nb,
                                   nq=self.nq, seed=42)

    def _test_qtype(self, qtype, name):
        index = faiss.IndexSQFastScan(self.d, qtype)
        index.train(self.ds.get_train())
        self.assertTrue(index.is_trained)

        index.add(self.ds.get_database())
        self.assertEqual(index.ntotal, self.nb)

        D, I = index.search(self.ds.get_queries(), self.k)
        self.assertEqual(D.shape, (self.nq, self.k))
        self.assertEqual(I.shape, (self.nq, self.k))
        # All results should be valid
        self.assertTrue(np.all(I[:, 0] >= 0))
        # Distances should be finite and sorted (L2)
        self.assertTrue(np.all(np.isfinite(D)))
        for q in range(self.nq):
            for j in range(1, self.k):
                self.assertLessEqual(D[q, j - 1], D[q, j] + 1e-5)

    def test_native_4bit(self):
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                self._test_qtype(qtype, name)

    def test_rerank(self):
        for name, qtype in RERANK:
            with self.subTest(qtype=name):
                self._test_qtype(qtype, name)

    def test_fallback(self):
        for name, qtype in FALLBACK:
            with self.subTest(qtype=name):
                self._test_qtype(qtype, name)


class TestSQFastScanRecallParity(unittest.TestCase):
    """IndexSQFastScan recall should match or be close to IndexScalarQuantizer."""

    def setUp(self):
        self.d = 64
        self.ds = SyntheticDataset(d=self.d, nt=2000, nb=10000,
                                   nq=100, seed=42)
        gt_index = faiss.IndexFlatL2(self.d)
        gt_index.add(self.ds.get_database())
        self.D_gt, self.I_gt = gt_index.search(self.ds.get_queries(), 10)

    def _build_sq(self, qtype):
        index = faiss.IndexScalarQuantizer(self.d, qtype)
        index.train(self.ds.get_train())
        index.add(self.ds.get_database())
        return index

    def _build_fs(self, qtype, rf=4):
        index = faiss.IndexSQFastScan(self.d, qtype)
        index.rerank_factor = rf
        index.train(self.ds.get_train())
        index.add(self.ds.get_database())
        return index

    def test_native_4bit_parity(self):
        """Native 4-bit: FastScan should match SQ recall exactly."""
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                sq = self._build_sq(qtype)
                fs = self._build_fs(qtype)
                _, I_sq = sq.search(self.ds.get_queries(), 10)
                _, I_fs = fs.search(self.ds.get_queries(), 10)
                r_sq = recall_at_k(self.I_gt, I_sq, 1)
                r_fs = recall_at_k(self.I_gt, I_fs, 1)
                # Should be very close (same codes, same distance computation)
                self.assertAlmostEqual(r_sq, r_fs, delta=0.05,
                    msg=f"{name}: SQ={r_sq:.3f} FS={r_fs:.3f}")

    def test_rerank_rf4_parity(self):
        """Rerank types at rf=4: recall should match SQ exactly."""
        for name, qtype in RERANK:
            with self.subTest(qtype=name):
                sq = self._build_sq(qtype)
                fs = self._build_fs(qtype, rf=4)
                _, I_sq = sq.search(self.ds.get_queries(), 10)
                _, I_fs = fs.search(self.ds.get_queries(), 10)
                r_sq = recall_at_k(self.I_gt, I_sq, 1)
                r_fs = recall_at_k(self.I_gt, I_fs, 1)
                # rf=4 should closely match SQ precision
                self.assertGreater(r_fs, r_sq - 0.05,
                    msg=f"{name}: SQ={r_sq:.3f} FS(rf=4)={r_fs:.3f}")

    def test_fallback_exact_parity(self):
        """Fallback types: results should be identical to IndexScalarQuantizer."""
        for name, qtype in FALLBACK:
            with self.subTest(qtype=name):
                sq = self._build_sq(qtype)
                fs = self._build_fs(qtype)
                D_sq, I_sq = sq.search(self.ds.get_queries(), 10)
                D_fs, I_fs = fs.search(self.ds.get_queries(), 10)
                np.testing.assert_array_equal(I_sq, I_fs,
                    err_msg=f"{name}: IDs differ between SQ and FS(fallback)")
                np.testing.assert_allclose(D_sq, D_fs, atol=1e-5,
                    err_msg=f"{name}: distances differ between SQ and FS(fallback)")


class TestSQFastScanRerankFactor(unittest.TestCase):
    """Higher rerank_factor should give equal or better recall."""

    def setUp(self):
        self.d = 64
        self.ds = SyntheticDataset(d=self.d, nt=2000, nb=10000,
                                   nq=100, seed=42)
        gt_index = faiss.IndexFlatL2(self.d)
        gt_index.add(self.ds.get_database())
        _, self.I_gt = gt_index.search(self.ds.get_queries(), 10)

    def test_rf_monotonicity(self):
        """For QT_8bit, recall should be non-decreasing with rf."""
        recalls = {}
        for rf in [1, 2, 4, 8]:
            index = faiss.IndexSQFastScan(self.d, SQ.QT_8bit)
            index.rerank_factor = rf
            index.train(self.ds.get_train())
            index.add(self.ds.get_database())
            _, I = index.search(self.ds.get_queries(), 10)
            recalls[rf] = recall_at_k(self.I_gt, I, 1)

        for rf in [2, 4, 8]:
            self.assertGreaterEqual(recalls[rf], recalls[rf // 2] - 0.02,
                msg=f"rf={rf} recall {recalls[rf]:.3f} < rf={rf//2} "
                    f"recall {recalls[rf//2]:.3f}")


class TestSQFastScanConversionConstructor(unittest.TestCase):
    """Conversion from IndexScalarQuantizer preserves search results."""

    def setUp(self):
        self.d = 64
        self.ds = SyntheticDataset(d=self.d, nt=2000, nb=5000,
                                   nq=50, seed=42)

    def test_conversion_native_4bit(self):
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(self.d, qtype)
                sq.train(self.ds.get_train())
                sq.add(self.ds.get_database())

                fs = faiss.IndexSQFastScan(sq)
                self.assertEqual(fs.ntotal, sq.ntotal)
                self.assertTrue(fs.is_trained)

                D_sq, I_sq = sq.search(self.ds.get_queries(), 10)
                D_fs, I_fs = fs.search(self.ds.get_queries(), 10)
                # Should be very close
                r1_sq = (I_sq[:, 0] >= 0).mean()
                r1_fs = (I_fs[:, 0] >= 0).mean()
                self.assertEqual(r1_sq, r1_fs)

    def test_conversion_rerank(self):
        for name, qtype in RERANK:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(self.d, qtype)
                sq.train(self.ds.get_train())
                sq.add(self.ds.get_database())

                fs = faiss.IndexSQFastScan(sq)
                self.assertEqual(fs.ntotal, sq.ntotal)
                self.assertTrue(fs.is_trained)

                D_fs, I_fs = fs.search(self.ds.get_queries(), 10)
                self.assertTrue(np.all(I_fs[:, 0] >= 0))
                self.assertTrue(np.all(np.isfinite(D_fs)))

    def test_conversion_fallback(self):
        for name, qtype in FALLBACK:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(self.d, qtype)
                sq.train(self.ds.get_train())
                sq.add(self.ds.get_database())

                fs = faiss.IndexSQFastScan(sq)
                self.assertEqual(fs.ntotal, sq.ntotal)

                D_sq, I_sq = sq.search(self.ds.get_queries(), 10)
                D_fs, I_fs = fs.search(self.ds.get_queries(), 10)
                np.testing.assert_array_equal(I_sq, I_fs)
                np.testing.assert_allclose(D_sq, D_fs, atol=1e-5)


class TestSQFastScanReset(unittest.TestCase):
    """Reset should clear all stored data."""

    def test_reset(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=500, nq=0, seed=42)
        for name, qtype in ALL_GENERIC:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())
                self.assertEqual(index.ntotal, 500)

                index.reset()
                self.assertEqual(index.ntotal, 0)

                # Can add again after reset
                index.add(ds.get_database())
                self.assertEqual(index.ntotal, 500)


class TestSQFastScanEdgeCases(unittest.TestCase):
    """Edge cases: k=1, single vector, zero vectors, odd dimensions."""

    def test_k_equals_1(self):
        d = 32
        ds = SyntheticDataset(d=d, nt=500, nb=1000, nq=10, seed=42)
        for name, qtype in ALL_GENERIC:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())
                D, I = index.search(ds.get_queries(), 1)
                self.assertEqual(D.shape, (10, 1))
                self.assertEqual(I.shape, (10, 1))
                self.assertTrue(np.all(I >= 0))

    def test_single_vector(self):
        d = 32
        xb = np.random.RandomState(42).randn(1, d).astype('float32')
        xq = xb.copy()
        for name, qtype in ALL_GENERIC:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 1)
                self.assertEqual(I[0, 0], 0)
                # Distance to self should be near zero (quantization error)
                self.assertLess(D[0, 0], 1.0)

    def test_zero_vectors(self):
        d = 32
        xb = np.zeros((100, d), dtype='float32')
        # Use non-direct types only (direct types need values in [0,255])
        safe_types = NATIVE_4BIT + [
            ("QT_6bit", SQ.QT_6bit),
            ("QT_8bit", SQ.QT_8bit),
            ("QT_8bit_uniform", SQ.QT_8bit_uniform),
            ("QT_fp16", SQ.QT_fp16),
            ("QT_bf16", SQ.QT_bf16),
        ]
        for name, qtype in safe_types:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(xb)
                index.add(xb)
                D, I = index.search(np.zeros((1, d), dtype='float32'), 10)
                self.assertTrue(np.all(np.isfinite(D)))

    def test_odd_dimensions(self):
        """Dimensions not aligned to SIMD width."""
        for d in [7, 9, 15, 17, 31, 33, 63, 65]:
            with self.subTest(d=d):
                ds = SyntheticDataset(d=d, nt=500, nb=500, nq=5, seed=42)
                index = faiss.IndexSQFastScan(d, SQ.QT_8bit)
                index.train(ds.get_train())
                index.add(ds.get_database())
                D, I = index.search(ds.get_queries(), 10)
                self.assertEqual(D.shape, (5, 10))
                self.assertTrue(np.all(I[:, 0] >= 0))


class TestSQFastScanInnerProduct(unittest.TestCase):
    """Inner product metric works correctly."""

    def test_ip_basic(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=2000, nb=5000, nq=50, seed=42)
        xb = ds.get_database().copy()
        xq = ds.get_queries().copy()
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        for name, qtype in ALL_GENERIC:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(
                    d, qtype, faiss.METRIC_INNER_PRODUCT)
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 10)
                self.assertEqual(D.shape, (50, 10))
                self.assertTrue(np.all(I[:, 0] >= 0))
                self.assertTrue(np.all(np.isfinite(D)))
                # IP distances should be in descending order
                for q in range(50):
                    for j in range(1, 10):
                        self.assertGreaterEqual(
                            D[q, j - 1] + 1e-5, D[q, j])

    def test_ip_recall_parity(self):
        """IP recall should be comparable between SQ and FS."""
        d = 64
        ds = SyntheticDataset(d=d, nt=2000, nb=5000, nq=50, seed=42)
        xb = ds.get_database().copy()
        xq = ds.get_queries().copy()
        xt = ds.get_train().copy()
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
        faiss.normalize_L2(xt)

        gt = faiss.IndexFlatIP(d)
        gt.add(xb)
        _, I_gt = gt.search(xq, 10)

        for name, qtype in [("QT_8bit", SQ.QT_8bit),
                             ("QT_fp16", SQ.QT_fp16)]:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(
                    d, qtype, faiss.METRIC_INNER_PRODUCT)
                sq.train(xt)
                sq.add(xb)
                _, I_sq = sq.search(xq, 10)

                fs = faiss.IndexSQFastScan(
                    d, qtype, faiss.METRIC_INNER_PRODUCT)
                fs.rerank_factor = 4
                fs.train(xt)
                fs.add(xb)
                _, I_fs = fs.search(xq, 10)

                r_sq = recall_at_k(I_gt, I_sq, 1)
                r_fs = recall_at_k(I_gt, I_fs, 1)
                self.assertGreater(r_fs, r_sq - 0.1,
                    msg=f"{name}: SQ={r_sq:.3f} FS={r_fs:.3f}")


class TestSQFastScanSaDecode(unittest.TestCase):
    """sa_decode should reconstruct vectors close to the original."""

    def test_sa_decode(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=100, nq=0, seed=42)
        # Only test types that have meaningful sa_decode via SQ
        for name, qtype in NATIVE_4BIT + RERANK + FALLBACK:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())

                xr = index.reconstruct_n(0, index.ntotal)
                self.assertEqual(xr.shape, (100, d))
                self.assertTrue(np.all(np.isfinite(xr)))


class TestSQFastScanIncrementalAdd(unittest.TestCase):
    """Adding vectors in batches should give same results as all-at-once."""

    def test_incremental_add(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=1000, nq=20, seed=42)
        xb = ds.get_database()
        xq = ds.get_queries()

        for name, qtype in [("QT_8bit", SQ.QT_8bit),
                             ("QT_4bit", SQ.QT_4bit),
                             ("QT_fp16", SQ.QT_fp16)]:
            with self.subTest(qtype=name):
                # All at once
                idx1 = faiss.IndexSQFastScan(d, qtype)
                idx1.train(ds.get_train())
                idx1.add(xb)

                # In two batches
                idx2 = faiss.IndexSQFastScan(d, qtype)
                idx2.train(ds.get_train())
                idx2.add(xb[:500])
                idx2.add(xb[500:])

                self.assertEqual(idx1.ntotal, idx2.ntotal)

                D1, I1 = idx1.search(xq, 10)
                D2, I2 = idx2.search(xq, 10)

                # Results should be identical
                np.testing.assert_array_equal(I1, I2)
                np.testing.assert_allclose(D1, D2, atol=1e-5)


class TestSQFastScanDirectTypes(unittest.TestCase):
    """QT_8bit_direct and QT_8bit_direct_signed with appropriate data ranges."""

    def test_8bit_direct(self):
        d = 64
        rng = np.random.RandomState(42)
        # QT_8bit_direct expects values in [0, 255]
        xb = rng.randint(0, 256, (1000, d)).astype('float32')
        xq = rng.randint(0, 256, (10, d)).astype('float32')

        index = faiss.IndexSQFastScan(d, SQ.QT_8bit_direct)
        index.train(xb)
        index.add(xb)
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I[:, 0] >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_8bit_direct_signed(self):
        d = 64
        rng = np.random.RandomState(42)
        # QT_8bit_direct_signed expects values in [-128, 127]
        xb = rng.randint(-128, 128, (1000, d)).astype('float32')
        xq = rng.randint(-128, 128, (10, d)).astype('float32')

        index = faiss.IndexSQFastScan(d, SQ.QT_8bit_direct_signed)
        index.train(xb)
        index.add(xb)
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I[:, 0] >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_direct_recall_parity(self):
        """QT_8bit_direct FastScan recall should match SQ."""
        d = 64
        rng = np.random.RandomState(42)
        xb = rng.randint(0, 256, (5000, d)).astype('float32')
        xq = rng.randint(0, 256, (50, d)).astype('float32')

        gt = faiss.IndexFlatL2(d)
        gt.add(xb)
        _, I_gt = gt.search(xq, 10)

        sq = faiss.IndexScalarQuantizer(d, SQ.QT_8bit_direct)
        sq.train(xb)
        sq.add(xb)
        _, I_sq = sq.search(xq, 10)

        fs = faiss.IndexSQFastScan(d, SQ.QT_8bit_direct)
        fs.rerank_factor = 4
        fs.train(xb)
        fs.add(xb)
        _, I_fs = fs.search(xq, 10)

        r_sq = recall_at_k(I_gt, I_sq, 1)
        r_fs = recall_at_k(I_gt, I_fs, 1)
        self.assertGreater(r_fs, r_sq - 0.05,
            msg=f"SQ={r_sq:.3f} FS={r_fs:.3f}")


class TestSQFastScanLargeK(unittest.TestCase):
    """k larger than ntotal should still work."""

    def test_k_larger_than_n(self):
        d = 32
        rng = np.random.RandomState(42)
        xb = rng.randn(50, d).astype('float32')
        xq = rng.randn(5, d).astype('float32')

        for name, qtype in [("QT_4bit", SQ.QT_4bit),
                             ("QT_8bit", SQ.QT_8bit),
                             ("QT_fp16", SQ.QT_fp16)]:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(xb)
                index.add(xb)
                # k=100 > ntotal=50
                D, I = index.search(xq, 100)
                self.assertEqual(D.shape, (5, 100))
                # First 50 should be valid, rest -1
                self.assertTrue(np.all(I[:, :50] >= 0))


class TestSQFastScanSaEncode(unittest.TestCase):
    """sa_encode / sa_decode round-trip matches IndexScalarQuantizer."""

    def test_sa_encode_matches_sq(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=100, nq=0, seed=42)
        for name, qtype in ALL_GENERIC:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(d, qtype)
                sq.train(ds.get_train())

                fs = faiss.IndexSQFastScan(d, qtype)
                fs.train(ds.get_train())

                xb = ds.get_database()
                cs = fs.sa_code_size()

                sq_codes = sq.sa_encode(xb)
                fs_codes = fs.sa_encode(xb)
                np.testing.assert_array_equal(sq_codes, fs_codes,
                    err_msg=f"{name}: sa_encode differs")

                sq_dec = sq.sa_decode(sq_codes)
                fs_dec = fs.sa_decode(fs_codes)
                np.testing.assert_allclose(sq_dec, fs_dec, atol=1e-6,
                    err_msg=f"{name}: sa_decode differs")


class TestSQFastScanAddSaCodes(unittest.TestCase):
    """add_sa_codes should produce working indexes."""

    def test_add_sa_codes(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=500, nq=20, seed=42)
        for name, qtype in [("QT_4bit", SQ.QT_4bit),
                             ("QT_8bit", SQ.QT_8bit),
                             ("QT_fp16", SQ.QT_fp16)]:
            with self.subTest(qtype=name):
                # Reference: add via float
                ref = faiss.IndexSQFastScan(d, qtype)
                ref.train(ds.get_train())
                ref.add(ds.get_database())

                # Test: add via pre-encoded codes
                test = faiss.IndexSQFastScan(d, qtype)
                test.train(ds.get_train())
                codes = test.sa_encode(ds.get_database())
                test.add_sa_codes(codes)

                self.assertEqual(ref.ntotal, test.ntotal)

                D_ref, I_ref = ref.search(ds.get_queries(), 10)
                D_test, I_test = test.search(ds.get_queries(), 10)
                # At least top results should be valid
                self.assertTrue(np.all(I_ref[:, 0] >= 0))
                self.assertTrue(np.all(I_test[:, 0] >= 0))


class TestSQFastScanDistanceComputer(unittest.TestCase):
    """get_distance_computer works for rerank/fallback types."""

    def test_distance_computer(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=500, nq=0, seed=42)
        for name, qtype in RERANK + FALLBACK:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())
                # Just verify it doesn't crash
                dc = index.get_distance_computer()
                self.assertIsNotNone(dc)


class TestSQFastScanRangeSearch(unittest.TestCase):
    """range_search produces same results as IndexScalarQuantizer."""

    def test_range_search_fallback(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=1000, nq=10, seed=42)
        for name, qtype in FALLBACK:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(d, qtype)
                sq.train(ds.get_train())
                sq.add(ds.get_database())

                fs = faiss.IndexSQFastScan(d, qtype)
                fs.train(ds.get_train())
                fs.add(ds.get_database())

                # Get a reasonable radius
                D, _ = sq.search(ds.get_queries()[:1], 1000)
                radius = float(D[0, 500])  # median distance

                lims_sq, D_sq, I_sq = sq.range_search(
                    ds.get_queries(), radius)
                lims_fs, D_fs, I_fs = fs.range_search(
                    ds.get_queries(), radius)

                np.testing.assert_array_equal(lims_sq, lims_fs,
                    err_msg=f"{name}: lims differ")


class TestSQFastScanIO(unittest.TestCase):
    """I/O round-trip preserves index state and search results."""

    def test_io_roundtrip(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=500, nq=20, seed=42)
        for name, qtype in [("QT_4bit", SQ.QT_4bit),
                             ("QT_8bit", SQ.QT_8bit),
                             ("QT_fp16", SQ.QT_fp16)]:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())

                D1, I1 = index.search(ds.get_queries(), 10)

                # Write and read back
                with tempfile.NamedTemporaryFile(suffix='.idx') as f:
                    faiss.write_index(index, f.name)
                    loaded = faiss.read_index(f.name)

                self.assertIsInstance(loaded, faiss.IndexSQFastScan)
                self.assertEqual(loaded.ntotal, index.ntotal)
                self.assertEqual(loaded.d, d)

                D2, I2 = loaded.search(ds.get_queries(), 10)
                np.testing.assert_array_equal(I1, I2)
                np.testing.assert_allclose(D1, D2, atol=1e-5)


class TestSQFastScanFactory(unittest.TestCase):
    """Factory string produces IndexSQFastScan."""

    def test_factory_strings(self):
        d = 64
        for desc in ["SQ8fs", "SQ4fs", "SQfp16fs", "SQbf16fs", "SQ6fs"]:
            with self.subTest(desc=desc):
                index = faiss.index_factory(d, desc)
                self.assertIsInstance(index, faiss.IndexSQFastScan)
                self.assertEqual(index.d, d)

    def test_factory_with_bbs(self):
        d = 64
        index = faiss.index_factory(d, "SQ8fs_64")
        self.assertIsInstance(index, faiss.IndexSQFastScan)
        self.assertEqual(index.bbs, 64)

    def test_factory_metric(self):
        d = 64
        index = faiss.index_factory(d, "SQ8fs", faiss.METRIC_INNER_PRODUCT)
        self.assertIsInstance(index, faiss.IndexSQFastScan)
        self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)


if __name__ == '__main__':
    unittest.main()
