# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import, division, print_function

import math
import os
import tempfile
import unittest

import numpy as np

import faiss
from faiss.contrib.datasets import SyntheticDataset


def recall_at_k(I_gt, I_test, k):
    """Fraction of queries whose true NN (gt[:,0]) appears in test[:,:k]."""
    return np.mean([I_gt[i, 0] in I_test[i, :k] for i in range(len(I_gt))])


# Native 4-bit types (only types supported by IndexSQFastScan)
SQ = faiss.ScalarQuantizer

NATIVE_4BIT = [
    ("QT_4bit",         SQ.QT_4bit),
    ("QT_4bit_uniform", SQ.QT_4bit_uniform),
]


class TestSQFastScanConstruction(unittest.TestCase):
    """IndexSQFastScan can be constructed for native 4-bit types only."""

    def test_native_4bit_construct(self):
        d = 64
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                self.assertEqual(index.d, d)
                self.assertFalse(index.is_trained)
                self.assertEqual(index.ntotal, 0)

    def test_default_construct(self):
        index = faiss.IndexSQFastScan()
        self.assertEqual(index.d, 0)
        self.assertEqual(index.ntotal, 0)

    def test_rejects_non_4bit(self):
        d = 64
        with self.assertRaises(RuntimeError):
            faiss.IndexSQFastScan(d, SQ.QT_8bit)
        with self.assertRaises(RuntimeError):
            faiss.IndexSQFastScan(d, SQ.QT_6bit)
        with self.assertRaises(RuntimeError):
            faiss.IndexSQFastScan(d, SQ.QT_fp16)
        with self.assertRaises(RuntimeError):
            faiss.IndexSQFastScan(d, SQ.QT_bf16)


class TestSQFastScanTrainAddSearch(unittest.TestCase):
    """Basic train/add/search works for native 4-bit types."""

    def setUp(self):
        self.d = 64
        self.nb = 5000
        self.nq = 50
        self.k = 10
        self.ds = SyntheticDataset(d=self.d, nt=2000, nb=self.nb,
                                   nq=self.nq, seed=42)

    def test_native_4bit(self):
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(self.d, qtype)
                index.train(self.ds.get_train())
                self.assertTrue(index.is_trained)

                index.add(self.ds.get_database())
                self.assertEqual(index.ntotal, self.nb)

                D, I = index.search(self.ds.get_queries(), self.k)
                self.assertEqual(D.shape, (self.nq, self.k))
                self.assertEqual(I.shape, (self.nq, self.k))
                self.assertTrue(np.all(I[:, 0] >= 0))
                self.assertTrue(np.all(np.isfinite(D)))
                for q in range(self.nq):
                    for j in range(1, self.k):
                        self.assertLessEqual(D[q, j - 1], D[q, j] + 1e-5)


class TestSQFastScanRecallParity(unittest.TestCase):
    """IndexSQFastScan recall should match IndexScalarQuantizer for 4-bit."""

    def setUp(self):
        self.d = 64
        self.ds = SyntheticDataset(d=self.d, nt=2000, nb=10000,
                                   nq=100, seed=42)
        gt_index = faiss.IndexFlatL2(self.d)
        gt_index.add(self.ds.get_database())
        self.D_gt, self.I_gt = gt_index.search(self.ds.get_queries(), 10)

    def test_native_4bit_parity(self):
        """Native 4-bit: FastScan should match SQ recall exactly."""
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(self.d, qtype)
                sq.train(self.ds.get_train())
                sq.add(self.ds.get_database())

                fs = faiss.IndexSQFastScan(self.d, qtype)
                fs.train(self.ds.get_train())
                fs.add(self.ds.get_database())

                _, I_sq = sq.search(self.ds.get_queries(), 10)
                _, I_fs = fs.search(self.ds.get_queries(), 10)
                r_sq = recall_at_k(self.I_gt, I_sq, 1)
                r_fs = recall_at_k(self.I_gt, I_fs, 1)
                self.assertAlmostEqual(r_sq, r_fs, delta=0.05,
                    msg=f"{name}: SQ={r_sq:.3f} FS={r_fs:.3f}")


class TestSQFastScanHighDimension(unittest.TestCase):
    """IndexSQFastScan sets M to d, so it reaches the uint16 accumulator bound.

    See TestManyPQSubQuantizers in test_fast_scan.py for the generic case.
    """

    def test_matches_scalar_quantizer(self):
        d=1024
        ds = SyntheticDataset(d=d, nt=2000, nb=5000, nq=50, seed=42)
        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(ds.get_database())
        _, I_gt = gt_index.search(ds.get_queries(), 10)

        sq = faiss.IndexScalarQuantizer(d, SQ.QT_4bit)
        sq.train(ds.get_train())
        sq.add(ds.get_database())

        fs = faiss.IndexSQFastScan(d, SQ.QT_4bit)
        fs.train(ds.get_train())
        fs.add(ds.get_database())

        _, I_sq = sq.search(ds.get_queries(), 10)
        _, I_fs = fs.search(ds.get_queries(), 10)
        r_sq = recall_at_k(I_gt, I_sq, 1)
        r_fs = recall_at_k(I_gt, I_fs, 1)
        self.assertAlmostEqual(
            r_sq, r_fs, delta=0.05,
            msg=f"d={d}: SQ={r_sq:.3f} FS={r_fs:.3f}")


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
                r1_sq = (I_sq[:, 0] >= 0).mean()
                r1_fs = (I_fs[:, 0] >= 0).mean()
                self.assertEqual(r1_sq, r1_fs)

    def test_conversion_rejects_non_4bit(self):
        sq = faiss.IndexScalarQuantizer(self.d, SQ.QT_8bit)
        sq.train(self.ds.get_train())
        with self.assertRaises(RuntimeError):
            faiss.IndexSQFastScan(sq)


class TestSQFastScanReset(unittest.TestCase):
    """Reset should clear all stored data."""

    def test_reset(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=500, nq=0, seed=42)
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())
                self.assertEqual(index.ntotal, 500)

                index.reset()
                self.assertEqual(index.ntotal, 0)

                index.add(ds.get_database())
                self.assertEqual(index.ntotal, 500)


class TestSQFastScanEdgeCases(unittest.TestCase):
    """Edge cases: k=1, single vector, zero vectors, odd dimensions."""

    def test_k_equals_1(self):
        d = 32
        ds = SyntheticDataset(d=d, nt=500, nb=1000, nq=10, seed=42)
        for name, qtype in NATIVE_4BIT:
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
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 1)
                self.assertEqual(I[0, 0], 0)
                self.assertLess(D[0, 0], 1.0)

    def test_zero_vectors(self):
        d = 32
        xb = np.zeros((100, d), dtype='float32')
        for name, qtype in NATIVE_4BIT:
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
                index = faiss.IndexSQFastScan(d, SQ.QT_4bit)
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

        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(
                    d, qtype, faiss.METRIC_INNER_PRODUCT)
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 10)
                self.assertEqual(D.shape, (50, 10))
                self.assertTrue(np.all(I[:, 0] >= 0))
                self.assertTrue(np.all(np.isfinite(D)))
                for q in range(50):
                    for j in range(1, 10):
                        self.assertGreaterEqual(
                            D[q, j - 1] + 1e-5, D[q, j])


class TestSQFastScanSaDecode(unittest.TestCase):
    """sa_decode should reconstruct vectors close to the original."""

    def test_sa_decode(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=100, nq=0, seed=42)
        for name, qtype in NATIVE_4BIT:
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

        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                idx1 = faiss.IndexSQFastScan(d, qtype)
                idx1.train(ds.get_train())
                idx1.add(xb)

                idx2 = faiss.IndexSQFastScan(d, qtype)
                idx2.train(ds.get_train())
                idx2.add(xb[:500])
                idx2.add(xb[500:])

                self.assertEqual(idx1.ntotal, idx2.ntotal)

                D1, I1 = idx1.search(xq, 10)
                D2, I2 = idx2.search(xq, 10)

                np.testing.assert_array_equal(I1, I2)
                np.testing.assert_allclose(D1, D2, atol=1e-5)


class TestSQFastScanSaEncode(unittest.TestCase):
    """sa_encode / sa_decode round-trip matches IndexScalarQuantizer."""

    def test_sa_encode_matches_sq(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=100, nq=0, seed=42)
        for name, qtype in NATIVE_4BIT:
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
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                ref = faiss.IndexSQFastScan(d, qtype)
                ref.train(ds.get_train())
                ref.add(ds.get_database())

                test = faiss.IndexSQFastScan(d, qtype)
                test.train(ds.get_train())
                codes = test.sa_encode(ds.get_database())
                test.add_sa_codes(codes)

                self.assertEqual(ref.ntotal, test.ntotal)

                D_ref, I_ref = ref.search(ds.get_queries(), 10)
                D_test, I_test = test.search(ds.get_queries(), 10)
                self.assertTrue(np.all(I_ref[:, 0] >= 0))
                self.assertTrue(np.all(I_test[:, 0] >= 0))


class TestSQFastScanPermuteEntries(unittest.TestCase):
    """permute_entries reorders the packed codes."""

    def test_permute_entries(self):
        d, nb = 64, 100
        ds = SyntheticDataset(d=d, nt=1000, nb=nb, nq=0, seed=42)
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())

                before = index.reconstruct_n(0, nb)
                perm = np.arange(nb - 1, -1, -1).astype("int64")
                index.permute_entries(perm)

                np.testing.assert_allclose(
                    index.reconstruct_n(0, nb), before[perm], atol=1e-5)


class TestSQFastScanDistanceComputer(unittest.TestCase):
    """get_distance_computer works for native 4-bit types."""

    def test_distance_computer(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=500, nq=0, seed=42)
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())
                dc = index.get_distance_computer()
                self.assertIsNotNone(dc)


class TestSQFastScanRangeSearch(unittest.TestCase):
    """range_search produces results consistent with kNN search."""

    def test_range_search(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=1000, nq=10, seed=42)
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                sq = faiss.IndexScalarQuantizer(d, qtype)
                sq.train(ds.get_train())
                sq.add(ds.get_database())

                fs = faiss.IndexSQFastScan(d, qtype)
                fs.train(ds.get_train())
                fs.add(ds.get_database())

                D, _ = sq.search(ds.get_queries()[:1], 1000)
                radius = float(D[0, 500])

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
        for name, qtype in NATIVE_4BIT:
            with self.subTest(qtype=name):
                index = faiss.IndexSQFastScan(d, qtype)
                index.train(ds.get_train())
                index.add(ds.get_database())

                D1, I1 = index.search(ds.get_queries(), 10)

                with tempfile.NamedTemporaryFile(suffix='.idx', delete=False) as f:
                    fname = f.name
                try:
                    faiss.write_index(index, fname)
                    loaded = faiss.read_index(fname)
                finally:
                    os.unlink(fname)

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
        index = faiss.index_factory(d, "SQ4fs")
        self.assertIsInstance(index, faiss.IndexSQFastScan)
        self.assertEqual(index.d, d)

    def test_factory_with_bbs(self):
        d = 64
        index = faiss.index_factory(d, "SQ4fs_64")
        self.assertIsInstance(index, faiss.IndexSQFastScan)
        self.assertEqual(index.bbs, 64)

    def test_factory_metric(self):
        d = 64
        index = faiss.index_factory(d, "SQ4fs", faiss.METRIC_INNER_PRODUCT)
        self.assertIsInstance(index, faiss.IndexSQFastScan)


class TestFastScanLUTScaleRounding(unittest.TestCase):
    """Rounding must not push the summed table over the uint16 limit.

    Each entry is rounded up by as much as 0.5, so M entries can add M/2 on
    top of the sum the scale was chosen to fit. The scale has to leave room
    for that.
    """

    def test_equal_spans_do_not_overflow_the_accumulator(self):
        # M equal spans is the worst case: with a scale of 65535 / total the
        # per-column maximum lands just under an integer and every one of them
        # rounds up together.
        for m in (256, 512, 1024, 2048):
            scale = faiss.fastscan_lut_scale(1.0, float(m), m)
            per_entry = math.floor(1.0 * scale + 0.5)
            self.assertLessEqual(
                per_entry * m,
                65535,
                f"M={m}: {m} entries of {per_entry} sum to {per_entry * m}",
            )
            self.assertLessEqual(per_entry, 255, f"M={m} overflows the uint8")

    def test_scale_still_uses_the_range_it_has(self):
        # The headroom must not collapse the scale: a 1024-column table should
        # still reach most of the accumulator.
        scale = faiss.fastscan_lut_scale(1.0, 1024.0, 1024)
        total = math.floor(1.0 * scale + 0.5) * 1024
        self.assertGreater(total, 60000, f"scale too conservative, total {total}")

    def test_sq_fastscan_recall_at_d1024(self):
        # d = 1024 means M = 1024, four times the M = 257 point where the
        # accumulator can wrap. A wrap makes the ranking noise, so R@1 collapses.
        d, n, nq = 1024, 2000, 50
        rng = np.random.default_rng(123)
        xb = rng.standard_normal((n, d), dtype=np.float32)
        xq = xb[:nq].copy()
        index = faiss.index_factory(d, "SQ4fs")
        index.train(xb)
        index.add(xb)
        _, ids = index.search(xq, 1)
        self.assertGreater(
            (ids[:, 0] == np.arange(nq)).mean(), 0.9, "R@1 collapsed, LUT wrapped"
        )


class TestRaBitQFastScanBiasBound(unittest.TestCase):
    """The per-probe bias is rounded into the same uint16 as the entries.

    The scale must bound span_j + probe_b[j] - glob_b, not span_j alone. A
    query whose probed lists have very different biases makes the difference
    large enough to wrap if only the span is bounded.
    """

    def test_matches_non_fastscan_with_uneven_probe_biases(self):
        """Guards the biased path at a dimension where the sum bound binds.

        At d = 1024 the scale comes from the sum constraint rather than the
        per-entry one, which is the regime where the per-probe bias added by
        simd_result_handlers (d0 += dbias16, into the same uint16) can push the
        accumulator over. This passes both with and without that bias in the
        bound, so it does NOT isolate the overflow; it is a regression guard on
        the path, not a boundary test. See the known limitation on large bias
        spreads below.
        """
        d, nlist, per, nq = 1024, 16, 40, 40
        rng = np.random.default_rng(7)
        centers = (np.arange(nlist, dtype=np.float32)[:, None] * 10.0) * np.ones(
            (1, d), dtype=np.float32
        )
        xb = np.repeat(centers, per, axis=0) + rng.standard_normal(
            (nlist * per, d)
        ).astype(np.float32)
        xq = xb[:nq].copy()

        def recall_at_1(factory):
            index = faiss.index_factory(d, factory)
            index.train(xb)
            index.add(xb)
            ivf = faiss.try_extract_index_ivf(index)
            ivf.nprobe = nlist
            _, ids = index.search(xq, 1)
            return (ids[:, 0] == np.arange(nq)).mean()

        ref = recall_at_1(f"HR,IVF{nlist},RaBitQ4")
        fs = recall_at_1(f"HR,IVF{nlist},RaBitQfs4")
        self.assertGreaterEqual(
            fs, ref - 0.1, f"fast scan R@1 {fs:.2f} trails the reference {ref:.2f}"
        )
