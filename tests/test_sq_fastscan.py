# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


SQ = faiss.ScalarQuantizer

NATIVE_4BIT = [
    ("QT_4bit",         SQ.QT_4bit),
    ("QT_4bit_uniform", SQ.QT_4bit_uniform),
]


class TestSQFastScanConstruction(unittest.TestCase):

    def test_accepts_reranked_types_rejects_fallback(self):
        d = 64
        with self.assertRaises(RuntimeError):
            faiss.IndexSQFastScan(d, SQ.QT_fp16)


class TestSQFastScanRecallParity(unittest.TestCase):

    def setUp(self):
        self.d = 64
        self.ds = SyntheticDataset(d=self.d, nt=2000, nb=10000,
                                   nq=100, seed=42)
        gt_index = faiss.IndexFlatL2(self.d)
        gt_index.add(self.ds.get_database())
        self.D_gt, self.I_gt = gt_index.search(self.ds.get_queries(), 10)

    def test_native_4bit_parity(self):
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


class TestSQFastScanConversionConstructor(unittest.TestCase):

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

                _, I_sq = sq.search(self.ds.get_queries(), 10)
                _, I_fs = fs.search(self.ds.get_queries(), 10)
                r1_sq = (I_sq[:, 0] >= 0).mean()
                r1_fs = (I_fs[:, 0] >= 0).mean()
                self.assertEqual(r1_sq, r1_fs)

    def test_conversion_accepts_8bit_and_rejects_fallback(self):
        sq = faiss.IndexScalarQuantizer(self.d, SQ.QT_8bit)
        sq.train(self.ds.get_train())
        sq.add(self.ds.get_database())
        converted = faiss.IndexSQFastScan(sq)
        self.assertEqual(converted.ntotal, sq.ntotal)

        fp16 = faiss.IndexScalarQuantizer(self.d, SQ.QT_fp16)
        fp16.train(self.ds.get_train())
        with self.assertRaises(RuntimeError):
            faiss.IndexSQFastScan(fp16)


class TestSQFastScanReset(unittest.TestCase):

    def test_reset(self):
        d = 64
        ds = SyntheticDataset(d=d, nt=1000, nb=500, nq=0, seed=42)
        index = faiss.IndexSQFastScan(d, SQ.QT_4bit)
        index.train(ds.get_train())
        index.add(ds.get_database())

        index.reset()
        self.assertEqual(index.ntotal, 0)

        index.add(ds.get_database())
        self.assertEqual(index.ntotal, 500)


class TestSQFastScanEdgeCases(unittest.TestCase):

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
        for d in [7, 9, 15, 17, 31, 33, 63, 65]:
            with self.subTest(d=d):
                ds = SyntheticDataset(d=d, nt=500, nb=500, nq=5, seed=42)
                index = faiss.IndexSQFastScan(d, SQ.QT_4bit)
                index.train(ds.get_train())
                index.add(ds.get_database())
                _, I = index.search(ds.get_queries(), 10)
                self.assertTrue(np.all(I[:, 0] >= 0))


class TestSQFastScanInnerProduct(unittest.TestCase):

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
                self.assertTrue(np.all(I[:, 0] >= 0))
                self.assertTrue(np.all(np.isfinite(D)))
                for q in range(50):
                    for j in range(1, 10):
                        self.assertGreaterEqual(
                            D[q, j - 1] + 1e-5, D[q, j])


class TestSQFastScanSaEncode(unittest.TestCase):

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

                sq_codes = sq.sa_encode(xb)
                fs_codes = fs.sa_encode(xb)
                np.testing.assert_array_equal(sq_codes, fs_codes,
                    err_msg=f"{name}: sa_encode differs")

                sq_dec = sq.sa_decode(sq_codes)
                fs_dec = fs.sa_decode(fs_codes)
                np.testing.assert_allclose(sq_dec, fs_dec, atol=1e-6,
                    err_msg=f"{name}: sa_decode differs")


class TestSQFastScanDistanceComputer(unittest.TestCase):

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


class TestSQFastScanIO(unittest.TestCase):

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

                D2, I2 = loaded.search(ds.get_queries(), 10)
                np.testing.assert_array_equal(I1, I2)
                np.testing.assert_allclose(D1, D2, atol=1e-5)


class TestSQFastScanFactory(unittest.TestCase):

    def test_factory_with_bbs(self):
        d = 64
        index = faiss.index_factory(d, "SQ4fs_64")
        self.assertIsInstance(index, faiss.IndexSQFastScan)
        self.assertEqual(index.bbs, 64)


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
        """Regression guard on the biased path at d = 1024, where the sum
        bound binds. It does not isolate the overflow."""
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


class TestSQFastScanSplitCodePaths(unittest.TestCase):
    """Paths that rebuild a full code from the split halves.

    QT_4bit hides every bug here, because its code is already one nibble per
    dimension and sq.code_size equals the fast-scan code size.
    """

    def _pair(self, qtype, d=32, n=200):
        rng = np.random.default_rng(0)
        xb = rng.standard_normal((n, d), dtype=np.float32)
        ref = faiss.IndexScalarQuantizer(d, qtype)
        ref.train(xb)
        ref.add(xb)
        fs = faiss.IndexSQFastScan(d, qtype)
        fs.train(xb)
        fs.add(xb)
        return xb, ref, fs

    def test_reconstruct_matches_reference(self):
        for qtype in (SQ.QT_4bit, SQ.QT_6bit, SQ.QT_8bit):
            _, ref, fs = self._pair(qtype)
            for i in (0, 7, 199):
                np.testing.assert_array_equal(
                    ref.reconstruct(i), fs.reconstruct(i)
                )

    def test_range_search_matches_reference(self):
        for qtype in (SQ.QT_4bit, SQ.QT_6bit, SQ.QT_8bit):
            xb, ref, fs = self._pair(qtype)
            lref, _, iref = ref.range_search(xb[:5], 20.0)
            lfs, _, ifs = fs.range_search(xb[:5], 20.0)
            np.testing.assert_array_equal(lref, lfs)
            self.assertEqual(sorted(iref.tolist()), sorted(ifs.tolist()))

    def test_permute_entries_moves_both_halves(self):
        for qtype in (SQ.QT_4bit, SQ.QT_6bit, SQ.QT_8bit):
            _, _, fs = self._pair(qtype)
            n = fs.ntotal
            before = fs.reconstruct_n(0, n)
            perm = np.arange(n - 1, -1, -1, dtype="int64")
            fs.permute_entries(perm)
            np.testing.assert_array_equal(fs.reconstruct_n(0, n), before[perm])

    def test_direct_qtypes_match_reference(self):
        rng = np.random.default_rng(0)
        d, n = 32, 200
        for qtype, lo in (
            (SQ.QT_8bit_direct, 0),
            (SQ.QT_8bit_direct_signed, -128),
        ):
            xb = rng.integers(lo, lo + 256, size=(n, d)).astype(np.float32)
            ref = faiss.IndexScalarQuantizer(d, qtype)
            ref.train(xb)
            ref.add(xb)
            fs = faiss.IndexSQFastScan(d, qtype)
            fs.train(xb)
            fs.add(xb)
            _, iref = ref.search(xb[:10], 5)
            _, ifs = fs.search(xb[:10], 5)
            hits = sum(len(set(a) & set(b)) for a, b in zip(iref, ifs))
            self.assertGreaterEqual(hits, 0.9 * iref.size)

    def test_lo_codes_track_ntotal_across_add_paths(self):
        rng = np.random.default_rng(0)
        d, qtype = 8, SQ.QT_8bit
        # over the blocking size of IndexFastScan::add, which re-enters add()
        n = 70000
        xb = rng.standard_normal((n, d), dtype=np.float32)
        ref = faiss.IndexScalarQuantizer(d, qtype)
        ref.train(xb)
        ref.add(xb)

        one = faiss.IndexSQFastScan(d, qtype)
        one.train(xb)
        one.add(xb)
        self.assertEqual(one.ntotal, n)

        split = faiss.IndexSQFastScan(d, qtype)
        split.train(xb)
        split.add(xb[:40000])
        split.add(xb[40000:])
        self.assertEqual(split.ntotal, n)

        for i in (0, 39999, 65536, n - 1):
            expect = ref.reconstruct(i)
            np.testing.assert_array_equal(expect, one.reconstruct(i))
            np.testing.assert_array_equal(expect, split.reconstruct(i))

    def test_add_sa_codes_fills_the_rerank_half(self):
        xb, ref, _ = self._pair(SQ.QT_8bit)
        fs = faiss.IndexSQFastScan(ref.d, SQ.QT_8bit)
        fs.train(xb)
        fs.add_sa_codes(ref.sa_encode(xb))
        self.assertEqual(fs.ntotal, ref.ntotal)
        for i in (0, 7, 199):
            np.testing.assert_array_equal(ref.reconstruct(i), fs.reconstruct(i))

    def test_merge_from_moves_the_rerank_half(self):
        rng = np.random.default_rng(0)
        d, n = 16, 100
        xb = rng.standard_normal((2 * n, d), dtype=np.float32)
        parts = []
        for lo, hi in ((0, n), (n, 2 * n)):
            idx = faiss.IndexSQFastScan(d, SQ.QT_8bit)
            idx.train(xb)
            idx.add(xb[lo:hi])
            parts.append(idx)
        parts[0].merge_from(parts[1])
        self.assertEqual(parts[0].ntotal, 2 * n)

        whole = faiss.IndexSQFastScan(d, SQ.QT_8bit)
        whole.train(xb)
        whole.add(xb)
        np.testing.assert_array_equal(
            whole.reconstruct_n(0, 2 * n), parts[0].reconstruct_n(0, 2 * n)
        )
