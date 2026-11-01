# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import numpy as np

import faiss
import unittest

from faiss.contrib.datasets import SyntheticDataset

from common_faiss_tests import for_all_simd_levels, NoneSIMDLevel


@for_all_simd_levels
class TestScalarQuantizerEncodeDecode(unittest.TestCase):

    def setUp(self):
        self.d = 32
        self.ds = SyntheticDataset(d=self.d, nt=0, nb=1000, nq=0, seed=42)
        self.xb = self.ds.get_database()

    def do_encode_decode(self, qtype, max_err):
        sq = faiss.ScalarQuantizer(self.d, qtype)
        sq.train(self.xb)
        codes = sq.compute_codes(self.xb)
        xb_decoded = sq.decode(codes)
        err = np.abs(self.xb - xb_decoded).max()
        self.assertLess(err, max_err)

    def test_4bit(self):
        self.do_encode_decode(faiss.ScalarQuantizer.QT_4bit, 0.1)

    def test_fp16(self):
        self.do_encode_decode(faiss.ScalarQuantizer.QT_fp16, 0.001)

    def test_8bit_uniform(self):
        self.do_encode_decode(faiss.ScalarQuantizer.QT_8bit_uniform, 0.01)

    def test_4bit_uniform(self):
        self.do_encode_decode(faiss.ScalarQuantizer.QT_4bit_uniform, 0.1)

    def test_tqmse(self):
        faiss.normalize_L2(self.xb)
        for qtype, max_err in (
            (faiss.ScalarQuantizer.QT_1bit_tqmse, 1.0),
            (faiss.ScalarQuantizer.QT_2bit_tqmse, 0.8),
            (faiss.ScalarQuantizer.QT_3bit_tqmse, 0.6),
            (faiss.ScalarQuantizer.QT_4bit_tqmse, 0.2),
            (faiss.ScalarQuantizer.QT_8bit_tqmse, 0.1),
        ):
            with self.subTest(qtype=qtype):
                self.do_encode_decode(qtype, max_err)

    def test_tqmse_preserves_norm(self):
        """tqmse has no per-vector scale: the Lloyd-Max table must be scaled
        by 1/sqrt(d) or the codebook collapses toward a sign quantizer."""
        for qtype, nbits in (
            (faiss.ScalarQuantizer.QT_1bit_tqmse, 1),
            (faiss.ScalarQuantizer.QT_2bit_tqmse, 2),
            (faiss.ScalarQuantizer.QT_3bit_tqmse, 3),
            (faiss.ScalarQuantizer.QT_4bit_tqmse, 4),
            (faiss.ScalarQuantizer.QT_8bit_tqmse, 8),
        ):
            for d in (32, 64, 768):
                with self.subTest(nbits=nbits, d=d):
                    rs = np.random.RandomState(123)
                    x = rs.randn(200, d).astype("float32")
                    faiss.normalize_L2(x)

                    sq = faiss.ScalarQuantizer(d, qtype)
                    sq.train(x)
                    decoded = sq.decode(sq.compute_codes(x))
                    norms = np.linalg.norm(decoded, axis=1)

                    # 1-bit cannot resolve magnitude at all (every component
                    # reconstructs to +/-c), so it gets a looser band.
                    tol = 0.25 if nbits == 1 else 0.1
                    self.assertLess(abs(np.median(norms) - 1.0), tol)

                    # Guard the collapse directly: a codebook matched to the
                    # data reconstructs most of its 2^nbits distinct levels.
                    # This is what catches 8-bit, whose norm stays near 1 even
                    # when collapsed (31 of 256 levels used at d=768).
                    used = len(np.unique(decoded))
                    self.assertGreaterEqual(used, 2**nbits // 2)

    def test_fp16_rounding(self):
        cases = {  # float32 bits: float16 bits
            0x3F801000: 0x3C00,  # tie between two normal halves
            0x33000000: 0x0000,  # tie between two subnormal halves
            0x7F802000: 0x7E01,  # signaling NaN keeps its payload
        }
        # d = 16, so that every level runs its own quantizer
        x = np.array(list(cases) * 6, dtype="uint32")[:16].view("float32")
        sq = faiss.ScalarQuantizer(16, faiss.ScalarQuantizer.QT_fp16)
        codes = sq.compute_codes(x[None]).view("uint16")[0]
        np.testing.assert_array_equal(codes, (list(cases.values()) * 6)[:16])

    def test_codes_match_none(self):
        """Encode must be bit-identical at every SIMD level."""
        if not faiss.SIMDConfig.is_simd_level_available(faiss.SIMDLevel_NONE):
            self.skipTest("SIMDLevel.NONE not available")
        for qtype in (
            faiss.ScalarQuantizer.QT_8bit,
            faiss.ScalarQuantizer.QT_4bit,
            faiss.ScalarQuantizer.QT_8bit_uniform,
            faiss.ScalarQuantizer.QT_4bit_uniform,
            faiss.ScalarQuantizer.QT_fp16,
        ):
            with self.subTest(qtype=qtype):
                sq = faiss.ScalarQuantizer(self.d, qtype)
                sq.train(self.xb)
                codes = sq.compute_codes(self.xb)
                with NoneSIMDLevel():
                    codes_none = sq.compute_codes(self.xb)
                np.testing.assert_array_equal(codes, codes_none)


@for_all_simd_levels
class TestScalarQuantizerSearch(unittest.TestCase):

    def setUp(self):
        self.d = 32
        self.ds = SyntheticDataset(d=self.d, nt=0, nb=10000, nq=100, seed=42)
        self.xb = self.ds.get_database()
        self.xq = self.ds.get_queries()

    def do_search(self, factory_str, min_recall):
        index_gt = faiss.IndexFlatL2(self.d)
        index_gt.add(self.xb)
        _, I_gt = index_gt.search(self.xq, 10)

        index = faiss.index_factory(self.d, factory_str)
        index.train(self.xb)
        index.add(self.xb)
        _, I = index.search(self.xq, 10)

        recall = (I_gt[:, 0] == I[:, 0]).mean()
        self.assertGreater(recall, min_recall)

    def test_SQ4(self):
        self.do_search("SQ4", 0.5)

    def test_SQfp16(self):
        self.do_search("SQfp16", 0.99)

    def test_tqmse_search(self):
        xb_unit = self.xb.copy()
        xq_unit = self.xq.copy()
        faiss.normalize_L2(xb_unit)
        faiss.normalize_L2(xq_unit)
        index_gt = faiss.IndexFlatL2(self.d)
        index_gt.add(xb_unit)
        _, I_gt = index_gt.search(xq_unit, 10)

        recalls = {}
        for factory_str in ["L2norm,RR,SQtqmse4", "L2norm,RR,SQtqmse8"]:
            index = faiss.index_factory(self.d, factory_str)
            index.train(self.xb)
            index.add(self.xb)
            _, I = index.search(self.xq, 10)
            recalls[factory_str] = (I_gt[:, 0] == I[:, 0]).mean()

        self.assertGreater(recalls["L2norm,RR,SQtqmse4"], 0.2)
        self.assertGreaterEqual(
            recalls["L2norm,RR,SQtqmse8"], recalls["L2norm,RR,SQtqmse4"]
        )


@for_all_simd_levels
class TestScalarQuantizerDistances(unittest.TestCase):

    def test_distance_matches_reconstruct(self):
        d = 32
        ds = SyntheticDataset(d=d, nt=0, nb=100, nq=1, seed=42)
        x = ds.get_queries()
        xb = ds.get_database()

        index = faiss.index_factory(d, "SQ8")
        index.train(xb)
        index.add(xb)

        D, I = index.search(x, 10)

        for i in range(10):
            xr = index.reconstruct(int(I[0, i]))
            dist = ((x[0] - xr) ** 2).sum()
            self.assertAlmostEqual(D[0, i], dist, places=4)

    def test_tqmse_distance_matches_reconstruct(self):
        d = 32
        ds = SyntheticDataset(d=d, nt=0, nb=100, nq=1, seed=42)
        x = ds.get_queries()
        xb = ds.get_database()
        faiss.normalize_L2(x)
        faiss.normalize_L2(xb)

        index = faiss.index_factory(d, "SQtqmse8")
        index.train(xb)
        index.add(xb)

        D, I = index.search(x, 10)

        for i in range(10):
            xr = index.reconstruct(int(I[0, i]))
            dist = ((x[0] - xr) ** 2).sum()
            self.assertAlmostEqual(D[0, i], dist, places=4)


@for_all_simd_levels
class TestScalarQuantizerEdgeCases(unittest.TestCase):

    def test_zero_vectors(self):
        d = 32
        xb = np.zeros((100, d), dtype="float32")
        index = faiss.index_factory(d, "SQ8")
        index.train(xb)
        index.add(xb)
        D, _ = index.search(np.zeros((1, d), dtype="float32"), 10)
        self.assertTrue(np.allclose(D, 0, atol=1e-5))

    def test_constant_vectors(self):
        d = 32
        xb = np.ones((100, d), dtype="float32") * 0.5
        index = faiss.index_factory(d, "SQ8")
        index.train(xb)
        index.add(xb)
        D, _ = index.search(np.ones((1, d), dtype="float32") * 0.5, 10)
        self.assertTrue(np.allclose(D, 0, atol=1e-3))

    def test_extreme_dims(self):
        for d in [1, 1024]:
            with self.subTest(d=d):
                ds = SyntheticDataset(d=d, nt=0, nb=100, nq=10, seed=42)
                xb = ds.get_database()
                xq = ds.get_queries()
                index = faiss.index_factory(d, "SQ8")
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 10)
                self.assertEqual(D.shape, (10, 10))
                self.assertEqual(I.shape, (10, 10))

    def test_non_simd_dims(self):
        for d in [7, 9, 15, 17, 31, 33, 63, 65]:
            with self.subTest(d=d):
                ds = SyntheticDataset(d=d, nt=0, nb=100, nq=5, seed=42)
                xb = ds.get_database()
                xq = ds.get_queries()
                index = faiss.index_factory(d, "SQ8")
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 10)
                self.assertEqual(D.shape, (5, 10))
                self.assertEqual(I.shape, (5, 10))

    def test_tqmse_non_simd_dims(self):
        factory_strings = [
            "SQtqmse1",
            "SQtqmse2",
            "SQtqmse3",
            "SQtqmse4",
            "SQtqmse8",
        ]
        for d in [7, 9, 15, 17, 31, 33, 63, 65]:
            for factory_str in factory_strings:
                with self.subTest(d=d, factory_str=factory_str):
                    ds = SyntheticDataset(d=d, nt=0, nb=100, nq=5, seed=42)
                    xb = ds.get_database()
                    xq = ds.get_queries()
                    faiss.normalize_L2(xb)
                    faiss.normalize_L2(xq)
                    index = faiss.index_factory(d, factory_str)
                    index.train(xb)
                    index.add(xb)
                    D, I = index.search(xq, 10)
                    self.assertEqual(D.shape, (5, 10))
                    self.assertEqual(I.shape, (5, 10))


class TestScalarQuantizerNonUniformTraining(unittest.TestCase):

    def train(self, x):
        sq = faiss.ScalarQuantizer(x.shape[1], faiss.ScalarQuantizer.QT_8bit)
        sq.rangestat = faiss.ScalarQuantizer.RS_meanstd
        sq.rangestat_arg = 2.0
        sq.train(x)
        return faiss.vector_to_array(sq.trained)

    def test_meanstd_matches_reference(self):
        x = np.random.RandomState(123).rand(200, 256).astype("float32")
        x[0] = 50
        mean, std = x.mean(0), x.std(0)
        ref = np.hstack([mean - 2 * std, 4 * std])
        np.testing.assert_allclose(self.train(x), ref, rtol=1e-4)

    def test_multithreaded_matches_single_threaded(self):
        x = np.random.RandomState(123).rand(3, 8192).astype("float32")
        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            ref = self.train(x)
            faiss.omp_set_num_threads(32)
            for _ in range(100):
                np.testing.assert_array_equal(self.train(x), ref)
        finally:
            faiss.omp_set_num_threads(nt)


@for_all_simd_levels
class TestScalarQuantizerIP(unittest.TestCase):

    def test_inner_product(self):
        d = 32
        ds = SyntheticDataset(d=d, nt=0, nb=1000, nq=10, seed=42)
        xb = ds.get_database().copy()
        xq = ds.get_queries().copy()
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        index = faiss.index_factory(d, "SQ8", faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
        D, _ = index.search(xq, 10)

        # normalized vectors: max IP should be close to 1
        self.assertTrue(np.all(D[:, 0] <= 1.1))
        self.assertTrue(np.all(D[:, 0] >= 0.5))


class TestTurboQUnbiasedness(unittest.TestCase):

    def test_reconstruction_quality(self):
        rng = np.random.RandomState(42)

        for d in [128, 256, 384, 512, 768]:
            n = 500
            xb = rng.randn(n, d).astype("float32")
            faiss.normalize_L2(xb)

            for qt_name, qt in [
                ("SQtq2", faiss.ScalarQuantizer.QT_2bit_tq),
                ("SQtq3", faiss.ScalarQuantizer.QT_3bit_tq),
            ]:
                index = faiss.IndexScalarQuantizer(d, qt)
                index.train(xb)
                index.add(xb)

                xr = np.vstack([index.reconstruct(i) for i in range(n)])

                self_ip = np.array([xb[i] @ xr[i] for i in range(n)])
                mean_self_ip = self_ip.mean()

                self.assertGreater(
                    mean_self_ip,
                    0.1,
                    msg=f"d={d} {qt_name}: self IP {mean_self_ip:.4f} too low",
                )


class TestTurboQFullFactory(unittest.TestCase):

    def test_factory_sqtq(self):
        cases = [
            ("SQtq2", faiss.ScalarQuantizer.QT_2bit_tq),
            ("SQtq3", faiss.ScalarQuantizer.QT_3bit_tq),
            ("SQtq4", faiss.ScalarQuantizer.QT_4bit_tq),
            ("SQtq5", faiss.ScalarQuantizer.QT_5bit_tq),
        ]
        for factory_str, qtype in cases:
            with self.subTest(factory_str=factory_str):
                index = faiss.index_factory(32, factory_str)
                self.assertIsInstance(index, faiss.IndexScalarQuantizer)
                self.assertEqual(index.sq.qtype, qtype)

    def test_factory_ivf_sqtq(self):
        d = 64
        for sq_str in ["SQtq2", "SQtq3", "SQtq4"]:
            with self.subTest(sq_str=sq_str):
                factory_str = f"IVF16,{sq_str}"
                index = faiss.index_factory(d, factory_str)
                self.assertIsInstance(index, faiss.IndexIVFScalarQuantizer)
                self.assertFalse(index.by_residual)


class TestTurboQFullEncodeDecode(unittest.TestCase):

    def test_encode_decode_roundtrip(self):
        for d in [7, 9, 15, 17, 32, 33, 64, 65, 128]:
            for qtype, max_err in [
                (faiss.ScalarQuantizer.QT_2bit_tq, 1.5),
                (faiss.ScalarQuantizer.QT_3bit_tq, 1.2),
                (faiss.ScalarQuantizer.QT_4bit_tq, 0.8),
            ]:
                with self.subTest(d=d, qtype=qtype):
                    rng = np.random.RandomState(42)
                    xb = rng.randn(200, d).astype("float32")
                    faiss.normalize_L2(xb)
                    sq = faiss.ScalarQuantizer(d, qtype)
                    sq.train(xb)
                    codes = sq.compute_codes(xb)
                    xr = sq.decode(codes)
                    err = np.abs(xb - xr).max()
                    self.assertLess(err, max_err)


class TestTurboQFullSearch(unittest.TestCase):

    def setUp(self):
        self.d = 64
        self.rng = np.random.RandomState(42)
        self.nb = 5000
        self.nq = 50
        self.xb = self.rng.randn(self.nb, self.d).astype("float32")
        self.xq = self.rng.randn(self.nq, self.d).astype("float32")
        faiss.normalize_L2(self.xb)
        faiss.normalize_L2(self.xq)

        gt_index = faiss.IndexFlatL2(self.d)
        gt_index.add(self.xb)
        _, self.I_gt = gt_index.search(self.xq, 10)

    def _recall_at_1(self, index):
        _, I = index.search(self.xq, 10)
        return (self.I_gt[:, 0] == I[:, 0]).mean()

    def test_flat_sqtq_search(self):
        recalls = {}
        for sq_str in ["SQtq2", "SQtq3", "SQtq4"]:
            index = faiss.index_factory(self.d, sq_str)
            index.train(self.xb)
            index.add(self.xb)
            recalls[sq_str] = self._recall_at_1(index)
            self.assertGreater(recalls[sq_str], 0.05)

        self.assertGreaterEqual(recalls["SQtq3"], recalls["SQtq2"] - 0.05)
        self.assertGreaterEqual(recalls["SQtq4"], recalls["SQtq3"] - 0.05)


class TestTurboQFullSerialization(unittest.TestCase):

    def _roundtrip(self, index):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)
            index2 = faiss.read_index(fname)
            return index2
        finally:
            os.unlink(fname)

    def test_flat_serialize_roundtrip(self):
        d = 64
        rng = np.random.RandomState(42)
        xb = rng.randn(500, d).astype("float32")
        xq = rng.randn(10, d).astype("float32")
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        for qtype in [
            faiss.ScalarQuantizer.QT_2bit_tq,
            faiss.ScalarQuantizer.QT_3bit_tq,
            faiss.ScalarQuantizer.QT_4bit_tq,
            faiss.ScalarQuantizer.QT_5bit_tq,
        ]:
            with self.subTest(qtype=qtype):
                index = faiss.IndexScalarQuantizer(d, qtype)
                index.train(xb)
                index.add(xb)

                D1, I1 = index.search(xq, 5)
                index2 = self._roundtrip(index)
                D2, I2 = index2.search(xq, 5)

                np.testing.assert_array_equal(I1, I2)
                np.testing.assert_allclose(D1, D2, atol=1e-6)

    def test_ivf_serialize_roundtrip(self):
        d = 64
        rng = np.random.RandomState(42)
        xb = rng.randn(1000, d).astype("float32")
        xq = rng.randn(10, d).astype("float32")
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        for sq_str in ["SQtq2", "SQtq3"]:
            with self.subTest(sq_str=sq_str):
                index = faiss.index_factory(d, f"IVF16,{sq_str}")
                index.train(xb)
                index.add(xb)
                index.nprobe = 4

                D1, I1 = index.search(xq, 5)
                self.assertTrue(np.all(I1[:, 0] >= 0))
                index2 = self._roundtrip(index)
                index2.nprobe = 4
                D2, I2 = index2.search(xq, 5)

                np.testing.assert_array_equal(I1, I2)
                np.testing.assert_allclose(D1, D2, atol=1e-6)

    def test_serialize_preserves_qtype(self):
        d = 32
        rng = np.random.RandomState(42)
        xb = rng.randn(100, d).astype("float32")
        faiss.normalize_L2(xb)

        for qtype in [
            faiss.ScalarQuantizer.QT_2bit_tq,
            faiss.ScalarQuantizer.QT_3bit_tq,
        ]:
            with self.subTest(qtype=qtype):
                index = faiss.IndexScalarQuantizer(d, qtype)
                index.train(xb)
                index.add(xb)

                index2 = self._roundtrip(index)
                self.assertEqual(index2.sq.qtype, qtype)
                self.assertEqual(index2.ntotal, index.ntotal)
                self.assertEqual(index2.d, d)


class TestTurboQFullDistances(unittest.TestCase):

    def test_search_distances_are_finite(self):
        d = 64
        rng = np.random.RandomState(42)
        xb = rng.randn(500, d).astype("float32")
        xq = rng.randn(10, d).astype("float32")
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        for qtype in [
            faiss.ScalarQuantizer.QT_2bit_tq,
            faiss.ScalarQuantizer.QT_3bit_tq,
        ]:
            with self.subTest(qtype=qtype):
                index = faiss.IndexScalarQuantizer(d, qtype)
                index.train(xb)
                index.add(xb)

                D, I = index.search(xq, 5)
                self.assertTrue(np.all(np.isfinite(D)))
                self.assertTrue(np.all(D[:, 0] >= 0))
                self.assertTrue(np.all(I >= 0))


@for_all_simd_levels
class TestSQByteDirectAcrossSIMDLevels(unittest.TestCase):
    """Every SIMD level must return the exact QT_8bit_direct{,_signed}
    distances.

    The distances are integers, so this class computes them from the
    definition and compares. Comparing against SIMDLevel.NONE instead would
    skip in static builds, which report only the one level they compile.

    d = 16 and d = 32 pass the d % 16 == 0 test in sq-dispatch.h. That test
    routes the query to DistanceComputerByte and DistanceComputerByteSigned.
    Two dispatch chains reach those kernels, and this class covers both.
    search() uses sq_select_InvertedListScanner. get_distance_computer() uses
    sq_select_distance_computer, which is the chain Refine(SQ8) follows.
    """

    def reference(self, xq, xb, metric):
        """Exact distances: all values stay below 2**24, so float32 is exact."""
        a = xq.astype("int64")
        b = xb.astype("int64")
        if metric == faiss.METRIC_L2:
            return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return a @ b.T

    def do_test(self, measure, expect):
        rng = np.random.RandomState(1234)
        for qtype in (
            faiss.ScalarQuantizer.QT_8bit_direct,
            faiss.ScalarQuantizer.QT_8bit_direct_signed,
        ):
            lo, hi = (
                (-128, 128)
                if qtype == faiss.ScalarQuantizer.QT_8bit_direct_signed
                else (0, 256)
            )
            for metric in (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT):
                for d in (16, 32):
                    with self.subTest(qtype=qtype, metric=metric, d=d):
                        xb = rng.randint(lo, hi, (500, d)).astype("float32")
                        xq = rng.randint(lo, hi, (10, d)).astype("float32")
                        index = faiss.IndexScalarQuantizer(d, qtype, metric)
                        index.add(xb)
                        got = measure(index, xq)
                        truth = self.reference(xq, xb, metric)
                        np.testing.assert_array_equal(got, expect(truth, metric))

    def test_scanner(self):
        k = 10

        def measure(index, xq):
            return index.search(xq, k)[0]

        def expect(truth, metric):
            # search returns L2 ascending and inner product descending.
            ordered = np.sort(truth, axis=1)
            if metric == faiss.METRIC_INNER_PRODUCT:
                ordered = ordered[:, ::-1]
            return ordered[:, :k]

        self.do_test(measure, expect)

    def test_distance_computer(self):
        def measure(index, xq):
            dc = index.get_distance_computer()
            out = np.empty((len(xq), index.ntotal), dtype="float32")
            for q in range(len(xq)):
                dc.set_query(faiss.swig_ptr(xq[q]))
                for i in range(index.ntotal):
                    out[q, i] = dc(int(i))
            return out

        self.do_test(measure, lambda truth, metric: truth)
