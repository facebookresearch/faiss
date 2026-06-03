# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

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
        self.assertEqual(xb_decoded.shape, self.xb.shape)
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

    def test_tqmse_1bit(self):
        faiss.normalize_L2(self.xb)
        self.do_encode_decode(faiss.ScalarQuantizer.QT_1bit_tqmse, 1.0)

    def test_tqmse_2bit(self):
        faiss.normalize_L2(self.xb)
        self.do_encode_decode(faiss.ScalarQuantizer.QT_2bit_tqmse, 0.8)

    def test_tqmse_3bit(self):
        faiss.normalize_L2(self.xb)
        self.do_encode_decode(faiss.ScalarQuantizer.QT_3bit_tqmse, 0.6)

    def test_tqmse_4bit(self):
        faiss.normalize_L2(self.xb)
        self.do_encode_decode(faiss.ScalarQuantizer.QT_4bit_tqmse, 0.2)

    def test_tqmse_8bit(self):
        faiss.normalize_L2(self.xb)
        self.do_encode_decode(faiss.ScalarQuantizer.QT_8bit_tqmse, 0.1)

    def test_codes_match_none(self):
        """SQ codes are integer; encode dispatch (sq-dispatch.h) must
        produce bit-identical output at every SIMD level. Catches drift in
        the per-ISA scalar-quantizer encode kernels."""
        if not faiss.SIMDConfig.is_simd_level_available(faiss.SIMDLevel_NONE):
            self.skipTest("SIMDLevel.NONE not available")
        for qtype in (
                faiss.ScalarQuantizer.QT_8bit,
                faiss.ScalarQuantizer.QT_4bit,
                faiss.ScalarQuantizer.QT_8bit_uniform,
                faiss.ScalarQuantizer.QT_4bit_uniform,
                faiss.ScalarQuantizer.QT_fp16):
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
        self.xb_unit = self.xb.copy()
        self.xq_unit = self.xq.copy()
        faiss.normalize_L2(self.xb_unit)
        faiss.normalize_L2(self.xq_unit)

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
        self.do_search('SQ4', 0.5)

    def test_SQfp16(self):
        self.do_search('SQfp16', 0.99)

    def test_tqmse_search(self):
        index_gt = faiss.IndexFlatL2(self.d)
        index_gt.add(self.xb_unit)
        _, I_gt = index_gt.search(self.xq_unit, 10)

        recalls = {}
        for factory_str in ['L2norm,RR,SQtqmse4', 'L2norm,RR,SQtqmse8']:
            index = faiss.index_factory(self.d, factory_str)
            index.train(self.xb)
            index.add(self.xb)
            _, I = index.search(self.xq, 10)
            recalls[factory_str] = (I_gt[:, 0] == I[:, 0]).mean()

        self.assertGreater(recalls['L2norm,RR,SQtqmse4'], 0.2)
        self.assertGreaterEqual(
            recalls['L2norm,RR,SQtqmse8'],
            recalls['L2norm,RR,SQtqmse4'])


@for_all_simd_levels
class TestScalarQuantizerDistances(unittest.TestCase):

    def test_distance_matches_reconstruct(self):
        d = 32
        ds = SyntheticDataset(d=d, nt=0, nb=100, nq=1, seed=42)
        x = ds.get_queries()
        xb = ds.get_database()

        index = faiss.index_factory(d, 'SQ8')
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

        index = faiss.index_factory(d, 'SQtqmse8')
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
        xb = np.zeros((100, d), dtype='float32')
        index = faiss.index_factory(d, 'SQ8')
        index.train(xb)
        index.add(xb)
        D, _ = index.search(np.zeros((1, d), dtype='float32'), 10)
        self.assertTrue(np.allclose(D, 0, atol=1e-5))

    def test_constant_vectors(self):
        d = 32
        xb = np.ones((100, d), dtype='float32') * 0.5
        index = faiss.index_factory(d, 'SQ8')
        index.train(xb)
        index.add(xb)
        D, _ = index.search(np.ones((1, d), dtype='float32') * 0.5, 10)
        self.assertTrue(np.allclose(D, 0, atol=1e-3))

    def test_extreme_dims(self):
        """Test extreme dimension values (very small and very large)."""
        for d in [1, 1024]:
            with self.subTest(d=d):
                ds = SyntheticDataset(d=d, nt=0, nb=100, nq=10, seed=42)
                xb = ds.get_database()
                xq = ds.get_queries()
                index = faiss.index_factory(d, 'SQ8')
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 10)
                self.assertEqual(D.shape, (10, 10))
                self.assertEqual(I.shape, (10, 10))

    def test_non_simd_dims(self):
        """Test dimensions not aligned to SIMD width (8/16)."""
        for d in [7, 9, 15, 17, 31, 33, 63, 65]:
            with self.subTest(d=d):
                ds = SyntheticDataset(d=d, nt=0, nb=100, nq=5, seed=42)
                xb = ds.get_database()
                xq = ds.get_queries()
                index = faiss.index_factory(d, 'SQ8')
                index.train(xb)
                index.add(xb)
                D, I = index.search(xq, 10)
                self.assertEqual(D.shape, (5, 10))
                self.assertEqual(I.shape, (5, 10))

    def test_tqmse_non_simd_dims(self):
        factory_strings = [
            'SQtqmse1',
            'SQtqmse2',
            'SQtqmse3',
            'SQtqmse4',
            'SQtqmse8',
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


@for_all_simd_levels
class TestScalarQuantizerIP(unittest.TestCase):

    def test_inner_product(self):
        d = 32
        ds = SyntheticDataset(d=d, nt=0, nb=1000, nq=10, seed=42)
        xb = ds.get_database().copy()
        xq = ds.get_queries().copy()
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        index = faiss.index_factory(d, 'SQ8', faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
        D, _ = index.search(xq, 10)

        # normalized vectors: max IP should be close to 1
        self.assertTrue(np.all(D[:, 0] <= 1.1))
        self.assertTrue(np.all(D[:, 0] >= 0.5))


class TestTurboQUnbiasedness(unittest.TestCase):
    """Verify TurboQuant reconstruction preserves inner products."""

    def test_reconstruction_quality(self):
        rng = np.random.RandomState(42)
        print("\n=== TurboQuant reconstruction quality ===")
        print(f"{'d':>6} {'qt':>8} {'self_ip':>10} {'cos_sim':>10}")
        print("-" * 42)

        for d in [128, 256, 384, 512, 768]:
            n = 500
            xb = rng.randn(n, d).astype('float32')
            faiss.normalize_L2(xb)

            for qt_name, qt in [
                ("SQtq2", faiss.ScalarQuantizer.QT_2bit_tq),
                ("SQtq3", faiss.ScalarQuantizer.QT_3bit_tq),
            ]:
                index = faiss.IndexScalarQuantizer(d, qt)
                index.train(xb)
                index.add(xb)

                xr = np.vstack([index.reconstruct(i) for i in range(n)])

                # Self inner product: <x, x_hat> for each vector
                self_ip = np.array([xb[i] @ xr[i] for i in range(n)])
                mean_self_ip = self_ip.mean()

                # Cosine similarity
                xr_norms = np.linalg.norm(xr, axis=1)
                cos_sim = np.array([
                    xb[i] @ xr[i] / (xr_norms[i] + 1e-30)
                    for i in range(n)
                ]).mean()

                print(
                    f"{d:>6} {qt_name:>8} "
                    f"{mean_self_ip:>10.4f} {cos_sim:>10.4f}"
                )

                # Reconstruction should point in the right direction
                self.assertGreater(
                    mean_self_ip, 0.1,
                    msg=f"d={d} {qt_name}: self IP {mean_self_ip:.4f} too low"
                )


class TestTurboQFullFactory(unittest.TestCase):
    """Test factory string construction for TurboQ full types."""

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
                ivf = faiss.downcast_index(index)
                self.assertFalse(ivf.by_residual)


class TestTurboQFullEncodeDecode(unittest.TestCase):
    """Encode/decode roundtrip for TurboQ full types."""

    def test_encode_decode_roundtrip(self):
        for d in [32, 64, 128]:
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
                    self.assertEqual(xr.shape, xb.shape)
                    err = np.abs(xb - xr).max()
                    self.assertLess(err, max_err)

    def test_non_simd_aligned_dims(self):
        for d in [7, 9, 15, 17, 33, 65]:
            with self.subTest(d=d):
                rng = np.random.RandomState(42)
                xb = rng.randn(100, d).astype("float32")
                faiss.normalize_L2(xb)
                sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_2bit_tq)
                sq.train(xb)
                codes = sq.compute_codes(xb)
                xr = sq.decode(codes)
                self.assertEqual(xr.shape, xb.shape)


class TestTurboQFullSearch(unittest.TestCase):
    """Search recall tests for TurboQ full types."""

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
        for sq_str in ["SQtq2", "SQtq3", "SQtq4"]:
            with self.subTest(sq_str=sq_str):
                index = faiss.index_factory(self.d, sq_str)
                index.train(self.xb)
                index.add(self.xb)
                recall = self._recall_at_1(index)
                self.assertGreater(recall, 0.05)

    def test_recall_increases_with_bits(self):
        recalls = {}
        for sq_str in ["SQtq2", "SQtq3", "SQtq4"]:
            index = faiss.index_factory(self.d, sq_str)
            index.train(self.xb)
            index.add(self.xb)
            recalls[sq_str] = self._recall_at_1(index)

        self.assertGreaterEqual(recalls["SQtq3"], recalls["SQtq2"] - 0.05)
        self.assertGreaterEqual(recalls["SQtq4"], recalls["SQtq3"] - 0.05)

    def test_ivf_sqtq_search(self):
        for sq_str in ["SQtq2", "SQtq3"]:
            with self.subTest(sq_str=sq_str):
                factory_str = f"IVF32,{sq_str}"
                index = faiss.index_factory(self.d, factory_str)
                index.train(self.xb)
                index.add(self.xb)
                index.nprobe = 8
                _, I = index.search(self.xq, 10)
                self.assertEqual(I.shape, (self.nq, 10))
                self.assertTrue(np.all(I[:, 0] >= 0))


class TestTurboQFullSerialization(unittest.TestCase):
    """Serialization roundtrip for TurboQ full types."""

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
    """Verify distance computation is consistent for TurboQ."""

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

    def test_search_distances_sorted(self):
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

                D, _ = index.search(xq, 10)
                for q in range(len(xq)):
                    for k in range(1, 10):
                        self.assertLessEqual(D[q, k - 1], D[q, k])
