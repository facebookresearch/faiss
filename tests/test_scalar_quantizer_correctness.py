# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

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
        self.do_encode_decode(faiss.ScalarQuantizer.QT_4bit_tqmse, 0.1)

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
