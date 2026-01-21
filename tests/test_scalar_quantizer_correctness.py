# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np

import faiss
import unittest


class TestScalarQuantizerEncodeDecode(unittest.TestCase):

    def setUp(self):
        self.d = 32
        self.n = 1000
        np.random.seed(42)
        self.xb = np.random.random((self.n, self.d)).astype('float32')

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


class TestScalarQuantizerSearch(unittest.TestCase):

    def setUp(self):
        self.d = 32
        np.random.seed(42)
        self.xb = np.random.random((10000, self.d)).astype('float32')
        self.xq = np.random.random((100, self.d)).astype('float32')

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


class TestScalarQuantizerDistances(unittest.TestCase):

    def test_distance_matches_reconstruct(self):
        d = 32
        np.random.seed(42)
        x = np.random.random((1, d)).astype('float32')
        xb = np.random.random((100, d)).astype('float32')

        index = faiss.index_factory(d, 'SQ8')
        index.train(xb)
        index.add(xb)

        D, I = index.search(x, 10)

        for i in range(10):
            xr = index.reconstruct(int(I[0, i]))
            dist = ((x[0] - xr) ** 2).sum()
            self.assertAlmostEqual(D[0, i], dist, places=4)


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

    def test_d1(self):
        d = 1
        np.random.seed(42)
        xb = np.random.random((100, d)).astype('float32')
        index = faiss.index_factory(d, 'SQ8')
        index.train(xb)
        index.add(xb)
        D, I = index.search(np.array([[0.5]], dtype='float32'), 10)
        self.assertEqual(D.shape, (1, 10))
        self.assertEqual(I.shape, (1, 10))

    def test_d1024(self):
        d = 1024
        np.random.seed(42)
        xb = np.random.random((100, d)).astype('float32')
        index = faiss.index_factory(d, 'SQ8')
        index.train(xb)
        index.add(xb)
        xq = np.random.random((10, d)).astype('float32')
        D, I = index.search(xq, 10)
        self.assertEqual(D.shape, (10, 10))
        self.assertEqual(I.shape, (10, 10))

    def test_non_simd_dims(self):
        # dimensions not aligned to SIMD width (8/16)
        np.random.seed(42)
        for d in [7, 9, 15, 17, 31, 33, 63, 65]:
            xb = np.random.random((100, d)).astype('float32')
            index = faiss.index_factory(d, 'SQ8')
            index.train(xb)
            index.add(xb)
            xq = np.random.random((5, d)).astype('float32')
            D, I = index.search(xq, 10)
            self.assertEqual(D.shape, (5, 10))
            self.assertEqual(I.shape, (5, 10))


class TestScalarQuantizerIP(unittest.TestCase):

    def test_inner_product(self):
        d = 32
        np.random.seed(42)
        xb = np.random.random((1000, d)).astype('float32')
        xq = np.random.random((10, d)).astype('float32')
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        index = faiss.index_factory(d, 'SQ8', faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
        D, _ = index.search(xq, 10)

        # normalized vectors: max IP should be close to 1
        self.assertTrue(np.all(D[:, 0] <= 1.1))
        self.assertTrue(np.all(D[:, 0] >= 0.8))
