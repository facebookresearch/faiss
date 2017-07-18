# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

"""this is a basic test script for simple indices work"""

import numpy as np
import unittest
import faiss


class EvalIVFPQAccuracy(unittest.TestCase):

    def get_dataset(self):
        d = 64
        nb = 1000
        nt = 1500
        nq = 200
        np.random.seed(123)
        xb = np.random.random(size=(nb, d)).astype('float32')
        xt = np.random.random(size=(nt, d)).astype('float32')
        xq = np.random.random(size=(nq, d)).astype('float32')

        return (xt, xb, xq)

    def test_IndexIVFPQ(self):
        (xt, xb, xq) = self.get_dataset()
        d = xt.shape[1]

        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, 25, 16, 8)
        index.train(xt)
        index.add(xb)
        index.nprobe = 5
        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()
        nq = xq.shape[0]

        self.assertGreater(n_ok, nq * 0.4)


class TestMultiIndexQuantizer(unittest.TestCase):

    def test_search_k1(self):

        # verify codepath for k = 1 and k > 1

        d = 64
        nt = 1500
        nq = 200
        np.random.seed(123)

        xt = np.random.random(size=(nt, d)).astype('float32')
        xq = np.random.random(size=(nq, d)).astype('float32')

        miq = faiss.MultiIndexQuantizer(d, 2, 6)

        miq.train(xt)

        D1, I1 = miq.search(xq, 1)

        D5, I5 = miq.search(xq, 5)

        self.assertEqual(np.abs(I1[:, :1] - I5[:, :1]).max(), 0)
        self.assertEqual(np.abs(D1[:, :1] - D5[:, :1]).max(), 0)


class TestScalarQuantizer(unittest.TestCase):

    def test_4variants_ivf(self):
        d = 32
        nt = 1500
        nq = 200
        nb = 10000

        np.random.seed(123)

        xt = np.random.random(size=(nt, d)).astype('float32')
        xq = np.random.random(size=(nq, d)).astype('float32')
        xb = np.random.random(size=(nb, d)).astype('float32')

        # common quantizer
        quantizer = faiss.IndexFlatL2(d)

        ncent = 128

        index_gt = faiss.IndexFlatL2(d)
        index_gt.add(xb)
        D, I_ref = index_gt.search(xq, 10)

        nok = {}

        index = faiss.IndexIVFFlat(quantizer, d, ncent,
                                   faiss.METRIC_L2)
        index.nprobe = 4
        index.train(xt)
        index.add(xb)
        D, I = index.search(xq, 10)
        nok['flat'] = (I[:, 0] == I_ref[:, 0]).sum()

        for qname in "QT_4bit QT_4bit_uniform QT_8bit QT_8bit_uniform".split():
            qtype = getattr(faiss.ScalarQuantizer, qname)
            index = faiss.IndexIVFScalarQuantizer(quantizer, d, ncent,
                                                  qtype, faiss.METRIC_L2)

            index.nprobe = 4
            index.train(xt)
            index.add(xb)
            D, I = index.search(xq, 10)

            nok[qname] = (I[:, 0] == I_ref[:, 0]).sum()

        print(nok)

        self.assertGreaterEqual(nok['flat'], nok['QT_8bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_4bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_8bit_uniform'])
        self.assertGreaterEqual(nok['QT_4bit'], nok['QT_4bit_uniform'])

    def test_4variants(self):
        d = 32
        nt = 1500
        nq = 200
        nb = 10000

        np.random.seed(123)

        xt = np.random.random(size=(nt, d)).astype('float32')
        xq = np.random.random(size=(nq, d)).astype('float32')
        xb = np.random.random(size=(nb, d)).astype('float32')

        index_gt = faiss.IndexFlatL2(d)
        index_gt.add(xb)
        D, I_ref = index_gt.search(xq, 10)

        nok = {}

        for qname in "QT_4bit QT_4bit_uniform QT_8bit QT_8bit_uniform".split():
            qtype = getattr(faiss.ScalarQuantizer, qname)
            index = faiss.IndexScalarQuantizer(d, qtype, faiss.METRIC_L2)
            index.train(xt)
            index.add(xb)
            D, I = index.search(xq, 10)

            nok[qname] = (I[:, 0] == I_ref[:, 0]).sum()

        print(nok)

        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_4bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_8bit_uniform'])
        self.assertGreaterEqual(nok['QT_4bit'], nok['QT_4bit_uniform'])



if __name__ == '__main__':
    unittest.main()
