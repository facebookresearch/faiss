# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

"""this is a basic test script for simple indices work"""

import numpy as np
import unittest
import faiss


def get_dataset(d, nb, nt, nq):
    rs = np.random.RandomState(123)
    xb = rs.rand(nb, d).astype('float32')
    xt = rs.rand(nt, d).astype('float32')
    xq = rs.rand(nq, d).astype('float32')

    return (xt, xb, xq)


def get_dataset_2(d, nb, nt, nq):
    """A dataset that is not completely random but still challenging to
    index
    """
    d1 = 10     # intrinsic dimension (more or less)
    n = nb + nt + nq
    rs = np.random.RandomState(1234)
    x = rs.normal(size=(n, d1))
    x = np.dot(x, rs.rand(d1, d))
    # now we have a d1-dim ellipsoid in d-dimensional space
    # higher factor (>4) -> higher frequency -> less linear
    x = x * (rs.rand(d) * 4 + 0.1)
    x = np.sin(x)
    x = x.astype('float32')
    return x[:nt], x[nt:-nq], x[-nq:]


class EvalIVFPQAccuracy(unittest.TestCase):

    def test_IndexIVFPQ(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nb, nt, nq)
        d = xt.shape[1]

        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, 32, 8, 8)
        index.train(xt)
        index.add(xb)
        index.nprobe = 4
        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()
        nq = xq.shape[0]

        self.assertGreater(n_ok, nq * 0.66)


class TestMultiIndexQuantizer(unittest.TestCase):

    def test_search_k1(self):

        # verify codepath for k = 1 and k > 1

        d = 64
        nb = 0
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        miq = faiss.MultiIndexQuantizer(d, 2, 6)

        miq.train(xt)

        D1, I1 = miq.search(xq, 1)

        D5, I5 = miq.search(xq, 5)

        self.assertEqual(np.abs(I1[:, :1] - I5[:, :1]).max(), 0)
        self.assertEqual(np.abs(D1[:, :1] - D5[:, :1]).max(), 0)


class TestScalarQuantizer(unittest.TestCase):

    def test_4variants_ivf(self):
        d = 32
        nt = 2500
        nq = 400
        nb = 5000

        (xt, xb, xq) = get_dataset_2(d, nb, nt, nq)

        # common quantizer
        quantizer = faiss.IndexFlatL2(d)

        ncent = 64

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
        print(nok, nq)
        self.assertGreaterEqual(nok['flat'], nq * 0.6)
        # The tests below are a bit fragile, it happens that the
        # ordering between uniform and non-uniform are reverted,
        # probably because the dataset is small, which introduces
        # jitter
        self.assertGreaterEqual(nok['flat'], nok['QT_8bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_4bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_8bit_uniform'])
        self.assertGreaterEqual(nok['QT_4bit'], nok['QT_4bit_uniform'])

    def test_4variants(self):
        d = 32
        nt = 2500
        nq = 400
        nb = 5000

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index_gt = faiss.IndexFlatL2(d)
        index_gt.add(xb)
        D_ref, I_ref = index_gt.search(xq, 10)

        nok = {}

        for qname in "QT_4bit QT_4bit_uniform QT_8bit QT_8bit_uniform".split():
            qtype = getattr(faiss.ScalarQuantizer, qname)
            index = faiss.IndexScalarQuantizer(d, qtype, faiss.METRIC_L2)
            index.train(xt)
            index.add(xb)

            D, I = index.search(xq, 10)

            nok[qname] = (I[:, 0] == I_ref[:, 0]).sum()

        self.assertGreaterEqual(nok['QT_8bit'], nq * 0.9)
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_4bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_8bit_uniform'])
        self.assertGreaterEqual(nok['QT_4bit'], nok['QT_4bit_uniform'])


class TestRangeSearch(unittest.TestCase):

    def test_range_search(self):
        d = 4
        nt = 100
        nq = 10
        nb = 50

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.IndexFlatL2(d)
        index.add(xb)

        Dref, Iref = index.search(xq, 5)

        thresh = 0.1   # *squared* distance
        lims, D, I = index.range_search(xq, thresh)

        for i in range(nq):
            Iline = I[lims[i]:lims[i + 1]]
            Dline = D[lims[i]:lims[i + 1]]
            for j, dis in zip(Iref[i], Dref[i]):
                if dis < thresh:
                    li, = np.where(Iline == j)
                    self.assertTrue(li.size == 1)
                    idx = li[0]
                    self.assertGreaterEqual(1e-4, abs(Dline[idx] - dis))


if __name__ == '__main__':
    unittest.main()
