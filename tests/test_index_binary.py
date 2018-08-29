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


def make_binary_dataset(d, nb, nt, nq):
    assert d % 8 == 0
    rs = np.random.RandomState(123)
    x = rs.randint(256, size=(nb + nq + nt, int(d / 8))).astype('uint8')
    return x[:nt], x[nt:-nq], x[-nq:]


def binary_to_float(x):
    n, d = x.shape
    x8 = x.reshape(n * d, -1)
    c8 = 2 * ((x8 >> np.arange(8)) & 1).astype('int8') - 1
    return c8.astype('float32').reshape(n, d * 8)


def binary_dis(x, y):
    return sum(faiss.popcount64(int(xi ^ yi)) for xi, yi in zip(x, y))


class TestBinaryPQ(unittest.TestCase):
    """ Use a PQ that mimicks a binary encoder """

    def test_encode_to_binary(self):
        d = 256
        nt = 256
        nb = 1500
        nq = 500
        (xt, xb, xq) = make_binary_dataset(d, nb, nt, nq)
        pq = faiss.ProductQuantizer(d, int(d / 8), 8)

        centroids = binary_to_float(
            np.tile(np.arange(256), int(d / 8)).astype('uint8').reshape(-1, 1))

        faiss.copy_array_to_vector(centroids.ravel(), pq.centroids)
        pq.is_trained = True

        codes = pq.compute_codes(binary_to_float(xb))

        assert np.all(codes == xb)

        indexpq = faiss.IndexPQ(d, int(d / 8), 8)
        indexpq.pq = pq
        indexpq.is_trained = True

        indexpq.add(binary_to_float(xb))
        D, I = indexpq.search(binary_to_float(xq), 3)

        for i in range(nq):
            for j, dj in zip(I[i], D[i]):
                ref_dis = binary_dis(xq[i], xb[j])
                assert 4 * ref_dis == dj

        nlist = 32
        quantizer = faiss.IndexFlatL2(d)
        # pretext class for training
        iflat = faiss.IndexIVFFlat(quantizer, d, nlist)
        iflat.train(binary_to_float(xt))

        indexivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, int(d / 8), 8)

        indexivfpq.pq = pq
        indexivfpq.is_trained = True
        indexivfpq.by_residual = False

        indexivfpq.add(binary_to_float(xb))
        indexivfpq.nprobe = 4

        D, I = indexivfpq.search(binary_to_float(xq), 3)

        for i in range(nq):
            for j, dj in zip(I[i], D[i]):
                ref_dis = binary_dis(xq[i], xb[j])
                assert 4 * ref_dis == dj


class TestBinaryFlat(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 0
        nb = 1500
        nq = 500

        (_, self.xb, self.xq) = make_binary_dataset(d, nb, nt, nq)

    def test_flat(self):
        d = self.xq.shape[1] * 8
        nq = self.xq.shape[0]

        index = faiss.IndexBinaryFlat(d)
        index.add(self.xb)
        D, I = index.search(self.xq, 3)

        for i in range(nq):
            for j, dj in zip(I[i], D[i]):
                ref_dis = binary_dis(self.xq[i], self.xb[j])
                assert dj == ref_dis


class TestBinaryIVF(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 200
        nb = 1500
        nq = 500

        (self.xt, self.xb, self.xq) = make_binary_dataset(d, nb, nt, nq)
        index = faiss.IndexBinaryFlat(d)
        index.add(self.xb)
        Dref, Iref = index.search(self.xq, 10)
        self.Dref = Dref

    def test_ivf_flat_exhaustive(self):
        d = self.xq.shape[1] * 8

        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, 8)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.nprobe = 8
        index.train(self.xt)
        index.add(self.xb)
        Divfflat, _ = index.search(self.xq, 10)

        np.testing.assert_array_equal(self.Dref, Divfflat)

    def test_ivf_flat2(self):
        d = self.xq.shape[1] * 8

        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, 8)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.nprobe = 4
        index.train(self.xt)
        index.add(self.xb)
        Divfflat, _ = index.search(self.xq, 10)

        self.assertEqual((self.Dref == Divfflat).sum(), 4110)



if __name__ == '__main__':
    unittest.main()
