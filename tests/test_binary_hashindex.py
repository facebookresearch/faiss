# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import faiss

from common_faiss_tests import make_binary_dataset


def bitvec_shuffle(a, order):
    n, d = a.shape
    db, = order.shape
    b = np.empty((n, db // 8), dtype='uint8')
    faiss.bitvec_shuffle(
        n, d * 8, db,
        faiss.swig_ptr(order),
        faiss.swig_ptr(a), faiss.swig_ptr(b))
    return b


class TestSmallFuncs(unittest.TestCase):

    def test_shuffle(self):
        d = 256
        n = 1000
        rs = np.random.RandomState(123)
        o = rs.permutation(d).astype('int32')

        x = rs.randint(256, size=(n, d // 8)).astype('uint8')

        y1 = bitvec_shuffle(x, o[:128])
        y2 = bitvec_shuffle(x, o[128:])
        y = np.hstack((y1, y2))

        oinv = np.empty(d, dtype='int32')
        oinv[o] = np.arange(d)
        z = bitvec_shuffle(y, oinv)

        np.testing.assert_array_equal(x, z)


class TestRange(unittest.TestCase):

    def test_hash(self):
        d = 128
        nq = 100
        nb = 2000

        (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)

        index_ref = faiss.IndexBinaryFlat(d)
        index_ref.add(xb)

        radius = 55

        Lref, Dref, Iref = index_ref.range_search(xq, radius)

        index = faiss.IndexBinaryHash(d, 10)
        index.add(xb)
        # index.display()
        nfound = []
        ndis = []
        stats = faiss.cvar.indexBinaryHash_stats
        for n_bitflips in range(index.b + 1):
            index.nflip = n_bitflips
            stats.reset()
            Lnew, Dnew, Inew = index.range_search(xq, radius)
            for i in range(nq):
                ref = Iref[Lref[i]:Lref[i + 1]]
                new = Inew[Lnew[i]:Lnew[i + 1]]
                snew = set(new)
                # no duplicates
                self.assertTrue(len(new) == len(snew))
                # subset of real results
                self.assertTrue(snew <= set(ref))
            nfound.append(Lnew[-1])
            ndis.append(stats.ndis)
        nfound = np.array(nfound)
        self.assertTrue(nfound[-1] == Lref[-1])
        self.assertTrue(np.all(nfound[1:] >= nfound[:-1]))

    def test_multihash(self):
        d = 128
        nq = 100
        nb = 2000

        (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)

        index_ref = faiss.IndexBinaryFlat(d)
        index_ref.add(xb)

        radius = 55

        Lref, Dref, Iref = index_ref.range_search(xq, radius)

        nfound = []
        ndis = []

        for nh in 1, 3, 5:
            index = faiss.IndexBinaryMultiHash(d, nh, 10)
            index.add(xb)
            # index.display()
            stats = faiss.cvar.indexBinaryHash_stats
            index.nflip = 2
            stats.reset()
            Lnew, Dnew, Inew = index.range_search(xq, radius)
            for i in range(nq):
                ref = Iref[Lref[i]:Lref[i + 1]]
                new = Inew[Lnew[i]:Lnew[i + 1]]
                snew = set(new)
                # no duplicates
                self.assertTrue(len(new) == len(snew))
                # subset of real results
                self.assertTrue(snew <= set(ref))
            nfound.append(Lnew[-1])
            ndis.append(stats.ndis)
        nfound = np.array(nfound)
        # self.assertTrue(nfound[-1] == Lref[-1])
        self.assertTrue(np.all(nfound[1:] >= nfound[:-1]))


class TestKnn(unittest.TestCase):

    def test_hash_and_multihash(self):
        d = 128
        nq = 100
        nb = 2000

        (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)

        index_ref = faiss.IndexBinaryFlat(d)
        index_ref.add(xb)
        k = 10
        Dref, Iref = index_ref.search(xq, k)

        nfound = {}
        for nh in 0, 1, 3, 5:

            for nbit in 4, 7:
                if nh == 0:
                    index = faiss.IndexBinaryHash(d, nbit)
                else:
                    index = faiss.IndexBinaryMultiHash(d, nh, nbit)
                index.add(xb)
                index.nflip = 2
                Dnew, Inew = index.search(xq, k)
                nf = 0
                for i in range(nq):
                    ref = Iref[i]
                    new = Inew[i]
                    snew = set(new)
                    # no duplicates
                    self.assertTrue(len(new) == len(snew))
                    nf += len(set(ref) & snew)
                nfound[(nh, nbit)] = nf
            self.assertGreater(nfound[(nh, 4)], nfound[(nh, 7)])

            # test serialization
            index2 = faiss.deserialize_index_binary(
                faiss.serialize_index_binary(index))

            D2, I2 = index2.search(xq, k)
            np.testing.assert_array_equal(Inew, I2)
            np.testing.assert_array_equal(Dnew, D2)

        self.assertGreater(3, abs(nfound[(0, 7)] - nfound[(1, 7)]))
        self.assertGreater(nfound[(3, 7)], nfound[(1, 7)])
        self.assertGreater(nfound[(5, 7)], nfound[(3, 7)])

    def subtest_result_order(self, nh):

        d = 128
        nq = 10
        nb = 200

        (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)

        nbit = 10
        if nh == 0:
            index = faiss.IndexBinaryHash(d, nbit)
        else:
            index = faiss.IndexBinaryMultiHash(d, nh, nbit)
        index.add(xb)
        index.nflip = 5
        k = 10
        Do, Io = index.search(xq, k)
        self.assertTrue(
            np.all(Do[:, 1:] >= Do[:, :-1])
        )

    def test_result_order_binhash(self):
        self.subtest_result_order(0)

    def test_result_order_miltihash(self):
        self.subtest_result_order(3)




"""
I suspect this test crashes CircleCI on Linux

# this is an expensive test, so we don't run it by default
class TestLargeIndexWrite:   # (unittest.TestCase):

    def test_write_580M(self):
        dim = 8
        nhash = 1
        num_million = 580 # changing to 570 works
        index1 = faiss.IndexBinaryMultiHash(dim, nhash, int(dim/nhash))
        random_hash_codes = np.random.randint(0, 256, (
            num_million * int(1e6), int(dim/8))).astype("uint8")
        index1.add(random_hash_codes)
        faiss.write_index_binary(index1, "/tmp/tmp.faiss")
        index2 = faiss.read_index_binary("/tmp/tmp.faiss")
"""
