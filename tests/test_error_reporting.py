# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This script tests a few failure cases of Faiss and whether they are handled
properly."""

import numpy as np
import unittest
import faiss

from common_faiss_tests import get_dataset_2
from faiss.contrib.datasets import SyntheticDataset


class TestValidIndexParams(unittest.TestCase):

    def test_IndexIVFPQ(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)

        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, 32, 8, 8)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.train(xt)
        index.add(xb)

        # invalid nprobe
        index.nprobe = 0
        k = 10
        self.assertRaises(RuntimeError, index.search, xq, k)

        # invalid k
        index.nprobe = 4
        k = -10
        self.assertRaises(AssertionError, index.search, xq, k)

        # valid params
        index.nprobe = 4
        k = 10
        D, nns = index.search(xq, k)

        self.assertEqual(D.shape[0], nq)
        self.assertEqual(D.shape[1], k)

    def test_IndexFlat(self):
        d = 32
        nb = 1000
        nt = 0
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)
        index = faiss.IndexFlat(d, faiss.METRIC_L2)

        index.add(xb)

        # invalid k
        k = -5
        self.assertRaises(AssertionError, index.search, xq, k)

        # valid k
        k = 5
        D, I = index.search(xq, k)

        self.assertEqual(D.shape[0], nq)
        self.assertEqual(D.shape[1], k)


class TestReconsException(unittest.TestCase):

    def test_recons_exception(self):

        d = 64                           # dimension
        nb = 1000
        rs = np.random.RandomState(1234)
        xb = rs.rand(nb, d).astype('float32')
        nlist = 10
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(xb)
        index.add(xb)
        index.make_direct_map()

        index.reconstruct(9)

        self.assertRaises(
            RuntimeError,
            index.reconstruct, 100001
        )

    def test_reconstuct_after_add(self):
        index = faiss.index_factory(10, 'IVF5,SQfp16')
        index.train(faiss.randn((100, 10), 123))
        index.add(faiss.randn((100, 10), 345))
        index.make_direct_map()
        index.add(faiss.randn((100, 10), 678))

        # should not raise an exception
        index.reconstruct(5)
        print(index.ntotal)
        index.reconstruct(150)


class TestNaN(unittest.TestCase):
    """ NaN values handling is transparent: they don't produce results
    but should not crash. The tests below cover a few common index types.
    """

    def do_test_train(self, factory_string):
        """ NaN and Inf should raise an exception at train time """
        ds = SyntheticDataset(32, 200, 20, 10)
        index = faiss.index_factory(ds.d, factory_string)
        # try to train with NaNs
        xt = ds.get_train().copy()
        xt[:, ::4] = np.nan
        self.assertRaises(RuntimeError, index.train, xt)

    def test_train_IVFSQ(self):
        self.do_test_train("IVF10,SQ8")

    def test_train_IVFPQ(self):
        self.do_test_train("IVF10,PQ4np")

    def test_train_SQ(self):
        self.do_test_train("SQ8")

    def do_test_add(self, factory_string):
        """ stored NaNs should not be returned at search time """
        ds = SyntheticDataset(32, 200, 20, 10)
        index = faiss.index_factory(ds.d, factory_string)
        if not index.is_trained:
            index.train(ds.get_train())
        xb = ds.get_database()
        xb[12, 3] = np.nan
        index.add(xb)
        D, I = index.search(ds.get_queries(), 20)
        self.assertTrue(np.where(I == 12)[0].size == 0)

    def test_add_Flat(self):
        self.do_test_add("Flat")

    def test_add_HNSW(self):
        self.do_test_add("HNSW32,Flat")

    def xx_test_add_SQ8(self):
        # this is expected to fail because:
        # in ASAN mode, the float NaN -> int conversion crashes
        # in opt mode it works but there is no way to encode the NaN,
        # so the value cannot be ignored.
        self.do_test_add("SQ8")

    def test_add_IVFFlat(self):
        self.do_test_add("IVF10,Flat")

    def do_test_search(self, factory_string):
        """ NaN query vectors should return -1 """
        ds = SyntheticDataset(32, 200, 20, 10)
        index = faiss.index_factory(ds.d, factory_string)
        if not index.is_trained:
            index.train(ds.get_train())
        index.add(ds.get_database())
        xq = ds.get_queries()
        xq[7, 3] = np.nan
        D, I = index.search(ds.get_queries(), 20)
        self.assertTrue(np.all(I[7] == -1))

    def test_search_Flat(self):
        self.do_test_search("Flat")

    def test_search_HNSW(self):
        self.do_test_search("HNSW32,Flat")

    def test_search_IVFFlat(self):
        self.do_test_search("IVF10,Flat")

    def test_search_SQ(self):
        self.do_test_search("SQ8")
