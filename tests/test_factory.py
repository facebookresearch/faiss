# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import unittest
import faiss

from faiss.contrib import factory_tools


class TestFactory(unittest.TestCase):

    def test_factory_1(self):

        index = faiss.index_factory(12, "IVF10,PQ4")
        assert index.do_polysemous_training

        index = faiss.index_factory(12, "IVF10,PQ4np")
        assert not index.do_polysemous_training

        index = faiss.index_factory(12, "PQ4")
        assert index.do_polysemous_training

        index = faiss.index_factory(12, "PQ4np")
        assert not index.do_polysemous_training

        try:
            index = faiss.index_factory(10, "PQ4")
        except RuntimeError:
            pass
        else:
            assert False, "should do a runtime error"

    def test_factory_2(self):

        index = faiss.index_factory(12, "SQ8")
        assert index.code_size == 12

    def test_factory_3(self):

        index = faiss.index_factory(12, "IVF10,PQ4")
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", 3)
        assert index.nprobe == 3

        index = faiss.index_factory(12, "PCAR8,IVF10,PQ4")
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", 3)
        assert faiss.downcast_index(index.index).nprobe == 3

    def test_factory_4(self):
        index = faiss.index_factory(12, "IVF10,FlatDedup")
        assert index.instances is not None

    def test_factory_HNSW(self):
        index = faiss.index_factory(12, "HNSW32")
        assert index.storage.sa_code_size() == 12 * 4
        index = faiss.index_factory(12, "HNSW32_SQ8")
        assert index.storage.sa_code_size() == 12
        index = faiss.index_factory(12, "HNSW32_PQ4")
        assert index.storage.sa_code_size() == 4

    def test_factory_HNSW_newstyle(self):
        index = faiss.index_factory(12, "HNSW32,Flat")
        assert index.storage.sa_code_size() == 12 * 4
        index = faiss.index_factory(12, "HNSW32,SQ8", faiss.METRIC_INNER_PRODUCT)
        assert index.storage.sa_code_size() == 12
        assert index.metric_type == faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(12, "HNSW32,PQ4")
        assert index.storage.sa_code_size() == 4
        index = faiss.index_factory(12, "HNSW32,PQ4np")
        indexpq = faiss.downcast_index(index.storage)
        assert not indexpq.do_polysemous_training

    def test_factory_NSG(self):
        index = faiss.index_factory(12, "NSG64")
        assert isinstance(index, faiss.IndexNSGFlat)
        assert index.nsg.R == 64

        index = faiss.index_factory(12, "NSG64", faiss.METRIC_INNER_PRODUCT)
        assert isinstance(index, faiss.IndexNSGFlat)
        assert index.nsg.R == 64
        assert index.metric_type == faiss.METRIC_INNER_PRODUCT

        index = faiss.index_factory(12, "NSG64,Flat")
        assert isinstance(index, faiss.IndexNSGFlat)
        assert index.nsg.R == 64

        index = faiss.index_factory(12, "IVF65536_NSG64,Flat")
        index_nsg = faiss.downcast_index(index.quantizer)
        assert isinstance(index, faiss.IndexIVFFlat)
        assert isinstance(index_nsg, faiss.IndexNSGFlat)
        assert index.nlist == 65536 and index_nsg.nsg.R == 64

        index = faiss.index_factory(12, "IVF65536_NSG64,PQ2x8")
        index_nsg = faiss.downcast_index(index.quantizer)
        assert isinstance(index, faiss.IndexIVFPQ)
        assert isinstance(index_nsg, faiss.IndexNSGFlat)
        assert index.nlist == 65536 and index_nsg.nsg.R == 64
        assert index.pq.M == 2 and index.pq.nbits == 8

    def test_factory_fast_scan(self):
        index = faiss.index_factory(56, "PQ28x4fs")
        self.assertEqual(index.bbs, 32)
        self.assertEqual(index.pq.nbits, 4)
        index = faiss.index_factory(56, "PQ28x4fs_64")
        self.assertEqual(index.bbs, 64)
        index = faiss.index_factory(56, "IVF50,PQ28x4fs_64", faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index.bbs, 64)
        self.assertEqual(index.nlist, 50)
        self.assertTrue(index.cp.spherical)
        index = faiss.index_factory(56, "PQ28x4fs,RFlat")
        self.assertEqual(index.k_factor, 1.0)

    def test_parenthesis(self):
        index = faiss.index_factory(50, "IVF32(PQ25),Flat")
        quantizer = faiss.downcast_index(index.quantizer)
        self.assertEqual(quantizer.pq.M, 25)

    def test_parenthesis_2(self):
        index = faiss.index_factory(50, "PCA30,IVF32(PQ15),Flat")
        index_ivf = faiss.extract_index_ivf(index)
        quantizer = faiss.downcast_index(index_ivf.quantizer)
        self.assertEqual(quantizer.pq.M, 15)
        self.assertEqual(quantizer.d, 30)

    def test_parenthesis_refine(self):
        index = faiss.index_factory(50, "IVF32,Flat,Refine(PQ25x12)")
        rf = faiss.downcast_index(index.refine_index)
        self.assertEqual(rf.pq.M, 25)
        self.assertEqual(rf.pq.nbits, 12)


    def test_parenthesis_refine_2(self):
        # Refine applies on the whole index including pre-transforms
        index = faiss.index_factory(50, "PCA32,IVF32,Flat,Refine(PQ25x12)")
        rf = faiss.downcast_index(index.refine_index)
        self.assertEqual(rf.pq.M, 25)

    def test_nested_parenteses(self):
        index = faiss.index_factory(50, "IVF1000(IVF20,SQ4,Refine(SQ8)),Flat")
        q = faiss.downcast_index(index.quantizer)
        qref = faiss.downcast_index(q.refine_index)
        # check we can access the scalar quantizer
        self.assertEqual(qref.sq.code_size, 50)

    def test_residual(self):
        index = faiss.index_factory(50, "IVF1000,PQ25x4fsr")
        self.assertTrue(index.by_residual)

class TestCodeSize(unittest.TestCase):

    def test_1(self):
        self.assertEqual(
            factory_tools.get_code_size(50, "IVF32,Flat,Refine(PQ25x12)"),
            50 * 4 + (25 * 12 + 7) // 8
        )


class TestCloneSize(unittest.TestCase):

    def test_clone_size(self):
        index = faiss.index_factory(20, 'PCA10,Flat')
        xb = faiss.rand((100, 20))
        index.train(xb)
        index.add(xb)
        index2 = faiss.clone_index(index)
        assert index2.ntotal == 100

class TestCloneIVFPQ(unittest.TestCase):

    def test_clone(self):
        index = faiss.index_factory(16, 'IVF10,PQ4np')
        xb = faiss.rand((1000, 16))
        index.train(xb)
        index.add(xb)
        index2 = faiss.clone_index(index)
        assert index2.ntotal == index.ntotal
