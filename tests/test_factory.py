# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import unittest
import gc
import faiss

from faiss.contrib import factory_tools
from faiss.contrib import datasets


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

    def test_factory_5(self):
        index = faiss.index_factory(128, "OPQ16,Flat")
        assert index.sa_code_size() == 128 * 4
        index = faiss.index_factory(128, "OPQ16_64,Flat")
        assert index.sa_code_size() == 64 * 4
        assert index.chain.at(0).d_out == 64

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
        index = faiss.index_factory(12, "HNSW32,SQ8",
                                    faiss.METRIC_INNER_PRODUCT)
        assert index.storage.sa_code_size() == 12
        assert index.metric_type == faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(12, "HNSW,PQ4")
        assert index.storage.sa_code_size() == 4
        self.assertEqual(index.hnsw.nb_neighbors(1), 32)
        index = faiss.index_factory(12, "HNSW32,PQ4np")
        indexpq = faiss.downcast_index(index.storage)
        assert not indexpq.do_polysemous_training
        index = faiss.index_factory(12, "HNSW32,PQ4x12np")
        indexpq = faiss.downcast_index(index.storage)
        self.assertEqual(indexpq.pq.nbits, 12)

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

        index = faiss.index_factory(12, "NSG64,PQ3x10")
        assert isinstance(index, faiss.IndexNSGPQ)
        assert index.nsg.R == 64
        indexpq = faiss.downcast_index(index.storage)
        self.assertEqual(indexpq.pq.nbits, 10)

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

    def test_factory_lsh(self):
        index = faiss.index_factory(128, 'LSHrt')
        self.assertEqual(index.nbits, 128)
        index = faiss.index_factory(128, 'LSH16rt')
        self.assertEqual(index.nbits, 16)

    def test_factory_fast_scan(self):
        index = faiss.index_factory(56, "PQ28x4fs")
        self.assertEqual(index.bbs, 32)
        self.assertEqual(index.pq.nbits, 4)
        index = faiss.index_factory(56, "PQ28x4fs_64")
        self.assertEqual(index.bbs, 64)
        index = faiss.index_factory(56, "IVF50,PQ28x4fs_64",
                                    faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index.bbs, 64)
        self.assertEqual(index.nlist, 50)
        self.assertTrue(index.cp.spherical)
        index = faiss.index_factory(56, "IVF50,PQ28x4fsr_64")
        self.assertTrue(index.by_residual)
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


class TestVTDowncast(unittest.TestCase):

    def test_itq_transform(self):
        codec = faiss.index_factory(16, "ITQ8,LSHt")

        itqt = faiss.downcast_VectorTransform(codec.chain.at(0))
        itqt.pca_then_itq


# tests after re-factoring
class TestFactoryV2(unittest.TestCase):

    def test_refine(self):
        index = faiss.index_factory(123, "Flat,RFlat")
        index.k_factor

    def test_refine_2(self):
        index = faiss.index_factory(123, "LSHrt,Refine(Flat)")
        index1 = faiss.downcast_index(index.base_index)
        self.assertTrue(index1.rotate_data)
        self.assertTrue(index1.train_thresholds)

    def test_pre_transform(self):
        index = faiss.index_factory(123, "PCAR100,L2Norm,PCAW50,LSHr")
        self.assertTrue(index.chain.size() == 3)

    def test_ivf(self):
        index = faiss.index_factory(123, "IVF456,Flat")
        self.assertEqual(index.__class__, faiss.IndexIVFFlat)

    def test_ivf_suffix_k(self):
        index = faiss.index_factory(123, "IVF3k,Flat")
        self.assertEqual(index.nlist, 3072)

    def test_ivf_suffix_M(self):
        index = faiss.index_factory(123, "IVF1M,Flat")
        self.assertEqual(index.nlist, 1024 * 1024)

    def test_ivf_suffix_HNSW_M(self):
        index = faiss.index_factory(123, "IVF1M_HNSW,Flat")
        self.assertEqual(index.nlist, 1024 * 1024)

    def test_idmap(self):
        index = faiss.index_factory(123, "Flat,IDMap")
        self.assertEqual(index.__class__, faiss.IndexIDMap)

    def test_idmap2_suffix(self):
        index = faiss.index_factory(123, "Flat,IDMap2")
        index = faiss.downcast_index(index)
        self.assertEqual(index.__class__, faiss.IndexIDMap2)

    def test_idmap2_prefix(self):
        index = faiss.index_factory(123, "IDMap2,Flat")
        index = faiss.downcast_index(index)
        self.assertEqual(index.__class__, faiss.IndexIDMap2)

    def test_idmap_refine(self):
        index = faiss.index_factory(8, "IDMap,PQ4x4fs,RFlat")
        self.assertEqual(index.__class__, faiss.IndexIDMap)
        refine_index = faiss.downcast_index(index.index)
        self.assertEqual(refine_index.__class__, faiss.IndexRefineFlat)
        base_index = faiss.downcast_index(refine_index.base_index)
        self.assertEqual(base_index.__class__, faiss.IndexPQFastScan)

        # Index now works with add_with_ids, but not with add
        index.train(np.zeros((16, 8)))
        index.add_with_ids(np.zeros((16, 8)), np.arange(16))
        self.assertRaises(RuntimeError, index.add, np.zeros((16, 8)))

    def test_ivf_hnsw(self):
        index = faiss.index_factory(123, "IVF100_HNSW,Flat")
        quantizer = faiss.downcast_index(index.quantizer)
        self.assertEqual(quantizer.hnsw.nb_neighbors(1), 32)

    def test_ivf_parent(self):
        index = faiss.index_factory(123, "IVF100(LSHr),Flat")
        quantizer = faiss.downcast_index(index.quantizer)
        self.assertEqual(quantizer.__class__, faiss.IndexLSH)


class TestAdditive(unittest.TestCase):

    def test_rcq(self):
        index = faiss.index_factory(12, "IVF256(RCQ2x4),RQ3x4")
        self.assertEqual(
            faiss.downcast_index(index.quantizer).__class__,
            faiss.ResidualCoarseQuantizer
        )

    def test_rq3(self):
        index = faiss.index_factory(5, "RQ2x16_3x8_6x4")

        np.testing.assert_array_equal(
            faiss.vector_to_array(index.rq.nbits),
            np.array([16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4])
        )

    def test_norm(self):
        index = faiss.index_factory(5, "RQ8x8_Nqint8")
        self.assertEqual(
            index.rq.search_type,
            faiss.AdditiveQuantizer.ST_norm_qint8)


class TestSpectralHash(unittest.TestCase):

    def test_sh(self):
        index = faiss.index_factory(123, "IVF256,ITQ64,SH1.2")
        self.assertEqual(index.__class__, faiss.IndexIVFSpectralHash)


class TestQuantizerClone(unittest.TestCase):

    def test_clone(self):
        ds = datasets.SyntheticDataset(32, 200, 10, 0)

        quant = faiss.ScalarQuantizer(32, faiss.ScalarQuantizer.QT_4bit)
        quant.train(ds.get_train())

        codes = quant.compute_codes(ds.get_database())

        quant2 = faiss.clone_Quantizer(quant)
        self.assertTrue(quant2.this.own())

        # make sure typemap works
        self.assertEqual(quant2.__class__, faiss.ScalarQuantizer)

        codes2 = quant2.compute_codes(ds.get_database())
        np.testing.assert_array_equal(codes, codes2)


class TestIVFSpectralHashOwnership(unittest.TestCase):

    def test_constructor(self):
        index = faiss.IndexIVFSpectralHash(faiss.IndexFlat(10), 10, 20, 10, 1)
        gc.collect()
        index.quantizer.ntotal   # this should not crash

    def test_replace_vt(self):
        index = faiss.IndexIVFSpectralHash(faiss.IndexFlat(10), 10, 20, 10, 1)
        index.replace_vt(faiss.ITQTransform(10, 10))
        gc.collect()
        index.vt.d_out  # this should not crash
