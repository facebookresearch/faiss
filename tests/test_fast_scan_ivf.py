# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import platform

import numpy as np
import faiss

from faiss.contrib import datasets
from faiss.contrib.inspect_tools import get_invlist


class TestLUTQuantization(unittest.TestCase):

    def compute_dis_float(self, codes, LUT, bias):
        nprobe, nt, M = codes.shape
        dis = np.zeros((nprobe, nt), dtype='float32')
        if bias is not None:
            dis[:] = bias.reshape(-1, 1)

        if LUT.ndim == 2:
            LUTp = LUT

        for p in range(nprobe):
            if LUT.ndim == 3:
                LUTp = LUT[p]

            for i in range(nt):
                dis[p, i] += LUTp[np.arange(M), codes[p, i]].sum()

        return dis

    def compute_dis_quant(self, codes, LUT, bias, a, b):
        nprobe, nt, M = codes.shape
        dis = np.zeros((nprobe, nt), dtype='uint16')
        if bias is not None:
            dis[:] = bias.reshape(-1, 1)

        if LUT.ndim == 2:
            LUTp = LUT

        for p in range(nprobe):
            if LUT.ndim == 3:
                LUTp = LUT[p]

            for i in range(nt):
                dis[p, i] += LUTp[np.arange(M), codes[p, i]].astype('uint16').sum()

        return dis / a + b

    def do_test(self, LUT, bias, nprobe, alt_3d=False):
        M, ksub = LUT.shape[-2:]
        nt = 200

        rs = np.random.RandomState(123)
        codes = rs.randint(ksub, size=(nprobe, nt, M)).astype('uint8')

        dis_ref = self.compute_dis_float(codes, LUT, bias)

        LUTq = np.zeros(LUT.shape, dtype='uint8')
        biasq = (
            np.zeros(bias.shape, dtype='uint16')
            if (bias is not None) and not alt_3d else None
        )
        atab = np.zeros(1, dtype='float32')
        btab = np.zeros(1, dtype='float32')

        def sp(x):
            return faiss.swig_ptr(x) if x is not None else None

        faiss.quantize_LUT_and_bias(
                nprobe, M, ksub, LUT.ndim == 3,
                sp(LUT), sp(bias), sp(LUTq), M, sp(biasq),
                sp(atab), sp(btab)
        )
        a = atab[0]
        b = btab[0]
        dis_new = self.compute_dis_quant(codes, LUTq, biasq, a, b)

        #    print(a, b, dis_ref.sum())
        avg_realtive_error = np.abs(dis_new - dis_ref).sum() / dis_ref.sum()
        # print('a=', a, 'avg_relative_error=', avg_realtive_error)
        self.assertLess(avg_realtive_error, 0.0005)

    def test_no_residual_ip(self):
        ksub = 16
        M = 20
        nprobe = 10
        rs = np.random.RandomState(1234)
        LUT = rs.rand(M, ksub).astype('float32')
        bias = None

        self.do_test(LUT, bias, nprobe)

    def test_by_residual_ip(self):
        ksub = 16
        M = 20
        nprobe = 10
        rs = np.random.RandomState(1234)
        LUT = rs.rand(M, ksub).astype('float32')
        bias = rs.rand(nprobe).astype('float32')
        bias *= 10

        self.do_test(LUT, bias, nprobe)

    def test_by_residual_L2(self):
        ksub = 16
        M = 20
        nprobe = 10
        rs = np.random.RandomState(1234)
        LUT = rs.rand(nprobe, M, ksub).astype('float32')
        bias = rs.rand(nprobe).astype('float32')
        bias *= 10

        self.do_test(LUT, bias, nprobe)

    def test_by_residual_L2_v2(self):
        ksub = 16
        M = 20
        nprobe = 10
        rs = np.random.RandomState(1234)
        LUT = rs.rand(nprobe, M, ksub).astype('float32')
        bias = rs.rand(nprobe).astype('float32')
        bias *= 10

        self.do_test(LUT, bias, nprobe, alt_3d=True)




##########################################################
# Tests for various IndexPQFastScan implementations
##########################################################

def verify_with_draws(testcase, Dref, Iref, Dnew, Inew):
    """ verify a list of results where there are draws in the distances (because
    they are integer). """
    np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)
    # here we have to be careful because of draws
    for i in range(len(Iref)):
        if np.all(Iref[i] == Inew[i]): # easy case
            continue
        # we can deduce nothing about the latest line
        skip_dis = Dref[i, -1]
        for dis in np.unique(Dref):
            if dis == skip_dis: continue
            mask = Dref[i, :] == dis
            testcase.assertEqual(set(Iref[i, mask]), set(Inew[i, mask]))

def three_metrics(Dref, Iref, Dnew, Inew):
    nq = Iref.shape[0]
    recall_at_1 = (Iref[:, 0] == Inew[:, 0]).sum() / nq
    recall_at_10 = (Iref[:, :1] == Inew[:, :10]).sum() / nq
    ninter = 0
    for i in range(nq):
        ninter += len(np.intersect1d(Inew[i], Iref[i]))
    intersection_at_10 = ninter / nq
    return recall_at_1, recall_at_10, intersection_at_10


##########################################################
# Tests for various IndexIVFPQFastScan implementations
##########################################################

class TestIVFImplem1(unittest.TestCase):
    """ Verify implem 1 (search from original invlists)
    against IndexIVFPQ """

    def do_test(self, by_residual, metric_type=faiss.METRIC_L2,
                use_precomputed_table=0):
        ds  = datasets.SyntheticDataset(32, 2000, 5000, 1000)

        index = faiss.index_factory(32, "IVF32,PQ16x4np", metric_type)
        index.use_precomputed_table
        index.use_precomputed_table = use_precomputed_table
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4
        index.by_residual = by_residual
        Da, Ia = index.search(ds.get_queries(), 10)

        index2 = faiss.IndexIVFPQFastScan(index)
        index2.implem = 1
        Db, Ib = index2.search(ds.get_queries(), 10)
        # self.assertLess((Ia != Ib).sum(), Ia.size * 0.005)
        np.testing.assert_array_equal(Ia, Ib)
        np.testing.assert_almost_equal(Da, Db, decimal=5)

    def test_no_residual(self):
        self.do_test(False)

    def test_by_residual(self):
        self.do_test(True)

    def test_by_residual_no_precomputed(self):
        self.do_test(True, use_precomputed_table=-1)

    def test_no_residual_ip(self):
        self.do_test(False, faiss.METRIC_INNER_PRODUCT)

    def test_by_residual_ip(self):
        self.do_test(True, faiss.METRIC_INNER_PRODUCT)



class TestIVFImplem2(unittest.TestCase):
    """ Verify implem 2 (search with original invlists with uint8 LUTs)
    against IndexIVFPQ. Entails some loss in accuracy. """

    def eval_quant_loss(self, by_residual, metric=faiss.METRIC_L2):
        ds  = datasets.SyntheticDataset(32, 2000, 5000, 1000)

        index = faiss.index_factory(32, "IVF32,PQ16x4np", metric)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4
        index.by_residual = by_residual
        Da, Ia = index.search(ds.get_queries(), 10)

        # loss due to int8 quantization of LUTs
        index2 = faiss.IndexIVFPQFastScan(index)
        index2.implem = 2
        Db, Ib = index2.search(ds.get_queries(), 10)

        m3 = three_metrics(Da, Ia, Db, Ib)


        # print(by_residual, metric, recall_at_1, recall_at_10, intersection_at_10)
        ref_results = {
            (True, 1): [0.985, 1.0, 9.872],
            (True, 0): [ 0.987, 1.0, 9.914],
            (False, 1): [0.991, 1.0, 9.907],
            (False, 0): [0.986, 1.0, 9.917],
        }

        ref = ref_results[(by_residual, metric)]

        self.assertGreaterEqual(m3[0], ref[0] * 0.995)
        self.assertGreaterEqual(m3[1], ref[1] * 0.995)
        self.assertGreaterEqual(m3[2], ref[2] * 0.995)


    def test_qloss_no_residual(self):
        self.eval_quant_loss(False)

    def test_qloss_by_residual(self):
        self.eval_quant_loss(True)

    def test_qloss_no_residual_ip(self):
        self.eval_quant_loss(False, faiss.METRIC_INNER_PRODUCT)

    def test_qloss_by_residual_ip(self):
        self.eval_quant_loss(True, faiss.METRIC_INNER_PRODUCT)

class TestEquivPQ(unittest.TestCase):

    def test_equiv_pq(self):
        ds  = datasets.SyntheticDataset(32, 2000, 200, 4)

        index = faiss.index_factory(32, "IVF1,PQ16x4np")
        index.by_residual = False
        # force coarse quantizer
        index.quantizer.add(np.zeros((1, 32), dtype='float32'))
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 4)

        index_pq = faiss.index_factory(32, "PQ16x4np")
        index_pq.pq = index.pq
        index_pq.is_trained = True
        index_pq.codes = faiss. downcast_InvertedLists(
            index.invlists).codes.at(0)
        index_pq.ntotal = index.ntotal
        Dnew, Inew = index_pq.search(ds.get_queries(), 4)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

        index_pq2 = faiss.IndexPQFastScan(index_pq)
        index_pq2.implem = 12
        Dref, Iref = index_pq2.search(ds.get_queries(), 4)

        index2 = faiss.IndexIVFPQFastScan(index)
        index2.implem = 12
        Dnew, Inew = index2.search(ds.get_queries(), 4)
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)


class TestIVFImplem12(unittest.TestCase):

    IMPLEM = 12

    def do_test(self, by_residual, metric=faiss.METRIC_L2, d=32):
        ds = datasets.SyntheticDataset(d, 2000, 5000, 200)

        index = faiss.index_factory(d, f"IVF32,PQ{d//2}x4np", metric)
        # force coarse quantizer
        # index.quantizer.add(np.zeros((1, 32), dtype='float32'))
        index.by_residual = by_residual
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4

        index2 = faiss.IndexIVFPQFastScan(index)
        index2.implem = 2
        Dref, Iref = index2.search(ds.get_queries(), 4)
        index2 = faiss.IndexIVFPQFastScan(index)
        index2.implem = self.IMPLEM
        Dnew, Inew = index2.search(ds.get_queries(), 4)

        verify_with_draws(self, Dref, Iref, Dnew, Inew)

        stats = faiss.cvar.indexIVF_stats
        stats.reset()

        # also verify with single result
        Dnew, Inew = index2.search(ds.get_queries(), 1)
        for q in range(len(Dref)):
            if Dref[q, 1] == Dref[q, 0]:
                # then we cannot conclude
                continue
            self.assertEqual(Iref[q, 0], Inew[q, 0])
            np.testing.assert_almost_equal(Dref[q, 0], Dnew[q, 0], decimal=5)

        self.assertGreater(stats.ndis, 0)

    def test_no_residual(self):
        self.do_test(False)

    def test_by_residual(self):
        self.do_test(True)

    def test_no_residual_ip(self):
        self.do_test(False, metric=faiss.METRIC_INNER_PRODUCT)

    def test_by_residual_ip(self):
        self.do_test(True, metric=faiss.METRIC_INNER_PRODUCT)

    def test_no_residual_odd_dim(self):
        self.do_test(False, d=30)

    def test_by_residual_odd_dim(self):
        self.do_test(True, d=30)


class TestIVFImplem10(TestIVFImplem12):
    IMPLEM = 10


class TestIVFImplem11(TestIVFImplem12):
    IMPLEM = 11

class TestIVFImplem13(TestIVFImplem12):
    IMPLEM = 13


class TestAdd(unittest.TestCase):

    def do_test(self, by_residual=False, metric=faiss.METRIC_L2, d=32, bbs=32):
        bbs = 32
        ds = datasets.SyntheticDataset(d, 2000, 5000, 200)

        index = faiss.index_factory(d, f"IVF32,PQ{d//2}x4np", metric)
        index.by_residual = by_residual
        index.train(ds.get_train())
        index.nprobe = 4

        xb = ds.get_database()
        index.add(xb[:1235])

        index2 = faiss.IndexIVFPQFastScan(index, bbs)

        index.add(xb[1235:])
        index3 = faiss.IndexIVFPQFastScan(index, bbs)
        Dref, Iref = index3.search(ds.get_queries(), 10)

        index2.add(xb[1235:])
        Dnew, Inew = index2.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(Dref, Dnew)
        np.testing.assert_array_equal(Iref, Inew)

        # direct verification of code content. Not sure the test is correct
        # if codes are shuffled.
        for list_no in range(32):
            ref_ids, ref_codes = get_invlist(index3.invlists, list_no)
            new_ids, new_codes = get_invlist(index2.invlists, list_no)
            self.assertEqual(set(ref_ids), set(new_ids))
            new_code_per_id = {
                new_ids[i]: new_codes[i // bbs, :, i % bbs]
                for i in range(new_ids.size)
            }
            for i, the_id in enumerate(ref_ids):
                ref_code_i = ref_codes[i // bbs, :, i % bbs]
                new_code_i = new_code_per_id[the_id]
                np.testing.assert_array_equal(ref_code_i, new_code_i)


    def test_add(self):
        self.do_test()

    def test_odd_d(self):
        self.do_test(d=30)

    def test_bbs64(self):
        self.do_test(bbs=64)


class TestTraining(unittest.TestCase):

    def do_test(self, by_residual=False, metric=faiss.METRIC_L2, d=32, bbs=32):
        bbs = 32
        ds = datasets.SyntheticDataset(d, 2000, 5000, 200)

        index = faiss.index_factory(d, f"IVF32,PQ{d//2}x4np", metric)
        index.by_residual = by_residual
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4
        Dref, Iref = index.search(ds.get_queries(), 10)

        index2 = faiss.IndexIVFPQFastScan(
            index.quantizer, d, 32, d // 2, 4, metric, bbs)
        index2.by_residual = by_residual
        index2.train(ds.get_train())

        index2.add(ds.get_database())
        index2.nprobe = 4
        Dnew, Inew = index2.search(ds.get_queries(), 10)

        m3 = three_metrics(Dref, Iref, Dnew, Inew)
        #   print((by_residual, metric, d), ":", m3)
        ref_m3_tab = {
            (True, 1, 32) : (0.995, 1.0, 9.91),
            (True, 0, 32) : (0.99, 1.0, 9.91),
            (True, 1, 30) : (0.99, 1.0, 9.885),
            (False, 1, 32) : (0.99, 1.0, 9.875),
            (False, 0, 32) : (0.99, 1.0, 9.92),
            (False, 1, 30) : (1.0, 1.0, 9.895)
        }
        ref_m3 = ref_m3_tab[(by_residual, metric, d)]
        self.assertGreater(m3[0], ref_m3[0] * 0.99)
        self.assertGreater(m3[1], ref_m3[1] * 0.99)
        self.assertGreater(m3[2], ref_m3[2] * 0.99)

        # Test I/O
        data = faiss.serialize_index(index2)
        index3 = faiss.deserialize_index(data)
        D3, I3 = index3.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(I3, Inew)
        np.testing.assert_array_equal(D3, Dnew)

    def test_no_residual(self):
        self.do_test(by_residual=False)

    def test_by_residual(self):
        self.do_test(by_residual=True)

    def test_no_residual_ip(self):
        self.do_test(by_residual=False, metric=faiss.METRIC_INNER_PRODUCT)

    def test_by_residual_ip(self):
        self.do_test(by_residual=True, metric=faiss.METRIC_INNER_PRODUCT)

    def test_no_residual_odd_dim(self):
        self.do_test(by_residual=False, d=30)

    def test_by_residual_odd_dim(self):
        self.do_test(by_residual=True, d=30)
