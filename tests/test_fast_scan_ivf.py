# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest
import tempfile

import numpy as np
import faiss

from faiss.contrib import datasets
from faiss.contrib.inspect_tools import get_invlist

# the tests tend to timeout in stress modes + dev otherwise
faiss.omp_set_num_threads(4)

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

        avg_realtive_error = np.abs(dis_new - dis_ref).sum() / dis_ref.sum()
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
        xq = ds.get_queries()

        index = faiss.index_factory(32, "IVF1,PQ16x4np")
        index.by_residual = False
        # force coarse quantizer
        index.quantizer.add(np.zeros((1, 32), dtype='float32'))
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(xq, 4)

        index_pq = faiss.index_factory(32, "PQ16x4np")
        index_pq.pq = index.pq
        index_pq.is_trained = True
        index_pq.codes = faiss. downcast_InvertedLists(
            index.invlists).codes.at(0)
        index_pq.ntotal = index.ntotal
        Dnew, Inew = index_pq.search(xq, 4)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

        index_pq2 = faiss.IndexPQFastScan(index_pq)
        index_pq2.implem = 12
        Dref, Iref = index_pq2.search(xq, 4)

        index2 = faiss.IndexIVFPQFastScan(index)
        index2.implem = 12
        Dnew, Inew = index2.search(xq, 4)
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

        # test encode and decode

        np.testing.assert_array_equal(
            index_pq.sa_encode(xq),
            index2.sa_encode(xq)
        )

        np.testing.assert_array_equal(
            index_pq.sa_decode(index_pq.sa_encode(xq)),
            index2.sa_decode(index2.sa_encode(xq))
        )

        np.testing.assert_array_equal(
            ((index_pq.sa_decode(index_pq.sa_encode(xq)) - xq) ** 2).sum(1),
            ((index2.sa_decode(index2.sa_encode(xq)) - xq) ** 2).sum(1)
        )

    def test_equiv_pq_encode_decode(self):
        ds = datasets.SyntheticDataset(32, 1000, 200, 10)
        xq = ds.get_queries()

        index_ivfpq = faiss.index_factory(ds.d, "IVF10,PQ8x4np")
        index_ivfpq.train(ds.get_train())

        index_ivfpqfs = faiss.IndexIVFPQFastScan(index_ivfpq)

        np.testing.assert_array_equal(
            index_ivfpq.sa_encode(xq),
            index_ivfpqfs.sa_encode(xq)
        )

        np.testing.assert_array_equal(
            index_ivfpq.sa_decode(index_ivfpq.sa_encode(xq)),
            index_ivfpqfs.sa_decode(index_ivfpqfs.sa_encode(xq))
        )

        np.testing.assert_array_equal(
            ((index_ivfpq.sa_decode(index_ivfpq.sa_encode(xq)) - xq) ** 2)
            .sum(1),
            ((index_ivfpqfs.sa_decode(index_ivfpqfs.sa_encode(xq)) - xq) ** 2)
            .sum(1)
        )


class TestIVFImplem12(unittest.TestCase):

    IMPLEM = 12

    def do_test(self, by_residual, metric=faiss.METRIC_L2, d=32, nq=200):
        ds = datasets.SyntheticDataset(d, 2000, 5000, nq)

        index = faiss.index_factory(d, f"IVF32,PQ{d//2}x4np", metric)
        # force coarse quantizer
        # index.quantizer.add(np.zeros((1, 32), dtype='float32'))
        index.by_residual = by_residual
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4

        # compare against implem = 2, which includes quantized LUTs
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

    # testin single query
    def test_no_residual_single_query(self):
        self.do_test(False, nq=1)

    def test_by_residual_single_query(self):
        self.do_test(True, nq=1)

    def test_no_residual_ip_single_query(self):
        self.do_test(False, metric=faiss.METRIC_INNER_PRODUCT, nq=1)

    def test_by_residual_ip_single_query(self):
        self.do_test(True, metric=faiss.METRIC_INNER_PRODUCT, nq=1)

    def test_no_residual_odd_dim_single_query(self):
        self.do_test(False, d=30, nq=1)

    def test_by_residual_odd_dim_single_query(self):
        self.do_test(True, d=30, nq=1)


class TestIVFImplem10(TestIVFImplem12):
    IMPLEM = 10


class TestIVFImplem11(TestIVFImplem12):
    IMPLEM = 11


class TestIVFImplem13(TestIVFImplem12):
    IMPLEM = 13


class TestIVFImplem14(TestIVFImplem12):
    IMPLEM = 14


class TestIVFImplem15(TestIVFImplem12):
    IMPLEM = 15


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
        ref_m3_tab = {
            (True, 1, 32): (0.995, 1.0, 9.91),
            (True, 0, 32): (0.99, 1.0, 9.91),
            (True, 1, 30): (0.989, 1.0, 9.885),
            (False, 1, 32): (0.99, 1.0, 9.875),
            (False, 0, 32): (0.99, 1.0, 9.92),
            (False, 1, 30): (1.0, 1.0, 9.895)
        }
        ref_m3 = ref_m3_tab[(by_residual, metric, d)]
        self.assertGreaterEqual(m3[0], ref_m3[0] * 0.99)
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


class TestIsTrained(unittest.TestCase):

    def test_issue_2019(self):
        index = faiss.index_factory(
            32,
            "PCAR16,IVF200(IVF10,PQ2x4fs,RFlat),PQ4x4fsr"
        )
        des = faiss.rand((1000, 32))
        index.train(des)


class TestIVFAQFastScan(unittest.TestCase):

    def subtest_accuracy(self, aq, st, by_residual, implem, metric_type='L2'):
        """
        Compare IndexIVFAdditiveQuantizerFastScan with
        IndexIVFAdditiveQuantizer
        """
        nlist, d = 16, 8
        ds = datasets.SyntheticDataset(d, 1000, 1000, 500, metric_type)
        gt = ds.get_groundtruth(k=1)

        if metric_type == 'L2':
            metric = faiss.METRIC_L2
            postfix1 = '_Nqint8'
            postfix2 = f'_N{st}2x4'
        else:
            metric = faiss.METRIC_INNER_PRODUCT
            postfix1 = postfix2 = ''

        index = faiss.index_factory(d, f'IVF{nlist},{aq}3x4{postfix1}', metric)
        index.by_residual = by_residual
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 16
        Dref, Iref = index.search(ds.get_queries(), 1)

        indexfs = faiss.index_factory(
            d, f'IVF{nlist},{aq}3x4fs_32{postfix2}', metric)
        indexfs.by_residual = by_residual
        indexfs.train(ds.get_train())
        indexfs.add(ds.get_database())
        indexfs.nprobe = 16
        indexfs.implem = implem
        D1, I1 = indexfs.search(ds.get_queries(), 1)

        nq = Iref.shape[0]
        recall_ref = (Iref == gt).sum() / nq
        recall1 = (I1 == gt).sum() / nq

        assert abs(recall_ref - recall1) < 0.051

    def xx_test_accuracy(self):
        # generated programatically below
        for metric in 'L2', 'IP':
            for byr in True, False:
                for implem in 0, 10, 11, 12, 13, 14, 15:
                    self.subtest_accuracy('RQ', 'rq', byr, implem, metric)
                    self.subtest_accuracy('LSQ', 'lsq', byr, implem, metric)

    def subtest_rescale_accuracy(self, aq, st, by_residual, implem):
        """
        we set norm_scale to 2 and compare it with IndexIVFAQ
        """
        nlist, d = 16, 8
        ds = datasets.SyntheticDataset(d, 1000, 1000, 500)
        gt = ds.get_groundtruth(k=1)

        metric = faiss.METRIC_L2
        postfix1 = '_Nqint8'
        postfix2 = f'_N{st}2x4'

        index = faiss.index_factory(
            d, f'IVF{nlist},{aq}3x4{postfix1}', metric)
        index.by_residual = by_residual
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 16
        Dref, Iref = index.search(ds.get_queries(), 1)

        indexfs = faiss.index_factory(
            d, f'IVF{nlist},{aq}3x4fs_32{postfix2}', metric)
        indexfs.by_residual = by_residual
        indexfs.norm_scale = 2
        indexfs.train(ds.get_train())
        indexfs.add(ds.get_database())
        indexfs.nprobe = 16
        indexfs.implem = implem
        D1, I1 = indexfs.search(ds.get_queries(), 1)

        nq = Iref.shape[0]
        recall_ref = (Iref == gt).sum() / nq
        recall1 = (I1 == gt).sum() / nq

        assert abs(recall_ref - recall1) < 0.05

    def xx_test_rescale_accuracy(self):
        for byr in True, False:
            for implem in 0, 10, 11, 12, 13, 14, 15:
                self.subtest_accuracy('RQ', 'rq', byr, implem, 'L2')
                self.subtest_accuracy('LSQ', 'lsq', byr, implem, 'L2')

    def subtest_from_ivfaq(self, implem):
        d = 8
        ds = datasets.SyntheticDataset(d, 1000, 2000, 1000, metric='IP')
        gt = ds.get_groundtruth(k=1)
        index = faiss.index_factory(d, 'IVF16,RQ8x4', faiss.METRIC_INNER_PRODUCT)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 16
        Dref, Iref = index.search(ds.get_queries(), 1)

        indexfs = faiss.IndexIVFAdditiveQuantizerFastScan(index)
        D1, I1 = indexfs.search(ds.get_queries(), 1)

        nq = Iref.shape[0]
        recall_ref = (Iref == gt).sum() / nq
        recall1 = (I1 == gt).sum() / nq
        assert abs(recall_ref - recall1) < 0.02

    def test_from_ivfaq(self):
        for implem in 0, 1, 2:
            self.subtest_from_ivfaq(implem)

    def subtest_factory(self, aq, M, bbs, st, r='r'):
        """
        Format: IVF{nlist},{AQ}{M}x4fs{r}_{bbs}_N{st}

            nlist (int): number of inverted lists
            AQ (str):    `LSQ` or `RQ`
            M (int):     number of sub-quantizers
            bbs (int):   build block size
            st (str):    search type, `lsq2x4` or `rq2x4`
            r  (str):    `r` or ``, by_residual or not
        """
        AQ = faiss.AdditiveQuantizer
        nlist, d = 128, 16

        if bbs > 0:
            index = faiss.index_factory(
                d, f'IVF{nlist},{aq}{M}x4fs{r}_{bbs}_N{st}2x4')
        else:
            index = faiss.index_factory(
                d, f'IVF{nlist},{aq}{M}x4fs{r}_N{st}2x4')
            bbs = 32

        assert index.nlist == nlist
        assert index.bbs == bbs
        q = faiss.downcast_Quantizer(index.aq)
        assert q.M == M

        if aq == 'LSQ':
            assert isinstance(q, faiss.LocalSearchQuantizer)
        if aq == 'RQ':
            assert isinstance(q, faiss.ResidualQuantizer)

        if st == 'lsq':
            assert q.search_type == AQ.ST_norm_lsq2x4
        if st == 'rq':
            assert q.search_type == AQ.ST_norm_rq2x4

        assert index.by_residual == (r == 'r')

    def test_factory(self):
        self.subtest_factory('LSQ', 16, 64, 'lsq')
        self.subtest_factory('LSQ', 16, 64, 'rq')
        self.subtest_factory('RQ', 16, 64, 'rq')
        self.subtest_factory('RQ', 16, 64, 'lsq')
        self.subtest_factory('LSQ', 64, 0, 'lsq')

        self.subtest_factory('LSQ', 64, 0, 'lsq', r='')

    def subtest_io(self, factory_str):
        d = 8
        ds = datasets.SyntheticDataset(d, 1000, 2000, 1000)

        index = faiss.index_factory(d, factory_str)
        index.train(ds.get_train())
        index.add(ds.get_database())
        D1, I1 = index.search(ds.get_queries(), 1)

        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)
            index2 = faiss.read_index(fname)
            D2, I2 = index2.search(ds.get_queries(), 1)
            np.testing.assert_array_equal(I1, I2)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_io(self):
        self.subtest_io('IVF16,LSQ4x4fs_Nlsq2x4')
        self.subtest_io('IVF16,LSQ4x4fs_Nrq2x4')
        self.subtest_io('IVF16,RQ4x4fs_Nrq2x4')
        self.subtest_io('IVF16,RQ4x4fs_Nlsq2x4')


# add more tests programatically

def add_TestIVFAQFastScan_subtest_accuracy(
        aq, st, by_residual, implem, metric='L2'):
    setattr(
        TestIVFAQFastScan,
        f"test_accuracy_{metric}_{aq}_implem{implem}_residual{by_residual}",
        lambda self:
        self.subtest_accuracy(aq, st, by_residual, implem, metric)
    )


def add_TestIVFAQFastScan_subtest_rescale_accuracy(aq, st, by_residual, implem):
    setattr(
        TestIVFAQFastScan,
        f"test_rescale_accuracy_{aq}_implem{implem}_residual{by_residual}",
        lambda self:
        self.subtest_rescale_accuracy(aq, st, by_residual, implem)
    )

for byr in True, False:
    for implem in 0, 10, 11, 12, 13, 14, 15:
        for mt in 'L2', 'IP':
            add_TestIVFAQFastScan_subtest_accuracy('RQ', 'rq', byr, implem, mt)
            add_TestIVFAQFastScan_subtest_accuracy('LSQ', 'lsq', byr, implem, mt)

        add_TestIVFAQFastScan_subtest_rescale_accuracy('LSQ', 'lsq', byr, implem)
        add_TestIVFAQFastScan_subtest_rescale_accuracy('RQ', 'rq', byr, implem)


class TestIVFPAQFastScan(unittest.TestCase):

    def subtest_accuracy(self, paq):
        """
        Compare IndexIVFAdditiveQuantizerFastScan with
        IndexIVFAdditiveQuantizer
        """
        nlist, d = 16, 8
        ds = datasets.SyntheticDataset(d, 1000, 1000, 500)
        gt = ds.get_groundtruth(k=1)

        index = faiss.index_factory(d, f'IVF{nlist},{paq}2x3x4_Nqint8')
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4
        Dref, Iref = index.search(ds.get_queries(), 1)

        indexfs = faiss.index_factory(d, f'IVF{nlist},{paq}2x3x4fsr_Nlsq2x4')
        indexfs.train(ds.get_train())
        indexfs.add(ds.get_database())
        indexfs.nprobe = 4
        D1, I1 = indexfs.search(ds.get_queries(), 1)

        nq = Iref.shape[0]
        recall_ref = (Iref == gt).sum() / nq
        recall1 = (I1 == gt).sum() / nq

        assert abs(recall_ref - recall1) < 0.05

    def test_accuracy_PLSQ(self):
        self.subtest_accuracy("PLSQ")

    def test_accuracy_PRQ(self):
        self.subtest_accuracy("PRQ")

    def subtest_factory(self, paq):
        nlist, d = 128, 16
        index = faiss.index_factory(d, f'IVF{nlist},{paq}2x3x4fsr_Nlsq2x4')
        q = faiss.downcast_Quantizer(index.aq)

        self.assertEqual(index.nlist, nlist)
        self.assertEqual(q.nsplits, 2)
        self.assertEqual(q.subquantizer(0).M, 3)
        self.assertTrue(index.by_residual)

    def test_factory(self):
        self.subtest_factory('PLSQ')
        self.subtest_factory('PRQ')

    def subtest_io(self, factory_str):
        d = 8
        ds = datasets.SyntheticDataset(d, 1000, 2000, 1000)

        index = faiss.index_factory(d, factory_str)
        index.train(ds.get_train())
        index.add(ds.get_database())
        D1, I1 = index.search(ds.get_queries(), 1)

        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)
            index2 = faiss.read_index(fname)
            D2, I2 = index2.search(ds.get_queries(), 1)
            np.testing.assert_array_equal(I1, I2)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_io(self):
        self.subtest_io('IVF16,PLSQ2x3x4fsr_Nlsq2x4')
        self.subtest_io('IVF16,PRQ2x3x4fs_Nrq2x4')


class TestSearchParams(unittest.TestCase):

    def test_search_params(self):
        ds = datasets.SyntheticDataset(32, 500, 100, 10)

        index = faiss.index_factory(ds.d, "IVF32,PQ16x4fs")
        index.train(ds.get_train())
        index.add(ds.get_database())

        index.nprobe
        index.nprobe = 4
        Dref4, Iref4 = index.search(ds.get_queries(), 10)
        # index.nprobe = 16
        # Dref16, Iref16 = index.search(ds.get_queries(), 10)

        index.nprobe = 1
        Dnew4, Inew4 = index.search(
            ds.get_queries(), 10, params=faiss.IVFSearchParameters(nprobe=4))
        np.testing.assert_array_equal(Dref4, Dnew4)
        np.testing.assert_array_equal(Iref4, Inew4)


class TestRangeSearchImplem12(unittest.TestCase):
    IMPLEM = 12

    def do_test(self, metric=faiss.METRIC_L2):
        ds = datasets.SyntheticDataset(32, 750, 200, 100)

        index = faiss.index_factory(ds.d, "IVF32,PQ16x4np", metric)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4

        # find a reasonable radius
        D, I = index.search(ds.get_queries(), 10)
        radius = np.median(D[:, -1])
        lims1, D1, I1 = index.range_search(ds.get_queries(), radius)

        index2 = faiss.IndexIVFPQFastScan(index)
        index2.implem = self.IMPLEM
        lims2, D2, I2 = index2.range_search(ds.get_queries(), radius)

        nmiss = 0
        nextra = 0

        for i in range(ds.nq):
            ref = set(I1[lims1[i]: lims1[i + 1]])
            new = set(I2[lims2[i]: lims2[i + 1]])
            nmiss += len(ref - new)
            nextra += len(new - ref)

        # need some tolerance because the look-up tables are quantized
        self.assertLess(nmiss, 10)
        self.assertLess(nextra, 10)

    def test_L2(self):
        self.do_test()

    def test_IP(self):
        self.do_test(metric=faiss.METRIC_INNER_PRODUCT)


class TestRangeSearchImplem10(TestRangeSearchImplem12):
    IMPLEM = 10


class TestRangeSearchImplem110(TestRangeSearchImplem12):
    IMPLEM = 110
