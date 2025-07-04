# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import time
import os
import tempfile

import numpy as np
import faiss

from faiss.contrib import datasets

# the tests tend to timeout in stress modes + dev otherwise
faiss.omp_set_num_threads(4)

class TestSearch(unittest.TestCase):

    def test_PQ4_accuracy(self):
        ds  = datasets.SyntheticDataset(32, 2000, 5000, 1000)

        index_gt = faiss.IndexFlatL2(32)
        index_gt.add(ds.get_database())
        Dref, Iref = index_gt.search(ds.get_queries(), 10)

        index = faiss.index_factory(32, 'PQ16x4fs')
        index.train(ds.get_train())
        index.add(ds.get_database())
        Da, Ia = index.search(ds.get_queries(), 10)

        nq = Iref.shape[0]
        recall_at_1 = (Iref[:, 0] == Ia[:, 0]).sum() / nq
        assert recall_at_1 > 0.6


    # This is an experiment to see if we can catch performance
    # regressions. It runs 2 codes, one should be faster than the
    # other by a factor ~10 in opt mode. We check for a factor 5.
    # hopefully the jitter in executtion time will not produce
    # too many spurious test failures. Unoptimized timings are
    # not exploitable, hence the flag test on that as well.
    @unittest.skipUnless(
        ('AVX2' in faiss.get_compile_options() or
        'AVX512' in faiss.get_compile_options() or
        'NEON' in faiss.get_compile_options()) and
        "OPTIMIZE" in faiss.get_compile_options(),
        "only test while building with avx2 or neon")
    def test_PQ4_speed(self):
        ds  = datasets.SyntheticDataset(32, 2000, 5000, 1000)
        xt = ds.get_train()
        xb = ds.get_database()
        xq = ds.get_queries()

        index = faiss.index_factory(32, 'PQ16x4')
        index.train(xt)
        index.add(xb)

        t0 = time.time()
        D1, I1 = index.search(xq, 10)
        t1 = time.time()
        pq_t = t1 - t0
        print('PQ16x4 search time:', pq_t)

        index2 = faiss.index_factory(32, 'PQ16x4fs')
        index2.train(xt)
        index2.add(xb)

        t0 = time.time()
        D2, I2 = index2.search(xq, 10)
        t1 = time.time()
        pqfs_t = t1 - t0
        print('PQ16x4fs search time:', pqfs_t)
        self.assertLess(pqfs_t * 4, pq_t)


class TestRounding(unittest.TestCase):

    def do_test_rounding(self, implem=4, metric=faiss.METRIC_L2):
        ds = datasets.SyntheticDataset(32, 2000, 5000, 200)

        index = faiss.index_factory(32, 'PQ16x4', metric)
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 10)
        nq = Iref.shape[0]

        index2 = faiss.IndexPQFastScan(index)

        # simply repro normal search
        index2.implem = 2
        D2, I2 = index2.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(I2, Iref)
        np.testing.assert_array_equal(D2, Dref)

        # rounded LUT with correction
        index2.implem = implem
        D4, I4 = index2.search(ds.get_queries(), 10)
        # check accuracy of indexes
        recalls = {}
        for rank in 1, 10:
            recalls[rank] = (Iref[:, :1] == I4[:, :rank]).sum() / nq

        min_r1 = 0.98 if metric == faiss.METRIC_INNER_PRODUCT else 0.99
        self.assertGreaterEqual(recalls[1], min_r1)
        self.assertGreater(recalls[10], 0.995)
        # check accuracy of distances
        # err3 = ((D3 - D2) ** 2).sum()
        err4 = ((D4 - D2) ** 2).sum()
        nf = (D2 ** 2).sum()
        self.assertLess(err4, nf * 1e-4)

    def test_implem_4(self):
        self.do_test_rounding(4)

    def test_implem_4_ip(self):
        self.do_test_rounding(4, faiss.METRIC_INNER_PRODUCT)

    def test_implem_12(self):
        self.do_test_rounding(12)

    def test_implem_12_ip(self):
        self.do_test_rounding(12, faiss.METRIC_INNER_PRODUCT)

    def test_implem_14(self):
        self.do_test_rounding(14)

    def test_implem_14_ip(self):
        self.do_test_rounding(12, faiss.METRIC_INNER_PRODUCT)


class TestReconstruct(unittest.TestCase):

    def test_pqfastscan(self):
        ds = datasets.SyntheticDataset(20, 1000, 1000, 0)

        index = faiss.index_factory(20, 'PQ5x4')
        index.train(ds.get_train())
        index.add(ds.get_database())
        recons = index.reconstruct_n(0, index.ntotal)

        index2 = faiss.IndexPQFastScan(index)
        recons2 = index2.reconstruct_n(0, index.ntotal)

        np.testing.assert_array_equal(recons, recons2)

    def test_aqfastscan(self):
        ds = datasets.SyntheticDataset(20, 1000, 1000, 0)

        index = faiss.index_factory(20, 'RQ5x4_Nrq2x4')
        index.train(ds.get_train())
        index.add(ds.get_database())
        recons = index.reconstruct_n(0, index.ntotal)

        index2 = faiss.IndexAdditiveQuantizerFastScan(index)
        recons2 = index2.reconstruct_n(0, index.ntotal)

        np.testing.assert_array_equal(recons, recons2)


#########################################################
# Kernel unit test
#########################################################


def reference_accu(codes, LUT):
    nq, nsp, is_16 = LUT.shape
    nb, nsp_2 = codes.shape
    assert is_16 == 16
    assert nsp_2 == nsp // 2
    accu = np.zeros((nq, nb), 'uint16')
    for i in range(nq):
        for j in range(nb):
            a = np.uint16(0)
            for sp in range(0, nsp, 2):
                c = codes[j, sp // 2]
                a += LUT[i, sp    , c & 15].astype('uint16')
                a += LUT[i, sp + 1, c >> 4].astype('uint16')
            accu[i, j] = a
    return accu


# disabled because the function to write to mem is not implemented currently
class ThisIsNotATestLoop5:    # (unittest.TestCase):

    def do_loop5_kernel(self, nq, bb):
        """ unit test for the accumulation kernel """
        nb = bb * 32  # databse size
        nsp = 24     # number of sub-quantizers

        rs = np.random.RandomState(123)
        codes = rs.randint(256, size=(nb, nsp // 2)).astype('uint8')
        LUT = rs.randint(256, size=(nq, nsp, 16)).astype('uint8')
        accu_ref = reference_accu(codes, LUT)

        def to_A(x):
            return faiss.array_to_AlignedTable(x.ravel())

        sp = faiss.swig_ptr

        LUT_a = faiss.AlignedTableUint8(LUT.size)
        faiss.pq4_pack_LUT(
            nq, nsp, sp(LUT),
            LUT_a.get()
        )

        codes_a = faiss.AlignedTableUint8(codes.size)
        faiss.pq4_pack_codes(
            sp(codes),
            nb, nsp, nb, nb, nsp,
            codes_a.get()
        )

        accu_a = faiss.AlignedTableUint16(nq * nb)
        accu_a.clear()
        faiss.loop5_kernel_accumulate_1_block_to_mem(
            nq, nb, nsp, codes_a.get(), LUT_a.get(), accu_a.get()
        )
        accu = faiss.AlignedTable_to_array(accu_a).reshape(nq, nb)
        np.testing.assert_array_equal(accu_ref, accu)

    def test_11(self):
        self.do_loop5_kernel(1, 1)

    def test_21(self):
        self.do_loop5_kernel(2, 1)

    def test_12(self):
        self.do_loop5_kernel(1, 2)

    def test_22(self):
        self.do_loop5_kernel(2, 2)


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
            if dis == skip_dis:
                continue
            mask = Dref[i, :] == dis
            testcase.assertEqual(set(Iref[i, mask]), set(Inew[i, mask]))


class TestImplems(unittest.TestCase):

    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.cache = {}
        self.k = 10

    def get_index(self, d, metric):
        if (d, metric) not in self.cache:
            ds = datasets.SyntheticDataset(d, 1000, 2000, 200)
            target_size = d // 2
            index = faiss.index_factory(d, 'PQ%dx4' % target_size, metric)
            index.train(ds.get_train())
            index.add(ds.get_database())

            index2 = faiss.IndexPQFastScan(index)
            # uint8 LUT but no SIMD
            index2.implem = 4
            Dref, Iref = index2.search(ds.get_queries(), 10)

            # check CodePacker
            codes_ref = faiss.vector_to_array(index.codes)
            codes_ref = codes_ref.reshape(-1, index.code_size)
            index2codes = faiss.vector_to_array(index2.codes)
            code_packer = index2.get_CodePacker()
            index2codes = index2codes.reshape(-1, code_packer.block_size)

            for i in range(0, len(codes_ref), 13):
                code_new = code_packer.unpack_1(index2codes, i)
                np.testing.assert_array_equal(codes_ref[i], code_new)

            self.cache[(d, metric)] = (ds, index, Dref, Iref)

        return self.cache[(d, metric)]

    def do_with_params(self, d, params, metric=faiss.METRIC_L2):
        ds, index, Dref, Iref = self.get_index(d, metric)

        index2 = self.build_fast_scan_index(index, params)

        Dnew, Inew = index2.search(ds.get_queries(), self.k)

        Dref = Dref[:, :self.k]
        Iref = Iref[:, :self.k]

        verify_with_draws(self, Dref, Iref, Dnew, Inew)

    def build_fast_scan_index(self, index, params):
        index2 = faiss.IndexPQFastScan(index)
        index2.implem = 5
        return index2


class TestImplem12(TestImplems):

    def build_fast_scan_index(self, index, qbs):
        index2 = faiss.IndexPQFastScan(index)
        index2.qbs = qbs
        index2.implem = 12
        return index2

    def test_qbs7(self):
        self.do_with_params(32, 0x223)

    def test_qbs7b(self):
        self.do_with_params(32, 0x133)

    def test_qbs6(self):
        self.do_with_params(32, 0x33)

    def test_qbs6_ip(self):
        self.do_with_params(32, 0x33, faiss.METRIC_INNER_PRODUCT)

    def test_qbs6b(self):
        # test codepath where qbs is not known at compile time
        self.do_with_params(32, 0x1113)

    def test_qbs6_odd_dim(self):
        self.do_with_params(30, 0x33)


class TestImplem13(TestImplems):

    def build_fast_scan_index(self, index, qbs):
        index2 = faiss.IndexPQFastScan(index)
        index2.qbs = qbs
        index2.implem = 13
        return index2

    def test_qbs7(self):
        self.do_with_params(32, 0x223)

    def test_qbs7_k1(self):
        self.k = 1
        self.do_with_params(32, 0x223)


class TestImplem14(TestImplems):

    def build_fast_scan_index(self, index, params):
        qbs, bbs = params
        index2 = faiss.IndexPQFastScan(index, bbs)
        index2.qbs = qbs
        index2.implem = 14
        return index2

    def test_1_32(self):
        self.do_with_params(32, (1, 32))

    def test_1_64(self):
        self.do_with_params(32, (1, 64))

    def test_2_32(self):
        self.do_with_params(32, (2, 32))

    def test_2_64(self):
        self.do_with_params(32, (2, 64))

    def test_qbs_1_32_k1(self):
        self.k = 1
        self.do_with_params(32, (1, 32))

    def test_qbs_1_64_k1(self):
        self.k = 1
        self.do_with_params(32, (1, 64))

    def test_1_32_odd_dim(self):
        self.do_with_params(30, (1, 32))

    def test_1_64_odd_dim(self):
        self.do_with_params(30, (1, 64))


class TestImplem15(TestImplems):

    def build_fast_scan_index(self, index, params):
        qbs, bbs = params
        index2 = faiss.IndexPQFastScan(index, bbs)
        index2.qbs = qbs
        index2.implem = 15
        return index2

    def test_1_32(self):
        self.do_with_params(32, (1, 32))

    def test_2_64(self):
        self.do_with_params(32, (2, 64))


class TestAdd(unittest.TestCase):

    def do_test_add(self, d, bbs):

        ds = datasets.SyntheticDataset(d, 2000, 5000, 200)

        index = faiss.index_factory(d, f'PQ{d//2}x4np')
        index.train(ds.get_train())

        xb = ds.get_database()
        index.add(xb[:1235])

        index2 = faiss.IndexPQFastScan(index, bbs)
        index2.add(xb[1235:])
        new_codes = faiss.AlignedTable_to_array(index2.codes)

        index.add(xb[1235:])
        index3 = faiss.IndexPQFastScan(index, bbs)
        ref_codes = faiss.AlignedTable_to_array(index3.codes)
        self.assertEqual(index3.ntotal, index2.ntotal)

        np.testing.assert_array_equal(ref_codes, new_codes)

    def test_add(self):
        self.do_test_add(32, 32)

    def test_add_bbs64(self):
        self.do_test_add(32, 64)

    def test_add_odd_d(self):
        self.do_test_add(30, 64)

    def test_constructor(self):
        d = 32
        ds = datasets.SyntheticDataset(d, 2000, 5000, 200)

        index = faiss.index_factory(d, f'PQ{d//2}x4np')
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 10)
        nq = Iref.shape[0]

        index2 = faiss.IndexPQFastScan(d, d // 2, 4)
        index2.train(ds.get_train())
        index2.add(ds.get_database())
        Dnew, Inew = index2.search(ds.get_queries(), 10)

        recall_at_1 = (Iref[:, 0] == Inew[:, 0]).sum() / nq

        self.assertGreaterEqual(recall_at_1, 0.99)

        data = faiss.serialize_index(index2)
        index3 = faiss.deserialize_index(data)

        self.assertEqual(index2.implem, index3.implem)

        D3, I3 = index3.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(D3, Dnew)
        np.testing.assert_array_equal(I3, Inew)


class TestAQFastScan(unittest.TestCase):

    def subtest_accuracy(self, aq, st, implem, metric_type='L2'):
        """
        Compare IndexAdditiveQuantizerFastScan with IndexAQ (qint8)
        """
        d = 16
        ds = datasets.SyntheticDataset(d, 1000, 1000, 500, metric_type)
        gt = ds.get_groundtruth(k=1)

        if metric_type == 'L2':
            metric = faiss.METRIC_L2
            postfix1 = '_Nqint8'
            postfix2 = f'_N{st}2x4'
        else:
            metric = faiss.METRIC_INNER_PRODUCT
            postfix1 = postfix2 = ''

        index = faiss.index_factory(d, f'{aq}3x4{postfix1}', metric)
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 1)

        indexfs = faiss.index_factory(d, f'{aq}3x4fs_32{postfix2}', metric)
        indexfs.train(ds.get_train())
        indexfs.add(ds.get_database())
        indexfs.implem = implem
        Da, Ia = indexfs.search(ds.get_queries(), 1)

        nq = Iref.shape[0]
        recall_ref = (Iref == gt).sum() / nq
        recall = (Ia == gt).sum() / nq

        assert abs(recall_ref - recall) < 0.05

    def xx_test_accuracy(self):
        for metric in 'L2', 'IP':
            for implem in 0, 12, 13, 14, 15:
                self.subtest_accuracy('RQ', 'rq', implem, metric)
                self.subtest_accuracy('LSQ', 'lsq', implem, metric)

    def subtest_from_idxaq(self, implem, metric):
        if metric == 'L2':
            metric_type = faiss.METRIC_L2
            st = '_Nrq2x4'
        else:
            metric_type = faiss.METRIC_INNER_PRODUCT
            st = ''

        d = 16
        ds = datasets.SyntheticDataset(d, 1000, 2000, 1000, metric=metric)
        gt = ds.get_groundtruth(k=1)
        index = faiss.index_factory(d, 'RQ8x4' + st, metric_type)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 16
        Dref, Iref = index.search(ds.get_queries(), 1)

        indexfs = faiss.IndexAdditiveQuantizerFastScan(index)
        indexfs.implem = implem
        D1, I1 = indexfs.search(ds.get_queries(), 1)

        nq = Iref.shape[0]
        recall_ref = (Iref == gt).sum() / nq
        recall1 = (I1 == gt).sum() / nq
        assert abs(recall_ref - recall1) < 0.05

    def xx_test_from_idxaq(self):
        for implem in 2, 3, 4:
            self.subtest_from_idxaq(implem, 'L2')
            self.subtest_from_idxaq(implem, 'IP')

    def subtest_factory(self, aq, M, bbs, st):
        """
        Format: {AQ}{M}x4fs_{bbs}_N{st}

            AQ (str):    `LSQ` or `RQ`
            M (int):     number of subquantizers
            bbs (int):   build block size
            st (str):    search type, `lsq2x4` or `rq2x4`
        """
        AQ = faiss.AdditiveQuantizer
        d = 16

        if bbs > 0:
            index = faiss.index_factory(d, f'{aq}{M}x4fs_{bbs}_N{st}2x4')
        else:
            index = faiss.index_factory(d, f'{aq}{M}x4fs_N{st}2x4')
            bbs = 32

        assert index.bbs == bbs
        aq = faiss.downcast_AdditiveQuantizer(index.aq)
        assert aq.M == M

        if aq == 'LSQ':
            assert isinstance(aq, faiss.LocalSearchQuantizer)
        if aq == 'RQ':
            assert isinstance(aq, faiss.ResidualQuantizer)

        if st == 'lsq':
            assert aq.search_type == AQ.ST_norm_lsq2x4
        if st == 'rq':
            assert aq.search_type == AQ.ST_norm_rq2x4

    def test_factory(self):
        self.subtest_factory('LSQ', 16, 64, 'lsq')
        self.subtest_factory('LSQ', 16, 64, 'rq')
        self.subtest_factory('RQ', 16, 64, 'rq')
        self.subtest_factory('RQ', 16, 64, 'lsq')
        self.subtest_factory('LSQ', 64, 0, 'lsq')

    def subtest_io(self, factory_str):
        d = 8
        ds = datasets.SyntheticDataset(d, 1000, 500, 100)

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
        self.subtest_io('LSQ4x4fs_Nlsq2x4')
        self.subtest_io('LSQ4x4fs_Nrq2x4')
        self.subtest_io('RQ4x4fs_Nrq2x4')
        self.subtest_io('RQ4x4fs_Nlsq2x4')


# programatically generate tests to get finer test granularity.

def add_TestAQFastScan_subset_accuracy(aq, st, implem, metric):
    setattr(
        TestAQFastScan,
        f"test_accuracy_{metric}_{aq}_implem{implem}",
        lambda self: self.subtest_accuracy(aq, st, implem, metric)
    )


for metric in 'L2', 'IP':
    for implem in 0, 12, 13, 14, 15:
        add_TestAQFastScan_subset_accuracy('LSQ', 'lsq', implem, metric)
        add_TestAQFastScan_subset_accuracy('RQ', 'rq', implem, metric)


def add_TestAQFastScan_subtest_from_idxaq(implem, metric):
    setattr(
        TestAQFastScan,
        f"test_from_idxaq_{metric}_implem{implem}",
        lambda self: self.subtest_from_idxaq(implem, metric)
    )


for implem in 2, 3, 4:
    add_TestAQFastScan_subtest_from_idxaq(implem, 'L2')
    add_TestAQFastScan_subtest_from_idxaq(implem, 'IP')


class TestPAQFastScan(unittest.TestCase):

    def subtest_accuracy(self, paq):
        """
        Compare IndexPAQFastScan with IndexPAQ (qint8)
        """
        d = 16
        ds = datasets.SyntheticDataset(d, 1000, 1000, 500)
        gt = ds.get_groundtruth(k=1)

        index = faiss.index_factory(d, f'{paq}2x3x4_Nqint8')
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 1)

        indexfs = faiss.index_factory(d, f'{paq}2x3x4fs_Nlsq2x4')
        indexfs.train(ds.get_train())
        indexfs.add(ds.get_database())
        Da, Ia = indexfs.search(ds.get_queries(), 1)

        nq = Iref.shape[0]
        recall_ref = (Iref == gt).sum() / nq
        recall = (Ia == gt).sum() / nq

        assert abs(recall_ref - recall) < 0.05

    def test_accuracy_PLSQ(self):
        self.subtest_accuracy("PLSQ")

    def test_accuracy_PRQ(self):
        self.subtest_accuracy("PRQ")

    def subtest_factory(self, paq):
        index = faiss.index_factory(16, f'{paq}2x3x4fs_Nlsq2x4')
        q = faiss.downcast_Quantizer(index.aq)
        self.assertEqual(q.nsplits, 2)
        self.assertEqual(q.subquantizer(0).M, 3)

    def test_factory(self):
        self.subtest_factory('PRQ')
        self.subtest_factory('PLSQ')

    def subtest_io(self, factory_str):
        d = 8
        ds = datasets.SyntheticDataset(d, 1000, 500, 100)

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
        self.subtest_io('PLSQ2x3x4fs_Nlsq2x4')
        self.subtest_io('PRQ2x3x4fs_Nrq2x4')


class TestBlockDecode(unittest.TestCase):

    def test_issue_2739(self):
        ds = datasets.SyntheticDataset(960, 200, 1, 0)
        M = 32
        index = faiss.index_factory(ds.d, f"PQ{M}x4fs")
        index.train(ds.get_train())
        index.add(ds.get_database())

        np.testing.assert_array_equal(
            index.pq.decode(index.pq.compute_codes(ds.get_database()))[0, ::100],
            index.reconstruct(0)[::100]
        )
