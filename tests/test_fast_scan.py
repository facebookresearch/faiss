# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import time

import numpy as np
import faiss

from faiss.contrib import datasets
import platform


class TestCompileOptions(unittest.TestCase):

    def test_compile_options(self):
        options = faiss.get_compile_options()
        options = options.split(' ')
        for option in options:
            assert option in ['AVX2', 'NEON', 'GENERIC', 'OPTIMIZE']


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
        # print(f'recall@1 = {recall_at_1:.3f}')


    # This is an experiment to see if we can catch performance
    # regressions. It runs 2 codes, one should be faster than the
    # other by a factor ~10 in opt mode. We check for a factor 5.
    # hopefully the jitter in executtion time will not produce
    # too many spurious test failures. Unoptimized timings are
    # not exploitable, hence the flag test on that as well.
    @unittest.skipUnless(
        'AVX2' in faiss.get_compile_options() and
        "OPTIMIZE" in faiss.get_compile_options(),
        "only test while building with avx2")
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
        self.assertLess(pqfs_t * 5, pq_t)


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
        self.assertGreater(recalls[1], min_r1)
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

        self.assertGreater(recall_at_1, 0.99)

        data = faiss.serialize_index(index2)
        index3 = faiss.deserialize_index(data)

        self.assertEqual(index2.implem, index3.implem)

        D3, I3 = index3.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(D3, Dnew)
        np.testing.assert_array_equal(I3, Inew)
