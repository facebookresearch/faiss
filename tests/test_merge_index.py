# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset

from common_faiss_tests import Randu10k

ru = Randu10k()
xb = ru.xb
xt = ru.xt
xq = ru.xq
nb, d = xb.shape
nq, d = xq.shape


class TestMerge1(unittest.TestCase):
    def make_index_for_merge(self, quant, index_type, master_index):
        ncent = 40
        if index_type == 1:
            index = faiss.IndexIVFFlat(quant, d, ncent, faiss.METRIC_L2)
            if master_index:
                index.is_trained = True
        elif index_type == 2:
            index = faiss.IndexIVFPQ(quant, d, ncent, 4, 8)
            if master_index:
                index.pq = master_index.pq
                index.is_trained = True
        elif index_type == 3:
            index = faiss.IndexIVFPQR(quant, d, ncent, 4, 8, 8, 8)
            if master_index:
                index.pq = master_index.pq
                index.refine_pq = master_index.refine_pq
                index.is_trained = True
        elif index_type == 4:
            # quant used as the actual index
            index = faiss.IndexIDMap(quant)
        return index

    def do_test_merge(self, index_type):
        k = 16
        quant = faiss.IndexFlatL2(d)
        ref_index = self.make_index_for_merge(quant, index_type, False)

        # trains the quantizer
        ref_index.train(xt)

        print('ref search')
        ref_index.add(xb)
        _Dref, Iref = ref_index.search(xq, k)
        print(Iref[:5, :6])

        indexes = []
        ni = 3
        for i in range(ni):
            i0 = int(i * nb / ni)
            i1 = int((i + 1) * nb / ni)
            index = self.make_index_for_merge(quant, index_type, ref_index)
            index.is_trained = True
            index.add(xb[i0:i1])
            indexes.append(index)

        index = indexes[0]

        for i in range(1, ni):
            print('merge ntotal=%d other.ntotal=%d ' % (
                index.ntotal, indexes[i].ntotal))
            index.merge_from(indexes[i], index.ntotal)

        _D, I = index.search(xq, k)

        ndiff = (I != Iref).sum()
        print('%d / %d differences' % (ndiff, nq * k))
        assert (ndiff < nq * k / 1000.)

    def test_merge(self):
        self.do_test_merge(1)
        self.do_test_merge(2)
        self.do_test_merge(3)

    ######################################
    # remove tests that piggyback on merge

    def do_test_remove(self, index_type):
        k = 16
        quant = faiss.IndexFlatL2(d)
        index = self.make_index_for_merge(quant, index_type, None)

        # trains the quantizer
        index.train(xt)

        if index_type < 4:
            index.add(xb)
        else:
            gen = np.random.RandomState(1234)
            id_list = gen.permutation(nb * 7)[:nb].astype('int64')
            index.add_with_ids(xb, id_list)

        print('ref search ntotal=%d' % index.ntotal)
        Dref, Iref = index.search(xq, k)

        toremove = np.zeros(nq * k, dtype='int64')
        nr = 0
        for i in range(nq):
            for j in range(k):
                # remove all even results (it's ok if there are duplicates
                # in the list of ids)
                if Iref[i, j] % 2 == 0:
                    nr = nr + 1
                    toremove[nr] = Iref[i, j]

        print('nr=', nr)

        idsel = faiss.IDSelectorBatch(
            nr, faiss.swig_ptr(toremove))

        for i in range(nr):
            assert (idsel.is_member(int(toremove[i])))

        nremoved = index.remove_ids(idsel)

        print('nremoved=%d ntotal=%d' % (nremoved, index.ntotal))

        D, I = index.search(xq, k)

        # make sure results are in the same order with even ones removed
        ndiff = 0
        for i in range(nq):
            j2 = 0
            for j in range(k):
                if Iref[i, j] % 2 != 0:
                    if I[i, j2] != Iref[i, j]:
                        ndiff += 1
                    assert abs(D[i, j2] - Dref[i, j]) < 1e-5
                    j2 += 1
        # draws are ordered arbitrarily
        assert ndiff < 5

    def test_remove(self):
        self.do_test_remove(1)
        self.do_test_remove(2)
        self.do_test_remove(4)


# Test merge_from method for all IndexFlatCodes Types
class TestMerge2(unittest.TestCase):

    def do_flat_codes_test(self, factory_key):
        ds = SyntheticDataset(32, 300, 300, 100)
        index1 = faiss.index_factory(ds.d, factory_key)
        index1.train(ds.get_train())
        index1.add(ds.get_database())
        _, Iref = index1.search(ds.get_queries(), 5)
        index1.reset()
        index2 = faiss.clone_index(index1)
        index1.add(ds.get_database()[:100])
        index2.add(ds.get_database()[100:])
        index1.merge_from(index2)
        _, Inew = index1.search(ds.get_queries(), 5)
        np.testing.assert_array_equal(Inew, Iref)

    def test_merge_IndexFlat(self):
        self.do_flat_codes_test("Flat")

    def test_merge_IndexPQ(self):
        self.do_flat_codes_test("PQ8np")

    def test_merge_IndexLSH(self):
        self.do_flat_codes_test("LSHr")

    def test_merge_IndexScalarQuantizer(self):
        self.do_flat_codes_test("SQ4")

    def test_merge_PreTransform(self):
        self.do_flat_codes_test("PCA16,SQ4")

    def do_fast_scan_test(self, factory_key, size1, with_add_id=False):
        ds = SyntheticDataset(110, 1000, 1000, 100)
        index_trained = faiss.index_factory(ds.d, factory_key)
        index_trained.train(ds.get_train())
        # test both clone and index_read/write
        if True:
            index1 = faiss.deserialize_index(
                faiss.serialize_index(index_trained))
        else:
            index1 = faiss.clone_index(index_trained)
        # assert index1.aq.qnorm.ntotal == index_trained.aq.qnorm.ntotal

        index1.add(ds.get_database())
        _, Iref = index1.search(ds.get_queries(), 5)
        index1.reset()
        index2 = faiss.clone_index(index_trained)
        index1.add(ds.get_database()[:size1])
        index2.add(ds.get_database()[size1:])
        if with_add_id:
            index1.merge_from(index2, add_id=index1.ntotal)
        else:
            index1.merge_from(index2)
        _, Inew = index1.search(ds.get_queries(), 5)
        np.testing.assert_array_equal(Inew, Iref)

    def test_merge_IndexFastScan_complete_block(self):
        self.do_fast_scan_test("PQ5x4fs", 320)

    def test_merge_IndexFastScan_not_complete_block(self):
        self.do_fast_scan_test("PQ11x4fs", 310)

    def test_merge_IndexFastScan_even_M(self):
        self.do_fast_scan_test("PQ10x4fs", 500)

    def test_merge_IndexAdditiveQuantizerFastScan(self):
        self.do_fast_scan_test("RQ10x4fs_32_Nrq2x4", 330)

    def test_merge_IVFFastScan(self):
        self.do_fast_scan_test("IVF20,PQ5x4fs", 123, with_add_id=True)

    def do_test_with_ids(self, factory_key):
        ds = SyntheticDataset(32, 300, 300, 100)
        rs = np.random.RandomState(123)
        ids = rs.choice(10000, ds.nb, replace=False).astype('int64')
        index1 = faiss.index_factory(ds.d, factory_key)
        index1.train(ds.get_train())
        index1.add_with_ids(ds.get_database(), ids)
        _, Iref = index1.search(ds.get_queries(), 5)
        index1.reset()
        index2 = faiss.clone_index(index1)
        index1.add_with_ids(ds.get_database()[:100], ids[:100])
        index2.add_with_ids(ds.get_database()[100:], ids[100:])
        index1.merge_from(index2)
        _, Inew = index1.search(ds.get_queries(), 5)
        np.testing.assert_array_equal(Inew, Iref)
        if "IDMap2" in factory_key:
            index1.check_consistency()

    def test_merge_IDMap(self):
        self.do_test_with_ids("Flat,IDMap")

    def test_merge_IDMap2(self):
        self.do_test_with_ids("Flat,IDMap2")


class TestRemoveFastScan(unittest.TestCase):

    def do_fast_scan_test(self,
                          factory_key,
                          with_ids=False,
                          direct_map_type=faiss.DirectMap.NoMap):
        ds = SyntheticDataset(110, 1000, 1000, 100)
        index = faiss.index_factory(ds.d, factory_key)
        index.train(ds.get_train())

        index.reset()
        tokeep = [i % 3 == 0 for i in range(ds.nb)]
        if with_ids:
            index.add_with_ids(ds.get_database()[tokeep], np.arange(ds.nb)[tokeep])
            faiss.extract_index_ivf(index).nprobe = 5
        else:
            index.add(ds.get_database()[tokeep])
        _, Iref = index.search(ds.get_queries(), 5)

        index.reset()
        if with_ids:
            index.add_with_ids(ds.get_database(), np.arange(ds.nb))
            index.set_direct_map_type(direct_map_type)
            faiss.extract_index_ivf(index).nprobe = 5
        else:
            index.add(ds.get_database())
        index.remove_ids(np.where(np.logical_not(tokeep))[0])
        _, Inew = index.search(ds.get_queries(), 5)
        np.testing.assert_array_equal(Inew, Iref)

    def test_remove_PQFastScan(self):
        # with_ids is not support for this type of index
        self.do_fast_scan_test("PQ5x4fs", False)

    def test_remove_IVFPQFastScan(self):
        self.do_fast_scan_test("IVF20,PQ5x4fs", True)

    def test_remove_IVFPQFastScan_2(self):
        self.assertRaisesRegex(Exception,
                               ".*not supported.*",
                               self.do_fast_scan_test,
                               "IVF20,PQ5x4fs",
                               True,
                               faiss.DirectMap.Hashtable)
