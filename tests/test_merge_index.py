# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset

from common_faiss_tests import for_all_simd_levels, Randu10k

ru = Randu10k()
xb = ru.xb
xt = ru.xt
xq = ru.xq
nb, d = xb.shape
nq, d = xq.shape


@for_all_simd_levels
class TestMerge1(unittest.TestCase):
    def make_index_for_merge(self, quant, master_index):
        index = faiss.IndexIVFPQR(quant, d, 40, 4, 8, 8, 8)
        if master_index:
            index.pq = master_index.pq
            index.refine_pq = master_index.refine_pq
            index.is_trained = True
        return index

    def test_merge(self):
        k = 16
        quant = faiss.IndexFlatL2(d)
        ref_index = self.make_index_for_merge(quant, None)
        ref_index.train(xt)
        ref_index.add(xb)
        _Dref, Iref = ref_index.search(xq, k)

        indexes = []
        ni = 3
        for i in range(ni):
            i0 = int(i * nb / ni)
            i1 = int((i + 1) * nb / ni)
            index = self.make_index_for_merge(quant, ref_index)
            index.is_trained = True
            index.add(xb[i0:i1])
            indexes.append(index)

        index = indexes[0]

        for i in range(1, ni):
            index.merge_from(indexes[i], index.ntotal)

        _D, I = index.search(xq, k)

        ndiff = (I != Iref).sum()
        assert ndiff < nq * k / 1000.0


@for_all_simd_levels
class TestMerge2(unittest.TestCase):

    def test_merge_PreTransform(self):
        ds = SyntheticDataset(32, 300, 300, 100)
        index1 = faiss.index_factory(ds.d, "PCA16,SQ4")
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

    def do_fast_scan_test(self, factory_key, size1, with_add_id=False):
        ds = SyntheticDataset(110, 1000, 1000, 100)
        index_trained = faiss.index_factory(ds.d, factory_key)
        index_trained.train(ds.get_train())
        index1 = faiss.deserialize_index(faiss.serialize_index(index_trained))

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

    def test_merge_IVFFastScan(self):
        self.do_fast_scan_test("IVF20,PQ5x4fs", 123, with_add_id=True)

    def do_test_with_ids(self, factory_key):
        ds = SyntheticDataset(32, 300, 300, 100)
        rs = np.random.RandomState(123)
        ids = rs.choice(10000, ds.nb, replace=False).astype("int64")
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


@for_all_simd_levels
class TestRemoveFastScan(unittest.TestCase):

    def do_fast_scan_test(
        self, factory_key, direct_map_type=faiss.DirectMap.NoMap
    ):
        ds = SyntheticDataset(110, 1000, 1000, 100)
        index = faiss.index_factory(ds.d, factory_key)
        index.train(ds.get_train())

        tokeep = [i % 3 == 0 for i in range(ds.nb)]
        index.add_with_ids(
            ds.get_database()[tokeep], np.arange(ds.nb)[tokeep]
        )
        faiss.extract_index_ivf(index).nprobe = 5
        _, Iref = index.search(ds.get_queries(), 5)

        index.reset()
        index.add_with_ids(ds.get_database(), np.arange(ds.nb))
        index.set_direct_map_type(direct_map_type)
        faiss.extract_index_ivf(index).nprobe = 5
        index.remove_ids(np.where(np.logical_not(tokeep))[0])
        _, Inew = index.search(ds.get_queries(), 5)
        np.testing.assert_array_equal(Inew, Iref)

    def test_remove_IVFPQFastScan(self):
        self.do_fast_scan_test("IVF20,PQ5x4fs")

    def test_remove_IVFPQFastScan_2(self):
        self.assertRaisesRegex(
            Exception,
            ".*not supported.*",
            self.do_fast_scan_test,
            "IVF20,PQ5x4fs",
            faiss.DirectMap.Hashtable,
        )
