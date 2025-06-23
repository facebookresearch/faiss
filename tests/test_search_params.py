# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest
import sys
import gc

from faiss.contrib import datasets
from faiss.contrib.evaluation import sort_range_res_2, check_ref_range_results

faiss.omp_set_num_threads(4)


class TestSelector(unittest.TestCase):
    """
    Test the IDSelector filtering for as many (index class, id selector class)
    combinations as possible.
    """

    def do_test_id_selector(
        self,
        index_key,
        id_selector_type="batch",
        mt=faiss.METRIC_L2,
        k=10,
        use_heap=True
    ):
        """ Verify that the id selector returns the subset of results that are
        members according to the IDSelector.
        Supports id_selector_type="batch", "bitmap", "range", "range_sorted", "and", "or", "xor"
        """
        d = 32  # make sure dimension is multiple of 8 for binary
        ds = datasets.SyntheticDataset(d, 1000, 100, 20)

        if index_key == "BinaryFlat":
            rs = np.random.RandomState(123)
            xb = rs.randint(256, size=(ds.nb, d // 8), dtype='uint8')
            xq = rs.randint(256, size=(ds.nq, d // 8), dtype='uint8')
            index = faiss.IndexBinaryFlat(d)
            index.use_heap = use_heap
            # Use smaller radius for Hamming distance
            base_radius = 4
            is_binary = True
        else:
            xb = ds.get_database()
            xq = ds.get_queries()
            xt = ds.get_train()
            index = faiss.index_factory(d, index_key, mt)
            index.train(xt)
            base_radius = float('inf')  # Will be set based on results
            is_binary = False

        # reference result
        if "range" in id_selector_type:
            subset = np.arange(30, 80).astype('int64')
        elif id_selector_type == "or":
            lhs_rs = np.random.RandomState(123)
            lhs_subset = lhs_rs.choice(ds.nb, 50, replace=False).astype("int64")
            rhs_rs = np.random.RandomState(456)
            rhs_subset = rhs_rs.choice(ds.nb, 20, replace=False).astype("int64")
            subset = np.union1d(lhs_subset, rhs_subset)
        elif id_selector_type == "and":
            lhs_rs = np.random.RandomState(123)
            lhs_subset = lhs_rs.choice(ds.nb, 50, replace=False).astype("int64")
            rhs_rs = np.random.RandomState(456)
            rhs_subset = rhs_rs.choice(ds.nb, 10, replace=False).astype("int64")
            subset = np.intersect1d(lhs_subset, rhs_subset)
        elif id_selector_type == "xor":
            lhs_rs = np.random.RandomState(123)
            lhs_subset = lhs_rs.choice(ds.nb, 50, replace=False).astype("int64")
            rhs_rs = np.random.RandomState(456)
            rhs_subset = rhs_rs.choice(ds.nb, 40, replace=False).astype("int64")
            subset = np.setxor1d(lhs_subset, rhs_subset)
        else:
            rs = np.random.RandomState(123)
            subset = rs.choice(ds.nb, 50, replace=False).astype('int64')

        index.add(xb[subset])
        if "IVF" in index_key and id_selector_type == "range_sorted":
            self.assertTrue(index.check_ids_sorted())
        Dref, Iref0 = index.search(xq, k)
        Iref = subset[Iref0]
        Iref[Iref0 < 0] = -1

        if base_radius == float('inf'):
            radius = float(Dref[Iref > 0].max()) * 1.01
        else:
            radius = base_radius

        try:
            Rlims_ref, RDref, RIref = index.range_search(xq, radius)
        except RuntimeError as e:
            if "not implemented" in str(e):
                have_range_search = False
            else:
                raise
        else:
            RIref = subset[RIref]
            # normalize the range search results
            RDref, RIref = sort_range_res_2(Rlims_ref, RDref, RIref)
            have_range_search = True

        # result with selector: fill full database and search with selector
        index.reset()
        index.add(xb)
        if id_selector_type == "range":
            sel = faiss.IDSelectorRange(30, 80)
        elif id_selector_type == "range_sorted":
            sel = faiss.IDSelectorRange(30, 80, True)
        elif id_selector_type == "array":
            sel = faiss.IDSelectorArray(subset)
        elif id_selector_type == "bitmap":
            bitmap = np.zeros(ds.nb, dtype=bool)
            bitmap[subset] = True
            bitmap = np.packbits(bitmap, bitorder='little')
            sel = faiss.IDSelectorBitmap(bitmap)
        elif id_selector_type == "not":
            ssubset = set(subset)
            inverse_subset = np.array([
                i for i in range(ds.nb)
                if i not in ssubset
            ]).astype('int64')
            sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(inverse_subset))
        elif id_selector_type == "or":
            sel = faiss.IDSelectorOr(
                faiss.IDSelectorBatch(lhs_subset),
                faiss.IDSelectorBatch(rhs_subset)
            )
        elif id_selector_type == "and":
            sel = faiss.IDSelectorAnd(
                faiss.IDSelectorBatch(lhs_subset),
                faiss.IDSelectorBatch(rhs_subset)
            )
        elif id_selector_type == "xor":
            sel = faiss.IDSelectorXOr(
                faiss.IDSelectorBatch(lhs_subset),
                faiss.IDSelectorBatch(rhs_subset)
            )
        else:
            sel = faiss.IDSelectorBatch(subset)

        params = (
            faiss.SearchParametersIVF(sel=sel) if "IVF" in index_key else
            faiss.SearchParametersPQ(sel=sel) if "PQ" in index_key else
            faiss.SearchParameters(sel=sel)
        )

        Dnew, Inew = index.search(xq, k, params=params)

        if is_binary:
            # For binary indexes, we need to check:
            # 1. All returned IDs are valid (in the subset or -1)
            # 2. The distances match

            # Check that all returned IDs are valid
            valid_ids = np.ones_like(Inew, dtype=bool)
            # Create a mask of valid IDs (those in subset)
            subset_set = set(subset)  # Convert to set for O(1) lookups
            # Handle -1 values separately (they're always valid)
            valid_ids = np.logical_or(
                Inew == -1,
                np.isin(Inew, list(subset_set))
            )

            self.assertTrue(np.all(valid_ids), "Some returned IDs are not in the subset")

            # Check that distances match
            np.testing.assert_almost_equal(Dref, Dnew, decimal=5)
        else:
            # For non-binary indexes, we can do exact comparison
            np.testing.assert_array_equal(Iref, Inew)
            np.testing.assert_almost_equal(Dref, Dnew, decimal=5)

        if have_range_search:
            Rlims_new, RDnew, RInew = index.range_search(xq, radius, params=params)
            np.testing.assert_array_equal(Rlims_ref, Rlims_new)
            RDref, RIref = sort_range_res_2(Rlims_ref, RDref, RIref)

            if is_binary:
                # For binary indexes, check that all returned IDs are valid
                valid_ids = np.ones(len(RInew), dtype=bool)
                # Use vectorized operation instead of loop
                subset_set = set(subset)  # Convert to set for O(1) lookups
                valid_ids = np.isin(RInew, list(subset_set))

                self.assertTrue(np.all(valid_ids), "Some range search IDs are not in the subset")

                # Check that distances match
                np.testing.assert_almost_equal(RDref, RDnew, decimal=5)
            else:
                # For non-binary indexes, we can do exact comparison
                np.testing.assert_array_equal(RIref, RInew)
                np.testing.assert_almost_equal(RDref, RDnew, decimal=5)

    def test_IVFFlat(self):
        self.do_test_id_selector("IVF32,Flat")

    def test_IVFFlat_range_sorted(self):
        self.do_test_id_selector("IVF32,Flat", id_selector_type="range_sorted")

    def test_IVFPQ(self):
        self.do_test_id_selector("IVF32,PQ4x4np")

    def test_IVFPQfs(self):
        self.do_test_id_selector("IVF32,PQ4x4fs")

    def test_IVFPQfs_k1(self):
        self.do_test_id_selector("IVF32,PQ4x4fs", k=1)

    def test_IVFPQfs_k40(self):
        # test reservoir codepath
        self.do_test_id_selector("IVF32,PQ4x4fs", k=40)

    def test_IVFSQ(self):
        self.do_test_id_selector("IVF32,SQ8")

    def test_pretrans(self):
        self.do_test_id_selector("PCA16,IVF32,Flat")

    def test_SQ(self):
        self.do_test_id_selector("SQ8")

    def test_Flat(self):
        self.do_test_id_selector("Flat")

    def test_Flat_IP(self):
        self.do_test_id_selector("Flat", mt=faiss.METRIC_INNER_PRODUCT)

    def test_Flat_id_range(self):
        self.do_test_id_selector("Flat", id_selector_type="range")

    def test_Flat_IP_id_range(self):
        self.do_test_id_selector(
            "Flat", id_selector_type="range",
            mt=faiss.METRIC_INNER_PRODUCT
        )

    def test_Flat_id_array(self):
        self.do_test_id_selector("Flat", id_selector_type="array")

    def test_Flat_IP_id_array(self):
        self.do_test_id_selector(
            "Flat", id_selector_type="array",
            mt=faiss.METRIC_INNER_PRODUCT
        )

    def test_Flat_id_bitmap(self):
        self.do_test_id_selector("Flat", id_selector_type="bitmap")

    def test_Flat_id_not(self):
        self.do_test_id_selector("Flat", id_selector_type="not")

    def test_Flat_id_or(self):
        self.do_test_id_selector("Flat", id_selector_type="or")

    # not implemented

    # def test_PQ(self):
    #    self.do_test_id_selector("PQ4x4np")

    # def test_AQ(self):
    #    self.do_test_id_selector("RQ3x4")

    def do_test_id_selector_weak(self, index_key):
        """ verify that the selected subset is the subset  in the list"""
        ds = datasets.SyntheticDataset(32, 1000, 100, 20)
        index = faiss.index_factory(ds.d, index_key)
        index.train(ds.get_train())
        index.add(ds.get_database())
        k = 10
        Dref, Iref = index.search(ds.get_queries(), k)

        # reference result
        rs = np.random.RandomState(123)
        subset = rs.choice(ds.nb, 50, replace=False).astype("int64")
        sel = faiss.IDSelectorBatch(subset)
        params = faiss.SearchParametersHNSW()
        params.sel = sel
        Dnew, Inew = index.search(ds.get_queries(), k, params=params)
        mask = np.zeros(ds.nb, dtype=bool)
        mask[subset] = True
        for q in range(len(Iref)):
            mask_q, = np.where(mask[Iref[q]])
            l = len(mask_q)
            np.testing.assert_array_equal(Iref[q, mask_q], Inew[q, :l])
            np.testing.assert_array_equal(Dref[q, mask_q], Dnew[q, :l])

    def test_HSNW(self):
        self.do_test_id_selector_weak("HNSW")

    def test_idmap(self):
        ds = datasets.SyntheticDataset(32, 100, 100, 20)
        rs = np.random.RandomState(123)
        ids = rs.choice(10000, size=100, replace=False)
        mask = ids % 2 == 0
        index = faiss.index_factory(ds.d, "IDMap,SQ8")
        index.train(ds.get_train())

        # ref result
        index.add_with_ids(ds.get_database()[mask], ids[mask])
        Dref, Iref = index.search(ds.get_queries(), 10)

        # with selector
        index.reset()
        index.add_with_ids(ds.get_database(), ids)

        valid_ids = ids[mask]
        sel = faiss.IDSelectorTranslated(
            index, faiss.IDSelectorBatch(valid_ids))

        Dnew, Inew = index.search(
            ds.get_queries(), 10,
            params=faiss.SearchParameters(sel=sel)
        )
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)

        # let the IDMap::search add the translation...
        Dnew, Inew = index.search(
            ds.get_queries(), 10,
            params=faiss.SearchParameters(sel=faiss.IDSelectorBatch(valid_ids))
        )
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)

    def test_bounds(self):
        # https://github.com/facebookresearch/faiss/issues/3156
        d = 64  # dimension
        nb = 100000  # database size
        xb = np.random.random((nb, d))
        index_ip = faiss.IndexFlatIP(d)
        index_ip.add(xb)
        index_l2 = faiss.IndexFlatIP(d)
        index_l2.add(xb)

        out_of_bounds_id = nb + 15  # + 14 or lower will work fine
        id_selector = faiss.IDSelectorArray([out_of_bounds_id])
        search_params = faiss.SearchParameters(sel=id_selector)

        # ignores out of bound, does not crash
        distances, indices = index_ip.search(xb[:2], k=3, params=search_params)
        distances, indices = index_l2.search(xb[:2], k=3, params=search_params)

    def test_BinaryFlat(self):
        self.do_test_id_selector("BinaryFlat")

    def test_BinaryFlat_id_range(self):
        self.do_test_id_selector("BinaryFlat", id_selector_type="range")

    def test_BinaryFlat_id_array(self):
        self.do_test_id_selector("BinaryFlat", id_selector_type="array")

    def test_BinaryFlat_no_heap(self):
        self.do_test_id_selector("BinaryFlat", use_heap=False)

class TestSearchParams(unittest.TestCase):

    def do_test_with_param(
            self, index_key, ps_params, params):
        """
        Test equivalence between setting
        1. param_name_2 = value with ParameterSpace
        2. pass in a SearchParameters with param_name = value
        """
        ds = datasets.SyntheticDataset(32, 1000, 100, 20)
        index = faiss.index_factory(ds.d, index_key)
        if index_key.startswith("PQ"):
            index.polysemous_training.n_iter = 50000
            index.polysemous_training.n_redo = 1
        index.train(ds.get_train())
        index.add(ds.get_database())

        I0, D0 = index.search(ds.get_queries(), 10)

        Dnew, Inew = index.search(ds.get_queries(), 10, params=params)

        # make sure rhe parameter does indeed change the result...
        self.assertFalse(np.all(Inew == I0))

        for param_name, value in ps_params.items():
            faiss.ParameterSpace().set_index_parameter(
                index, param_name, value)
        Dref, Iref = index.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

    def test_nprobe(self):
        self.do_test_with_param(
                "IVF32,Flat", {"nprobe": 3},
                faiss.SearchParametersIVF(nprobe=3))

    def test_efSearch(self):
        self.do_test_with_param(
            "HNSW", {"efSearch": 4},
            faiss.SearchParametersHNSW(efSearch=4))

    def test_quantizer_hnsw(self):
        self.do_test_with_param(
            "IVF200_HNSW,Flat",
            {"quantizer_efSearch": 5, "nprobe": 10},
            faiss.SearchParametersIVF(
                nprobe=10,
                quantizer_params=faiss.SearchParametersHNSW(
                    efSearch=5)
            )
        )

    def test_PQ_polysemous_ht(self):
        self.do_test_with_param(
            "PQ4x8",
            {"ht": 10},
            faiss.SearchParametersPQ(
                polysemous_ht=10,
                search_type=faiss.IndexPQ.ST_polysemous
            )
        )

    def test_max_codes(self):
        " tests whether the max nb codes is taken into account "
        ds = datasets.SyntheticDataset(32, 1000, 100, 20)
        index = faiss.index_factory(ds.d, "IVF32,Flat")
        index.train(ds.get_train())
        index.add(ds.get_database())

        stats = faiss.cvar.indexIVF_stats
        stats.reset()
        D0, I0 = index.search(
            ds.get_queries(), 10,
            params=faiss.SearchParametersIVF(nprobe=8)
        )
        ndis0 = stats.ndis
        target_ndis = ndis0 // ds.nq  # a few queries will be below, a few above
        for q in range(ds.nq):
            stats.reset()
            Dq, Iq = index.search(
                ds.get_queries()[q:q + 1], 10,
                params=faiss.SearchParametersIVF(
                    nprobe=8, max_codes=target_ndis
                )
            )
            self.assertLessEqual(stats.ndis, target_ndis)
            if stats.ndis < target_ndis:
                np.testing.assert_equal(I0[q], Iq[0])

    def test_ownership(self):
        # see https://github.com/facebookresearch/faiss/issues/2996
        subset = np.arange(0, 50)
        sel = faiss.IDSelectorBatch(subset)
        self.assertTrue(sel.this.own())
        params = faiss.SearchParameters(sel=sel)
        self.assertTrue(sel.this.own())  # otherwise mem leak!
        # this is a somewhat fragile test because it assumes the
        # gc decreases refcounts immediately.
        prev_count = sys.getrefcount(sel)
        del params
        new_count = sys.getrefcount(sel)
        self.assertEqual(new_count, prev_count - 1)

        # check for other objects as well
        sel1 = faiss.IDSelectorBatch([1, 2, 3])
        sel2 = faiss.IDSelectorBatch([4, 5, 6])
        sel = faiss.IDSelectorAnd(sel1, sel2)
        # make storage is still managed by python
        self.assertTrue(sel1.this.own())
        self.assertTrue(sel2.this.own())


class TestSelectorCallback(unittest.TestCase):

    def test(self):
        ds = datasets.SyntheticDataset(32, 1000, 100, 20)
        index = faiss.index_factory(ds.d, "IVF32,Flat")
        index.train(ds.get_train())
        k = 10
        rs = np.random.RandomState(123)
        subset = rs.choice(ds.nb, 50, replace=False)

        params = faiss.SearchParametersIVF(
            sel=faiss.IDSelectorBatch(subset),
            nprobe=4
        )

        Dref, Iref = index.search(ds.get_queries(), k, params=params)

        def is_member(idx):
            return idx in subset

        params = faiss.SearchParametersIVF(
            sel=faiss.PyCallbackIDSelector(is_member),
            nprobe=4
        )

        Dnew, Inew = index.search(ds.get_queries(), k, params=params)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_almost_equal(Dref, Dnew, decimal=5)


class TestSortedIDSelectorRange(unittest.TestCase):
    """ to test the sorted id bounds, there are a few cases to consider """

    def do_test_sorted(self, imin, imax, n=100):
        selr = faiss.IDSelectorRange(imin, imax, True)
        sp = faiss.swig_ptr
        for seed in range(10):
            rs = np.random.RandomState(seed)
            ids = rs.choice(30, n).astype('int64')
            ids.sort()
            j01 = np.zeros(2, dtype='uint64')
            selr.find_sorted_ids_bounds(
                len(ids), sp(ids), sp(j01[:1]), sp(j01[1:]))
            j0, j1 = j01.astype(int)
            ref_idx, = np.where((ids >= imin) & (ids < imax))
            np.testing.assert_array_equal(ref_idx, np.arange(j0, j1))

    def test_sorted_in_range(self):
        self.do_test_sorted(10, 20)

    def test_sorted_out_0(self):
        self.do_test_sorted(-10, 20)

    def test_sorted_out_1(self):
        self.do_test_sorted(10, 40)

    def test_sorted_in_range_smalln(self):
        self.do_test_sorted(10, 20, n=5)

    def test_12_92(self):
        selr = faiss.IDSelectorRange(30, 80, True)
        ids = np.array([12, 92], dtype='int64')
        j01 = np.zeros(2, dtype='uint64')
        sp = faiss.swig_ptr
        selr.find_sorted_ids_bounds(
            len(ids), sp(ids), sp(j01[:1]), sp(j01[1:]))
        assert j01[0] >= j01[1]


class TestPrecomputed(unittest.TestCase):

    def do_test_knn_and_range(self, factory, range=True):
        ds = datasets.SyntheticDataset(32, 10000, 100, 20)
        index = faiss.index_factory(ds.d, factory)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 5
        Dref, Iref = index.search(ds.get_queries(), 10)

        Dq, Iq = index.quantizer.search(ds.get_queries(), index.nprobe)
        Dnew, Inew = index.search_preassigned(ds.get_queries(), 10, Iq, Dq)
        np.testing.assert_equal(Iref, Inew)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-5)

        if range:
            r2 = float(np.median(Dref[:, 5]))
            Lref, Dref, Iref = index.range_search(ds.get_queries(), r2)
            assert Lref.size > 10   # make sure there is something to test...

            Lnew, Dnew, Inew = index.range_search_preassigned(ds.get_queries(), r2, Iq, Dq)
            check_ref_range_results(
                Lref, Dref, Iref,
                Lnew, Dnew, Inew
            )

    def test_knn_and_range_Flat(self):
        self.do_test_knn_and_range("IVF32,Flat")

    def test_knn_and_range_SQ(self):
        self.do_test_knn_and_range("IVF32,SQ8")

    def test_knn_and_range_PQ(self):
        self.do_test_knn_and_range("IVF32,PQ8x4np")

    def test_knn_and_range_FS(self):
        self.do_test_knn_and_range("IVF32,PQ8x4fs", range=False)
