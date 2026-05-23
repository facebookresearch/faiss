# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest
import sys

from common_faiss_tests import for_all_simd_levels
from faiss.contrib import datasets
from faiss.contrib.evaluation import sort_range_res_2, check_ref_range_results

faiss.omp_set_num_threads(4)


@for_all_simd_levels
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


@for_all_simd_levels
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

        # make sure the parameter does indeed change the result...
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


class TestIVFEarlyTermination(unittest.TestCase):
    """
    Coverage for the new SearchParametersIVF early-stop fields:
      - ensure_topk_full
      - max_empty_result_buckets
      - max_lists_num
    """

    def _build(self, factory, mt=faiss.METRIC_L2, d=32):
        ds = datasets.SyntheticDataset(d, 2000, 500, 30)
        index = faiss.index_factory(d, factory, mt)
        index.train(ds.get_train())
        index.add(ds.get_database())
        return ds, index

    def test_ensure_topk_full_fills_heap(self):
        ds, index = self._build("IVF32,Flat")
        k = 10
        params = faiss.SearchParametersIVF()
        params.nprobe = 32
        params.max_codes = k // 2    # tight budget: forces truncation
        params.ensure_topk_full = False
        _, I_tight = index.search(ds.get_queries(), k, params=params)

        params.ensure_topk_full = True
        _, I_full = index.search(ds.get_queries(), k, params=params)

        # ensure_topk_full=true must produce at least as many valid slots.
        miss_tight = np.sum(I_tight == -1)
        miss_full = np.sum(I_full == -1)
        self.assertLessEqual(miss_full, miss_tight)
        # Test precondition: tight budget produced some misses.
        self.assertGreater(miss_tight, 0)

    def test_range_default_matches_baseline(self):
        """Default max_empty_result_buckets=0 must preserve pre-existing
        range_search results byte-for-byte."""
        ds, index = self._build("IVF32,Flat")
        # Set nprobe via the index attribute for the no-params baseline.
        index.nprobe = 8
        radius = 2.0
        Lref, Dref, Iref = index.range_search(ds.get_queries(), radius)

        params = faiss.SearchParametersIVF()
        params.nprobe = 8
        # all new fields are at defaults
        Lnew, Dnew, Inew = index.range_search(
            ds.get_queries(), radius, params=params
        )

        np.testing.assert_array_equal(Lref, Lnew)
        # Per-query (id, distance) pairs must match as sorted sets since
        # within-query order is not guaranteed across OMP threads.
        for q in range(ds.nq):
            a = sorted(zip(
                Iref[Lref[q]: Lref[q + 1]].tolist(),
                Dref[Lref[q]: Lref[q + 1]].tolist(),
            ))
            b = sorted(zip(
                Inew[Lnew[q]: Lnew[q + 1]].tolist(),
                Dnew[Lnew[q]: Lnew[q + 1]].tolist(),
            ))
            self.assertEqual(a, b)

    def test_range_early_exit_is_subset(self):
        ds, index = self._build("IVF32,Flat")
        # Radius chosen empirically against this synthetic dataset
        # (d=32, nb=500, IVF32) so that several probed buckets per
        # query yield zero in-radius hits — required to exercise
        # the empty-bucket exit path. Too-tight or too-wide radii
        # leave the test vacuous.
        radius = 15.0
        p_full = faiss.SearchParametersIVF()
        p_full.nprobe = 32
        p_full.max_empty_result_buckets = 0    # disabled
        Lf, Df, If = index.range_search(
            ds.get_queries(), radius, params=p_full
        )

        p_early = faiss.SearchParametersIVF()
        p_early.nprobe = 32
        p_early.max_empty_result_buckets = 1   # aggressive early-exit
        Le, De, Ie = index.range_search(
            ds.get_queries(), radius, params=p_early
        )

        # Early-exit must never expand the result set.
        self.assertTrue(np.all(np.diff(Le) <= np.diff(Lf)))
        # And each per-query id-set must be a subset of the full result.
        for q in range(ds.nq):
            full = set(If[Lf[q]: Lf[q + 1]].tolist())
            early = set(Ie[Le[q]: Le[q + 1]].tolist())
            self.assertTrue(early.issubset(full))
        # At least one query must strictly shrink.
        any_shrunk = bool(np.any(np.diff(Le) < np.diff(Lf)))
        self.assertTrue(
            any_shrunk,
            "no query shrank: max_empty_result_buckets has no effect",
        )

    def test_range_early_exit_counter_resets_on_hit(self):
        """Tightening max_empty_result_buckets (1 vs 3) must never
        produce *more* results. With 3 the loop tolerates two
        intervening empty buckets between non-empty ones; with 1 it
        terminates on the first empty bucket. Per-query result
        counts must therefore satisfy tight ≤ relaxed."""
        ds, index = self._build("IVF32,Flat")
        radius = 15.0
        p_tight = faiss.SearchParametersIVF()
        p_tight.nprobe = 32
        p_tight.max_empty_result_buckets = 1
        Lt, _, _ = index.range_search(
            ds.get_queries(), radius, params=p_tight
        )

        p_relaxed = faiss.SearchParametersIVF()
        p_relaxed.nprobe = 32
        p_relaxed.max_empty_result_buckets = 3
        Lr, _, _ = index.range_search(
            ds.get_queries(), radius, params=p_relaxed
        )

        self.assertTrue(np.all(np.diff(Lt) <= np.diff(Lr)))

    def test_range_rejects_non_default_parallel_mode(self):
        ds, index = self._build("IVF32,Flat")
        ivf = faiss.downcast_index(index)
        ivf.parallel_mode = 1

        params = faiss.SearchParametersIVF()
        params.nprobe = 4
        params.max_empty_result_buckets = 2
        with self.assertRaises(RuntimeError):
            index.range_search(ds.get_queries()[:1], 1.0, params=params)

    def test_ndis_stats_post_filter(self):
        """IndexIVFStats::ndis is documented as 'nb of distances
        computed' — a post-filter quantity. With a 50%-keep selector,
        ndis must drop because rejected codes don't have a distance
        computed."""
        ds, index = self._build("IVF32,Flat")
        index.nprobe = 32
        stats = faiss.cvar.indexIVF_stats

        stats.reset()
        index.search(ds.get_queries(), 10)
        ndis_no_sel = stats.ndis
        self.assertGreater(ndis_no_sel, 0)

        # 50%-keep batch selector.
        keep = np.arange(0, ds.nb, 2).astype("int64")
        sel = faiss.IDSelectorBatch(keep)
        params = faiss.SearchParametersIVF(nprobe=32, sel=sel)
        stats.reset()
        index.search(ds.get_queries(), 10, params=params)
        ndis_with_sel = stats.ndis

        self.assertLess(ndis_with_sel, ndis_no_sel)
        # Loose-bound sanity: ~50% retention should drop ndis at least
        # noticeably below the no-selector value.
        self.assertLess(ndis_with_sel * 4, ndis_no_sel * 3)

    def test_ensure_topk_full_with_restrictive_selector(self):
        """Under a 10%-keep selector and
        max_codes < k, the heap must still be filled (no -1 entries)
        because the post-filter break condition keeps probing until k
        valid candidates have been observed."""
        ds, index = self._build("IVF32,Flat")
        # 10%-keep selector — 50 valid ids in 500-vector dataset.
        keep = np.arange(0, ds.nb, 10).astype("int64")
        sel = faiss.IDSelectorBatch(keep)

        params = faiss.SearchParametersIVF()
        params.nprobe = 32
        params.max_codes = 5            # tight pre-filter budget
        params.ensure_topk_full = True  # post-filter soft cap
        params.sel = sel
        _, I = index.search(ds.get_queries(), 10, params=params)

        self.assertEqual(int(np.sum(I == -1)), 0)

    def test_fields_exposed_and_defaulted(self):
        p = faiss.SearchParametersIVF()
        self.assertEqual(p.max_lists_num, 0)
        self.assertIs(p.ensure_topk_full, False)
        self.assertEqual(p.max_empty_result_buckets, 0)
        p.max_lists_num = 5
        p.ensure_topk_full = True
        p.max_empty_result_buckets = 2
        self.assertEqual(p.max_lists_num, 5)
        self.assertIs(p.ensure_topk_full, True)
        self.assertEqual(p.max_empty_result_buckets, 2)
