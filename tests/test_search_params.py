# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest

from faiss.contrib import datasets
from faiss.contrib.evaluation import sort_range_res_2

faiss.omp_set_num_threads(4)


class TestSelector(unittest.TestCase):

    def do_test_id_selector(self, index_key):
        """ Verify that the id selector returns the subset of results that are
        members according to the IDSelector
        """
        ds = datasets.SyntheticDataset(32, 1000, 100, 20)
        index = faiss.index_factory(ds.d, index_key)
        index.train(ds.get_train())
        k = 10

        # reference result
        rs = np.random.RandomState(123)
        subset = rs.choice(ds.nb, 50, replace=False).astype("int64")
        # add_with_ids not supported for all index types
        # index.add_with_ids(ds.get_database()[subset], subset)
        index.add(ds.get_database()[subset])
        Dref, Iref0 = index.search(ds.get_queries(), k)
        Iref = subset[Iref0]
        Iref[Iref0 < 0] = -1

        radius = float(Dref[Iref > 0].max())
        try:
            Rlims_ref, RDref, RIref = index.range_search(
                ds.get_queries(), radius)
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
        index.add(ds.get_database())
        sel = faiss.IDSelectorBatch(
            len(subset),
            faiss.swig_ptr(subset)
        )
        params = faiss.IVFSearchParameters()
        params.sel = sel
        Dnew, Inew = index.search(ds.get_queries(), k, params=params)
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

        if have_range_search:
            Rlims_new, RDnew, RInew = index.range_search(
                ds.get_queries(), radius, params=params)
            np.testing.assert_array_equal(Rlims_ref, Rlims_new)
            RDref, RIref = sort_range_res_2(Rlims_ref, RDref, RIref)
            np.testing.assert_array_equal(RIref, RInew)
            np.testing.assert_array_equal(RDref, RDnew)

    def test_IVFFlat(self):
        self.do_test_id_selector("IVF32,Flat")

    def test_IVFPQ(self):
        self.do_test_id_selector("IVF32,PQ4x4np")

    def test_IVFSQ(self):
        self.do_test_id_selector("IVF32,SQ8")

    def test_pretrans(self):
        self.do_test_id_selector("PCA16,IVF32,Flat")

    def test_SQ(self):
        self.do_test_id_selector("SQ8")

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
        sel = faiss.IDSelectorBatch(
            len(subset),
            faiss.swig_ptr(subset)
        )
        params = faiss.SearchParametersHNSW()
        params.sel = sel
        Dnew, Inew = index.search(ds.get_queries(), k, params=params)
        mask = np.zeros(ds.nb, dtype=bool)
        mask[subset] = True
        for q in range(len(Iref)):
            mask_q, = np.where(mask[Iref[q]])
            np.testing.assert_array_equal(Iref[q, mask_q], Inew[q, :len(mask_q)])
            np.testing.assert_array_equal(Dref[q, mask_q], Dnew[q, :len(mask_q)])

    def test_HSNW(self):
        self.do_test_id_selector_weak("HNSW")


class TestParamsWrappers(unittest.TestCase):

    def do_test_with_param(
            self, index_key, ps_params, params):
        """
        Test equivalence between setting
        1. param_name_2 = value with ParameterSpace
        2. pass in a SearchParameters with param_name = value
        """
        ds = datasets.SyntheticDataset(32, 1000, 100, 20)
        index = faiss.index_factory(ds.d, index_key)
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
