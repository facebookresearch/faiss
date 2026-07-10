# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import unittest
import faiss

from common_faiss_tests import for_all_simd_levels
from faiss.contrib import datasets, evaluation


@for_all_simd_levels
class TestDistanceComputer(unittest.TestCase):

    def do_test(self, factory_string, metric_type=faiss.METRIC_L2):
        ds = datasets.SyntheticDataset(32, 1000, 200, 20)

        index = faiss.index_factory(32, factory_string, metric_type)
        index.train(ds.get_train())
        index.add(ds.get_database())
        xq = ds.get_queries()
        Dref, Iref = index.search(xq, 10)

        for is_FlatCodesDistanceComputer in False, True:
            if not is_FlatCodesDistanceComputer:
                dc = index.get_distance_computer()
            else:
                if not isinstance(index, faiss.IndexFlatCodes):
                    continue
                dc = index.get_FlatCodesDistanceComputer()
            self.assertTrue(dc.this.own())
            for q in range(ds.nq):
                dc.set_query(faiss.swig_ptr(xq[q]))
                for j in range(10):
                    ref_dis = Dref[q, j]
                    new_dis = dc(int(Iref[q, j]))
                    np.testing.assert_almost_equal(new_dis, ref_dis, decimal=5)

    def test_distance_computer_PQ(self):
        self.do_test("PQ8np")

    def test_distance_computer_SQ(self):
        self.do_test("SQ8")

    def test_distance_computer_SQ6(self):
        self.do_test("SQ6")

    def test_distance_computer_PQbit6(self):
        self.do_test("PQ8x6np")

    def test_distance_computer_PQbit6_ip(self):
        self.do_test("PQ8x6np", faiss.METRIC_INNER_PRODUCT)

    def test_distance_computer_VT(self):
        self.do_test("PCA20,SQ8")

    def test_distance_computer_AQ_decompress(self):
        self.do_test("RQ3x4")  # test decompress path

    def test_distance_computer_AQ_LUT(self):
        self.do_test("RQ3x4_Nqint8")  # test LUT path

    def test_distance_computer_AQ_LUT_IP(self):
        self.do_test("RQ3x4_Nqint8", faiss.METRIC_INNER_PRODUCT)


@for_all_simd_levels
class TestIndexRefineSearchParams(unittest.TestCase):

    def do_test(self, factory_string):
        ds = datasets.SyntheticDataset(32, 256, 100, 40)

        index = faiss.index_factory(32, factory_string)
        index.train(ds.get_train())
        index.add(ds.get_database())

        # Set nprobe on the base index (for IndexRefine, nprobe belongs to
        # the IVF base index)
        if hasattr(index, "base_index") and hasattr(index.base_index, "nprobe"):
            index.base_index.nprobe = 4
        elif hasattr(index, "nprobe"):
            index.nprobe = 4

        xq = ds.get_queries()

        # do a search with k_factor = 1
        D1, I1 = index.search(xq, 10)
        inter1 = faiss.eval_intersection(I1, ds.get_groundtruth(10))

        # do a search with k_factor = 1.5
        params = faiss.IndexRefineSearchParameters(k_factor=1.1)
        D2, I2 = index.search(xq, 10, params=params)
        inter2 = faiss.eval_intersection(I2, ds.get_groundtruth(10))

        # do a search with k_factor = 2
        params = faiss.IndexRefineSearchParameters(k_factor=2)
        D3, I3 = index.search(xq, 10, params=params)
        inter3 = faiss.eval_intersection(I3, ds.get_groundtruth(10))

        # make sure that the recall rate increases with k_factor
        self.assertGreater(inter2, inter1)
        self.assertGreater(inter3, inter2)

        # make sure that the baseline k_factor is unchanged
        self.assertEqual(index.k_factor, 1)

        # try passing params for the baseline index, change nprobe
        base_params = faiss.IVFSearchParameters(nprobe=10)
        params = faiss.IndexRefineSearchParameters(
            k_factor=1, base_index_params=base_params
        )
        D4, I4 = index.search(xq, 10, params=params)
        inter4 = faiss.eval_intersection(I4, ds.get_groundtruth(10))

        base_params = faiss.IVFSearchParameters(nprobe=2)
        params = faiss.IndexRefineSearchParameters(
            k_factor=1, base_index_params=base_params
        )
        D5, I5 = index.search(xq, 10, params=params)
        inter5 = faiss.eval_intersection(I5, ds.get_groundtruth(10))

        # make sure that the recall rate changes
        self.assertNotEqual(inter4, inter5)

    def test_rflat(self):
        # flat is handled by the IndexRefineFlat class
        self.do_test("IVF8,PQ2x4np,RFlat")

    def test_refine_sq8(self):
        # this case uses the IndexRefine class
        self.do_test("IVF8,PQ2x4np,Refine(SQ8)")


@for_all_simd_levels
class TestIndexRefineRangeSearch(unittest.TestCase):

    def do_test(self, factory_string):
        d = 32
        radius = 8

        ds = datasets.SyntheticDataset(d, 1024, 512, 256)

        index = faiss.index_factory(d, factory_string)
        index.train(ds.get_train())
        index.add(ds.get_database())
        xq = ds.get_queries()
        xb = ds.get_database()

        # perform a range_search
        lims_1, D1, I1 = index.range_search(xq, radius)

        # create a baseline (FlatL2)
        index_flat = faiss.IndexFlatL2(d)
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())

        lims_ref, Dref, Iref = index_flat.range_search(xq, radius)

        # add a refine index on top of the index
        index_r = faiss.IndexRefine(index, index_flat)
        lims_2, D2, I2 = index_r.range_search(xq, radius)

        # validate: refined range_search() keeps indices untouched
        precision_1, recall_1 = evaluation.range_PR(lims_ref, Iref, lims_1, I1)

        precision_2, recall_2 = evaluation.range_PR(lims_ref, Iref, lims_2, I2)

        self.assertAlmostEqual(recall_1, recall_2)

        # validate: refined range_search() updates distances, and new distances are correct L2 distances
        for iq in range(0, ds.nq):
            start_lim = lims_2[iq]
            end_lim = lims_2[iq + 1]
            for i_lim in range(start_lim, end_lim):
                idx = I2[i_lim]
                l2_dis = np.sum(
                    np.square(xq[iq : iq + 1,] - xb[idx : idx + 1,])
                )

                self.assertAlmostEqual(l2_dis, D2[i_lim], places=4)

                # every returned result must be within the radius: candidates
                # picked by approximate distance can exceed it once refined,
                # and those must be dropped (issue #5367)
                self.assertLessEqual(D2[i_lim], radius)

    def test_refine_1(self):
        self.do_test("SQ4")

    def test_refine_sq6(self):
        # coarser quantization makes approximate/exact distance mismatches
        # near the radius boundary more likely, exercising the filtering
        self.do_test("SQ6")

    def do_test_k_factor(
        self,
        factory_string,
        metric=faiss.METRIC_L2,
        radius=8,
    ):
        d = 32

        ds = datasets.SyntheticDataset(d, 1024, 512, 256)

        index = faiss.index_factory(d, factory_string, metric)
        index.train(ds.get_train())
        index.add(ds.get_database())
        xq = ds.get_queries()

        is_similarity = metric == faiss.METRIC_INNER_PRODUCT
        index_flat = (
            faiss.IndexFlatIP(d) if is_similarity else faiss.IndexFlatL2(d)
        )
        index_flat.add(ds.get_database())
        lims_ref, Dref, Iref = index_flat.range_search(xq, radius)

        # guard against a vacuous test: the radius must select some but not all
        # candidates, otherwise the exact filter and k_factor widening below are
        # never meaningfully exercised
        self.assertGreater(len(Iref), 0)
        self.assertLess(len(Iref), ds.nq * ds.nb)

        index_r = faiss.IndexRefine(index, index_flat)

        # k_factor scales the radius used to query the base index (base_radius =
        # radius * k_factor); the results are still filtered at the requested
        # radius, so every returned distance must stay on the correct side of it
        # regardless of k_factor (issue #5367). For an L2 base a larger k_factor
        # widens the base search and can only add candidates, so recall must not
        # decrease; for a similarity base radius * k_factor does not necessarily
        # widen the search, so no recall guarantee is asserted there.
        recalls = []
        for k_factor in (1, 4):
            index_r.k_factor = k_factor
            lims, D, I = index_r.range_search(xq, radius)
            _, recall = evaluation.range_PR(lims_ref, Iref, lims, I)
            recalls.append(recall)
            for iq in range(ds.nq):
                for i_lim in range(lims[iq], lims[iq + 1]):
                    if is_similarity:
                        self.assertGreaterEqual(D[i_lim], radius)
                    else:
                        self.assertLessEqual(D[i_lim], radius)

        if not is_similarity:
            # widening the L2 base radius does not lose any results
            self.assertGreaterEqual(recalls[1] + 1e-6, recalls[0])

    def test_refine_k_factor(self):
        self.do_test_k_factor("SQ4")

    def test_refine_k_factor_ip(self):
        # an inner-product base exercises the is_similarity (CMin) branch of the
        # exact radius filter, with both a positive and a negative radius.
        # Inner products on this dataset span roughly [-5, +5], so these radii
        # select non-trivial result sets on either side of zero.
        self.do_test_k_factor("SQ4", faiss.METRIC_INNER_PRODUCT, radius=4)
        self.do_test_k_factor("SQ4", faiss.METRIC_INNER_PRODUCT, radius=-2)
