# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import faiss

from common_faiss_tests import for_all_simd_levels
from faiss.contrib import datasets


@for_all_simd_levels
class TestParameterSpace(unittest.TestCase):

    def test_nprobe(self):
        index = faiss.index_factory(32, "IVF32,Flat")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        self.assertEqual(index.nprobe, 5)

    def test_nprobe_2(self):
        index = faiss.index_factory(32, "IDMap,IVF32,Flat")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_index(index.index)
        self.assertEqual(index2.nprobe, 5)

    def test_nprobe_3(self):
        index = faiss.index_factory(32, "IVF32,SQ8,RFlat")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_index(index.base_index)
        self.assertEqual(index2.nprobe, 5)

    def test_nprobe_4(self):
        index = faiss.index_factory(32, "PCAR32,IVF32,SQ8,RFlat")
        ps = faiss.ParameterSpace()

        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_index(index.base_index)
        index2 = faiss.downcast_index(index2.index)
        self.assertEqual(index2.nprobe, 5)

    def test_efSearch(self):
        index = faiss.index_factory(32, "IVF32_HNSW32,SQ8")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "quantizer_efSearch", 5)
        index2 = faiss.downcast_index(index.quantizer)
        self.assertEqual(index2.hnsw.efSearch, 5)

    def test_update_search_parameters(self):
        """update_search_parameters sets params.nprobe to the value encoded in cno."""
        index = faiss.index_factory(32, "IVF32,Flat")
        ps = faiss.ParameterSpace()
        ps.initialize(index)

        # find the nprobe ParameterRange and its values
        nprobe_range = next(
            ps.parameter_ranges.at(i)
            for i in range(ps.parameter_ranges.size())
            if ps.parameter_ranges.at(i).name == "nprobe"
        )

        params = faiss.SearchParametersIVF()
        for cno in range(ps.n_combinations()):
            expected_nprobe = int(nprobe_range.values.at(cno))
            ps.update_search_parameters(params, cno)
            self.assertEqual(params.nprobe, expected_nprobe)

class TestAutoTuneCriterion(unittest.TestCase):

    def test_set_groundtruth_with_distances(self):
        nq, gt_nnn = 10, 5
        crit = faiss.OneRecallAtRCriterion(nq, gt_nnn)
        rng = np.random.default_rng(42)
        gt_I = rng.integers(0, 100, size=(nq, gt_nnn)).astype("int64")
        gt_D = rng.random(size=(nq, gt_nnn)).astype("float32")
        crit.set_groundtruth(gt_D, gt_I)
        self.assertEqual(crit.gt_nnn, gt_nnn)
        self.assertEqual(crit.gt_I.size(), nq * gt_nnn)

    def test_set_groundtruth_null_distances(self):
        # gt_D is documented as optional (None skips the distances copy).
        nq, gt_nnn = 10, 5
        crit = faiss.OneRecallAtRCriterion(nq, gt_nnn)
        rng = np.random.default_rng(42)
        gt_I = rng.integers(0, 100, size=(nq, gt_nnn)).astype("int64")
        crit.set_groundtruth(None, gt_I)
        self.assertEqual(crit.gt_nnn, gt_nnn)
        self.assertEqual(crit.gt_I.size(), nq * gt_nnn)


class TestOperatingPoints(unittest.TestCase):
    def _keys(self, ops):
        n = ops.optimal_pts.size()
        return [ops.optimal_pts.at(i).key for i in range(n)]

    def test_equal_time_better_perf_removes_dominated(self):
        # B (perf=0.9, t=1.0) dominates A (perf=0.5, t=1.0) — same time,
        # strictly better perf.  A must be pruned from the optimal frontier.
        ops = faiss.OperatingPoints()
        self.assertTrue(ops.add(0.5, 1.0, "A"))
        self.assertTrue(ops.add(0.9, 1.0, "B"))
        keys = self._keys(ops)
        # origin ("") + B only — A must not survive
        self.assertEqual(len(keys), 2)
        self.assertNotIn("A", keys, "A dominated by B; must not remain")
        self.assertIn("B", keys)

    def test_middle_insert_equal_time_removes_dominated(self):
        # Frontier has P1=(0.5, t=1.0) and P2=(0.9, t=2.0).  Adding
        # P3=(0.7, t=1.0) inserts P3 in the middle and must prune P1 (same
        # time, worse perf) — exercises the else-branch insertion path.
        ops = faiss.OperatingPoints()
        self.assertTrue(ops.add(0.5, 1.0, "P1"))
        self.assertTrue(ops.add(0.9, 2.0, "P2"))
        self.assertTrue(ops.add(0.7, 1.0, "P3"))
        keys = self._keys(ops)
        # origin ("") + P3 + P2 — P1 dominated by P3 at equal time
        self.assertEqual(len(keys), 3)
        self.assertNotIn("P1", keys, "P1 dominated by P3; must not remain")
        self.assertIn("P3", keys)
        self.assertIn("P2", keys)

    def test_normal_pareto_frontier_preserved(self):
        # Two points where neither dominates the other: higher perf at higher
        # cost vs lower perf at lower cost.  Both must remain on the frontier.
        ops = faiss.OperatingPoints()
        self.assertTrue(ops.add(0.5, 0.5, "approximate"))
        self.assertTrue(ops.add(0.9, 2.0, "accurate"))
        keys = self._keys(ops)
        self.assertIn("approximate", keys)
        self.assertIn("accurate", keys)

    def test_worse_perf_at_equal_time_rejected(self):
        # A point with worse perf and equal time is not admitted.
        ops = faiss.OperatingPoints()
        self.assertTrue(ops.add(0.9, 1.0, "good"))
        admitted = ops.add(0.5, 1.0, "bad")
        self.assertFalse(admitted)
        self.assertNotIn("bad", self._keys(ops))
