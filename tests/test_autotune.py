# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import numpy as np
import faiss


class TestParameterSpace(unittest.TestCase):

    def test_nprobe_2(self):
        index = faiss.index_factory(32, "IDMap,IVF32,Flat")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_index(index.index)
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


class TestAutoTuneCriterion(unittest.TestCase):

    def test_set_groundtruth(self):
        nq, gt_nnn = 10, 5
        crit = faiss.OneRecallAtRCriterion(nq, gt_nnn)
        rng = np.random.default_rng(42)
        gt_I = rng.integers(0, 100, size=(nq, gt_nnn)).astype("int64")
        gt_D = rng.random(size=(nq, gt_nnn)).astype("float32")
        # gt_D is documented as optional (None skips the distances copy).
        crit.set_groundtruth(None, gt_I)
        self.assertEqual(crit.gt_nnn, gt_nnn)
        self.assertEqual(crit.gt_I.size(), nq * gt_nnn)
        crit.set_groundtruth(gt_D, gt_I)
        self.assertEqual(crit.gt_D.size(), nq * gt_nnn)


class TestOperatingPoints(unittest.TestCase):
    def _keys(self, ops):
        n = ops.optimal_pts.size()
        return [ops.optimal_pts.at(i).key for i in range(n)]

    def test_middle_insert_equal_time_removes_dominated(self):
        # P3 is inserted before P2 (else-branch) and prunes P1 (same time)
        ops = faiss.OperatingPoints()
        self.assertTrue(ops.add(0.5, 1.0, "P1"))
        self.assertTrue(ops.add(0.9, 2.0, "P2"))
        self.assertTrue(ops.add(0.7, 1.0, "P3"))
        self.assertEqual(self._keys(ops), ["", "P3", "P2"])

    def test_worse_perf_at_equal_time_rejected(self):
        ops = faiss.OperatingPoints()
        self.assertTrue(ops.add(0.9, 1.0, "good"))
        admitted = ops.add(0.5, 1.0, "bad")
        self.assertFalse(admitted)
        self.assertNotIn("bad", self._keys(ops))
