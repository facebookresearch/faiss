# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import numpy as np
import faiss

from common_faiss_tests import for_all_simd_levels


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
