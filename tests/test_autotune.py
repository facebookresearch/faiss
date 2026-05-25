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

    def test_explore_with_params(self):
        """SearchParameters forwarded through explore restricts results."""
        d = 32
        ds = datasets.SyntheticDataset(d, 1000, 200, 50)
        index = faiss.index_factory(d, "IVF32,Flat")
        index.train(ds.get_train())
        index.add(ds.get_database())

        xq = ds.get_queries()
        k = 10

        gt_D, gt_I = index.search(xq, k)
        crit = faiss.OneRecallAtRCriterion(ds.nq, k)
        crit.set_groundtruth(gt_D, gt_I)

        ps = faiss.ParameterSpace()
        ps.initialize(index)
        ps.n_experiments = 0
        ps.verbose = 0

        def best_perf(ops):
            n = ops.optimal_pts.size()
            return ops.optimal_pts.at(n - 1).perf

        # without params: should achieve high recall
        ops_full = ps.explore(index, xq, crit)
        self.assertGreater(best_perf(ops_full), 0.9)

        # exclude all ground-truth IDs: recall must drop to 0
        gt_ids = set(int(v) for v in gt_I.flatten() if v >= 0)
        non_gt = np.array(
            [i for i in range(ds.nb) if i not in gt_ids],
            dtype='int64'
        )
        sel = faiss.IDSelectorBatch(non_gt)
        params = faiss.SearchParametersIVF(sel=sel)
        ops_sel = ps.explore(index, xq, crit, params=params)
        self.assertEqual(best_perf(ops_sel), 0.0)
