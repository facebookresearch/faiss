# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss

from common_faiss_tests import for_all_simd_levels
from faiss.contrib import datasets

faiss.omp_set_num_threads(4)


@for_all_simd_levels
class TestFastScanFiltering(unittest.TestCase):

    def do_test_filter(
        self,
        index_key,
        id_selector_type="batch",
        mt=faiss.METRIC_L2,
        k=10,
        nb=1000,
    ):
        """
        Build a fastscan IVF index and search with IDSelector filtering.
        Verify all returned results are in the allowed set.
        """
        d = 32
        ds = datasets.SyntheticDataset(d, 2000, nb, 20, metric=mt)

        index = faiss.index_factory(d, index_key, mt)
        index.train(ds.get_train())
        index.add(ds.get_database())

        # Create selector and allowed set
        rs = np.random.RandomState(123)
        if id_selector_type == "batch":
            subset = rs.choice(nb, nb // 3, replace=False).astype("int64")
            sel = faiss.IDSelectorBatch(subset)
            allowed = set(subset.tolist())
        elif id_selector_type == "range":
            lo, hi = nb // 4, 3 * nb // 4
            sel = faiss.IDSelectorRange(lo, hi)
            allowed = set(range(lo, hi))
        elif id_selector_type == "not_batch":
            # Exclude entire 32-vector blocks to test block-skip
            excluded = np.concatenate(
                [
                    np.arange(0, 32, dtype="int64"),
                    np.arange(64, 96, dtype="int64"),
                ]
            )
            inner_sel = faiss.IDSelectorBatch(excluded)
            sel = faiss.IDSelectorNot(inner_sel)
            allowed = {i for i in range(nb) if i not in excluded}
        elif id_selector_type == "partial_batch":
            # Exclude a few IDs within a single block (not a whole block)
            excluded = np.array([5, 10, 20, 31], dtype="int64")
            inner_sel = faiss.IDSelectorBatch(excluded)
            sel = faiss.IDSelectorNot(inner_sel)
            allowed = {i for i in range(nb) if i not in excluded}
        elif id_selector_type == "empty":
            sel = faiss.IDSelectorBatch(np.array([], dtype="int64"))
            allowed = set()
        else:
            raise ValueError(f"Unknown selector type: {id_selector_type}")

        params = faiss.SearchParametersIVF(sel=sel, nprobe=8)
        Dfs, Ifs = index.search(ds.get_queries(), k, params=params)

        # Check: all results are in the allowed set or -1
        for q in range(ds.nq):
            for j in range(k):
                idx = int(Ifs[q, j])
                if idx >= 0:
                    self.assertIn(
                        idx,
                        allowed,
                        f"Query {q}, rank {j}: got id {idx} not in allowed set",
                    )

        # If allowed set is large enough, expect some valid results
        if len(allowed) > k:
            valid = np.sum(Ifs >= 0)
            self.assertGreater(valid, 0, "Expected some valid results")

        return Ifs

    def test_IVFPQfs_batch_IP(self):
        self.do_test_filter(
            "IVF32,PQ4x4fs", "batch", faiss.METRIC_INNER_PRODUCT
        )

    def test_IVFPQfs_range(self):
        self.do_test_filter("IVF32,PQ4x4fs", "range")

    def test_IVFPQfs_not_batch(self):
        """IDSelectorNot excluding whole blocks -> block-skip path"""
        self.do_test_filter("IVF32,PQ4x4fs", "not_batch")

    def test_IVFPQfs_partial_batch(self):
        """IDSelectorNot excluding a few IDs within a block"""
        self.do_test_filter("IVF32,PQ4x4fs", "partial_batch")

    def test_IVFPQfs_empty_selector(self):
        """Empty selector accepts nothing -> all results -1"""
        Ifs = self.do_test_filter("IVF32,PQ4x4fs", "empty")
        np.testing.assert_array_equal(Ifs, -1)
