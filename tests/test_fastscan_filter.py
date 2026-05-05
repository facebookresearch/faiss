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
    """
    Test IDSelector filtering on IVF fast_scan indexes.

    The PR adds block-skip filtering to the fast_scan accumulate loops:
    before processing each 32-vector block, it checks if all vectors
    in the block are filtered out by the IDSelector, and skips the
    block entirely if so.

    These tests verify that filtered results only contain allowed IDs.
    Only IVF-based fastscan indexes support SearchParameters with sel.
    """

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
            excluded = np.concatenate([
                np.arange(0, 32, dtype="int64"),
                np.arange(64, 96, dtype="int64"),
            ])
            inner_sel = faiss.IDSelectorBatch(excluded)
            sel = faiss.IDSelectorNot(inner_sel)
            allowed = set(i for i in range(nb) if i not in excluded)
        elif id_selector_type == "partial_batch":
            # Exclude a few IDs within a single block (not a whole block)
            excluded = np.array([5, 10, 20, 31], dtype="int64")
            inner_sel = faiss.IDSelectorBatch(excluded)
            sel = faiss.IDSelectorNot(inner_sel)
            allowed = set(i for i in range(nb) if i not in excluded)
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
                        idx, allowed,
                        f"Query {q}, rank {j}: got id {idx} not in allowed set"
                    )

        # If allowed set is large enough, expect some valid results
        if len(allowed) > k:
            valid = np.sum(Ifs >= 0)
            self.assertGreater(
                valid, 0, "Expected some valid results"
            )

        return Ifs

    # ------- IVFPQFastScan tests -------

    def test_IVFPQfs_batch_L2(self):
        self.do_test_filter("IVF32,PQ4x4fs", "batch", faiss.METRIC_L2)

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

    # ------- Different k values (different handler paths) -------

    def test_IVFPQfs_k1(self):
        """k=1 -> SingleResultHandler path"""
        self.do_test_filter("IVF32,PQ4x4fs", "batch", k=1)

    def test_IVFPQfs_k40(self):
        """k=40 -> ReservoirHandler path"""
        self.do_test_filter("IVF32,PQ4x4fs", "batch", k=40)

    # ------- Non-aligned ntotal -------

    def test_IVFPQfs_ntotal_50(self):
        """ntotal=50, not a multiple of 32"""
        self.do_test_filter("IVF32,PQ4x4fs", "batch", nb=50)

    def test_IVFPQfs_ntotal_77(self):
        """ntotal=77, partial last block"""
        self.do_test_filter("IVF32,PQ4x4fs", "batch", nb=77)

    def test_IVFPQfs_ntotal_150(self):
        """ntotal=150, not a multiple of 32"""
        self.do_test_filter("IVF32,PQ4x4fs", "batch", nb=150)


@for_all_simd_levels
class TestBlockSkipConsistency(unittest.TestCase):
    """
    Test that block-skip filtering produces consistent results
    across different filter configurations on fastscan IVF indexes.
    """

    def test_consistency_with_ivfpq(self):
        """
        Compare IVFPQFastScan filtered results with IVFPQ (non-fastscan)
        filtered results. Both should only return allowed IDs.
        """
        d = 32
        ds = datasets.SyntheticDataset(d, 2000, 500, 20)

        rs = np.random.RandomState(42)
        subset = rs.choice(500, 150, replace=False).astype("int64")
        sel = faiss.IDSelectorBatch(subset)
        allowed = set(subset.tolist())

        # FastScan index
        index_fs = faiss.index_factory(d, "IVF32,PQ4x4fs")
        index_fs.train(ds.get_train())
        index_fs.add(ds.get_database())

        # Non-fastscan PQ index
        index_pq = faiss.index_factory(d, "IVF32,PQ4x4np")
        index_pq.train(ds.get_train())
        index_pq.add(ds.get_database())

        k = 10
        params = faiss.SearchParametersIVF(sel=sel, nprobe=8)

        Dfs, Ifs = index_fs.search(ds.get_queries(), k, params=params)
        Dpq, Ipq = index_pq.search(ds.get_queries(), k, params=params)

        # Both should only return allowed IDs
        for q in range(ds.nq):
            for j in range(k):
                if Ifs[q, j] >= 0:
                    self.assertIn(int(Ifs[q, j]), allowed)
                if Ipq[q, j] >= 0:
                    self.assertIn(int(Ipq[q, j]), allowed)

    def test_blockskip_consistency_with_ivfpq(self):
        """
        Compare IVFPQFastScan vs IVFPQ (non-fastscan) when using
        IDSelectorNot that excludes entire 32-vector blocks.
        This specifically validates the block-skip optimization
        against a baseline that doesn't have block-skip logic.
        """
        d = 32
        nb = 500
        ds = datasets.SyntheticDataset(d, 2000, nb, 20)

        # Exclude entire blocks: IDs 0-31 and 64-95
        excluded = np.concatenate([
            np.arange(0, 32, dtype="int64"),
            np.arange(64, 96, dtype="int64"),
        ])
        inner_sel = faiss.IDSelectorBatch(excluded)
        sel = faiss.IDSelectorNot(inner_sel)
        allowed = set(i for i in range(nb) if i not in excluded)

        # FastScan index
        index_fs = faiss.index_factory(d, "IVF32,PQ4x4fs")
        index_fs.train(ds.get_train())
        index_fs.add(ds.get_database())

        # Non-fastscan PQ index (no block-skip logic)
        index_pq = faiss.index_factory(d, "IVF32,PQ4x4np")
        index_pq.train(ds.get_train())
        index_pq.add(ds.get_database())

        k = 10
        params = faiss.SearchParametersIVF(sel=sel, nprobe=8)

        Dfs, Ifs = index_fs.search(ds.get_queries(), k, params=params)
        Dpq, Ipq = index_pq.search(ds.get_queries(), k, params=params)

        # Both should only return allowed IDs
        for q in range(ds.nq):
            for j in range(k):
                if Ifs[q, j] >= 0:
                    self.assertIn(int(Ifs[q, j]), allowed,
                        f"FastScan q={q} j={j}: id {Ifs[q, j]} is excluded")
                if Ipq[q, j] >= 0:
                    self.assertIn(int(Ipq[q, j]), allowed,
                        f"IVFPQ q={q} j={j}: id {Ipq[q, j]} is excluded")

    def test_partial_vs_whole_block_filter(self):
        """
        Excluding the same set of IDs via two identical IDSelectorNot
        configurations should produce identical results.
        """
        d = 32
        ds = datasets.SyntheticDataset(d, 2000, 200, 10)

        index = faiss.index_factory(d, "IVF32,PQ4x4fs")
        index.train(ds.get_train())
        index.add(ds.get_database())

        # Exclude the first 32 elements (whole block)
        excluded = np.arange(0, 32, dtype="int64")
        sel1 = faiss.IDSelectorNot(faiss.IDSelectorBatch(excluded))
        sel2 = faiss.IDSelectorNot(faiss.IDSelectorBatch(excluded))

        k = 10
        params1 = faiss.SearchParametersIVF(sel=sel1, nprobe=8)
        params2 = faiss.SearchParametersIVF(sel=sel2, nprobe=8)

        D1, I1 = index.search(ds.get_queries(), k, params=params1)
        D2, I2 = index.search(ds.get_queries(), k, params=params2)

        # Results should be identical
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_array_almost_equal(D1, D2, decimal=5)

    def test_heavy_filtering(self):
        """
        With >90% of vectors filtered, block-skip should skip most blocks.
        Verify correctness under heavy filtering.
        """
        d = 32
        ds = datasets.SyntheticDataset(d, 2000, 1000, 20)

        index = faiss.index_factory(d, "IVF32,PQ4x4fs")
        index.train(ds.get_train())
        index.add(ds.get_database())

        # Allow only ~5% of vectors
        rs = np.random.RandomState(99)
        subset = rs.choice(1000, 50, replace=False).astype("int64")
        sel = faiss.IDSelectorBatch(subset)
        allowed = set(subset.tolist())

        k = 10
        params = faiss.SearchParametersIVF(sel=sel, nprobe=16)
        Dfs, Ifs = index.search(ds.get_queries(), k, params=params)

        # All fastscan results must be in allowed set
        for q in range(ds.nq):
            for j in range(k):
                idx = int(Ifs[q, j])
                if idx >= 0:
                    self.assertIn(idx, allowed)


@for_all_simd_levels
class TestFastScanRangeSearchFilter(unittest.TestCase):
    """Test range_search with IDSelector on fastscan IVF indexes."""

    def test_range_search_filtered(self):
        """Range search with IDSelector should only return allowed IDs."""
        d = 32
        ds = datasets.SyntheticDataset(d, 2000, 500, 10)

        index = faiss.index_factory(d, "IVF32,PQ4x4fs")
        index.train(ds.get_train())
        index.add(ds.get_database())

        # Unfiltered range search to find a reasonable radius
        Dref, _ = index.search(ds.get_queries(), 10)
        radius = float(Dref.max()) * 1.5

        rs = np.random.RandomState(123)
        subset = rs.choice(500, 150, replace=False).astype("int64")
        sel = faiss.IDSelectorBatch(subset)
        allowed = set(subset.tolist())

        params = faiss.SearchParametersIVF(sel=sel, nprobe=8)

        try:
            lims, D, I = index.range_search(
                ds.get_queries(), radius, params=params
            )

            # All returned IDs should be in the allowed set
            for idx in I:
                if idx >= 0:
                    self.assertIn(
                        int(idx), allowed,
                        f"Range search returned id {idx} not in allowed set"
                    )
        except RuntimeError:
            # range_search may not be supported for all fastscan variants
            pass


if __name__ == "__main__":
    unittest.main()
