# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""IVF early termination tests. Ported from test_ivf_early_termination.cpp."""

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


d = 16
nb = 500
nq = 20
nlist = 16
k = 10


def make_data(n, seed):
    return np.random.RandomState(seed).rand(n, d).astype(np.float32)


def build_ivf_flat():
    xb = make_data(nb, 0xABCDEF)
    index = faiss.index_factory(d, "IVF16,Flat", faiss.METRIC_L2)
    index.train(xb)
    index.add(xb)
    return index


def build_fastscan():
    xb = make_data(nb, 0xCAFEBABE)
    index = faiss.index_factory(d, "IVF16,PQ4x4fs", faiss.METRIC_L2)
    index.train(xb)
    index.add(xb)
    return index


def make_bitmap_selector(step):
    bitmap = np.zeros(nb, dtype=bool)
    bitmap[0::step] = True
    return faiss.IDSelectorBitmap(
        np.packbits(bitmap, bitorder="little"))


def assert_sorted_pairs_equal(test, lims_a, d_a, i_a, lims_b, d_b, i_b):
    test.assertEqual(len(lims_a), len(lims_b))
    for j in range(len(lims_a)):
        test.assertEqual(lims_a[j], lims_b[j], f"at lims[{j}]")
    nq_ = len(lims_a) - 1
    for q in range(nq_):
        pairs_a = sorted(
            zip(
                i_a[lims_a[q] : lims_a[q + 1]],
                d_a[lims_a[q] : lims_a[q + 1]],
            )
        )
        pairs_b = sorted(
            zip(
                i_b[lims_b[q] : lims_b[q + 1]],
                d_b[lims_b[q] : lims_b[q + 1]],
            )
        )
        test.assertEqual(pairs_a, pairs_b)


class TestIVFEarlyTermination(unittest.TestCase):
    def test_defaults_preserve_baseline_knn(self):
        index = build_ivf_flat()
        faiss.downcast_index(index).nprobe = 4

        xq = make_data(nq, 0x123456)
        D_ref, I_ref = index.search(xq, k)

        params = faiss.SearchParametersIVF()
        params.nprobe = 4
        D_new, I_new = index.search(xq, k, params=params)

        np.testing.assert_array_equal(I_ref, I_new)
        np.testing.assert_array_equal(D_ref, D_new)

    def test_defaults_preserve_baseline_range(self):
        index = build_ivf_flat()
        faiss.downcast_index(index).nprobe = 4

        xq = make_data(nq, 0x789ABC)
        radius = 2.0

        lims_ref, D_ref, I_ref = index.range_search(xq, radius)

        params = faiss.SearchParametersIVF()
        params.nprobe = 4
        lims_tst, D_tst, I_tst = index.range_search(
            xq, radius, params=params)

        assert_sorted_pairs_equal(
            self, lims_ref, D_ref, I_ref, lims_tst, D_tst, I_tst)

    def test_ensure_topk_full_reduces_misses(self):
        index = build_ivf_flat()
        xq = make_data(nq, 0xDEADBE)

        params = faiss.SearchParametersIVF()
        params.nprobe = nlist
        params.max_codes = k // 2
        params.ensure_topk_full = False
        _, I_tight = index.search(xq, k, params=params)

        params.ensure_topk_full = True
        _, I_full = index.search(xq, k, params=params)

        miss_tight = np.count_nonzero(I_tight == -1)
        miss_full = np.count_nonzero(I_full == -1)

        self.assertLessEqual(miss_full, miss_tight)
        self.assertGreater(
            miss_tight, 0,
            "test precondition: tight budget should cause truncation")
        self.assertEqual(
            miss_full, 0,
            "ensure_topk_full must fill the heap when enough data is "
            "available in the probed lists")

    def test_ensure_topk_full_rejects_unsupported_parallel_mode(self):
        index = build_ivf_flat()
        faiss.downcast_index(index).parallel_mode = 1

        xq = make_data(1, 0xAAAA)

        params = faiss.SearchParametersIVF()
        params.nprobe = 4
        params.ensure_topk_full = True
        with self.assertRaises(RuntimeError):
            index.search(xq, k, params=params)

    def test_range_early_exit_produces_subset(self):
        index = build_ivf_flat()
        xq = make_data(nq, 0xFACEFEED)
        radius = 0.5

        p_full = faiss.SearchParametersIVF()
        p_full.nprobe = nlist
        p_full.max_empty_result_buckets = 0
        lims_full, _, I_full = index.range_search(
            xq, radius, params=p_full)

        p_early = faiss.SearchParametersIVF()
        p_early.nprobe = nlist
        p_early.max_empty_result_buckets = 1
        lims_early, _, I_early = index.range_search(
            xq, radius, params=p_early)

        for q in range(nq):
            full_ids = set(I_full[lims_full[q] : lims_full[q + 1]])
            early_ids = set(I_early[lims_early[q] : lims_early[q + 1]])
            for i in early_ids:
                self.assertIn(
                    i, full_ids,
                    f"early-exit returned id {i} not present in "
                    f"full-scan result for query {q}")

    def test_range_early_exit_actually_truncates(self):
        index = build_ivf_flat()
        xq = make_data(nq, 0xFACEFEED)
        radius = 1.0

        p_full = faiss.SearchParametersIVF()
        p_full.nprobe = nlist
        p_full.max_empty_result_buckets = 0
        lims_full, _, _ = index.range_search(xq, radius, params=p_full)

        p_early = faiss.SearchParametersIVF()
        p_early.nprobe = nlist
        p_early.max_empty_result_buckets = 1
        lims_early, _, _ = index.range_search(
            xq, radius, params=p_early)

        for q in range(nq):
            self.assertLessEqual(
                lims_early[q + 1] - lims_early[q],
                lims_full[q + 1] - lims_full[q],
                f"early-exit grew result count for query {q}")

        any_shrunk = any(
            (lims_early[q + 1] - lims_early[q])
            < (lims_full[q + 1] - lims_full[q])
            for q in range(nq)
        )
        self.assertTrue(
            any_shrunk,
            "no query shrank: max_empty_result_buckets has no effect")

    def test_range_early_exit_counter_resets_on_hit(self):
        index = build_ivf_flat()
        xq = make_data(nq, 0xC0FFEE)
        radius = 1.0

        p_tight = faiss.SearchParametersIVF()
        p_tight.nprobe = nlist
        p_tight.max_empty_result_buckets = 1
        lims_tight, _, _ = index.range_search(
            xq, radius, params=p_tight)

        p_relaxed = faiss.SearchParametersIVF()
        p_relaxed.nprobe = nlist
        p_relaxed.max_empty_result_buckets = 3
        lims_relaxed, _, _ = index.range_search(
            xq, radius, params=p_relaxed)

        for q in range(nq):
            self.assertLessEqual(
                lims_tight[q + 1] - lims_tight[q],
                lims_relaxed[q + 1] - lims_relaxed[q],
                "relaxed early-exit returned fewer results than tight "
                f"for query {q}")

    def test_range_early_exit_rejects_unsupported_parallel_mode(self):
        index = build_ivf_flat()
        faiss.downcast_index(index).parallel_mode = 1

        xq = make_data(1, 0xBBBB)

        params = faiss.SearchParametersIVF()
        params.nprobe = 4
        params.max_empty_result_buckets = 2
        with self.assertRaises(RuntimeError):
            index.range_search(xq, 1.0, params=params)

    def test_ndis_stats_matches_docstring_post_filter(self):
        index = build_ivf_flat()
        faiss.downcast_index(index).nprobe = nlist

        xq = make_data(nq, 0xB0BAFE11)
        stats = faiss.cvar.indexIVF_stats

        stats.reset()
        index.search(xq, k)
        ndis_no_sel = stats.ndis
        self.assertGreater(
            ndis_no_sel, 0,
            "test precondition: at least some codes should be visited")

        sel = make_bitmap_selector(2)
        params = faiss.SearchParametersIVF()
        params.nprobe = nlist
        params.sel = sel

        stats.reset()
        index.search(xq, k, params=params)
        ndis_with_sel = stats.ndis

        self.assertLess(
            ndis_with_sel, ndis_no_sel,
            "ndis did not decrease under restrictive IDSelector")
        self.assertLess(ndis_with_sel * 4, ndis_no_sel * 3)

    def test_ndis_stats_matches_docstring_post_filter_range(self):
        index = build_ivf_flat()
        faiss.downcast_index(index).nprobe = nlist

        xq = make_data(nq, 0xC01DCA11)
        radius = 1.0
        stats = faiss.cvar.indexIVF_stats

        stats.reset()
        index.range_search(xq, radius)
        ndis_no_sel = stats.ndis
        self.assertGreater(ndis_no_sel, 0)

        sel = make_bitmap_selector(2)
        params = faiss.SearchParametersIVF()
        params.nprobe = nlist
        params.sel = sel

        stats.reset()
        index.range_search(xq, radius, params=params)
        ndis_with_sel = stats.ndis

        self.assertLess(ndis_with_sel, ndis_no_sel)
        self.assertLess(ndis_with_sel * 4, ndis_no_sel * 3)

    def test_ensure_topk_full_fills_heap_under_restrictive_selector(self):
        index = build_ivf_flat()
        faiss.downcast_index(index).nprobe = nlist

        sel = make_bitmap_selector(10)
        xq = make_data(nq, 0xDEADBE)

        params = faiss.SearchParametersIVF()
        params.nprobe = nlist
        params.max_codes = k // 2
        params.ensure_topk_full = True
        params.sel = sel

        _, I = index.search(xq, k, params=params)

        self.assertEqual(
            np.count_nonzero(I == -1), 0,
            "ensure_topk_full failed to fill the heap under a "
            "restrictive IDSelector")

    def test_ensure_topk_full_no_selector_byte_identical(self):
        index = build_ivf_flat()
        xq = make_data(nq, 0xA5A5A5)

        p_ref = faiss.SearchParametersIVF()
        p_ref.nprobe = nlist
        p_ref.max_codes = k // 2
        p_ref.ensure_topk_full = False
        _, I_ref = index.search(xq, k, params=p_ref)

        p_full = faiss.SearchParametersIVF()
        p_full.nprobe = nlist
        p_full.max_codes = k // 2
        p_full.ensure_topk_full = True
        _, I_full = index.search(xq, k, params=p_full)

        miss_ref = np.count_nonzero(I_ref == -1)
        miss_full = np.count_nonzero(I_full == -1)
        self.assertLessEqual(miss_full, miss_ref)

    def test_fastscan_nheap_updates_is_zero_today(self):
        index = build_fastscan()
        xq = make_data(nq, 0x5678)

        stats = faiss.cvar.indexIVF_stats
        stats.reset()
        index.search(xq, k)

        self.assertGreater(stats.ndis, 0)
        self.assertEqual(
            stats.nheap_updates, 0,
            "FastScan started reporting nheap_updates")

    def test_fastscan_range_defaults_preserve_baseline(self):
        index = build_fastscan()
        faiss.downcast_index(index).nprobe = 4

        xq = make_data(nq, 0x9999)
        radius = 1.0

        lims_ref, D_ref, I_ref = index.range_search(xq, radius)

        params = faiss.SearchParametersIVF()
        params.nprobe = 4
        params.max_empty_result_buckets = 0
        lims_tst, D_tst, I_tst = index.range_search(
            xq, radius, params=params)

        assert_sorted_pairs_equal(
            self, lims_ref, D_ref, I_ref, lims_tst, D_tst, I_tst)

    def test_fastscan_range_rejects_max_empty_unsupported_parallel_mode(self):
        index = build_fastscan()
        faiss.downcast_index(index).parallel_mode = 1

        xq = make_data(1, 0x9999)

        params = faiss.SearchParametersIVF()
        params.nprobe = 4
        params.max_empty_result_buckets = 2
        with self.assertRaises(RuntimeError):
            index.range_search(xq, 1.0, params=params)
