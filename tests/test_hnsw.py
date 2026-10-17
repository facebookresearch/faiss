# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for HNSW behavior. Ported from test_hnsw.cpp."""

import math
import threading
import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


def get_neighbor_range(hnsw, no, layer_no):
    begin = np.zeros(1, dtype=np.uint64)
    end = np.zeros(1, dtype=np.uint64)
    hnsw.neighbor_range(
        no, layer_no, faiss.swig_ptr(begin), faiss.swig_ptr(end)
    )
    return int(begin[0]), int(end[0])


def get_neighbors(hnsw):
    return faiss.rev_swig_ptr(hnsw.neighbors.data(), hnsw.neighbors.size())


class TestHNSWLpMetric(unittest.TestCase):
    def test_index_hnsw_metric_lp(self):
        index = faiss.IndexHNSWFlat(1, 32, faiss.METRIC_Lp)
        index.storage.metric_arg = 3

        index.add(np.array([[0.0]], dtype=np.float32))
        D, I = index.search(np.array([[2.0]], dtype=np.float32), 1)

        # distance is 2**3
        self.assertAlmostEqual(D[0][0], 8.0, places=5)
        self.assertEqual(I[0][0], 0)


class TestHNSWStats(unittest.TestCase):
    def set_search_stats_enabled(self, enabled):
        previous = faiss.get_search_stats_enabled()
        faiss.set_search_stats_enabled(enabled)
        self.addCleanup(faiss.set_search_stats_enabled, previous)

    def make_index(self):
        index = faiss.IndexBinaryHNSW(8, 2)
        database = np.zeros((1, 1), dtype=np.uint8)
        index.add(database)
        index.hnsw.efSearch = 1
        return index, database

    def test_disabled_collection_skips_global_stats(self):
        index, database = self.make_index()
        self.set_search_stats_enabled(False)
        faiss.cvar.hnsw_stats.reset()

        index.search(database, 1)

        self.assertEqual(0, faiss.cvar.hnsw_stats.n1)

    def test_enabled_collection_aggregates_concurrent_searches_exactly(self):
        num_threads = 16
        searches_per_thread = 100

        index, database = self.make_index()
        self.set_search_stats_enabled(True)
        self.addCleanup(faiss.omp_set_num_threads, faiss.omp_get_max_threads())
        faiss.cvar.hnsw_stats.reset()

        def run_searches():
            faiss.omp_set_num_threads(1)
            for _ in range(searches_per_thread):
                index.search(database, 1)

        threads = [
            threading.Thread(target=run_searches) for _ in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(
            num_threads * searches_per_thread, faiss.cvar.hnsw_stats.n1
        )


def copy_base_level_only(src, dst):
    n = src.ntotal
    d = src.d
    M = src.hnsw.nb_neighbors(0) // 2
    graph_degree = src.hnsw.nb_neighbors(0)

    storage = faiss.IndexFlatL2(d)
    storage.this.disown()
    dst.storage = storage
    dst.own_fields = True
    dst.d = d
    dst.metric_type = src.metric_type
    dst.is_trained = True
    dst.keep_max_size_level0 = True

    dst.hnsw.reset()
    dst.hnsw.assign_probas.clear()
    dst.hnsw.cum_nneighbor_per_level.clear()
    dst.hnsw.set_default_probas(M, 1.0 / math.log(M))

    dst.hnsw.prepare_level_tab(n, False)

    src_flat = faiss.downcast_index(src.storage)
    assert isinstance(src_flat, faiss.IndexFlat)
    xb = faiss.rev_swig_ptr(src_flat.get_xb(), n * d).reshape(n, d)
    dst.storage.add(xb)
    dst.ntotal = n

    src_neighbors = get_neighbors(src.hnsw)
    dst_neighbors = get_neighbors(dst.hnsw)
    for i in range(n):
        src_begin, _ = get_neighbor_range(src.hnsw, i, 0)
        dst_begin, dst_end = get_neighbor_range(dst.hnsw, i, 0)
        for j in range(min(graph_degree, dst_end - dst_begin)):
            dst_neighbors[dst_begin + j] = src_neighbors[src_begin + j]

    dst.base_level_only = True


class TestHNSW(unittest.TestCase):
    def test_index_hnsw_cagra_base_level_only_range_search(self):
        d = 8
        nb = 100
        nq = 5
        M = 4

        xb = faiss.rand(nb * d, 1234).reshape(nb, d)
        xq = faiss.rand(nq * d, 4321).reshape(nq, d)

        index = faiss.IndexHNSWCagra(d, M, faiss.METRIC_L2)
        index.add(xb)
        index.base_level_only = True
        index.num_base_level_search_entrypoints = 8

        dst_index = faiss.IndexHNSWCagra()
        copy_base_level_only(index, dst_index)
        dst_index.num_base_level_search_entrypoints = 8

        radius = 1e9
        lims, _, _ = dst_index.range_search(xq, radius)

        for i in range(nq):
            self.assertGreater(lims[i + 1] - lims[i], 0)


def reference_greedy_update_nearest(hnsw, qdis, level, nearest, d_nearest):
    stats = faiss.HNSWStats()
    neighbors = get_neighbors(hnsw)

    while True:
        prev_nearest = nearest
        begin, end = get_neighbor_range(hnsw, nearest, level)

        ndis = 0
        for i in range(begin, end):
            v = int(neighbors[i])
            if v < 0:
                break
            ndis += 1
            dis = qdis(v)
            if dis < d_nearest:
                nearest = v
                d_nearest = dis
        stats.ndis += ndis
        stats.nhops += 1

        if nearest == prev_nearest:
            return stats, nearest, d_nearest


class TestHNSWTest(unittest.TestCase):
    d = 64
    nb = 2000
    M = 4
    nq = 10
    k = 10

    def setUp(self):
        self.xb = faiss.rand(self.d * self.nb, 12345).reshape(self.nb, self.d)
        self.index = faiss.IndexHNSWFlat(self.d, self.M)
        self.index.add(self.xb)
        self.xq = faiss.rand(self.d * self.nq, 12345).reshape(self.nq, self.d)
        self.dis = self.index.storage.get_distance_computer()
        self.dis.set_query(faiss.swig_ptr(self.xq[0]))

        self.addCleanup(faiss.omp_set_num_threads, faiss.omp_get_max_threads())
        faiss.omp_set_num_threads(1)

    def test_greedy_update_nearest(self):
        hnsw = self.index.hnsw
        nearest = np.array([hnsw.entry_point], dtype=np.int32)
        d_nearest = np.array([self.dis(int(nearest[0]))], dtype=np.float32)
        reference_nearest = hnsw.entry_point
        reference_d_nearest = np.float32(self.dis(reference_nearest))

        stats = faiss.greedy_update_nearest(
            hnsw,
            self.dis,
            0,
            faiss.swig_ptr(nearest),
            faiss.swig_ptr(d_nearest),
        )

        reference_stats, reference_nearest, reference_d_nearest = (
            reference_greedy_update_nearest(
                hnsw, self.dis, 0, reference_nearest, reference_d_nearest
            )
        )
        self.assertEqual(stats.ndis, reference_stats.ndis)
        self.assertEqual(stats.nhops, reference_stats.nhops)
        self.assertEqual(stats.n1, reference_stats.n1)
        self.assertEqual(stats.n2, reference_stats.n2)
        self.assertAlmostEqual(d_nearest[0], reference_d_nearest, delta=0.01)
        self.assertEqual(nearest[0], reference_nearest)

    def test_nb_neighbors_bound(self):
        hnsw = self.index.hnsw
        self.assertEqual(hnsw.nb_neighbors(0), 8)
        self.assertEqual(hnsw.nb_neighbors(1), 4)
        self.assertEqual(hnsw.nb_neighbors(2), 4)
        self.assertEqual(hnsw.nb_neighbors(3), 4)
        # picking a large number to trigger an exception based on checking
        # bounds
        with self.assertRaises(RuntimeError):
            hnsw.nb_neighbors(100)


class TestVisitedTableReuse(unittest.TestCase):
    def test_reuses_grows_and_resets(self):
        # get_reusable() hands back the same thread-local object across calls
        # (no per-call allocation) ...
        t1 = faiss.VisitedTable.get_reusable(100)
        t2 = faiss.VisitedTable.get_reusable(100)
        self.assertEqual(int(t1.this), int(t2.this))

        # ... and hands it back clean: a mark set before re-acquiring must not
        # survive, since a reused table may carry stamps from a prior search.
        t2.set(42)
        self.assertTrue(t2.get(42))
        t3 = faiss.VisitedTable.get_reusable(100)
        self.assertEqual(int(t2.this), int(t3.this))
        self.assertFalse(t3.get(42))

        # Growing to a larger size keeps the same object and exposes the new
        # range.
        t4 = faiss.VisitedTable.get_reusable(1000)
        self.assertEqual(int(t3.this), int(t4.this))
        self.assertFalse(t4.get(900))
        self.assertTrue(t4.set(900))
        self.assertTrue(t4.get(900))
