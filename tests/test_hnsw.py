# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

    def setUp(self):
        self.xb = faiss.rand(self.d * self.nb, 12345).reshape(self.nb, self.d)
        self.index = faiss.IndexHNSWFlat(self.d, self.M)
        self.index.add(self.xb)
        self.xq = faiss.rand(self.d * self.nq, 12345).reshape(self.nq, self.d)
        self.dis = self.index.storage.get_distance_computer()
        self.dis.set_query(faiss.swig_ptr(self.xq[0]))

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
        self.assertAlmostEqual(d_nearest[0], reference_d_nearest, delta=0.01)
        self.assertEqual(nearest[0], reference_nearest)

    def test_nb_neighbors_bound(self):
        hnsw = self.index.hnsw
        self.assertEqual(hnsw.nb_neighbors(0), 8)
        self.assertEqual(hnsw.nb_neighbors(1), 4)
        self.assertEqual(hnsw.nb_neighbors(2), 4)
        with self.assertRaises(RuntimeError):
            hnsw.nb_neighbors(100)


class TestVisitedTableReuse(unittest.TestCase):
    def test_reuses_grows_and_resets(self):
        # get_reusable() hands back the same thread-local object.
        t1 = faiss.VisitedTable.get_reusable(100)
        t2 = faiss.VisitedTable.get_reusable(100)
        self.assertEqual(int(t1.this), int(t2.this))

        # It comes back clean: a reused table may carry stamps from a prior
        # search.
        t2.set(42)
        self.assertTrue(t2.get(42))
        t3 = faiss.VisitedTable.get_reusable(100)
        self.assertEqual(int(t2.this), int(t3.this))
        self.assertFalse(t3.get(42))

        # Growing keeps the same object.
        t4 = faiss.VisitedTable.get_reusable(1000)
        self.assertEqual(int(t3.this), int(t4.this))
        self.assertFalse(t4.get(900))
        self.assertTrue(t4.set(900))
        self.assertTrue(t4.get(900))


def plant_sink_at_entry_point(hnsw):
    """Removes all links of the entry point, so that every search stops
    there."""
    neighbors = get_neighbors(hnsw)
    for level in range(hnsw.max_level + 1):
        begin, end = get_neighbor_range(hnsw, hnsw.entry_point, level)
        neighbors[begin:end] = -1


def near_nodes(nodes, n, seed):
    """Vectors near the nodes, cycling through all of them."""
    noise = np.random.default_rng(seed).standard_normal((n, nodes.shape[1]))
    x = nodes[np.arange(n) % len(nodes)] + 0.01 * noise
    return x.astype(np.float32)


def recall_at_1(index, xq):
    """Fraction of the queries for which the index returns the exact nearest
    node."""
    _, labels = index.search(xq, 1)
    _, exact = index.storage.search(xq, 1)
    return (labels == exact).mean()


def near_codes(nodes, n, seed):
    """Binary codes near the nodes, cycling through all of them: 4 random
    bits differ."""
    rng = np.random.default_rng(seed)
    codes = nodes[np.arange(n) % len(nodes)].copy()
    bits = rng.integers(0, nodes.shape[1] * 8, size=(n, 4))
    for f in range(4):
        np.bitwise_xor.at(
            codes,
            (np.arange(n), bits[:, f] // 8),
            (1 << (bits[:, f] % 8)).astype(np.uint8),
        )
    return codes


class TestHNSWLevel0LinksFromSample(unittest.TestCase):
    d = 8
    nnodes = 4096

    def nodes(self):
        rng = np.random.default_rng(1)
        return rng.standard_normal((self.nnodes, self.d)).astype(np.float32)

    def check_planted_sink_removed(self, index, add_links):
        """The intact recall is below 1 when the HNSW search and the storage
        search use different distance estimates, for example with RaBitQ."""
        nodes = self.nodes()
        index.train(nodes)
        index.add(nodes)
        queries = near_nodes(nodes, 8192, 3)
        intact_recall = recall_at_1(index, queries)
        hnsw = index.hnsw
        plant_sink_at_entry_point(hnsw)
        self.assertLess(recall_at_1(index, queries), 0.01)

        stats = add_links(index, near_nodes(nodes, 40000, 2))

        self.assertEqual(stats.n_sinks_found, 1)
        self.assertEqual(stats.n_sinks_remaining, 0)
        self.assertGreater(recall_at_1(index, queries), 0.95 * intact_recall)

    def test_removes_planted_sink_l2(self):
        self.check_planted_sink_removed(
            faiss.IndexHNSWFlat(self.d, 32),
            lambda index, x: index.add_level_0_links_from_sample(x),
        )

    def test_removes_planted_sink_inner_product(self):
        self.check_planted_sink_removed(
            faiss.IndexHNSWFlat(self.d, 32, faiss.METRIC_INNER_PRODUCT),
            lambda index, x: index.add_level_0_links_from_sample(x),
        )

    def test_removes_planted_sink_rabitq_storage(self):
        self.check_planted_sink_removed(
            faiss.IndexHNSWRaBitQ(self.d, 32),
            lambda index, x: index.add_level_0_links_from_sample(x),
        )

    def test_removes_planted_sink_panorama(self):
        self.check_planted_sink_removed(
            faiss.IndexHNSWFlatPanorama(self.d, 32, 2),
            lambda index, x: index.add_level_0_links_from_sample(x),
        )

    def test_removes_planted_sink_float16_samples(self):
        self.check_planted_sink_removed(
            faiss.IndexHNSWFlat(self.d, 32),
            lambda index, x: index.add_level_0_links_from_sample(
                x.astype(np.float16), numeric_type=faiss.Float16
            ),
        )

    def test_removes_planted_sink_encoded_samples(self):
        def add_links(index, x):
            codec = faiss.IndexScalarQuantizer(
                self.d, faiss.ScalarQuantizer.QT_8bit
            )
            codec.train(x)
            codes = codec.sa_encode(x)
            return index.add_level_0_links_from_sample_encoded(
                len(x), faiss.swig_ptr(codes), codec
            )

        self.check_planted_sink_removed(
            faiss.IndexHNSWFlat(self.d, 32), add_links
        )

    def test_removes_planted_sink_binary(self):
        nbits = 128
        rng = np.random.default_rng(1)
        nodes = rng.integers(
            0, 256, size=(self.nnodes, nbits // 8), dtype=np.uint8
        )
        index = faiss.IndexBinaryHNSW(nbits, 32)
        index.add(nodes)
        queries = near_codes(nodes, 8192, 3)

        # Hamming distances have ties, so compare distances, not IDs.
        def recall():
            distances, _ = index.search(queries, 1)
            exact, _ = index.storage.search(queries, 1)
            return (distances == exact).mean()

        intact_recall = recall()
        hnsw = index.hnsw
        plant_sink_at_entry_point(hnsw)
        self.assertLess(recall(), 0.01)

        stats = index.add_level_0_links_from_sample(
            near_codes(nodes, 40000, 2)
        )

        self.assertEqual(stats.n_sinks_found, 1)
        self.assertEqual(stats.n_sinks_remaining, 0)
        self.assertGreater(recall(), 0.95 * intact_recall)

    def test_does_not_change_intact_graph(self):
        nodes = self.nodes()
        index = faiss.IndexHNSWFlat(self.d, 32)
        index.add(nodes)
        hnsw = index.hnsw
        before = get_neighbors(hnsw).copy()

        stats = index.add_level_0_links_from_sample(
            near_nodes(nodes, 40000, 2)
        )

        self.assertEqual(stats.n_sinks_found, 0)
        np.testing.assert_array_equal(get_neighbors(hnsw), before)

    def test_same_links_for_any_thread_count(self):
        nodes = self.nodes()
        samples = near_nodes(nodes, 40000, 2)
        previous = faiss.omp_get_max_threads()
        self.addCleanup(faiss.omp_set_num_threads, previous)

        def linked(threads):
            faiss.omp_set_num_threads(threads)
            index = faiss.IndexHNSWFlat(self.d, 32)
            index.add(nodes)
            hnsw = index.hnsw
            plant_sink_at_entry_point(hnsw)
            index.add_level_0_links_from_sample(samples)
            return get_neighbors(hnsw).copy()

        np.testing.assert_array_equal(linked(1), linked(8))
