# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
import faiss

from common_faiss_tests import get_dataset_2, for_all_simd_levels


def check_io_and_clone(index, xq, D, I):
    index2 = faiss.deserialize_index(faiss.serialize_index(index))
    index2b = faiss.deserialize_index(faiss.serialize_index(index2))
    for index3 in index2, index2b, faiss.clone_index(index):
        D3, I3 = index3.search(xq, 1)
        np.testing.assert_array_equal(D3, D)
        np.testing.assert_array_equal(I3, I)


@for_all_simd_levels
class TestHNSW(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 0
        nb = 1500
        nq = 500

        (_, self.xb, self.xq) = get_dataset_2(d, nt, nb, nq)
        index = faiss.IndexFlatL2(d)
        index.add(self.xb)
        _, Iref = index.search(self.xq, 1)
        self.Iref = Iref

    def test_hnsw(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 460)

        check_io_and_clone(index, self.xq, Dhnsw, Ihnsw)

    def test_deterministic_build(self):
        # nb must be large enough that the batch-size cap (2% of n) exceeds
        # the >100 threshold gating the parallel build region; otherwise the
        # build runs single-threaded and determinism is trivially untested.
        d = 32
        nb = 20000
        xb = np.random.RandomState(123).random((nb, d)).astype("float32")

        # build() mutates the global OpenMP thread count: restore it.
        self.addCleanup(faiss.omp_set_num_threads, faiss.omp_get_max_threads())

        def build(nthreads):
            faiss.omp_set_num_threads(nthreads)
            index = faiss.IndexHNSWFlat(d, 16)
            index.add(xb)
            return index

        # `index.hnsw` returns a fresh (copied) proxy and `.neighbors` /
        # `.offsets` are views into it, so chaining `index.hnsw.neighbors`
        # inline reads a freed temporary. Bind the hnsw proxy to keep it alive
        # across the (copying) vector_to_array calls.
        def graph_of(index):
            hnsw = index.hnsw
            return (
                hnsw.max_level,
                faiss.vector_to_array(hnsw.neighbors),
                faiss.vector_to_array(hnsw.offsets),
            )

        idx8 = build(8)
        max_level, g8a, offs = graph_of(idx8)

        # graph is non-trivial: multi-level, many real edges
        self.assertGreaterEqual(max_level, 1)
        self.assertGreater(int((g8a >= 0).sum()), nb)

        g8b = graph_of(build(8))[1]
        g1 = graph_of(build(1))[1]
        np.testing.assert_array_equal(g8a, g8b)
        np.testing.assert_array_equal(g8a, g1)

        for i in range(nb):
            seg = g8a[int(offs[i]) : int(offs[i + 1])]
            self.assertNotIn(i, seg[seg >= 0].tolist())

    def test_range_search(self):
        index_flat = faiss.IndexFlat(self.xb.shape[1])
        index_flat.add(self.xb)
        D, _ = index_flat.search(self.xq, 10)
        radius = np.median(D[:, -1])
        lims_ref, _, Iref = index_flat.range_search(self.xq, radius)

        index = faiss.IndexHNSWFlat(self.xb.shape[1], 16)
        index.add(self.xb)
        lims, D, I = index.range_search(self.xq, radius)

        nmiss = 0
        for i in range(len(self.xq)):
            ref = Iref[lims_ref[i] : lims_ref[i + 1]]
            new = I[lims[i] : lims[i + 1]]
            self.assertLessEqual(set(new), set(ref))
            nmiss += len(ref) - len(new)
        # currently we miss 405 / 6019 neighbors
        self.assertLessEqual(nmiss, lims_ref[-1] * 0.1)

    def test_hnsw_unbounded_queue(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        index.hnsw.search_bounded_queue = False
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 460)

        check_io_and_clone(index, self.xq, Dhnsw, Ihnsw)

    def test_hnsw_no_init_level0(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.init_level0 = False
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        # This is expected to be smaller because we are not initializing
        # vectors into level 0.
        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 25)

        check_io_and_clone(index, self.xq, Dhnsw, Ihnsw)

    def test_hnsw_2level(self):
        d = self.xq.shape[1]

        quant = faiss.IndexFlatL2(d)

        index = faiss.IndexHNSW2Level(quant, 256, 8, 8)
        index.train(self.xb)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 307)

        check_io_and_clone(index, self.xq, Dhnsw, Ihnsw)

    def test_hnsw_2level_mixed_search(self):
        d = self.xq.shape[1]

        quant = faiss.IndexFlatL2(d)

        storage = faiss.IndexIVFPQ(quant, d, 32, 8, 8)
        storage.make_direct_map()
        index = faiss.IndexHNSW2Level(quant, 32, 8, 8)
        index.storage = storage
        index.train(self.xb)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        # It is expected that the mixed search will perform worse.
        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 200)

        check_io_and_clone(index, self.xq, Dhnsw, Ihnsw)

    def test_add_0_vecs(self):
        index = faiss.IndexHNSWFlat(10, 16)
        zero_vecs = np.zeros((0, 10), dtype="float32")
        # used to loop forever
        index.add(zero_vecs)
        self.assertEqual(index.ntotal, 0)

    def test_hnsw_IP(self):
        d = self.xq.shape[1]

        index_IP = faiss.IndexFlatIP(d)
        index_IP.add(self.xb)
        Dref, Iref = index_IP.search(self.xq, 1)

        index = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((Iref == Ihnsw).sum(), 470)

        mask = Iref[:, 0] == Ihnsw[:, 0]
        self.assertTrue(np.allclose(Dref[mask, 0], Dhnsw[mask, 0]))

    def test_ndis_stats(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        stats = faiss.cvar.hnsw_stats
        self.addCleanup(
            faiss.set_search_stats_enabled, faiss.get_search_stats_enabled()
        )
        faiss.set_search_stats_enabled(True)
        stats.reset()
        index.search(self.xq, 1)
        self.assertGreater(stats.ndis, len(self.xq) * index.hnsw.efSearch)

    def test_io_no_storage(self):
        d = self.xq.shape[1]
        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)

        Dref, Iref = index.search(self.xq, 5)

        index2 = faiss.deserialize_index(
            faiss.serialize_index(index, faiss.IO_FLAG_SKIP_STORAGE)
        )
        self.assertEqual(index2.storage, None)
        self.assertRaises(RuntimeError, index2.search, self.xb, 1)

        # make sure we can store an index with empty storage
        faiss.deserialize_index(faiss.serialize_index(index2))

        index.storage = faiss.clone_index(index.storage)
        index.own_fields = True

        Dnew, Inew = index.search(self.xq, 5)
        np.testing.assert_array_equal(Dnew, Dref)
        np.testing.assert_array_equal(Inew, Iref)

    def test_hnsw_reset(self):
        d = self.xb.shape[1]
        index_flat = faiss.IndexFlat(d)
        index_flat.add(self.xb)
        index_hnsw = faiss.IndexHNSW(index_flat)
        index_hnsw.add(self.xb)
        # * 2 because we add to storage twice. This is just for testing
        # that storage gets cleared correctly.
        self.assertEqual(index_hnsw.ntotal, self.xb.shape[0] * 2)

        index_hnsw.reset()

        self.assertEqual(index_flat.ntotal, 0)
        self.assertEqual(index_hnsw.ntotal, 0)


class TestHNSWSimilarity(unittest.TestCase):
    """HNSW variants with METRIC_INNER_PRODUCT: is_similarity dispatch."""

    def setUp(self):
        d = 32
        nt = 1500
        nb = 1500
        nq = 200
        (self.xt, self.xb, self.xq) = get_dataset_2(d, nt, nb, nq)
        ref = faiss.IndexFlatIP(d)
        ref.add(self.xb)
        _, self.Iref_ip = ref.search(self.xq, 1)

    def _check_quantized_IP(self, index, min_agreement):
        """An ordering regression collapses agreement to ~0."""
        self.assertTrue(index.hnsw.is_similarity)
        if not index.is_trained:
            index.train(self.xt)
        index.add(self.xb)

        Dhnsw, Ihnsw = index.search(self.xq, 1)
        agreement = (self.Iref_ip[:, 0] == Ihnsw[:, 0]).sum()
        self.assertGreaterEqual(
            agreement,
            min_agreement,
            f"{type(index).__name__} agreement={agreement} < {min_agreement}",
        )

        # `hnsw.is_similarity` is not serialized; index_read must
        # re-derive it from the persisted metric_type.
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertTrue(index2.hnsw.is_similarity)
        self.assertEqual(index2.metric_type, faiss.METRIC_INNER_PRODUCT)
        D2, I2 = index2.search(self.xq, 1)
        np.testing.assert_array_equal(I2, Ihnsw)
        np.testing.assert_array_equal(D2, Dhnsw)

    # Agreement floors below are calibrated against measured values
    # (Flat: 199, PQ: 26-30, SQ: 195, Cagra: 200 over nq=200) with a
    # small headroom for RNG noise.

    def test_hnsw_flat_IP(self):
        d = self.xb.shape[1]
        self._check_quantized_IP(
            faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT),
            # observed ~199/200; headroom for OpenMP-parallel-build
            # nondeterminism
            min_agreement=180,
        )

    def test_hnsw_pq_IP(self):
        d = self.xb.shape[1]
        self._check_quantized_IP(
            faiss.IndexHNSWPQ(d, 8, 16, 8, faiss.METRIC_INNER_PRODUCT),
            min_agreement=20,
        )

    def test_hnsw_sq_IP(self):
        d = self.xb.shape[1]
        self._check_quantized_IP(
            faiss.IndexHNSWSQ(
                d,
                faiss.ScalarQuantizer.QT_8bit,
                16,
                faiss.METRIC_INNER_PRODUCT,
            ),
            min_agreement=190,
        )

    def test_hnsw_cagra_IP(self):
        d = self.xb.shape[1]
        self._check_quantized_IP(
            faiss.IndexHNSWCagra(d, 16, faiss.METRIC_INNER_PRODUCT),
            min_agreement=195,
        )

    def test_hnsw_cagra_IP_base_level_only(self):
        """IndexHNSWCagra.base_level_only takes a separate code path
        (random entry-point selection + search_level_0) that also
        needs to dispatch on is_similarity."""
        d = self.xb.shape[1]
        index = faiss.IndexHNSWCagra(d, 16, faiss.METRIC_INNER_PRODUCT)
        index.add(self.xb)
        index.base_level_only = True
        index.num_base_level_search_entrypoints = 64

        _, Ihnsw = index.search(self.xq, 1)
        agreement = (self.Iref_ip[:, 0] == Ihnsw[:, 0]).sum()
        # Measured value on this dataset is 200/200; leave headroom for
        # the RNG-driven entry-point selection (this path is unseeded).
        self.assertGreaterEqual(
            agreement,
            190,
            f"Cagra base_only agreement={agreement}",
        )


class TestHNSWNaN(unittest.TestCase):
    """Adding a NaN vector used to crash: NaN distances corrupted the
    MinimaxHeap ordering in HNSW."""

    def test_add_nan_vector_to_ivf_hnsw(self):
        d = 64
        nt = 2000
        nb = 1000
        xt = np.random.default_rng(42).random((nt, d), dtype="float32")
        xb = np.random.default_rng(43).random((nb, d), dtype="float32")

        index = faiss.index_factory(d, "IVF256_HNSW32,SQ8")
        index.train(xt)
        index.add(xb)

        vec = np.zeros((1, d), dtype="float32")
        vec[0, 0] = np.nan

        index.add(vec)
        self.assertEqual(index.ntotal, nb + 1)


class Issue3684(unittest.TestCase):

    def test_issue3684(self):
        np.random.seed(1234)
        d = 256
        nb = 10
        nq = 2
        xb = np.random.random((nb, d)).astype("float32")
        xq = np.random.random((nq, d)).astype("float32")

        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        hnsw_index_ip = faiss.IndexHNSWFlat(256, 16, faiss.METRIC_INNER_PRODUCT)
        hnsw_index_ip.hnsw.efConstruction = 512
        hnsw_index_ip.hnsw.efSearch = 512
        hnsw_index_ip.add(xb)

        D, I = hnsw_index_ip.search(xq, 10)
        self.assertTrue(np.all(D[:, :-1] >= D[:, 1:]))

        radius = 0.74
        lims, D, I = hnsw_index_ip.range_search(xq, radius)
        self.assertTrue(np.all(D >= radius))


class TestNSG(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 0
        nb = 1500
        nq = 500
        self.GK = 32

        _, self.xb, self.xq = get_dataset_2(d, nt, nb, nq)

    def make_knn_graph(self, metric):
        n = self.xb.shape[0]
        d = self.xb.shape[1]
        index = faiss.IndexFlat(d, metric)
        index.add(self.xb)
        _, I = index.search(self.xb, self.GK + 1)
        knn_graph = np.zeros((n, self.GK), dtype=np.int64)

        # For the inner product distance, the distance between a vector and
        # itself may not be the smallest, so it is not guaranteed that I[:, 0]
        # is the query itself.
        for i in range(n):
            cnt = 0
            for j in range(self.GK + 1):
                if I[i, j] != i:
                    knn_graph[i, cnt] = I[i, j]
                    cnt += 1
                if cnt == self.GK:
                    break
        return knn_graph

    def subtest_connectivity(self, index, nb):
        vt = faiss.VisitedTableVector(nb)
        count = index.nsg.dfs(vt, index.nsg.enterpoint, 0)
        self.assertEqual(count, nb)

    def subtest_add(self, build_type, thresh, metric=faiss.METRIC_L2):
        d = self.xq.shape[1]

        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        _, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16, metric)
        index.verbose = True
        index.build_type = build_type
        index.GK = self.GK
        index.add(self.xb)
        Dnsg, Insg = index.search(self.xq, 1)

        recalls = (Iref == Insg).sum()
        self.assertGreaterEqual(recalls, thresh)
        self.subtest_connectivity(index, self.xb.shape[0])
        check_io_and_clone(index, self.xq, Dnsg, Insg)

    def subtest_build(self, knn_graph, thresh, metric=faiss.METRIC_L2):
        d = self.xq.shape[1]

        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        _, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16, metric)
        index.verbose = True

        index.build(self.xb, knn_graph)
        _, Insg = index.search(self.xq, 1)

        recalls = (Iref == Insg).sum()
        self.assertGreaterEqual(recalls, thresh)
        self.subtest_connectivity(index, self.xb.shape[0])

    def test_add_bruteforce_L2(self):
        self.subtest_add(0, 475, faiss.METRIC_L2)

    def test_add_nndescent_L2(self):
        self.subtest_add(1, 475, faiss.METRIC_L2)

    def test_add_bruteforce_IP(self):
        self.subtest_add(0, 480, faiss.METRIC_INNER_PRODUCT)

    def test_add_nndescent_IP(self):
        self.subtest_add(1, 480, faiss.METRIC_INNER_PRODUCT)

    def test_build_invalid_knng(self):
        """Make some invalid entries in the input knn graph.

        It would cause a warning but IndexNSG should be able
        to handle this.
        """
        knn_graph = self.make_knn_graph(faiss.METRIC_L2)
        knn_graph[:100, 5] = -111
        self.subtest_build(knn_graph, 475, faiss.METRIC_L2)

        knn_graph = self.make_knn_graph(faiss.METRIC_INNER_PRODUCT)
        knn_graph[:100, 5] = -111
        self.subtest_build(knn_graph, 480, faiss.METRIC_INNER_PRODUCT)

    def test_reset(self):
        d = self.xq.shape[1]

        metric = faiss.METRIC_L2
        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        _, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16)
        index.verbose = True
        index.GK = 32

        index.add(self.xb)

        index.reset()
        index.add(self.xb)
        _, Insg = index.search(self.xq, 1)
        recalls = (Iref == Insg).sum()
        self.assertGreaterEqual(recalls, 475)
        self.subtest_connectivity(index, self.xb.shape[0])

    def test_order(self):
        """make sure that output results are sorted"""
        d = self.xq.shape[1]
        index = faiss.IndexNSGFlat(d, 32)

        index.train(self.xb)
        index.add(self.xb)

        k = 10
        nq = self.xq.shape[0]
        D, _ = index.search(self.xq, k)

        indices = np.argsort(D, axis=1)
        gt = np.arange(0, k)[np.newaxis, :]  # [1, k]
        gt = np.repeat(gt, nq, axis=0)  # [nq, k]
        np.testing.assert_array_equal(indices, gt)

    def test_nsg_with_pre_built_knn_graph_throws_when_rebuilding_via_add(self):
        knn_graph = self.make_knn_graph(faiss.METRIC_L2)

        d = self.xq.shape[1]
        index = faiss.IndexNSGFlat(d, 16)
        index.build(self.xb, knn_graph)
        self.assertTrue(index.is_built)

        with self.assertRaises(RuntimeError) as context:
            index.add(self.xb)

        self.assertIn(
            "NSG does not support incremental addition", str(context.exception)
        )

    def test_nsg_rebuild_throws_with_pre_built_knn_graph(self):
        knn_graph = self.make_knn_graph(faiss.METRIC_L2)
        d = self.xq.shape[1]
        index = faiss.IndexNSGFlat(d, 16)
        index.build(self.xb, knn_graph)

        with self.assertRaises(RuntimeError) as context:
            index.build(self.xb, knn_graph)

        self.assertIn("The IndexNSG is already built", str(context.exception))

    def test_nsg_pq(self):
        d = self.xq.shape[1]
        R, pq_M = 32, 4
        index = faiss.index_factory(d, f"NSG{R}_PQ{pq_M}np")
        self.assertIsInstance(index, faiss.IndexNSGPQ)
        idxpq = faiss.downcast_index(index.storage)
        self.assertEqual(index.nsg.R, R)
        self.assertEqual(idxpq.pq.M, pq_M)

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        _, Iref = flat_index.search(self.xq, k=1)

        index.GK = 32
        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        recalls = (Iref == I).sum()
        # observed ~193/500; PQ4 is coarse -- headroom for OpenMP
        # FP-reduction / tie-break nondeterminism in NSG build + PQ assign
        self.assertGreaterEqual(recalls, 180)

        check_io_and_clone(index, self.xq, D, I)

    def test_nsg_sq(self):
        d = self.xq.shape[1]
        R = 32
        index = faiss.index_factory(d, f"NSG{R}_SQ8")
        self.assertIsInstance(index, faiss.IndexNSGSQ)
        idxsq = faiss.downcast_index(index.storage)
        self.assertEqual(index.nsg.R, R)
        self.assertEqual(idxsq.sq.qtype, faiss.ScalarQuantizer.QT_8bit)

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        _, Iref = flat_index.search(self.xq, k=1)

        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        recalls = (Iref == I).sum()
        # nominal ~411/500; loosened for OpenMP graph-build nondeterminism
        self.assertGreaterEqual(recalls, 395)

        check_io_and_clone(index, self.xq, D, I)


class TestNNDescent(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 0
        nb = 1500
        nq = 500
        self.GK = 32

        _, self.xb, self.xq = get_dataset_2(d, nt, nb, nq)

    def test_nndescentflat(self):
        d = self.xq.shape[1]
        index = faiss.IndexNNDescentFlat(d, 32)
        index.nndescent.search_L = 8

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        _, Iref = flat_index.search(self.xq, k=1)

        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        recalls = (Iref == I).sum()
        self.assertGreaterEqual(recalls, 450)  # 462

        check_io_and_clone(index, self.xq, D, I)

    def test_order(self):
        """make sure that output results are sorted"""
        d = self.xq.shape[1]
        index = faiss.IndexNNDescentFlat(d, 32)

        index.train(self.xb)
        index.add(self.xb)

        k = 10
        nq = self.xq.shape[0]
        D, _ = index.search(self.xq, k)

        indices = np.argsort(D, axis=1)
        gt = np.arange(0, k)[np.newaxis, :]  # [1, k]
        gt = np.repeat(gt, nq, axis=0)  # [nq, k]
        np.testing.assert_array_equal(indices, gt)


class TestNNDescentGenRandom(unittest.TestCase):
    """Regression tests for gen_random edge cases in NNDescent."""

    def test_search_L_equals_ntotal(self):
        """gen_random(size, N) crashed with division by zero when size == N,
        which search() reaches when search_L >= ntotal."""
        d = 32
        nb = 200  # just above NUM_EVAL_POINTS=100
        xb = np.random.default_rng(42).random((nb, d)).astype("float32")
        xq = np.random.default_rng(43).random((10, d)).astype("float32")

        index = faiss.IndexNNDescentFlat(d, 32)
        index.nndescent.search_L = nb  # triggers gen_random(size=nb, N=nb)
        index.train(xb)
        index.add(xb)

        index.search(xq, k=1)


class TestNNDescentKNNG(unittest.TestCase):

    def test_knng_L2(self):
        self.subtest(32, 10, faiss.METRIC_L2)

    def test_knng_IP(self):
        self.subtest(32, 10, faiss.METRIC_INNER_PRODUCT)

    def subtest(self, d, K, metric):
        nb = 1000
        _, xb, _ = get_dataset_2(d, 0, nb, 0)

        _, knn = faiss.knn(xb, xb, K + 1, metric)
        knn = knn[:, 1:]

        index = faiss.IndexNNDescentFlat(d, K, metric)
        index.nndescent.S = 10
        index.nndescent.R = 32
        index.nndescent.L = K + 20
        index.nndescent.iter = 5
        index.verbose = True

        index.add(xb)
        graph = index.nndescent.final_graph
        graph = faiss.vector_to_array(graph)
        graph = graph.reshape(nb, K)

        recalls = 0
        for i in range(nb):
            for j in range(K):
                for k in range(K):
                    if graph[i, j] == knn[i, k]:
                        recalls += 1
                        break
        recall = 1.0 * recalls / (nb * K)
        self.assertGreater(recall, 0.99)

    def test_small_nndescent(self):
        """building a too small graph used to crash, make sure it raises
        an exception instead.
        TODO: build the exact knn graph for small cases
        """
        d = 32
        K = 10
        index = faiss.IndexNNDescentFlat(d, K, faiss.METRIC_L2)
        index.nndescent.S = 10
        index.nndescent.R = 32
        index.nndescent.L = K + 20
        index.nndescent.iter = 5
        index.verbose = True

        xb = np.zeros((78, d), dtype="float32")
        self.assertRaises(RuntimeError, index.add, xb)
