# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" a few tests for graph-based indices (HNSW, nndescent and NSG)"""

import numpy as np
import unittest
import faiss
import tempfile
import os

from common_faiss_tests import get_dataset_2


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
        Dref, Iref = index.search(self.xq, 1)
        self.Iref = Iref

    def test_hnsw(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 460)

        self.io_and_retest(index, Dhnsw, Ihnsw)

    def test_range_search(self):
        index_flat = faiss.IndexFlat(self.xb.shape[1])
        index_flat.add(self.xb)
        D, _ = index_flat.search(self.xq, 10)
        radius = np.median(D[:, -1])
        lims_ref, Dref, Iref = index_flat.range_search(self.xq, radius)

        index = faiss.IndexHNSWFlat(self.xb.shape[1], 16)
        index.add(self.xb)
        lims, D, I = index.range_search(self.xq, radius)

        nmiss = 0
        # check if returned resutls are a subset of the reference results
        for i in range(len(self.xq)):
            ref = Iref[lims_ref[i]: lims_ref[i + 1]]
            new = I[lims[i]: lims[i + 1]]
            self.assertLessEqual(set(new), set(ref))
            nmiss += len(ref) - len(new)
        # currenly we miss 405 / 6019 neighbors
        self.assertLessEqual(nmiss, lims_ref[-1] * 0.1)

    def test_hnsw_unbounded_queue(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        index.search_bounded_queue = False
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 460)

        self.io_and_retest(index, Dhnsw, Ihnsw)

    def test_hnsw_no_init_level0(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.init_level0 = False
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        # This is expected to be smaller because we are not initializing
        # vectors into level 0.
        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 25)

        self.io_and_retest(index, Dhnsw, Ihnsw)

    def io_and_retest(self, index, Dhnsw, Ihnsw):
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        Dhnsw2, Ihnsw2 = index2.search(self.xq, 1)

        self.assertTrue(np.all(Dhnsw2 == Dhnsw))
        self.assertTrue(np.all(Ihnsw2 == Ihnsw))

        # also test clone
        index3 = faiss.clone_index(index)
        Dhnsw3, Ihnsw3 = index3.search(self.xq, 1)

        self.assertTrue(np.all(Dhnsw3 == Dhnsw))
        self.assertTrue(np.all(Ihnsw3 == Ihnsw))

    def test_hnsw_2level(self):
        d = self.xq.shape[1]

        quant = faiss.IndexFlatL2(d)

        index = faiss.IndexHNSW2Level(quant, 256, 8, 8)
        index.train(self.xb)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 307)

        self.io_and_retest(index, Dhnsw, Ihnsw)

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

        self.io_and_retest(index, Dhnsw, Ihnsw)

    def test_add_0_vecs(self):
        index = faiss.IndexHNSWFlat(10, 16)
        zero_vecs = np.zeros((0, 10), dtype='float32')
        # infinite loop
        index.add(zero_vecs)

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
        assert np.allclose(Dref[mask, 0], Dhnsw[mask, 0])

    def test_ndis_stats(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        stats = faiss.cvar.hnsw_stats
        stats.reset()
        Dhnsw, Ihnsw = index.search(self.xq, 1)
        self.assertGreater(stats.ndis, len(self.xq) * index.hnsw.efSearch)

    def test_io_no_storage(self):
        d = self.xq.shape[1]
        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)

        Dref, Iref = index.search(self.xq, 5)

        # test writing without storage
        index2 = faiss.deserialize_index(
            faiss.serialize_index(index, faiss.IO_FLAG_SKIP_STORAGE)
        )
        self.assertEqual(index2.storage, None)
        self.assertRaises(
            RuntimeError,
            index2.search, self.xb, 1)

        # make sure we can store an index with empty storage
        index4 = faiss.deserialize_index(
            faiss.serialize_index(index2))

        # add storage afterwards
        index.storage = faiss.clone_index(index.storage)
        index.own_fields = True

        Dnew, Inew = index.search(self.xq, 5)
        np.testing.assert_array_equal(Dnew, Dref)
        np.testing.assert_array_equal(Inew, Iref)

        if False:
            # test reading without storage
            # not implemented because it is hard to skip over an index
            index3 = faiss.deserialize_index(
                faiss.serialize_index(index), faiss.IO_FLAG_SKIP_STORAGE
            )
            self.assertEqual(index3.storage, None)

    def test_abs_inner_product(self):
        """Test HNSW with abs inner product (not a real distance, so dubious that triangular inequality works)"""
        d = self.xq.shape[1]
        xb = self.xb - self.xb.mean(axis=0)  # need to be centered to give interesting directions
        xq = self.xq - self.xq.mean(axis=0)
        Dref, Iref = faiss.knn(xq, xb, 10, faiss.METRIC_ABS_INNER_PRODUCT)

        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_ABS_INNER_PRODUCT)
        index.add(xb)
        Dnew, Inew = index.search(xq, 10)

        inter = faiss.eval_intersection(Iref, Inew)
        # 4769 vs. 500*10
        self.assertGreater(inter, Iref.size * 0.9)

    def test_hnsw_reset(self):
        d = self.xb.shape[1]
        index_flat = faiss.IndexFlat(d)
        index_flat.add(self.xb)
        self.assertEqual(index_flat.ntotal, self.xb.shape[0])
        index_hnsw = faiss.IndexHNSW(index_flat)
        index_hnsw.add(self.xb)
        # * 2 because we add to storage twice. This is just for testing
        # that storage gets cleared correctly.
        self.assertEqual(index_hnsw.ntotal, self.xb.shape[0] * 2)

        index_hnsw.reset()

        self.assertEqual(index_flat.ntotal, 0)
        self.assertEqual(index_hnsw.ntotal, 0)

class Issue3684(unittest.TestCase):

    def test_issue3684(self):
        np.random.seed(1234)  # For reproducibility
        d = 256  # Example dimension
        nb = 10  # Number of database vectors
        nq = 2   # Number of query vectors
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')

        faiss.normalize_L2(xb)  # Normalize both query and database vectors
        faiss.normalize_L2(xq)

        hnsw_index_ip = faiss.IndexHNSWFlat(256, 16, faiss.METRIC_INNER_PRODUCT)
        hnsw_index_ip.hnsw.efConstruction = 512
        hnsw_index_ip.hnsw.efSearch = 512
        hnsw_index_ip.add(xb)

        # test knn 
        D, I = hnsw_index_ip.search(xq, 10)
        self.assertTrue(np.all(D[:, :-1] >= D[:, 1:]))

        # test range search 
        radius = 0.74  # Cosine similarity threshold
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

    def subtest_io_and_clone(self, index, Dnsg, Insg):
        fd, tmpfile = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, tmpfile)
            index2 = faiss.read_index(tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

        Dnsg2, Insg2 = index2.search(self.xq, 1)
        np.testing.assert_array_equal(Dnsg2, Dnsg)
        np.testing.assert_array_equal(Insg2, Insg)

        # also test clone
        index3 = faiss.clone_index(index)
        Dnsg3, Insg3 = index3.search(self.xq, 1)
        np.testing.assert_array_equal(Dnsg3, Dnsg)
        np.testing.assert_array_equal(Insg3, Insg)

    def subtest_connectivity(self, index, nb):
        vt = faiss.VisitedTable(nb)
        count = index.nsg.dfs(vt, index.nsg.enterpoint, 0)
        self.assertEqual(count, nb)

    def subtest_add(self, build_type, thresh, metric=faiss.METRIC_L2):
        d = self.xq.shape[1]
        metrics = {faiss.METRIC_L2: 'L2',
                   faiss.METRIC_INNER_PRODUCT: 'IP'}

        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16, metric)
        index.verbose = True
        index.build_type = build_type
        index.GK = self.GK
        index.add(self.xb)
        Dnsg, Insg = index.search(self.xq, 1)

        recalls = (Iref == Insg).sum()
        self.assertGreaterEqual(recalls, thresh)
        self.subtest_connectivity(index, self.xb.shape[0])
        self.subtest_io_and_clone(index, Dnsg, Insg)

    def subtest_build(self, knn_graph, thresh, metric=faiss.METRIC_L2):
        d = self.xq.shape[1]
        metrics = {faiss.METRIC_L2: 'L2',
                   faiss.METRIC_INNER_PRODUCT: 'IP'}

        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16, metric)
        index.verbose = True

        index.build(self.xb, knn_graph)
        Dnsg, Insg = index.search(self.xq, 1)

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

    def test_build_L2(self):
        knn_graph = self.make_knn_graph(faiss.METRIC_L2)
        self.subtest_build(knn_graph, 475, faiss.METRIC_L2)

    def test_build_IP(self):
        knn_graph = self.make_knn_graph(faiss.METRIC_INNER_PRODUCT)
        self.subtest_build(knn_graph, 480, faiss.METRIC_INNER_PRODUCT)

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
        """test IndexNSG.reset()"""
        d = self.xq.shape[1]
        metrics = {faiss.METRIC_L2: 'L2',
                   faiss.METRIC_INNER_PRODUCT: 'IP'}

        metric = faiss.METRIC_L2
        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16)
        index.verbose = True
        index.GK = 32

        index.add(self.xb)
        Dnsg, Insg = index.search(self.xq, 1)
        recalls = (Iref == Insg).sum()
        self.assertGreaterEqual(recalls, 475)
        self.subtest_connectivity(index, self.xb.shape[0])

        index.reset()
        index.add(self.xb)
        Dnsg, Insg = index.search(self.xq, 1)
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

    def test_nsg_pq(self):
        """Test IndexNSGPQ"""
        d = self.xq.shape[1]
        R, pq_M = 32, 4
        index = faiss.index_factory(d, f"NSG{R}_PQ{pq_M}np")
        assert isinstance(index, faiss.IndexNSGPQ)
        idxpq = faiss.downcast_index(index.storage)
        assert index.nsg.R == R and idxpq.pq.M == pq_M

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, k=1)

        index.GK = 32
        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        # test accuracy
        recalls = (Iref == I).sum()
        self.assertGreaterEqual(recalls, 190)  # 193

        # test I/O
        self.subtest_io_and_clone(index, D, I)

    def test_nsg_sq(self):
        """Test IndexNSGSQ"""
        d = self.xq.shape[1]
        R = 32
        index = faiss.index_factory(d, f"NSG{R}_SQ8")
        assert isinstance(index, faiss.IndexNSGSQ)
        idxsq = faiss.downcast_index(index.storage)
        assert index.nsg.R == R
        assert idxsq.sq.qtype == faiss.ScalarQuantizer.QT_8bit

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, k=1)

        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        # test accuracy
        recalls = (Iref == I).sum()
        self.assertGreaterEqual(recalls, 405)  # 411

        # test I/O
        self.subtest_io_and_clone(index, D, I)


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
        Dref, Iref = flat_index.search(self.xq, k=1)

        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        # test accuracy
        recalls = (Iref == I).sum()
        self.assertGreaterEqual(recalls, 450)  # 462

        # do some IO tests
        fd, tmpfile = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, tmpfile)
            index2 = faiss.read_index(tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

        D2, I2 = index2.search(self.xq, 1)
        np.testing.assert_array_equal(D2, D)
        np.testing.assert_array_equal(I2, I)

        # also test clone
        index3 = faiss.clone_index(index)
        D3, I3 = index3.search(self.xq, 1)
        np.testing.assert_array_equal(D3, D)
        np.testing.assert_array_equal(I3, I)

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


class TestNNDescentKNNG(unittest.TestCase):

    def test_knng_L2(self):
        self.subtest(32, 10, faiss.METRIC_L2)

    def test_knng_IP(self):
        self.subtest(32, 10, faiss.METRIC_INNER_PRODUCT)

    def subtest(self, d, K, metric):
        metric_names = {faiss.METRIC_L1: 'L1',
                        faiss.METRIC_L2: 'L2',
                        faiss.METRIC_INNER_PRODUCT: 'IP'}

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
        assert recall > 0.99

    def test_small_nndescent(self):
        """ building a too small graph used to crash, make sure it raises
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

        xb = np.zeros((78, d), dtype='float32')
        self.assertRaises(RuntimeError, index.add, xb)
