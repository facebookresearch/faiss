# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for IndexPiPNN (Flat and SQ storage)."""

import unittest

import faiss
import numpy as np

from common_faiss_tests import get_dataset_2

DIM = 32
NB = 20000
NQ = 500
K = 10


def recall(index, xb, xq):
    _, gt = faiss.knn(xq, xb, K, index.metric_type)
    _, labels = index.search(xq, K)
    return faiss.eval_intersection(labels, gt) / labels.size


def ip_dataset(nb, nq):
    # unnormalized, mixed-sign inner products
    _, xb, xq = get_dataset_2(DIM, 0, nb, nq)
    scale = np.random.RandomState(123).uniform(0.5, 2.0, size=(nb, 1))
    return (xb * scale).astype("float32"), xq


def graph_of(index):
    return faiss.vector_to_array(index.pipnn.graph), index.pipnn.entry_point


class TestPiPNNBuild(unittest.TestCase):

    def build(self, xb, metric, **params):
        index = faiss.IndexPiPNNFlat(DIM, 16, metric)
        # many leaves, merges and depth >= 3
        index.pipnn.c_max = 64
        index.pipnn.c_min = 16
        for name, value in params.items():
            setattr(index.pipnn, name, value)
        index.add(xb)
        return index

    def test_deterministic_build(self):
        self.addCleanup(faiss.omp_set_num_threads, faiss.omp_get_max_threads())
        _, xb_l2, _ = get_dataset_2(DIM, 0, 3000, 0)
        xb_ip, _ = ip_dataset(3000, 0)
        for metric, xb in (
            (faiss.METRIC_L2, xb_l2),
            (faiss.METRIC_INNER_PRODUCT, xb_ip),
        ):
            with self.subTest(metric=metric):
                faiss.omp_set_num_threads(1)
                i1 = self.build(xb, metric, replicas=2)
                faiss.omp_set_num_threads(8)
                i8 = self.build(xb, metric, replicas=2, partition_wave_budget=1)
                np.testing.assert_equal(graph_of(i8), graph_of(i1))
                # the entry point is the L2 medoid
                d2 = ((xb - xb.mean(axis=0, dtype=np.float64)) ** 2).sum(1)
                self.assertEqual(i1.pipnn.entry_point, np.argmin(d2))

    def test_second_replica_adds_neighbours(self):
        _, xb, _ = get_dataset_2(DIM, 0, 3000, 0)
        _, nn = faiss.knn(xb, xb, 2)

        def nn_in_row(replicas):
            g, _ = graph_of(self.build(xb, faiss.METRIC_L2, replicas=replicas))
            return (g.reshape(len(xb), -1) == nn[:, 1:]).any(axis=1).mean()

        self.assertGreater(nn_in_row(2), nn_in_row(1))

    def test_effective_alpha(self):
        p = faiss.PiPNN()
        self.assertAlmostEqual(p.effective_alpha(faiss.METRIC_L2), 1.2, 6)
        self.assertEqual(p.effective_alpha(faiss.METRIC_INNER_PRODUCT), 1.0)
        p.alpha = 1.5
        self.assertEqual(p.effective_alpha(faiss.METRIC_INNER_PRODUCT), 1.5)
        self.assertRaises(RuntimeError, p.effective_alpha, faiss.METRIC_L1)

    def test_sq8_train_add_search(self):
        _, xb, xq = get_dataset_2(DIM, 0, 10000, 200)
        index = faiss.IndexPiPNNSQ(DIM, faiss.ScalarQuantizer.QT_8bit, 32)
        self.assertFalse(index.is_trained)
        with self.assertRaisesRegex(RuntimeError, "is_trained"):
            index.add(xb)
        self.assertRaises(RuntimeError, index.search, xq, K)
        index.train(xb)
        index.add(xb)
        self.assertGreaterEqual(recall(index, xb, xq), 0.96)


class TestPiPNNRecall(unittest.TestCase):

    def test_recall_L2(self):
        _, xb, xq = get_dataset_2(DIM, 0, NB, NQ)
        index = faiss.IndexPiPNNFlat(DIM, 64)
        index.add(xb)
        self.assertGreaterEqual(recall(index, xb, xq), 0.97)

    def test_recall_IP(self):
        xb, xq = ip_dataset(NB, NQ)
        index = faiss.IndexPiPNNFlat(DIM, 64, faiss.METRIC_INNER_PRODUCT)
        index.add(xb)
        self.assertGreaterEqual(recall(index, xb, xq), 0.96)
        D, I = index.search(xq, K)
        # inner products are returned positive, not negated
        np.testing.assert_allclose(
            D[:, 0], (xq * xb[I[:, 0]]).sum(axis=1), rtol=1e-4, atol=1e-4
        )

    def test_recall_one_leaf(self):
        # nb <= c_max: each point offers its reservoir_size nearest, not leaf_k
        _, xb, xq = get_dataset_2(DIM, 0, 500, NQ)
        index = faiss.IndexPiPNNFlat(DIM, 32)
        index.add(xb)
        index.pipnn.search_L = 16
        self.assertGreaterEqual(recall(index, xb, xq), 0.95)


class TestPiPNNSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, cls.xb, cls.xq = get_dataset_2(DIM, 0, 10000, 200)
        cls.index = faiss.IndexPiPNNFlat(DIM, 32)
        cls.index.add(cls.xb)

    def test_id_selector(self):
        subset = np.sort(
            np.random.RandomState(5).choice(10000, 3000, replace=False)
        )
        params = faiss.SearchParametersPiPNN(
            sel=faiss.IDSelectorBatch(subset), search_L=128
        )
        _, labels = self.index.search(self.xq, K, params=params)
        self.assertTrue(np.all(np.isin(labels[labels >= 0], subset)))
        _, gt = faiss.knn(self.xq, self.xb[subset], K)
        r = faiss.eval_intersection(labels, subset[gt]) / labels.size
        self.assertGreaterEqual(r, 0.90)

    def test_search_L(self):
        index, P = self.index, faiss.SearchParametersPiPNN
        self.addCleanup(setattr, index.pipnn, "search_L", index.pipnn.search_L)
        index.pipnn.search_L = 16
        _, ref = index.search(self.xq, K)
        # params with search_L = 0 inherit the index value
        _, inherited = index.search(self.xq, K, params=P())
        np.testing.assert_array_equal(inherited, ref)
        _, wide = index.search(self.xq, K, params=P(search_L=128))
        self.assertFalse(np.array_equal(wide, ref))
        # search_L < k searches with a beam of k
        _, at_k = index.search(self.xq, K, params=P(search_L=K))
        _, below_k = index.search(self.xq, K, params=P(search_L=-1))
        np.testing.assert_array_equal(below_k, at_k)

    def test_k_above_ntotal(self):
        for nb in (1, 20):
            with self.subTest(nb=nb):
                _, xb, xq = get_dataset_2(DIM, 0, nb, 5)
                index = faiss.IndexPiPNNFlat(DIM, 32)
                index.add(xb)
                D, I = index.search(xq, nb + 5)
                Dref, Iref = faiss.knn(xq, xb, nb)
                np.testing.assert_array_equal(I[:, :nb], Iref)
                np.testing.assert_allclose(D[:, :nb], Dref, rtol=1e-5)
                np.testing.assert_array_equal(I[:, nb:], -1)
                np.testing.assert_array_equal(
                    D[:, nb:], np.finfo(np.float32).max
                )


class TestPiPNNErrors(unittest.TestCase):

    def setUp(self):
        _, self.xb, self.xq = get_dataset_2(DIM, 0, 3000, 10)

    def test_build_lifecycle(self):
        prefilled = faiss.IndexFlatL2(DIM)
        prefilled.add(self.xb)
        from_storage = faiss.IndexPiPNN(prefilled, 32)
        self.assertEqual(from_storage.ntotal, len(self.xb))
        deferred = faiss.IndexPiPNNFlat(DIM, 32)
        deferred.defer_build = True
        deferred.add(self.xb)
        for index in (deferred, from_storage):
            with self.assertRaisesRegex(RuntimeError, r"call build\(\)"):
                index.search(self.xq, K)
            index.build()
            with self.assertRaisesRegex(RuntimeError, "incremental addition"):
                index.add(self.xq)
            _, labels = index.search(self.xq, K)
            self.assertTrue(np.all(labels >= 0))
            index.storage.add(self.xq)
            with self.assertRaisesRegex(RuntimeError, r"call build\(\)"):
                index.search(self.xq, K)

    def test_unsupported_storage_ctor(self):
        SQ = faiss.ScalarQuantizer
        for storage in (
            faiss.IndexFlatL2Panorama(DIM, 4),
            faiss.IndexScalarQuantizer(DIM, SQ.QT_0bit),
            faiss.IndexScalarQuantizer(DIM, SQ.QT_4bit_tq),
            faiss.IndexFlat(DIM, faiss.METRIC_L1),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported storage"):
                faiss.IndexPiPNN(storage, 32)

    def test_reset_and_rebuild(self):
        index = faiss.IndexPiPNNFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
        index.add(self.xb)
        _, Iref = index.search(self.xq, K)
        index.reset()
        Dempty, Iempty = index.search(self.xq, K)
        np.testing.assert_array_equal(Iempty, -1)
        np.testing.assert_array_equal(Dempty, -np.finfo(np.float32).max)
        index.add(self.xb)
        np.testing.assert_array_equal(index.search(self.xq, K)[1], Iref)

    def test_invalid_params(self):
        with self.assertRaisesRegex(RuntimeError, "has no storage"):
            faiss.IndexPiPNN(None, 32)
        index = faiss.IndexPiPNNFlat(DIM, 32)
        index.pipnn.hash_bits = 17
        with self.assertRaisesRegex(RuntimeError, "hash_bits"):
            index.add(self.xb)

        bad = [
            ("R", 0),
            ("R", 65),  # > reservoir_size
            ("leaf_k", 0),
            ("leaf_k", 1025),  # > c_max
            ("hash_bits", 0),
            ("reservoir_size", 0),
            ("reservoir_size", 256),
            ("replicas", 0),
            ("c_min", 0),
            ("c_min", 2000),  # > c_max
            ("c_max", 4097),
            ("leader_cap", 0),
            ("leader_cap", 65536),
            ("leader_fraction", -0.1),
            ("leader_fraction", 1.5),
            ("alpha", 0.5),
            ("alpha", float("inf")),
        ]
        for name, value in bad:
            with self.subTest(name=name, value=value):
                p = faiss.PiPNN()
                setattr(p, name, value)
                self.assertRaises(RuntimeError, p.check_params)
        p = faiss.PiPNN()
        p.fanout.push_back(0)
        self.assertRaises(RuntimeError, p.check_params)
        limits = {
            "leaf_k": 4096,
            "hash_bits": 16,
            "reservoir_size": 255,
            "c_min": 4096,
            "c_max": 4096,
            "leader_cap": 65535,
            "leader_fraction": 0,
            "alpha": 1,
        }
        p = faiss.PiPNN()
        for name, value in limits.items():
            setattr(p, name, value)
        p.check_params()

    def test_interrupts(self):
        index = faiss.IndexPiPNNFlat(DIM, 32)
        index.add(self.xb)
        with faiss.TimeoutGuard(1e-6):
            with self.assertRaises(RuntimeError):
                index.search(self.xq, K)
