# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# noqa E741

import numpy as np

import faiss
import unittest

from common_faiss_tests import get_dataset_2

from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.evaluation import check_ref_knn_with_draws

import scipy.spatial.distance


class TestExtraDistances(unittest.TestCase):
    """ check wrt. the scipy implementation """

    def make_example(self):
        rs = np.random.RandomState(123)
        x = rs.rand(5, 32).astype('float32')
        y = rs.rand(3, 32).astype('float32')
        return x, y

    def run_simple_dis_test(self, ref_func, metric_type):
        xq, yb = self.make_example()
        ref_dis = np.array([
            [ref_func(x, y) for y in yb]
            for x in xq
        ])
        new_dis = faiss.pairwise_distances(xq, yb, metric_type)
        self.assertTrue(np.allclose(ref_dis, new_dis))

    def test_L1(self):
        self.run_simple_dis_test(scipy.spatial.distance.cityblock,
                                 faiss.METRIC_L1)

    def test_Linf(self):
        self.run_simple_dis_test(scipy.spatial.distance.chebyshev,
                                 faiss.METRIC_Linf)

    def test_L2(self):
        xq, yb = self.make_example()
        ref_dis = np.array([
            [scipy.spatial.distance.sqeuclidean(x, y) for y in yb]
            for x in xq
        ])
        new_dis = faiss.pairwise_distances(xq, yb, faiss.METRIC_L2)
        self.assertTrue(np.allclose(ref_dis, new_dis))

        ref_dis = np.array([
            [scipy.spatial.distance.euclidean(x, y) for y in yb]
            for x in xq
        ])
        new_dis = np.sqrt(new_dis)        # post processing
        self.assertTrue(np.allclose(ref_dis, new_dis))

    def test_Lp(self):
        p = 1.5
        xq, yb = self.make_example()
        ref_dis = np.array([
            [scipy.spatial.distance.minkowski(x, y, p) for y in yb]
            for x in xq
        ])
        new_dis = faiss.pairwise_distances(xq, yb, faiss.METRIC_Lp, p)
        new_dis = new_dis ** (1 / p)     # post processing
        self.assertTrue(np.allclose(ref_dis, new_dis))

    def test_canberra(self):
        self.run_simple_dis_test(scipy.spatial.distance.canberra,
                                 faiss.METRIC_Canberra)

    def test_braycurtis(self):
        self.run_simple_dis_test(scipy.spatial.distance.braycurtis,
                                 faiss.METRIC_BrayCurtis)

    def xx_test_jensenshannon(self):
        # this distance does not seem to be implemented in scipy
        # vectors should probably be L1 normalized
        self.run_simple_dis_test(scipy.spatial.distance.jensenshannon,
                                 faiss.METRIC_JensenShannon)

    def test_jaccard(self):
        xq, yb = self.make_example()
        ref_dis = np.array([
            [
                (np.min([x, y], axis=0).sum() / np.max([x, y], axis=0).sum())
                for y in yb
            ]
            for x in xq
        ])
        new_dis = faiss.pairwise_distances(xq, yb, faiss.METRIC_Jaccard)
        self.assertTrue(np.allclose(ref_dis, new_dis))

    def test_nan_euclidean(self):
        xq, yb = self.make_example()
        ref_dis = np.array([
            [scipy.spatial.distance.sqeuclidean(x, y) for y in yb]
            for x in xq
        ])
        new_dis = faiss.pairwise_distances(xq, yb, faiss.METRIC_NaNEuclidean)
        self.assertTrue(np.allclose(ref_dis, new_dis))

        x = [[3, np.nan, np.nan, 6]]
        q = [[1, np.nan, np.nan, 5]]
        dis = [(4 / 2 * ((3 - 1)**2 + (6 - 5)**2))]
        new_dis = faiss.pairwise_distances(x, q, faiss.METRIC_NaNEuclidean)
        self.assertTrue(np.allclose(new_dis, dis))

        x = [[np.nan] * 4]
        q = [[np.nan] * 4]
        new_dis = faiss.pairwise_distances(x, q, faiss.METRIC_NaNEuclidean)
        self.assertTrue(np.isnan(new_dis[0]))

    def test_gower(self):
        # Create test data with mixed numeric and categorical features
        # First 2 dimensions are numeric (0-1), last 2 are categorical
        # (negative integers)
        xq = np.array(
            [
                [0.5, 0.3, -1, -2],  # First query vector
                [0.7, 0.8, -1, -3],  # Second query vector
            ],
            dtype="float32",
        )

        yb = np.array(
            [
                [0.4, 0.2, -1, -2],  # Same categories, similar numeric values
                [0.9, 0.1, -1, -2],  # Same categories, different numeric values
                [0.5, 0.3, -2, -2],  # Different first category, same second
                [0.5, 0.3, -2, -3],  # Different categories
            ],
            dtype="float32",
        )

        # Compute distances using FAISS
        dis = faiss.pairwise_distances(xq, yb, faiss.METRIC_GOWER)

        # Expected distances:
        # For first query [0.5, 0.3, -1, -2]:
        # - [0.4, 0.2, -1, -2]: (|0.5-0.4| + |0.3-0.2| + 0 + 0) / 4 = 0.05
        # - [0.9, 0.1, -1, -2]: (|0.5-0.9| + |0.3-0.1| + 0 + 0) / 4 = 0.15
        # - [0.5, 0.3, -2, -2]: (|0.5-0.5| + |0.3-0.3| + 1 + 0) / 4 = 0.25
        # - [0.5, 0.3, -2, -3]: (|0.5-0.5| + |0.3-0.3| + 1 + 1) / 4 = 0.5

        # For second query [0.7, 0.8, -1, -3]:
        # - [0.4, 0.2, -1, -2]: (|0.7-0.4| + |0.8-0.2| + 0 + 1) / 4 = 0.475
        # - [0.9, 0.1, -1, -2]: (|0.7-0.9| + |0.8-0.1| + 0 + 1) / 4 = 0.475
        # - [0.5, 0.3, -2, -2]: (|0.7-0.5| + |0.8-0.3| + 1 + 1) / 4 = 0.675
        # - [0.5, 0.3, -2, -3]: (|0.7-0.5| + |0.8-0.3| + 1 + 0) / 4 = 0.425

        expected = np.array(
            [
                [0.05, 0.15, 0.25, 0.50],  # Distances for first query
                [0.475, 0.475, 0.675, 0.425],  # Distances for second query
            ],
            dtype="float32",
        )

        self.assertTrue(np.allclose(dis, expected, rtol=1e-5))

        # Test with NaN values
        xq_nan = np.array(
            [
                [0.5, np.nan, -1, -2],
                [0.7, 0.8, -1, -3],
            ],
            dtype="float32",
        )

        yb_nan = np.array(
            [
                [0.4, 0.2, -1, -2],
                [0.9, np.nan, -1, -2],
            ],
            dtype="float32",
        )

        dis_nan = faiss.pairwise_distances(xq_nan, yb_nan, faiss.METRIC_GOWER)

        # For first query [0.5, nan, -1, -2]:
        # - [0.4, 0.2, -1, -2]: (|0.5-0.4| + 0 + 0) / 3 = 0.0333...
        # - [0.9, nan, -1, -2]: (|0.5-0.9| + 0 + 0) / 3 = 0.1333...

        # For second query [0.7, 0.8, -1, -3]:
        # - [0.4, 0.2, -1, -2]: (|0.7-0.4| + |0.8-0.2| + 0 + 1) / 4 = 0.475
        # - [0.9, nan, -1, -2]: (|0.7-0.9| + 0 + 0 + 1) / 3 = 0.4

        expected_nan = np.array(
            [
                [0.033333, 0.133333],
                [0.475, 0.4],
            ],
            dtype="float32",
        )

        self.assertTrue(np.allclose(dis_nan, expected_nan, rtol=1e-5))

        # Test error case: mixing numeric and categorical values
        xq_mixed = np.array(
            [
                # Second value is categorical but first is numeric
                [0.5, -1, -1, -2],
            ],
            dtype="float32",
        )

        yb_mixed = np.array(
            [
                [0.4, 0.2, -1, -2],  # Second value is numeric
            ],
            dtype="float32",
        )

        dis_mixed = faiss.pairwise_distances(xq_mixed, yb_mixed, faiss.METRIC_GOWER)

        self.assertTrue(np.all(np.isnan(dis_mixed)))

        # Test error case: numeric values outside [0,1] range
        xq_out_of_range = np.array(
            [
                [1.5, 0.3, -1, -2],  # First value is outside [0,1]
            ],
            dtype="float32",
        )

        yb_out_of_range = np.array(
            [
                [0.4, 0.2, -1, -2],
            ],
            dtype="float32",
        )

        # Should return NaN for invalid data (consistent with other metrics)
        dis_out_of_range = faiss.pairwise_distances(
            xq_out_of_range, yb_out_of_range, faiss.METRIC_GOWER
        )
        self.assertTrue(np.all(np.isnan(dis_out_of_range)))

class TestKNN(unittest.TestCase):
    """ test that the knn search gives the same as distance matrix + argmin """

    def do_test_knn(self, mt):
        d = 10
        nb = 100
        nq = 50
        nt = 0
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        index = faiss.IndexFlat(d, mt)
        index.add(xb)

        D, I = index.search(xq, 10)

        dis = faiss.pairwise_distances(xq, xb, mt)
        o = dis.argsort(axis=1)
        assert np.all(I == o[:, :10])

        for q in range(nq):
            assert np.all(D[q] == dis[q, I[q]])

        index2 = faiss.deserialize_index(faiss.serialize_index(index))

        D2, I2 = index2.search(xq, 10)

        self.assertTrue(np.all(I == I2))

    def test_L1(self):
        self.do_test_knn(faiss.METRIC_L1)

    def test_Linf(self):
        self.do_test_knn(faiss.METRIC_Linf)


class TestHNSW(unittest.TestCase):
    """ since it has a distance computer, HNSW should work """

    def test_hnsw(self):

        d = 10
        nb = 1000
        nq = 100
        nt = 0
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        mt = faiss.METRIC_L1

        index = faiss.IndexHNSW(faiss.IndexFlat(d, mt))
        index.add(xb)

        D, I = index.search(xq, 10)

        dis = faiss.pairwise_distances(xq, xb, mt)

        for q in range(nq):
            assert np.all(D[q] == dis[q, I[q]])


class TestIVF(unittest.TestCase):
    """ since it has a distance computer, IVF should work """

    def test_ivf(self):

        ds = SyntheticDataset(10, 1000, 200, 20)
        mt = faiss.METRIC_L1

        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), 10, mt)
        index = faiss.IndexIVFFlat(faiss.IndexFlat(ds.d), ds.d, 20, mt)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 10

        Dnew, Inew = index.search(ds.get_queries(), 10)
        inter = faiss.eval_intersection(Iref, Inew)
        self.assertGreater(inter, Iref.size * 0.9)

        index.nprobe = index.nlist 
        Dnew, Inew = index.search(ds.get_queries(), 10)

        check_ref_knn_with_draws(Dref, Iref, Dnew, Inew) 
