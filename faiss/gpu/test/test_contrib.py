import faiss
import unittest
import numpy as np

from faiss.contrib import datasets
from faiss.contrib.exhaustive_search import knn_ground_truth, knn_gpu

from common import get_dataset_2


class TestComputeGT(unittest.TestCase):

    def test_compute_GT(self):
        d = 64
        xt, xb, xq = get_dataset_2(d, 0, 10000, 100)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        # iterator function on the matrix

        def matrix_iterator(xb, bs):
            for i0 in range(0, xb.shape[0], bs):
                yield xb[i0:i0 + bs]

        Dnew, Inew = knn_ground_truth(xq, matrix_iterator(xb, 1000), 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_almost_equal(Dref, Dnew, decimal=4)

class TestBfKnnNumpy(unittest.TestCase):

    def test_bf_knn(self):
        d = 64
        k = 10
        xt, xb, xq = get_dataset_2(d, 0, 10000, 100)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        Dref, Iref = index.search(xq, k)

        res = faiss.StandardGpuResources()

        D, I = knn_gpu(res, xb, xq, k)

        np.testing.assert_array_equal(Iref, I)
        np.testing.assert_almost_equal(Dref, D, decimal=4)

        # Test transpositions
        xbt = np.ascontiguousarray(xb.T)

        D, I = knn_gpu(res, xbt.T, xq, k)

        np.testing.assert_array_equal(Iref, I)
        np.testing.assert_almost_equal(Dref, D, decimal=4)

        xqt = np.ascontiguousarray(xq.T)

        D, I = knn_gpu(res, xb, xqt.T, k)

        np.testing.assert_array_equal(Iref, I)
        np.testing.assert_almost_equal(Dref, D, decimal=4)

        D, I = knn_gpu(res, xbt.T, xqt.T, k)

        np.testing.assert_array_equal(Iref, I)
        np.testing.assert_almost_equal(Dref, D, decimal=4)

        # Test f16 data types
        xb16 = xb.astype(np.float16)
        xq16 = xq.astype(np.float16)

        D, I = knn_gpu(res, xb, xq, k)

        np.testing.assert_array_equal(Iref, I)
        np.testing.assert_almost_equal(Dref, D, decimal=4)

        # Test i32 indices
        I32 = np.empty((xq.shape[0], k), dtype=np.int32)

        D, _ = knn_gpu(res, xb, xq, k, I=I32)

        np.testing.assert_array_equal(Iref, I32)
        np.testing.assert_almost_equal(Dref, D, decimal=4)
