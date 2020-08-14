import faiss
import unittest
import numpy as np

from common import get_dataset_2
from faiss.contrib.exhaustive_search import knn_ground_truth

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
        np.testing.assert_almost_equal(Dref, Dnew, decimal=5)
