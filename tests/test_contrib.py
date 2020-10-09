import faiss
import unittest
import numpy as np
import platform

from faiss.contrib import datasets

from common import get_dataset_2
try:
    from faiss.contrib.exhaustive_search import knn_ground_truth, knn

except:
    pass  # Submodule import broken in python 2.

@unittest.skipIf(platform.python_version_tuple()[0] < '3', \
                 'Submodule import broken in python 2.')
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
        # decimal = 4 required when run on GPU
        np.testing.assert_almost_equal(Dref, Dnew, decimal=4)


class TestDatasets(unittest.TestCase):
    """here we test only the synthetic dataset. Datasets that require
    disk or manifold access are in
    //deeplearning/projects/faiss-forge/test_faiss_datasets/:test_faiss_datasets
    """

    def test_synthetic(self):
        ds = datasets.SynteticDataset(32, 1000, 2000, 10)
        xq = ds.get_queries()
        self.assertEqual(xq.shape, (10, 32))
        xb = ds.get_database()
        self.assertEqual(xb.shape, (2000, 32))
        ds.check_sizes()

    def test_synthetic_iterator(self):
        ds = datasets.SynteticDataset(32, 1000, 2000, 10)
        xb = ds.get_database()
        xb2 = []
        for xbi in ds.database_iterator():
            xb2.append(xbi)
        xb2 = np.vstack(xb2)
        np.testing.assert_array_equal(xb, xb2)


class TestExhaustiveSearch(unittest.TestCase):

    def test_knn_cpu(self):

        xb = np.random.rand(200, 32).astype('float32')
        xq = np.random.rand(100, 32).astype('float32')


        index = faiss.IndexFlatL2(32)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        Dnew, Inew = knn(xq, xb, 10)

        assert np.all(Inew == Iref)
        assert np.allclose(Dref, Dnew)


        index = faiss.IndexFlatIP(32)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        Dnew, Inew = knn(xq, xb, 10, distance_type=faiss.METRIC_INNER_PRODUCT)

        assert np.all(Inew == Iref)
        assert np.allclose(Dref, Dnew)
