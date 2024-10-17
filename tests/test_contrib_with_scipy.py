# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import unittest
import numpy as np

from faiss.contrib import datasets
from faiss.contrib import clustering

import scipy.sparse

# this test is not in test_contrib because it depends on scipy


class TestClustering(unittest.TestCase):

    def test_sparse_routines(self):
        """ the sparse assignment routine """
        ds = datasets.SyntheticDataset(1000, 2000, 0, 200)
        xt = ds.get_train().copy()
        faiss.normalize_L2(xt)

        mask = np.abs(xt) > 0.045
        xt[np.logical_not(mask)] = 0

        centroids = ds.get_queries()
        assert len(centroids) == 200

        xsparse = scipy.sparse.csr_matrix(xt)

        Dref, Iref = faiss.knn(xsparse.todense(), centroids, 1)
        D, I = clustering.sparse_assign_to_dense(xsparse, centroids)

        np.testing.assert_array_equal(Iref.ravel(), I)
        np.testing.assert_array_almost_equal(Dref.ravel(), D, decimal=3)

        D, I = clustering.sparse_assign_to_dense_blocks(
            xsparse, centroids, qbs=123, bbs=33, nt=4)

        np.testing.assert_array_equal(Iref.ravel(), I)
        np.testing.assert_array_almost_equal(Dref.ravel(), D, decimal=3)

    def test_sparse_kmeans(self):
        """ demo on how to cluster sparse data into dense clusters """

        ds = datasets.SyntheticDataset(1000, 1500, 0, 0)
        xt = ds.get_train().copy()
        faiss.normalize_L2(xt)

        mask = np.abs(xt) > 0.045
        xt[np.logical_not(mask)] = 0

        km = faiss.Kmeans(ds.d, 50)
        km.train(xt)
        ref_err = km.iteration_stats[-1]["obj"]

        xsparse = scipy.sparse.csr_matrix(xt)

        centroids, iteration_stats = clustering.kmeans(
            50, clustering.DatasetAssignSparse(xsparse), return_stats=True)

        new_err = iteration_stats[-1]["obj"]

        self.assertLess(new_err, ref_err * 1.1)
