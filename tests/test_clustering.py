# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np

import faiss
import unittest
import array

from common_faiss_tests import get_dataset_2


class TestClustering(unittest.TestCase):

    def test_clustering(self):
        d = 64
        n = 1000
        rs = np.random.RandomState(123)
        x = rs.uniform(size=(n, d)).astype('float32')

        x *= 10

        km = faiss.Kmeans(d, 32, niter=10)
        err32 = km.train(x)

        # check that objective is decreasing
        prev = 1e50
        for o in km.obj:
            self.assertGreater(prev, o)
            prev = o

        km = faiss.Kmeans(d, 64, niter=10)
        err64 = km.train(x)

        # check that 64 centroids give a lower quantization error than 32
        self.assertGreater(err32, err64)

        km = faiss.Kmeans(d, 32, niter=10, int_centroids=True)
        err_int = km.train(x)

        # check that integer centoids are not as good as float ones
        self.assertGreater(err_int, err32)
        self.assertTrue(np.all(km.centroids == np.floor(km.centroids)))


    def test_nasty_clustering(self):
        d = 2
        rs = np.random.RandomState(123)
        x = np.zeros((100, d), dtype='float32')
        for i in range(5):
            x[i * 20:i * 20 + 20] = rs.uniform(size=d)

        # we have 5 distinct points but ask for 10 centroids...
        km = faiss.Kmeans(d, 10, niter=10, verbose=True)
        km.train(x)




    def test_1ptpercluster(self):
        # https://github.com/facebookresearch/faiss/issues/842
        X = np.random.randint(0, 1, (5, 10)).astype('float32')
        k = 5
        niter = 10
        verbose = True
        kmeans = faiss.Kmeans(X.shape[1], k, niter=niter, verbose=verbose)
        kmeans.train(X)
        l2_distances, I = kmeans.index.search(X, 1)

    def test_weighted(self):
        d = 32
        sigma = 0.1

        # Data is naturally clustered in 10 clusters.
        # 5 clusters have 100 points
        # 5 clusters have 10 points
        # run k-means with 5 clusters

        ccent = faiss.randn((10, d), 123)
        faiss.normalize_L2(ccent)
        x = [ccent[i] + sigma * faiss.randn((100, d), 1234 + i) for i in range(5)]
        x += [ccent[i] + sigma * faiss.randn((10, d), 1234 + i) for i in range(5, 10)]
        x = np.vstack(x)

        clus = faiss.Clustering(d, 5)
        index = faiss.IndexFlatL2(d)
        clus.train(x, index)
        cdis1, perm1 = index.search(ccent, 1)

        # distance^2 of ground-truth centroids to clusters
        cdis1_first = cdis1[:5].sum()
        cdis1_last = cdis1[5:].sum()

        # now assign weight 0.1 to the 5 first clusters and weight 10
        # to the 5 last ones and re-run k-means
        weights = np.ones(100 * 5 + 10 * 5, dtype='float32')
        weights[:100 * 5] = 0.1
        weights[100 * 5:] = 10

        clus = faiss.Clustering(d, 5)
        index = faiss.IndexFlatL2(d)
        clus.train(x, index, weights=weights)
        cdis2, perm2 = index.search(ccent, 1)

        # distance^2 of ground-truth centroids to clusters
        cdis2_first = cdis2[:5].sum()
        cdis2_last = cdis2[5:].sum()

        # with the new clustering, the last should be much (*2) closer
        # to their centroids
        self.assertGreater(cdis1_last, cdis1_first * 2)
        self.assertGreater(cdis2_first, cdis2_last * 2)

    def test_encoded(self):
        d = 32
        k = 5
        xt, xb, xq = get_dataset_2(d, 1000, 0, 0)

        # make sure that training on a compressed then decompressed
        # dataset gives the same result as decompressing on-the-fly

        codec = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_4bit)
        codec.train(xt)
        codes = codec.sa_encode(xt)

        xt2 = codec.sa_decode(codes)

        clus = faiss.Clustering(d, k)
        # clus.verbose = True
        clus.niter = 0
        index = faiss.IndexFlatL2(d)
        clus.train(xt2, index)
        ref_centroids = faiss.vector_to_array(clus.centroids).reshape(-1, d)

        _, ref_errs = index.search(xt2, 1)

        clus = faiss.Clustering(d, k)
        # clus.verbose = True
        clus.niter = 0
        clus.decode_block_size = 120
        index = faiss.IndexFlatL2(d)
        clus.train_encoded(codes, codec, index)
        new_centroids = faiss.vector_to_array(clus.centroids).reshape(-1, d)

        _, new_errs = index.search(xt2, 1)

        # It's the same operation, so should be bit-exact the same
        self.assertTrue(np.all(ref_centroids == new_centroids))

    def test_init(self):
        d = 32
        k = 5
        xt, xb, xq = get_dataset_2(d, 1000, 0, 0)
        km = faiss.Kmeans(d, k, niter=4)
        km.train(xt)

        km2 = faiss.Kmeans(d, k, niter=4)
        km2.train(xt, init_centroids=km.centroids)

        # check that the initial objective is better for km2 than km
        self.assertGreater(km.obj[0], km2.obj[0] * 1.01)

    def test_stats(self):
        d = 32
        k = 5
        xt, xb, xq = get_dataset_2(d, 1000, 0, 0)
        km = faiss.Kmeans(d, k, niter=4)
        km.train(xt)
        assert list(km.obj) == [st['obj'] for st in km.iteration_stats]


class TestCompositeClustering(unittest.TestCase):

    def test_redo(self):
        d = 64
        n = 1000

        rs = np.random.RandomState(123)
        x = rs.uniform(size=(n, d)).astype('float32')

        # make sure that doing 10 redos yields a better objective than just 1

        clus = faiss.Clustering(d, 20)
        clus.nredo = 1
        clus.train(x, faiss.IndexFlatL2(d))
        obj1 = clus.iteration_stats.at(clus.iteration_stats.size() - 1).obj

        clus = faiss.Clustering(d, 20)
        clus.nredo = 10
        clus.train(x, faiss.IndexFlatL2(d))
        obj10 = clus.iteration_stats.at(clus.iteration_stats.size() - 1).obj

        self.assertGreater(obj1, obj10)

    def test_redo_cosine(self):
        # test redo with cosine distance (inner prod, so objectives are reversed)
        d = 64
        n = 1000

        rs = np.random.RandomState(123)
        x = rs.uniform(size=(n, d)).astype('float32')
        faiss.normalize_L2(x)

        # make sure that doing 10 redos yields a better objective than just 1
        # for cosine distance, it is IP so higher is better

        clus = faiss.Clustering(d, 20)
        clus.nredo = 1
        clus.train(x, faiss.IndexFlatIP(d))
        obj1 = clus.iteration_stats.at(clus.iteration_stats.size() - 1).obj

        clus = faiss.Clustering(d, 20)
        clus.nredo = 10
        clus.train(x, faiss.IndexFlatIP(d))
        obj10 = clus.iteration_stats.at(clus.iteration_stats.size() - 1).obj

        self.assertGreater(obj10, obj1)

    def test_progressive_dim(self):
        d = 32
        n = 10000
        k = 50
        xt, _, _ = get_dataset_2(d, n, 0, 0)

        # basic kmeans
        kmeans = faiss.Kmeans(d, k)
        kmeans.train(xt)

        clus = faiss.ProgressiveDimClustering(d, k)
        clus.verbose
        clus.verbose = True
        clus.progressive_dim_steps
        clus.progressive_dim_steps = 5
        fac = faiss.ProgressiveDimIndexFactory()
        clus.train(n, faiss.swig_ptr(xt), fac)

        stats = clus.iteration_stats
        stats = [stats.at(i) for i in range(stats.size())]
        obj = np.array([st.obj for st in stats])
        # clustering objective should be a tad better
        self.assertLess(obj[-1], kmeans.obj[-1])

        # same test w/ Kmeans wrapper
        kmeans2 = faiss.Kmeans(d, k, progressive_dim_steps=5)
        kmeans2.train(xt)
        self.assertLess(kmeans2.obj[-1], kmeans.obj[-1])


class TestClustering1D(unittest.TestCase):

    def evaluate_obj(self, centroids, x):
        index = faiss.IndexFlatL2(1)
        index.add(centroids)
        D, I = index.search(x, k=1)
        return D.sum()

    def subtest_cluster1d(self, n, k):
        rs = np.random.RandomState(123)
        x = rs.uniform(size=(n, 1)).astype('float32')

        clus = faiss.Clustering1D(k)
        clus.train_exact(x)
        centroids = faiss.vector_to_array(clus.centroids).reshape((-1, 1))
        obj = self.evaluate_obj(centroids, x)

        clus2 = faiss.Kmeans(1, k)
        clus2.train(x)
        obj2 = self.evaluate_obj(clus2.centroids, x)
        self.assertLessEqual(obj, obj2)

    def test_cluster1d(self):
        self.subtest_cluster1d(20, 20)
        self.subtest_cluster1d(1000, 20)
        self.subtest_cluster1d(1000, 256)

    def test_smawk(self):
        # example in http://web.cs.unlv.edu/larmore/Courses/CSC477/monge.pdf.
        A = [[ 25, 21, 13,10,20,13,19,35,37,41,58,66,82,99,124,133,156,178],
             [ 42, 35, 26,20,29,21,25,37,36,39,56,64,76,91,116,125,146,164],
             [ 57, 48, 35,28,33,24,28,40,37,37,54,61,72,83,107,113,131,146],
             [ 78, 65, 51,42,44,35,38,48,42,42,55,61,70,80,100,106,120,135],
             [ 90, 76, 58,48,49,39,42,48,39,35,47,51,56,63, 80, 86, 97,110],
             [103, 85, 67,56,55,44,44,49,39,33,41,44,49,56, 71, 75, 84, 96],
             [123,105, 86,75,73,59,57,62,51,44,50,52,55,59, 72, 74, 80, 92],
             [142,123,100,86,82,65,61,62,50,43,47,45,46,46, 58, 59, 65, 73],
             [151,130,104,88,80,59,52,49,37,29,29,24,23,20, 28, 25, 31, 39]];

        sp = faiss.swig_ptr
        A = np.array(A).astype(np.float32)
        nrows, ncols = A.shape
        argmins = np.zeros(nrows).astype(np.int64)

        faiss.smawk(nrows, ncols, sp(A), sp(argmins))
        argmins_ref = np.argmin(A, axis=1)
        assert np.array_equal(argmins, argmins_ref)
