# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

# not linting this file because it imports * from swigfaiss, which
# causes a ton of useless warnings.

import numpy as np

from faiss.loader import *

import faiss

###########################################
# Wrapper for a few functions
###########################################


def kmin(array, k):
    """return k smallest values (and their indices) of the lines of a
    float32 array"""
    array = np.ascontiguousarray(array, dtype='float32')
    m, n = array.shape
    I = np.zeros((m, k), dtype='int64')
    D = np.zeros((m, k), dtype='float32')
    ha = faiss.float_maxheap_array_t()
    ha.ids = swig_ptr(I)
    ha.val = swig_ptr(D)
    ha.nh = m
    ha.k = k
    ha.heapify()
    ha.addn(n, swig_ptr(array))
    ha.reorder()
    return D, I


def kmax(array, k):
    """return k largest values (and their indices) of the lines of a
    float32 array"""
    array = np.ascontiguousarray(array, dtype='float32')
    m, n = array.shape
    I = np.zeros((m, k), dtype='int64')
    D = np.zeros((m, k), dtype='float32')
    ha = faiss.float_minheap_array_t()
    ha.ids = swig_ptr(I)
    ha.val = swig_ptr(D)
    ha.nh = m
    ha.k = k
    ha.heapify()
    ha.addn(n, swig_ptr(array))
    ha.reorder()
    return D, I


def pairwise_distances(xq, xb, mt=METRIC_L2, metric_arg=0):
    """compute the whole pairwise distance matrix between two sets of
    vectors"""
    xq = np.ascontiguousarray(xq, dtype='float32')
    xb = np.ascontiguousarray(xb, dtype='float32')
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2
    dis = np.empty((nq, nb), dtype='float32')
    if mt == METRIC_L2:
        pairwise_L2sqr(
            d, nq, swig_ptr(xq),
            nb, swig_ptr(xb),
            swig_ptr(dis))
    else:
        pairwise_extra_distances(
            d, nq, swig_ptr(xq),
            nb, swig_ptr(xb),
            mt, metric_arg,
            swig_ptr(dis))
    return dis


def rand(n, seed=12345):
    res = np.empty(n, dtype='float32')
    float_rand(swig_ptr(res), res.size, seed)
    return res


def randint(n, seed=12345, vmax=None):
    res = np.empty(n, dtype='int64')
    if vmax is None:
        int64_rand(swig_ptr(res), res.size, seed)
    else:
        int64_rand_max(swig_ptr(res), res.size, vmax, seed)
    return res


lrand = randint


def randn(n, seed=12345):
    res = np.empty(n, dtype='float32')
    float_randn(swig_ptr(res), res.size, seed)
    return res


rand_smooth_vectors_c = rand_smooth_vectors


def rand_smooth_vectors(n, d, seed=1234):
    res = np.empty((n, d), dtype='float32')
    rand_smooth_vectors_c(n, d, swig_ptr(res), seed)
    return res


def eval_intersection(I1, I2):
    """ size of intersection between each line of two result tables"""
    I1 = np.ascontiguousarray(I1, dtype='int64')
    I2 = np.ascontiguousarray(I2, dtype='int64')
    n = I1.shape[0]
    assert I2.shape[0] == n
    k1, k2 = I1.shape[1], I2.shape[1]
    ninter = 0
    for i in range(n):
        ninter += ranklist_intersection_size(
            k1, swig_ptr(I1[i]), k2, swig_ptr(I2[i]))
    return ninter


def normalize_L2(x):
    fvec_renorm_L2(x.shape[1], x.shape[0], swig_ptr(x))


###########################################
# ResultHeap
###########################################

class ResultHeap:
    """Accumulate query results from a sliced dataset. The final result will
    be in self.D, self.I."""

    def __init__(self, nq, k, keep_max=False):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        if keep_max:
            heaps = float_minheap_array_t()
        else:
            heaps = float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = swig_ptr(self.D)
        heaps.ids = swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_result(self, D, I):
        """D, I do not need to be in a particular order (heap or sorted)"""
        nq, kd = D.shape
        D = np.ascontiguousarray(D, dtype='float32')
        I = np.ascontiguousarray(I, dtype='int64')
        assert I.shape == (nq, kd)
        assert nq == self.nq
        self.heaps.addn_with_ids(
            kd, swig_ptr(D),
            swig_ptr(I), kd)

    def finalize(self):
        self.heaps.reorder()



######################################################
# KNN function
######################################################

def knn(xq, xb, k, metric=METRIC_L2):
    """
    Compute the k nearest neighbors of a vector without constructing an index


    Parameters
    ----------
    xq : array_like
        Query vectors, shape (nq, d) where d is appropriate for the index.
        `dtype` must be float32.
    xb : array_like
        Database vectors, shape (nb, d) where d is appropriate for the index.
        `dtype` must be float32.
    k : int
        Number of nearest neighbors.
    distance_type : MetricType, optional
        distance measure to use (either METRIC_L2 or METRIC_INNER_PRODUCT)

    Returns
    -------
    D : array_like
        Distances of the nearest neighbors, shape (nq, k)
    I : array_like
        Labels of the nearest neighbors, shape (nq, k)
    """
    xq = np.ascontiguousarray(xq, dtype='float32')
    xb = np.ascontiguousarray(xb, dtype='float32')
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2

    I = np.empty((nq, k), dtype='int64')
    D = np.empty((nq, k), dtype='float32')

    if metric == METRIC_L2:
        knn_L2sqr(
            swig_ptr(xq), swig_ptr(xb),
            d, nq, nb, k, swig_ptr(D), swig_ptr(I)
        )
    elif metric == METRIC_INNER_PRODUCT:
        knn_inner_product(
            swig_ptr(xq), swig_ptr(xb),
            d, nq, nb, k, swig_ptr(D), swig_ptr(I)
        )
    else:
        raise NotImplementedError("only L2 and INNER_PRODUCT are supported")
    return D, I


###########################################
# Kmeans object
###########################################


class Kmeans:
    """Object that performs k-means clustering and manages the centroids.
    The `Kmeans` class is essentially a wrapper around the C++ `Clustering` object.

    Parameters
    ----------
    d : int
       dimension of the vectors to cluster
    k : int
       number of clusters
    gpu: bool or int, optional
       False: don't use GPU
       True: use all GPUs
       number: use this many GPUs
    progressive_dim_steps:
        use a progressive dimension clustering (with that number of steps)

    Subsequent parameters are fields of the Clustring object. The most important are:

    niter: int, optional
       clustering iterations
    nredo: int, optional
       redo clustering this many times and keep best
    verbose: bool, optional
    spherical: bool, optional
       do we want normalized centroids?
    int_centroids: bool, optional
       round centroids coordinates to integer
    seed: int, optional
       seed for the random number generator

    """

    def __init__(self, d, k, **kwargs):
        """d: input dimension, k: nb of centroids. Additional
         parameters are passed on the ClusteringParameters object,
         including niter=25, verbose=False, spherical = False
        """
        self.d = d
        self.k = k
        self.gpu = False
        if "progressive_dim_steps" in kwargs:
            self.cp = ProgressiveDimClusteringParameters()
        else:
            self.cp = ClusteringParameters()
        for k, v in kwargs.items():
            if k == 'gpu':
                if v == True or v == -1:
                    v = get_num_gpus()
                self.gpu = v
            else:
                # if this raises an exception, it means that it is a non-existent field
                getattr(self.cp, k)
                setattr(self.cp, k, v)
        self.centroids = None

    def train(self, x, weights=None, init_centroids=None):
        """ Perform k-means clustering.
        On output of the function call:

        - the centroids are in the centroids field of size (`k`, `d`).

        - the objective value at each iteration is in the array obj (size `niter`)

        - detailed optimization statistics are in the array iteration_stats.

        Parameters
        ----------
        x : array_like
            Training vectors, shape (n, d), `dtype` must be float32 and n should
            be larger than the number of clusters `k`.
        weights : array_like
            weight associated to each vector, shape `n`
        init_centroids : array_like
            initial set of centroids, shape (n, d)

        Returns
        -------
        final_obj: float
            final optimization objective

        """
        x = np.ascontiguousarray(x, dtype='float32')
        n, d = x.shape
        assert d == self.d

        if self.cp.__class__ == ClusteringParameters:
            # regular clustering
            clus = Clustering(d, self.k, self.cp)
            if init_centroids is not None:
                nc, d2 = init_centroids.shape
                assert d2 == d
                faiss.copy_array_to_vector(init_centroids.ravel(), clus.centroids)
            if self.cp.spherical:
                self.index = IndexFlatIP(d)
            else:
                self.index = IndexFlatL2(d)
            if self.gpu:
                self.index = faiss.index_cpu_to_all_gpus(self.index, ngpu=self.gpu)
            clus.train(x, self.index, weights)
        else:
            # not supported for progressive dim
            assert weights is None
            assert init_centroids is None
            assert not self.cp.spherical
            clus = ProgressiveDimClustering(d, self.k, self.cp)
            if self.gpu:
                fac = GpuProgressiveDimIndexFactory(ngpu=self.gpu)
            else:
                fac = ProgressiveDimIndexFactory()
            clus.train(n, swig_ptr(x), fac)

        centroids = faiss.vector_float_to_array(clus.centroids)

        self.centroids = centroids.reshape(self.k, d)
        stats = clus.iteration_stats
        stats = [stats.at(i) for i in range(stats.size())]
        self.obj = np.array([st.obj for st in stats])
        # copy all the iteration_stats objects to a python array
        stat_fields = 'obj time time_search imbalance_factor nsplit'.split()
        self.iteration_stats = [
            {field: getattr(st, field) for field in stat_fields}
            for st in stats
        ]
        return self.obj[-1] if self.obj.size > 0 else 0.0

    def assign(self, x):
        x = np.ascontiguousarray(x, dtype='float32')
        assert self.centroids is not None, "should train before assigning"
        self.index.reset()
        self.index.add(self.centroids)
        D, I = self.index.search(x, 1)
        return D.ravel(), I.ravel()
