# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This contrib module contains a few routines useful to do clustering variants.
"""

import numpy as np
import faiss
import time
from multiprocessing.pool import ThreadPool


try:
    import scipy.sparse
except ImportError:
    print("scipy not accessible, Python k-means will not work")

def print_nop(*arg, **kwargs):
    pass

def two_level_clustering(xt, nc1, nc2, rebalance=True, clustering_niter=25, **args):
    """
    perform 2-level clustering on a training set xt
    nc1 and nc2 are the number of clusters at each level, the final number of
    clusters is nc2. Additional arguments are passed to the Kmeans object.

    Rebalance allocates the number of sub-clusters depending on the number of
    first-level assignment.
    """
    d = xt.shape[1]

    verbose = args.get("verbose", False)

    log = print if verbose else print_nop

    log(f"2-level clustering of {xt.shape} nb 1st level clusters = {nc1} total {nc2}")
    log("perform coarse training")

    km = faiss.Kmeans(
        d, nc1, niter=clustering_niter,
        max_points_per_centroid=2000,
        **args
    )
    km.train(xt)

    iteration_stats = [km.iteration_stats]
    log()

    # coarse centroids
    centroids1 = km.centroids

    log("assigning the training set")
    t0 = time.time()
    _, assign1 = km.assign(xt)
    bc = np.bincount(assign1, minlength=nc1)
    log(f"done in {time.time() - t0:.2f} s. Sizes of clusters {min(bc)}-{max(bc)}")
    o = assign1.argsort()
    del km

    if not rebalance:
        # make sure the sub-clusters sum up to exactly nc2
        cc = np.arange(nc1 + 1) * nc2 // nc1
        all_nc2 = cc[1:] - cc[:-1]
    else:
        bc_sum = np.cumsum(bc)
        all_nc2 = bc_sum * nc2 // bc_sum[-1]
        all_nc2[1:] -= all_nc2[:-1]
        assert sum(all_nc2) == nc2
        log(f"nb 2nd-level centroids {min(all_nc2)}-{max(all_nc2)}")

    # train sub-clusters
    i0 = 0
    c2 = []
    t0 = time.time()
    for c1 in range(nc1):
        nc2 = int(all_nc2[c1])
        log(f"[{time.time() - t0:.2f} s] training sub-cluster {c1}/{nc1} nc2={nc2}\r", end="", flush=True)
        i1 = i0 + bc[c1]
        subset = o[i0:i1]
        assert np.all(assign1[subset] == c1)
        km = faiss.Kmeans(d, nc2, **args)
        xtsub = xt[subset]
        km.train(xtsub)
        iteration_stats.append(km.iteration_stats)
        c2.append(km.centroids)
        del km
        i0 = i1
    log(f"done in {time.time() - t0:.2f} s")
    return np.vstack(c2), iteration_stats


def train_ivf_index_with_2level(index, xt, **args):
    """
    Applies 2-level clustering to an index_ivf embedded in an index.
    """
    # handle PreTransforms
    index = faiss.downcast_index(index)
    if isinstance(index, faiss.IndexPreTransform):
        for i in range(index.chain.size()):
            vt = index.chain.at(i)
            vt.train(xt)
            xt = vt.apply(xt)
        train_ivf_index_with_2level(index.index, xt, **args)
        index.is_trained = True
        return
    assert isinstance(index, faiss.IndexIVF)
    assert index.metric_type == faiss.METRIC_L2
    # now do 2-level clustering
    nc1 = int(np.sqrt(index.nlist))
    print("REBALANCE=", args)

    centroids, _ = two_level_clustering(xt, nc1, index.nlist, **args)
    index.quantizer.train(centroids)
    index.quantizer.add(centroids)
    # finish training
    index.train(xt)


###############################################################################
# K-means implementation in Python
#
# It relies on DatasetAssign, an abstraction of the training vectors that offers
# the minimal set of operations to perform k-means clustering.
###############################################################################


class DatasetAssign:
    """Wrapper for a matrix that offers a function to assign the vectors
    to centroids. All other implementations offer the same interface"""

    def __init__(self, x):
        self.x = np.ascontiguousarray(x, dtype='float32')

    def count(self):
        return self.x.shape[0]

    def dim(self):
        return self.x.shape[1]

    def get_subset(self, indices):
        return self.x[indices]

    def perform_search(self, centroids):
        return faiss.knn(self.x, centroids, 1)

    def assign_to(self, centroids, weights=None):
        D, I = self.perform_search(centroids)

        I = I.ravel()
        D = D.ravel()
        n = len(self.x)
        if weights is None:
            weights = np.ones(n, dtype='float32')
        nc = len(centroids)
        m = scipy.sparse.csc_matrix(
            (weights, I, np.arange(n + 1)),
            shape=(nc, n))
        sum_per_centroid = m * self.x

        return I, D, sum_per_centroid


class DatasetAssignGPU(DatasetAssign):
    """ GPU version of the previous """

    def __init__(self, x, gpu_id, verbose=False):
        DatasetAssign.__init__(self, x)
        index = faiss.IndexFlatL2(x.shape[1])
        if gpu_id >= 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                gpu_id, index)
        else:
            # -1 -> assign to all GPUs
            self.index = faiss.index_cpu_to_all_gpus(index)

    def perform_search(self, centroids):
        self.index.reset()
        self.index.add(centroids)
        return self.index.search(self.x, 1)


def sparse_assign_to_dense(xq, xb, xq_norms=None, xb_norms=None):
    """ assignment function for xq is sparse, xb is dense
    uses a matrix multiplication. The squared norms can be provided if available.
    """
    nq = xq.shape[0]
    nb = xb.shape[0]
    if xb_norms is None:
        xb_norms = (xb ** 2).sum(1)
    if xq_norms is None:
        xq_norms = np.array(xq.power(2).sum(1))
    d2 =  xb_norms - 2 * xq @ xb.T
    I = d2.argmin(axis=1)
    D = d2.ravel()[I + np.arange(nq) * nb] + xq_norms.ravel()
    return D, I


def sparse_assign_to_dense_blocks(
        xq, xb, xq_norms=None, xb_norms=None, qbs=16384, bbs=16384, nt=None):
    """
    decomposes the sparse_assign_to_dense function into blocks to avoid a
    possible memory blow up. Can be run in multithreaded mode, because scipy's
    sparse-dense matrix multiplication is single-threaded.
    """
    nq = xq.shape[0]
    nb = xb.shape[0]
    D = np.empty(nq, dtype="float32")
    D.fill(np.inf)
    I = -np.ones(nq, dtype=int)

    if xb_norms is None:
        xb_norms = (xb ** 2).sum(1)

    def handle_query_block(i):
        xq_block = xq[i : i + qbs]
        Iblock = I[i : i + qbs]
        Dblock = D[i : i + qbs]
        if xq_norms is None:
            xq_norms_block = np.array(xq_block.power(2).sum(1))
        else:
            xq_norms_block = xq_norms[i : i + qbs]
        for j in range(0, nb, bbs):
            Di, Ii = sparse_assign_to_dense(
                xq_block,
                xb[j : j + bbs],
                xq_norms=xq_norms_block,
                xb_norms=xb_norms[j : j + bbs],
            )
            if j == 0:
                Iblock[:] = Ii
                Dblock[:] = Di
            else:
                mask = Di < Dblock
                Iblock[mask] = Ii[mask] + j
                Dblock[mask] = Di[mask]

    if nt == 0 or nt == 1 or nq <= qbs:
        list(map(handle_query_block, range(0, nq, qbs)))
    else:
        pool = ThreadPool(nt)
        pool.map(handle_query_block, range(0, nq, qbs))

    return D, I


class DatasetAssignSparse(DatasetAssign):
    """Wrapper for a matrix that offers a function to assign the vectors
    to centroids. All other implementations offer the same interface"""

    def __init__(self, x):
        assert x.__class__ == scipy.sparse.csr_matrix
        self.x = x
        self.squared_norms = np.array(x.power(2).sum(1))

    def get_subset(self, indices):
        return np.array(self.x[indices].todense())

    def perform_search(self, centroids):
        return sparse_assign_to_dense_blocks(
            self.x, centroids, xq_norms=self.squared_norms)

    def assign_to(self, centroids, weights=None):
        D, I = self.perform_search(centroids)

        I = I.ravel()
        D = D.ravel()
        n = self.x.shape[0]
        if weights is None:
            weights = np.ones(n, dtype='float32')
        nc = len(centroids)
        m = scipy.sparse.csc_matrix(
            (weights, I, np.arange(n + 1)),
            shape=(nc, n))
        sum_per_centroid = np.array((m * self.x).todense())

        return I, D, sum_per_centroid


def imbalance_factor(k, assign):
    assign = np.ascontiguousarray(assign, dtype='int64')
    return faiss.imbalance_factor(len(assign), k, faiss.swig_ptr(assign))


def reassign_centroids(hassign, centroids, rs=None):
    """ reassign centroids when some of them collapse """
    if rs is None:
        rs = np.random
    k, d = centroids.shape
    nsplit = 0
    empty_cents = np.where(hassign == 0)[0]

    if empty_cents.size == 0:
        return 0

    fac = np.ones(d)
    fac[::2] += 1 / 1024.
    fac[1::2] -= 1 / 1024.

    # this is a single pass unless there are more than k/2
    # empty centroids
    while empty_cents.size > 0:
        # choose which centroids to split
        probas = hassign.astype('float') - 1
        probas[probas < 0] = 0
        probas /= probas.sum()
        nnz = (probas > 0).sum()

        nreplace = min(nnz, empty_cents.size)
        cjs = rs.choice(k, size=nreplace, p=probas)

        for ci, cj in zip(empty_cents[:nreplace], cjs):

            c = centroids[cj]
            centroids[ci] = c * fac
            centroids[cj] = c / fac

            hassign[ci] = hassign[cj] // 2
            hassign[cj] -= hassign[ci]
            nsplit += 1

        empty_cents = empty_cents[nreplace:]

    return nsplit


def kmeans(k, data, niter=25, seed=1234, checkpoint=None, verbose=True,
           return_stats=False):
    """Pure python kmeans implementation. Follows the Faiss C++ version
    quite closely, but takes a DatasetAssign instead of a training data
    matrix. Also redo is not implemented. """
    n, d = data.count(), data.dim()

    log = print if verbose else print_nop

    log(("Clustering %d points in %dD to %d clusters, " +
            "%d iterations seed %d") % (n, d, k, niter, seed))

    rs = np.random.RandomState(seed)
    print("preproc...")
    t0 = time.time()
    # initialization
    perm = rs.choice(n, size=k, replace=False)
    centroids = data.get_subset(perm)

    iteration_stats = []

    log("  done")
    t_search_tot = 0
    obj = []
    for i in range(niter):
        t0s = time.time()

        log('assigning', end='\r', flush=True)
        assign, D, sums = data.assign_to(centroids)

        log('compute centroids', end='\r', flush=True)

        t_search_tot += time.time() - t0s;

        err = D.sum()
        obj.append(err)

        hassign = np.bincount(assign, minlength=k)

        fac = hassign.reshape(-1, 1).astype('float32')
        fac[fac == 0] = 1 # quiet warning

        centroids = sums / fac

        nsplit = reassign_centroids(hassign, centroids, rs)

        s = {
            "obj": err,
            "time": (time.time() - t0),
            "time_search": t_search_tot,
            "imbalance_factor": imbalance_factor (k, assign),
            "nsplit": nsplit
        }

        log(("  Iteration %d (%.2f s, search %.2f s): "
             "objective=%g imbalance=%.3f nsplit=%d") % (
                   i, s["time"], s["time_search"],
                   err, s["imbalance_factor"],
                   nsplit)
        )
        iteration_stats.append(s)

        if checkpoint is not None:
            log('storing centroids in', checkpoint)
            np.save(checkpoint, centroids)

    if return_stats:
        return centroids, iteration_stats
    else:
        return centroids
