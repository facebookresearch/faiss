# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import time
import numpy as np

import logging

LOG = logging.getLogger(__name__)

def knn_ground_truth(xq, db_iterator, k, metric_type=faiss.METRIC_L2):
    """Computes the exact KNN search results for a dataset that possibly
    does not fit in RAM but for which we have an iterator that
    returns it block by block.
    """
    LOG.info("knn_ground_truth queries size %s k=%d" % (xq.shape, k))
    t0 = time.time()
    nq, d = xq.shape
    rh = faiss.ResultHeap(nq, k)

    index = faiss.IndexFlat(d, metric_type)
    if faiss.get_num_gpus():
        LOG.info('running on %d GPUs' % faiss.get_num_gpus())
        index = faiss.index_cpu_to_all_gpus(index)

    # compute ground-truth by blocks, and add to heaps
    i0 = 0
    for xbi in db_iterator:
        ni = xbi.shape[0]
        index.add(xbi)
        D, I = index.search(xq, k)
        I += i0
        rh.add_result(D, I)
        index.reset()
        i0 += ni
        LOG.info("%d db elements, %.3f s" % (i0, time.time() - t0))

    rh.finalize()
    LOG.info("GT time: %.3f s (%d vectors)" % (time.time() - t0, i0))

    return rh.D, rh.I

# knn function used to be here
knn = faiss.knn




def range_search_gpu(xq, r2, index_gpu, index_cpu):
    """GPU does not support range search, so we emulate it with
    knn search + fallback to CPU index.

    The index_cpu can either be a CPU index or a numpy table that will
    be used to construct a Flat index if needed.
    """
    nq, d = xq.shape
    LOG.debug("GPU search %d queries" % nq)
    k = min(index_gpu.ntotal, 1024)
    D, I = index_gpu.search(xq, k)
    if index_gpu.metric_type == faiss.METRIC_L2:
        mask = D[:, k - 1] < r2
    else:
        mask = D[:, k - 1] > r2
    if mask.sum() > 0:
        LOG.debug("CPU search remain %d" % mask.sum())
        if isinstance(index_cpu, np.ndarray):
            # then it in fact an array that we have to make flat
            xb = index_cpu
            index_cpu = faiss.IndexFlat(d, index_gpu.metric_type)
            index_cpu.add(xb)
        lim_remain, D_remain, I_remain = index_cpu.range_search(xq[mask], r2)
    LOG.debug("combine")
    D_res, I_res = [], []
    nr = 0
    for i in range(nq):
        if not mask[i]:
            if index_gpu.metric_type == faiss.METRIC_L2:
                nv = (D[i, :] < r2).sum()
            else:
                nv = (D[i, :] > r2).sum()
            D_res.append(D[i, :nv])
            I_res.append(I[i, :nv])
        else:
            l0, l1 = lim_remain[nr], lim_remain[nr + 1]
            D_res.append(D_remain[l0:l1])
            I_res.append(I_remain[l0:l1])
            nr += 1
    lims = np.cumsum([0] + [len(di) for di in D_res])
    return lims, np.hstack(D_res), np.hstack(I_res)


def range_ground_truth(xq, db_iterator, threshold, metric_type=faiss.METRIC_L2,
                       shard=False, ngpu=-1):
    """Computes the range-search search results for a dataset that possibly
    does not fit in RAM but for which we have an iterator that
    returns it block by block.
    """
    nq, d = xq.shape
    t0 = time.time()
    xq = np.ascontiguousarray(xq, dtype='float32')

    index = faiss.IndexFlat(d, metric_type)
    if ngpu == -1:
        ngpu = faiss.get_num_gpus()
    if ngpu:
        LOG.info('running on %d GPUs' % ngpu)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)

    # compute ground-truth by blocks
    i0 = 0
    D = [[] for _i in range(nq)]
    I = [[] for _i in range(nq)]
    for xbi in db_iterator:
        ni = xbi.shape[0]
        if ngpu > 0:
            index_gpu.add(xbi)
            lims_i, Di, Ii = range_search_gpu(xq, threshold, index_gpu, xbi)
            index_gpu.reset()
        else:
            index.add(xbi)
            lims_i, Di, Ii = index.range_search(xq, threshold)
            index.reset()
        Ii += i0
        for j in range(nq):
            l0, l1 = lims_i[j], lims_i[j + 1]
            if l1 > l0:
                D[j].append(Di[l0:l1])
                I[j].append(Ii[l0:l1])
        i0 += ni
        LOG.info("%d db elements, %.3f s" % (i0, time.time() - t0))

    empty_I = np.zeros(0, dtype='int64')
    empty_D = np.zeros(0, dtype='float32')
    # import pdb; pdb.set_trace()
    D = [(np.hstack(i) if i != [] else empty_D) for i in D]
    I = [(np.hstack(i) if i != [] else empty_I) for i in I]
    sizes = [len(i) for i in I]
    assert len(sizes) == nq
    lims = np.zeros(nq + 1, dtype="uint64")
    lims[1:] = np.cumsum(sizes)
    return lims, np.hstack(D), np.hstack(I)


def threshold_radius_nres(nres, dis, ids, thresh):
    """ select a set of results """
    mask = dis < thresh
    new_nres = np.zeros_like(nres)
    o = 0
    for i, nr in enumerate(nres):
        nr = int(nr)   # avoid issues with int64 + uint64
        new_nres[i] = mask[o : o + nr].sum()
        o += nr
    return new_nres, dis[mask], ids[mask]


def threshold_radius(lims, dis, ids, thresh):
    """ restrict range-search results to those below a given radius """
    mask = dis < thresh
    new_lims = np.zeros_like(lims)
    n = len(lims) - 1
    for i in range(n):
        l0, l1 = lims[i], lims[i + 1]
        new_lims[i + 1] = new_lims[i] + mask[l0:l1].sum()
    return new_lims, dis[mask], ids[mask]


def apply_maxres(res_batches, target_nres):
    """find radius that reduces number of results to target_nres, and
    applies it in-place to the result batches used in range_search_max_results"""
    alldis = np.hstack([dis for _, dis, _ in res_batches])
    alldis.partition(target_nres)
    radius = alldis[target_nres]

    if alldis.dtype == 'float32':
        radius = float(radius)
    else:
        radius = int(radius)
    LOG.debug('   setting radius to %s' % radius)
    totres = 0
    for i, (nres, dis, ids) in enumerate(res_batches):
        nres, dis, ids = threshold_radius_nres(nres, dis, ids, radius)
        totres += len(dis)
        res_batches[i] = nres, dis, ids
    LOG.debug('   updated previous results, new nb results %d' % totres)
    return radius, totres


def range_search_max_results(index, query_iterator, radius,
                             max_results=None, min_results=None,
                             shard=False, ngpu=0):
    """Performs a range search with many queries (given by an iterator)
    and adjusts the threshold on-the-fly so that the total results
    table does not grow larger than max_results.

    If ngpu != 0, the function moves the index to this many GPUs to
    speed up search.
    """

    if max_results is not None:
        if min_results is None:
            min_results = int(0.8 * max_results)

    if ngpu == -1:
        ngpu = faiss.get_num_gpus()

    if ngpu:
        LOG.info('running on %d GPUs' % ngpu)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)

    t_start = time.time()
    t_search = t_post_process = 0
    qtot = totres = raw_totres = 0
    res_batches = []

    for xqi in query_iterator:
        t0 = time.time()
        if ngpu > 0:
            lims_i, Di, Ii = range_search_gpu(xqi, radius, index_gpu, index)
        else:
            lims_i, Di, Ii = index.range_search(xqi, radius)

        nres_i = lims_i[1:] - lims_i[:-1]
        raw_totres += len(Di)
        qtot += len(xqi)

        t1 = time.time()
        if xqi.dtype != np.float32:
            # for binary indexes
            # weird Faiss quirk that returns floats for Hamming distances
            Di = Di.astype('int16')

        totres += len(Di)
        res_batches.append((nres_i, Di, Ii))

        if max_results is not None and totres > max_results:
            LOG.info('too many results %d > %d, scaling back radius' %
                     (totres, max_results))
            radius, totres = apply_maxres(res_batches, min_results)
        t2 = time.time()
        t_search += t1 - t0
        t_post_process += t2 - t1
        LOG.debug('   [%.3f s] %d queries done, %d results' % (
            time.time() - t_start, qtot, totres))

    LOG.info('   search done in %.3f s + %.3f s, total %d results, end threshold %g' % (
        t_search, t_post_process, totres, radius))

    nres = np.hstack([nres_i for nres_i, dis_i, ids_i in res_batches])
    dis = np.hstack([dis_i for nres_i, dis_i, ids_i in res_batches])
    ids = np.hstack([ids_i for nres_i, dis_i, ids_i in res_batches])

    lims = np.zeros(len(nres) + 1, dtype='uint64')
    lims[1:] = np.cumsum(nres)

    return radius, lims, dis, ids
