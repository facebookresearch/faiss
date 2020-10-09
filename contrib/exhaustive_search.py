#! /usr/bin/env python3

import faiss
import time
import numpy as np

import logging

LOG = logging.getLogger(__name__)

def knn_ground_truth(xq, db_iterator, k):
    """Computes the exact KNN search results for a dataset that possibly
    does not fit in RAM but for whihch we have an iterator that
    returns it block by block.
    """
    t0 = time.time()
    nq, d = xq.shape
    rh = faiss.ResultHeap(nq, k)

    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus():
        LOG.info('running on %d GPUs' % faiss.get_num_gpus())
        index = faiss.index_cpu_to_all_gpus(index)

    # compute ground-truth by blocks of bs, and add to heaps
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

def knn(xq, xb, k, distance_type=faiss.METRIC_L2):
    """ wrapper around the faiss knn functions without index """
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2

    I = np.empty((nq, k), dtype='int64')
    D = np.empty((nq, k), dtype='float32')

    if distance_type == faiss.METRIC_L2:
        heaps = faiss.float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(D)
        heaps.ids = faiss.swig_ptr(I)
        faiss.knn_L2sqr(
            faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            d, nq, nb, heaps
        )
    elif distance_type == faiss.METRIC_INNER_PRODUCT:
        heaps = faiss.float_minheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(D)
        heaps.ids = faiss.swig_ptr(I)
        faiss.knn_inner_product(
            faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            d, nq, nb, heaps
        )
    return D, I


def knn_gpu(res, xb, xq, k, D=None, I=None, metric=faiss.METRIC_L2):
    """Brute-force k-nearest neighbor on the GPU using CPU-resident numpy arrays
    Supports float16 arrays and Fortran-order arrays.
    """
    if xb.ndim != 2 or xq.ndim != 2:
        raise TypeError('xb and xq must be matrices')

    nb, d = xb.shape
    nq, d2 = xq.shape
    if d != d2:
        raise TypeError('xq not the same dimension as xb')

    if xb.flags.c_contiguous:
        xb_row_major = True
    elif xb.flags.f_contiguous:
        xb = xb.T
        xb_row_major = False
    else:
        raise TypeError('xb must be either C or Fortran contiguous')

    if xq.flags.c_contiguous:
        xq_row_major = True
    elif xq.flags.f_contiguous:
        xq = xq.T
        xq_row_major = False
    else:
        raise TypeError('xq must be either C or Fortran contiguous')

    if xb.dtype == np.float32 and xq.dtype == np.float32:
        xb_xq_type = faiss.DistanceDataType_F32
    elif xb.dtype == np.float16 and xq.dtype == np.float16:
        xb_xq_type = faiss.DistanceDataType_F16
    else:
        raise TypeError('xb and xq must both be np.float32 or np.float16')

    if D is None:
        D = np.empty((nq, k), dtype=np.float32)
    else:
        assert D.shape == (nq, k)
        assert D.dtype == np.float32

    if I is None:
        I = np.empty((nq, k), dtype=np.int64)
        indices_type = faiss.IndicesDataType_I64
    else:
        assert I.shape == (nq, k)
        if I.dtype == np.int64:
            indices_type = faiss.IndicesDataType_I64
        elif I.dtype == np.int32:
            indices_type = faiss.IndicesDataType_I32
        else:
            raise TypeError('I must be either np.int64 or np.int32')

    print('row major', xb_row_major, xq_row_major)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = k
    args.dims = d
    args.vectors = faiss.swig_ptr(xb)
    args.vectorType = xb_xq_type
    args.vectorsRowMajor = xb_row_major
    args.numVectors = nb
    args.queries = faiss.swig_ptr(xq)
    args.queryType = xb_xq_type
    args.queriesRowMajor = xq_row_major
    args.numQueries = nq
    args.outDistances = faiss.swig_ptr(D)
    args.outIndices = faiss.swig_ptr(I)
    args.outIndicesType = indices_type
    faiss.bfKnn(res, args)

    return D, I
