#! /usr/bin/env python3

import faiss
import time

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
