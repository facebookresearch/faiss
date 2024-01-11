# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from time import perf_counter
import logging
from multiprocessing.pool import ThreadPool
import numpy as np
import faiss  # @manual=//faiss/python:pyfaiss_gpu
import functools

logger = logging.getLogger(__name__)

def timer(name, func, once=False) -> float:
    logger.info(f"Measuring {name}")
    t1 = perf_counter()
    res = func()
    t2 = perf_counter()
    t = t2 - t1
    repeat = 1
    if not once and t < 1.0:
        repeat = int(2.0 // t)
        logger.info(
            f"Time for {name}: {t:.3f} seconds, repeating {repeat} times"
        )
        t1 = perf_counter()
        for _ in range(repeat):
            res = func()
        t2 = perf_counter()
        t = (t2 - t1) / repeat
    logger.info(f"Time for {name}: {t:.3f} seconds")
    return res, t, repeat


def refine_distances_knn(
    xq: np.ndarray, xb: np.ndarray, I: np.ndarray, metric,
):
    """ Recompute distances between xq[i] and xb[I[i, :]] """
    nq, k = I.shape
    xq = np.ascontiguousarray(xq, dtype='float32')
    nq2, d = xq.shape
    xb = np.ascontiguousarray(xb, dtype='float32')
    nb, d2 = xb.shape
    I = np.ascontiguousarray(I, dtype='int64')
    assert nq2 == nq
    assert d2 == d
    D = np.empty(I.shape, dtype='float32')
    D[:] = np.inf
    if metric == faiss.METRIC_L2:
        faiss.fvec_L2sqr_by_idx(
            faiss.swig_ptr(D), faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            faiss.swig_ptr(I), d, nq, k
        )
    else:
        faiss.fvec_inner_products_by_idx(
            faiss.swig_ptr(D), faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            faiss.swig_ptr(I), d, nq, k
        )
    return D


def refine_distances_range(
    lims: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    xq: np.ndarray,
    xb: np.ndarray,
    metric,
):
    with ThreadPool(32) as pool:
        R = pool.map(
            lambda i: (
                np.sum(np.square(xq[i] - xb[I[lims[i] : lims[i + 1]]]), axis=1)
                if metric == faiss.METRIC_L2
                else np.tensordot(
                    xq[i], xb[I[lims[i] : lims[i + 1]]], axes=(0, 1)
                )
            )
            if lims[i + 1] > lims[i]
            else [],
            range(len(lims) - 1),
        )
    return np.hstack(R)


def distance_ratio_measure(I, R, D_GT, metric):
    sum_of_R = np.sum(np.where(I >= 0, R, 0))
    sum_of_D_GT = np.sum(np.where(I >= 0, D_GT, 0))
    if metric == faiss.METRIC_INNER_PRODUCT:
        return (sum_of_R / sum_of_D_GT).item()
    elif metric == faiss.METRIC_L2:
        return (sum_of_D_GT / sum_of_R).item()
    else:
        raise RuntimeError(f"unknown metric {metric}")


@functools.cache
def get_cpu_info():
    return [l for l in open("/proc/cpuinfo", "r") if "model name" in l][0][13:].strip()

def dict_merge(target, source):
    for k, v in source.items():
        if isinstance(v, dict) and k in target:
            dict_merge(target[k], v)
        else:
            target[k] = v
