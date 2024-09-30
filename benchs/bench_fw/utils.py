# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
from enum import Enum
from multiprocessing.pool import ThreadPool
from time import perf_counter

import faiss  # @manual=//faiss/python:pyfaiss
import numpy as np

from faiss.contrib.evaluation import (  # @manual=//faiss/contrib:faiss_contrib
    OperatingPoints,
)

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
    xq: np.ndarray,
    xb: np.ndarray,
    I: np.ndarray,
    metric,
):
    """Recompute distances between xq[i] and xb[I[i, :]]"""
    nq, k = I.shape
    xq = np.ascontiguousarray(xq, dtype="float32")
    nq2, d = xq.shape
    xb = np.ascontiguousarray(xb, dtype="float32")
    nb, d2 = xb.shape
    I = np.ascontiguousarray(I, dtype="int64")
    assert nq2 == nq
    assert d2 == d
    D = np.empty(I.shape, dtype="float32")
    D[:] = np.inf
    if metric == faiss.METRIC_L2:
        faiss.fvec_L2sqr_by_idx(
            faiss.swig_ptr(D),
            faiss.swig_ptr(xq),
            faiss.swig_ptr(xb),
            faiss.swig_ptr(I),
            d,
            nq,
            k,
        )
    else:
        faiss.fvec_inner_products_by_idx(
            faiss.swig_ptr(D),
            faiss.swig_ptr(xq),
            faiss.swig_ptr(xb),
            faiss.swig_ptr(I),
            d,
            nq,
            k,
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
    return [l for l in open("/proc/cpuinfo", "r") if "model name" in l][0][
        13:
    ].strip()


def dict_merge(target, source):
    for k, v in source.items():
        if isinstance(v, dict) and k in target:
            dict_merge(target[k], v)
        else:
            target[k] = v


class Cost:
    def __init__(self, values):
        self.values = values

    def __le__(self, other):
        return all(
            v1 <= v2 for v1, v2 in zip(self.values, other.values, strict=True)
        )

    def __lt__(self, other):
        return all(
            v1 < v2 for v1, v2 in zip(self.values, other.values, strict=True)
        )


class ParetoMode(Enum):
    DISABLE = 1  # no Pareto filtering
    INDEX = 2  # index-local optima
    GLOBAL = 3  # global optima


class ParetoMetric(Enum):
    TIME = 0  # time vs accuracy
    SPACE = 1  # space vs accuracy
    TIME_SPACE = 2  # (time, space) vs accuracy


def range_search_recall_at_precision(experiment, precision):
    return round(
        max(
            r
            for r, p in zip(
                experiment["range_search_pr"]["recall"],
                experiment["range_search_pr"]["precision"],
            )
            if p > precision
        ),
        6,
    )


def filter_results(
    results,
    evaluation,
    accuracy_metric,  # str or func
    time_metric=None,  # func or None -> use default
    space_metric=None,  # func or None -> use default
    min_accuracy=0,
    max_space=0,
    max_time=0,
    scaling_factor=1.0,
    name_filter=None,  # func
    pareto_mode=ParetoMode.DISABLE,
    pareto_metric=ParetoMetric.TIME,
):
    if isinstance(accuracy_metric, str):
        accuracy_key = accuracy_metric
        accuracy_metric = lambda v: v[accuracy_key]

    if time_metric is None:
        time_metric = lambda v: v["time"] * scaling_factor + (
            v["quantizer"]["time"] if "quantizer" in v else 0
        )

    if space_metric is None:
        space_metric = lambda v: results["indices"][v["codec"]]["code_size"]

    fe = []
    ops = {}
    if pareto_mode == ParetoMode.GLOBAL:
        op = OperatingPoints()
        ops["global"] = op
    for k, v in results["experiments"].items():
        if f".{evaluation}" in k:
            accuracy = accuracy_metric(v)
            if min_accuracy > 0 and accuracy < min_accuracy:
                continue
            space = space_metric(v)
            if space is None:
                space = 0
            if max_space > 0 and space > max_space:
                continue
            time = time_metric(v)
            if max_time > 0 and time > max_time:
                continue
            idx_name = v["index"] + (
                "snap"
                if "search_params" in v and v["search_params"]["snap"] == 1
                else ""
            )
            if name_filter is not None and not name_filter(idx_name):
                continue
            experiment = (accuracy, space, time, k, v)
            if pareto_mode == ParetoMode.DISABLE:
                fe.append(experiment)
                continue
            if pareto_mode == ParetoMode.INDEX:
                if idx_name not in ops:
                    ops[idx_name] = OperatingPoints()
                op = ops[idx_name]
            if pareto_metric == ParetoMetric.TIME:
                op.add_operating_point(experiment, accuracy, time)
            elif pareto_metric == ParetoMetric.SPACE:
                op.add_operating_point(experiment, accuracy, space)
            else:
                op.add_operating_point(
                    experiment, accuracy, Cost([time, space])
                )

    if ops:
        for op in ops.values():
            for v, _, _ in op.operating_points:
                fe.append(v)

    fe.sort()
    return fe
