# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import logging
import time
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from statistics import median, mean
from typing import Any, List, Optional
from .descriptors import DatasetDescriptor, IndexDescriptor

import faiss  # @manual=//faiss/python:pyfaiss_gpu
from faiss.contrib.evaluation import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    knn_intersection_measure,
    OperatingPointsWithRanges,
)
from faiss.contrib.ivf_tools import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    add_preassigned,
)

import numpy as np

from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def refine_distances_knn(
    D: np.ndarray, I: np.ndarray, xq: np.ndarray, xb: np.ndarray, metric
):
    return np.where(
        I >= 0,
        np.square(np.linalg.norm(xq[:, None] - xb[I], axis=2))
        if metric == faiss.METRIC_L2
        else np.einsum("qd,qkd->qk", xq, xb[I]),
        D,
    )


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
                np.sum(np.square(xq[i] - xb[I[lims[i]:lims[i + 1]]]), axis=1)
                if metric == faiss.METRIC_L2
                else np.tensordot(
                    xq[i], xb[I[lims[i]:lims[i + 1]]], axes=(0, 1)
                )
            )
            if lims[i + 1] > lims[i]
            else [],
            range(len(lims) - 1),
        )
    return np.hstack(R)


def range_search_pr_curve(
    dist_ann: np.ndarray, metric_score: np.ndarray, gt_rsm: float
):
    assert dist_ann.shape == metric_score.shape
    assert dist_ann.ndim == 1
    sort_by_dist_ann = dist_ann.argsort()
    dist_ann = dist_ann[sort_by_dist_ann]
    metric_score = metric_score[sort_by_dist_ann]
    cum_score = np.cumsum(metric_score)
    precision = cum_score / np.arange(1, len(cum_score) + 1)
    recall = cum_score / gt_rsm
    unique_key = np.round(precision * 100) * 100 + np.round(recall * 100)
    tbl = np.vstack(
        [dist_ann, metric_score, cum_score, precision, recall, unique_key]
    )
    group_by_dist_max_cum_score = np.empty(len(dist_ann), np.bool)
    group_by_dist_max_cum_score[-1] = True
    group_by_dist_max_cum_score[:-1] = dist_ann[1:] != dist_ann[:-1]
    tbl = tbl[:, group_by_dist_max_cum_score]
    _, unique_key_idx = np.unique(tbl[5], return_index=True)
    dist_ann, metric_score, cum_score, precision, recall, unique_key = tbl[
        :, np.sort(unique_key_idx)
    ].tolist()
    return {
        "dist_ann": dist_ann,
        "metric_score_sample": metric_score,
        "cum_score": cum_score,
        "precision": precision,
        "recall": recall,
        "unique_key": unique_key,
    }


def set_index_parameter(index, name, val):
    index = faiss.downcast_index(index)

    if isinstance(index, faiss.IndexPreTransform):
        set_index_parameter(index.index, name, val)
    elif name.startswith("quantizer_"):
        index_ivf = faiss.extract_index_ivf(index)
        set_index_parameter(
            index_ivf.quantizer, name[name.find("_") + 1:], val
        )
    elif name == "efSearch":
        index.hnsw.efSearch
        index.hnsw.efSearch = int(val)
    elif name == "nprobe":
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe
        index_ivf.nprobe = int(val)
    elif name == "noop":
        pass
    else:
        raise RuntimeError(f"could not set param {name} on {index}")


def optimizer(codec, search, cost_metric, perf_metric):
    op = OperatingPointsWithRanges()
    op.add_range("noop", [0])
    codec_ivf = faiss.try_extract_index_ivf(codec)
    if codec_ivf is not None:
        op.add_range(
            "nprobe",
            [2**i for i in range(12) if 2**i < codec_ivf.nlist * 0.1],
        )

    totex = op.num_experiments()
    rs = np.random.RandomState(123)
    if totex > 1:
        experiments = rs.permutation(totex - 2) + 1
        experiments = [0, totex - 1] + list(experiments)
    else:
        experiments = [0]

    print(f"total nb experiments {totex}, running {len(experiments)}")

    for cno in experiments:
        key = op.cno_to_key(cno)
        parameters = op.get_parameters(key)

        (max_perf, min_cost) = op.predict_bounds(key)
        if not op.is_pareto_optimal(max_perf, min_cost):
            logger.info(
                f"{cno=:4d} {str(parameters):50}: SKIP, {max_perf=:.3f} {min_cost=:.3f}",
            )
            continue

        logger.info(f"{cno=:4d} {str(parameters):50}: RUN")
        cost, perf = search(
            parameters,
            cost_metric,
            perf_metric,
        )
        logger.info(
            f"{cno=:4d} {str(parameters):50}: DONE, {cost=:.3f} {perf=:.3f}"
        )
        op.add_operating_point(key, perf, cost)


def distance_ratio_measure(R, D_GT, metric):
    if metric == faiss.METRIC_INNER_PRODUCT:
        return (np.sum(R) / np.sum(D_GT)).item()
    elif metric == faiss.METRIC_L2:
        return (np.sum(D_GT) / np.sum(R)).item()
    else:
        raise RuntimeError(f"unknown metric {metric}")


# range_metric possible values:
#
# radius
#    [0..radius) -> 1
#    [radius..inf) -> 0
#
# [[radius1, score1], ...]
#    [0..radius1) -> score1
#    [radius1..radius2) -> score2
#
# [[radius1_from, radius1_to, score1], ...]
#    [radius1_from, radius1_to) -> score1,
#    [radius2_from, radius2_to) -> score2
def get_range_search_metric_function(range_metric, D, R):
    if D is not None:
        assert R is not None
        assert D.shape == R.shape
    if isinstance(range_metric, list):
        aradius, ascore = [], []
        radius_to = 0
        for rsd in range_metric:
            assert isinstance(rsd, list)
            if len(rsd) == 3:
                radius_from, radius_to, score = rsd
            elif len(rsd) == 2:
                radius_from = radius_to
                radius_to, score = rsd
            else:
                raise AssertionError(f"invalid range definition {rsd}")
            # radius_from and radius_to are compressed distances,
            # we need to convert them to real embedding distances.
            if D is not None:
                sample_idxs = np.argwhere((D <= radius_to) & (D > radius_from))
                assert len(sample_idxs) > 0
                real_radius = np.mean(R[sample_idxs]).item()
            else:
                real_radius = mean([radius_from, radius_to])
            logger.info(
                f"range_search_metric_function {radius_from=} {radius_to=} {real_radius=} {score=}"
            )
            aradius.append(real_radius)
            ascore.append(score)

        def sigmoid(x, a, b, c):
            return a / (1 + np.exp(b * x - c))

        cutoff = max(aradius)
        popt, _ = curve_fit(sigmoid, aradius, ascore, [1, 5, 5])

        for r in np.arange(0, cutoff + 0.05, 0.05):
            logger.info(
                f"range_search_metric_function {r=} {sigmoid(r, *popt)=}"
            )

        assert isinstance(cutoff, float)
        return (
            cutoff,
            lambda x: np.where(x < cutoff, sigmoid(x, *popt), 0),
            popt.tolist(),
        )
    else:
        # Assuming that the range_metric is a float,
        # so the range is [0..range_metric).
        # D is the result of a range_search with a radius of range_metric,
        # but both range_metric and D may be compressed distances.
        # We approximate the real embedding distance as max(R).
        if R is not None:
            real_range = np.max(R).item()
        else:
            real_range = range_metric
        logger.info(
            f"range_search_metric_function {range_metric=} {real_range=}"
        )
        assert isinstance(real_range, float)
        return real_range * 2, lambda x: np.where(x < real_range, 1, 0), []


@dataclass
class Benchmark:
    training_vectors: Optional[DatasetDescriptor] = None
    db_vectors: Optional[DatasetDescriptor] = None
    query_vectors: Optional[DatasetDescriptor] = None
    index_descs: Optional[List[IndexDescriptor]] = None
    range_ref_index_desc: Optional[str] = None
    k: Optional[int] = None
    distance_metric: str = "METRIC_L2"

    def __post_init__(self):
        if self.distance_metric == "METRIC_INNER_PRODUCT":
            self.distance_metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.distance_metric == "METRIC_L2":
            self.distance_metric_type = faiss.METRIC_L2
        else:
            raise ValueError
        self.cached_index_key = None

    def set_io(self, benchmark_io):
        self.io = benchmark_io
        self.io.distance_metric = self.distance_metric
        self.io.distance_metric_type = self.distance_metric_type

    def get_index_desc(self, factory: str) -> Optional[IndexDescriptor]:
        for desc in self.index_descs:
            if desc.factory == factory:
                return desc
        return None

    def get_index(self, index_desc: IndexDescriptor):
        if self.cached_index_key != index_desc.factory:
            xb = self.io.get_dataset(self.db_vectors)
            index = faiss.clone_index(
                self.io.get_codec(index_desc, xb.shape[1])
            )
            assert index.ntotal == 0
            logger.info("Adding vectors to index")
            index_ivf = faiss.try_extract_index_ivf(index)
            if index_ivf is not None:
                QD, QI, _, QP = self.knn_search(
                    index_desc,
                    parameters=None,
                    db_vectors=None,
                    query_vectors=self.db_vectors,
                    k=1,
                    index=index_ivf.quantizer,
                    level=1,
                )
                print(f"{QI.ravel().shape=}")
                add_preassigned(index_ivf, xb, QI.ravel())
            else:
                index.add(xb)
            assert index.ntotal == xb.shape[0]
            logger.info("Added vectors to index")
            self.cached_index_key = index_desc.factory
            self.cached_index = index
        return self.cached_index

    def range_search_reference(self, index_desc, range_metric):
        logger.info("range_search_reference: begin")
        if isinstance(range_metric, list):
            assert len(range_metric) > 0
            ri = len(range_metric[0]) - 1
            m_radius = (
                max(range_metric, key=itemgetter(ri))[ri]
                if self.distance_metric_type == faiss.METRIC_L2
                else min(range_metric, key=itemgetter(ri))[ri]
            )
        else:
            m_radius = range_metric

        lims, D, I, R, P = self.range_search(
            index_desc,
            index_desc.parameters,
            radius=m_radius,
        )
        flat = index_desc.factory == "Flat"
        (
            gt_radius,
            range_search_metric_function,
            coefficients,
        ) = get_range_search_metric_function(
            range_metric,
            D if not flat else None,
            R if not flat else None,
        )
        logger.info("range_search_reference: end")
        return gt_radius, range_search_metric_function, coefficients

    def estimate_range(self, index_desc, parameters, range_scoring_radius):
        D, I, R, P = self.knn_search(
            index_desc, parameters, self.db_vectors, self.query_vectors
        )
        samples = []
        for i, j in np.argwhere(R < range_scoring_radius):
            samples.append((R[i, j].item(), D[i, j].item()))
        samples.sort(key=itemgetter(0))
        return median(r for _, r in samples[-3:])

    def range_search(
        self,
        index_desc: IndexDescriptor,
        parameters: Optional[dict[str, int]],
        radius: Optional[float] = None,
        gt_radius: Optional[float] = None,
    ):
        logger.info("range_search: begin")
        flat = index_desc.factory == "Flat"
        if radius is None:
            assert gt_radius is not None
            radius = (
                gt_radius
                if flat
                else self.estimate_range(index_desc, parameters, gt_radius)
            )
        logger.info(f"Radius={radius}")
        filename = self.io.get_filename_range_search(
            factory=index_desc.factory,
            parameters=parameters,
            level=0,
            db_vectors=self.db_vectors,
            query_vectors=self.query_vectors,
            r=radius,
        )
        if self.io.file_exist(filename):
            logger.info(f"Using cached results for {index_desc.factory}")
            lims, D, I, R, P = self.io.read_file(
                filename, ["lims", "D", "I", "R", "P"]
            )
        else:
            xq = self.io.get_dataset(self.query_vectors)
            index = self.get_index(index_desc)
            if parameters:
                for name, val in parameters.items():
                    set_index_parameter(index, name, val)

            index_ivf = faiss.try_extract_index_ivf(index)
            if index_ivf is not None:
                QD, QI, _, QP = self.knn_search(
                    index_desc,
                    parameters=None,
                    db_vectors=None,
                    query_vectors=self.query_vectors,
                    k=index.nprobe,
                    index=index_ivf.quantizer,
                    level=1,
                )
                # QD = QD[:, :index.nprobe]
                # QI = QI[:, :index.nprobe]
                logger.info("Timing range_search_preassigned")
                faiss.cvar.indexIVF_stats.reset()
                t0 = time.time()
                lims, D, I = index.range_search_preassigned(xq, radius, QI, QD)
                t = time.time() - t0
            else:
                logger.info("Timing range_search")
                t0 = time.time()
                lims, D, I = index.range_search(xq, radius)
                t = time.time() - t0
            if flat:
                R = D
            else:
                xb = self.io.get_dataset(self.db_vectors)
                R = refine_distances_range(
                    lims, D, I, xq, xb, self.distance_metric_type
                )
            P = {
                "time": t,
                "radius": radius,
                "count": lims[-1].item(),
                "parameters": parameters,
                "index": index_desc.factory,
            }
            if index_ivf is not None:
                stats = faiss.cvar.indexIVF_stats
                P |= {
                    "quantizer": QP,
                    "nq": stats.nq,
                    "nlist": stats.nlist,
                    "ndis": stats.ndis,
                    "nheap_updates": stats.nheap_updates,
                    "quantization_time": stats.quantization_time,
                    "search_time": stats.search_time,
                }
            self.io.write_file(
                filename, ["lims", "D", "I", "R", "P"], [lims, D, I, R, P]
            )
        logger.info("range_seach: end")
        return lims, D, I, R, P

    def range_ground_truth(self, gt_radius, range_search_metric_function):
        logger.info("range_ground_truth: begin")
        flat_desc = self.get_index_desc("Flat")
        lims, D, I, R, P = self.range_search(
            flat_desc,
            flat_desc.parameters,
            radius=gt_radius,
        )
        gt_rsm = np.sum(range_search_metric_function(R)).item()
        logger.info("range_ground_truth: end")
        return gt_rsm

    def range_search_benchmark(
        self,
        results: dict[str, Any],
        index_desc: IndexDescriptor,
        metric_key: str,
        gt_radius: float,
        range_search_metric_function,
        gt_rsm: float,
    ):
        logger.info(f"range_search_benchmark: begin {index_desc.factory=}")
        xq = self.io.get_dataset(self.query_vectors)
        (nq, d) = xq.shape
        logger.info(
            f"Searching {index_desc.factory} with {nq} vectors of dimension {d}"
        )
        codec = self.io.get_codec(index_desc, d)
        faiss.omp_set_num_threads(16)

        def experiment(parameters, cost_metric, perf_metric):
            nonlocal results
            key = self.io.get_filename_evaluation_name(
                factory=index_desc.factory,
                parameters=parameters,
                level=0,
                db_vectors=self.db_vectors,
                query_vectors=self.query_vectors,
                evaluation_name=metric_key,
            )
            if key in results["experiments"]:
                metrics = results["experiments"][key]
            else:
                lims, D, I, R, P = self.range_search(
                    index_desc, parameters, gt_radius=gt_radius
                )
                range_search_metric = range_search_metric_function(R)
                range_search_pr = range_search_pr_curve(
                    D, range_search_metric, gt_rsm
                )
                range_score_sum = np.sum(range_search_metric).item()
                metrics = P | {
                    "range_score_sum": range_score_sum,
                    "range_score_max_recall": range_score_sum / gt_rsm,
                    "range_search_pr": range_search_pr,
                }
                results["experiments"][key] = metrics
            return metrics[cost_metric], metrics[perf_metric]

        for cost_metric in ["time"]:
            for perf_metric in ["range_score_max_recall"]:
                optimizer(
                    codec,
                    experiment,
                    cost_metric,
                    perf_metric,
                )
        logger.info("range_search_benchmark: end")
        return results

    def knn_search(
        self,
        index_desc: IndexDescriptor,
        parameters: Optional[dict[str, int]],
        db_vectors: Optional[DatasetDescriptor],
        query_vectors: DatasetDescriptor,
        k: Optional[int] = None,
        index: Optional[faiss.Index] = None,
        level: int = 0,
    ):
        assert level >= 0
        if level == 0:
            assert index is None
            assert db_vectors is not None
        else:
            assert index is not None  # quantizer
            assert db_vectors is None
        logger.info("knn_seach: begin")
        k = k if k is not None else self.k
        flat = index_desc.factory == "Flat"
        filename = self.io.get_filename_knn_search(
            factory=index_desc.factory,
            parameters=parameters,
            level=level,
            db_vectors=db_vectors,
            query_vectors=query_vectors,
            k=k,
        )
        if self.io.file_exist(filename):
            logger.info(f"Using cached results for {index_desc.factory}")
            D, I, R, P = self.io.read_file(filename, ["D", "I", "R", "P"])
        else:
            xq = self.io.get_dataset(query_vectors)
            if index is None:
                index = self.get_index(index_desc)
            if parameters:
                for name, val in parameters.items():
                    set_index_parameter(index, name, val)

            index_ivf = faiss.try_extract_index_ivf(index)
            if index_ivf is not None:
                QD, QI, _, QP = self.knn_search(
                    index_desc,
                    parameters=None,
                    db_vectors=None,
                    query_vectors=query_vectors,
                    k=index.nprobe,
                    index=index_ivf.quantizer,
                    level=level + 1,
                )
                # QD = QD[:, :index.nprobe]
                # QI = QI[:, :index.nprobe]
                logger.info("Timing knn search_preassigned")
                faiss.cvar.indexIVF_stats.reset()
                t0 = time.time()
                D, I = index.search_preassigned(xq, k, QI, QD)
                t = time.time() - t0
            else:
                logger.info("Timing knn search")
                t0 = time.time()
                D, I = index.search(xq, k)
                t = time.time() - t0
            if flat or level > 0:
                R = D
            else:
                xb = self.io.get_dataset(db_vectors)
                R = refine_distances_knn(
                    D, I, xq, xb, self.distance_metric_type
                )
            P = {
                "time": t,
                "parameters": parameters,
                "index": index_desc.factory,
                "level": level,
            }
            if index_ivf is not None:
                stats = faiss.cvar.indexIVF_stats
                P |= {
                    "quantizer": QP,
                    "nq": stats.nq,
                    "nlist": stats.nlist,
                    "ndis": stats.ndis,
                    "nheap_updates": stats.nheap_updates,
                    "quantization_time": stats.quantization_time,
                    "search_time": stats.search_time,
                }
            self.io.write_file(filename, ["D", "I", "R", "P"], [D, I, R, P])
        logger.info("knn_seach: end")
        return D, I, R, P

    def knn_ground_truth(self):
        logger.info("knn_ground_truth: begin")
        flat_desc = self.get_index_desc("Flat")
        self.gt_knn_D, self.gt_knn_I, _, _ = self.knn_search(
            flat_desc,
            flat_desc.parameters,
            self.db_vectors,
            self.query_vectors,
        )
        logger.info("knn_ground_truth: end")

    def knn_search_benchmark(
        self, results: dict[str, Any], index_desc: IndexDescriptor
    ):
        logger.info(f"knn_search_benchmark: begin {index_desc.factory=}")
        xq = self.io.get_dataset(self.query_vectors)
        (nq, d) = xq.shape
        logger.info(
            f"Searching {index_desc.factory} with {nq} vectors of dimension {d}"
        )
        codec = self.io.get_codec(index_desc, d)
        codec_ivf = faiss.try_extract_index_ivf(codec)
        if codec_ivf is not None:
            results["indices"][index_desc.factory] = {"nlist": codec_ivf.nlist}

        faiss.omp_set_num_threads(16)

        def experiment(parameters, cost_metric, perf_metric):
            nonlocal results
            key = self.io.get_filename_evaluation_name(
                factory=index_desc.factory,
                parameters=parameters,
                level=0,
                db_vectors=self.db_vectors,
                query_vectors=self.query_vectors,
                evaluation_name="knn",
            )
            if key in results["experiments"]:
                metrics = results["experiments"][key]
            else:
                D, I, R, P = self.knn_search(
                    index_desc, parameters, self.db_vectors, self.query_vectors
                )
                metrics = P | {
                    "knn_intersection": knn_intersection_measure(
                        I, self.gt_knn_I
                    ),
                    "distance_ratio": distance_ratio_measure(
                        R, self.gt_knn_D, self.distance_metric_type
                    ),
                }
                results["experiments"][key] = metrics
            return metrics[cost_metric], metrics[perf_metric]

        for cost_metric in ["time"]:
            for perf_metric in ["knn_intersection", "distance_ratio"]:
                optimizer(
                    codec,
                    experiment,
                    cost_metric,
                    perf_metric,
                )
        logger.info("knn_search_benchmark: end")
        return results

    def benchmark(self) -> str:
        logger.info("begin evaluate")
        results = {"indices": {}, "experiments": {}}
        if self.get_index_desc("Flat") is None:
            self.index_descs.append(IndexDescriptor(factory="Flat"))
        self.knn_ground_truth()
        for index_desc in self.index_descs:
            results = self.knn_search_benchmark(
                results=results,
                index_desc=index_desc,
            )

        if self.range_ref_index_desc is not None:
            index_desc = self.get_index_desc(self.range_ref_index_desc)
            if index_desc is None:
                raise ValueError(
                    f"Unknown range index {self.range_ref_index_desc}"
                )
            if index_desc.range_metrics is None:
                raise ValueError(
                    f"Range index {index_desc.factory} has no radius_score"
                )
            results["metrics"] = {}
            for metric_key, range_metric in index_desc.range_metrics.items():
                (
                    gt_radius,
                    range_search_metric_function,
                    coefficients,
                ) = self.range_search_reference(index_desc, range_metric)
                results["metrics"][metric_key] = coefficients
                gt_rsm = self.range_ground_truth(
                    gt_radius, range_search_metric_function
                )
                for index_desc in self.index_descs:
                    results = self.range_search_benchmark(
                        results=results,
                        index_desc=index_desc,
                        metric_key=metric_key,
                        gt_radius=gt_radius,
                        range_search_metric_function=range_search_metric_function,
                        gt_rsm=gt_rsm,
                    )
        self.io.write_json(results, "result.json", overwrite=True)
        logger.info("end evaluate")
        return json.dumps(results)
