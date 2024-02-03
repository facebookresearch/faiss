# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import copy
from dataclasses import dataclass
from operator import itemgetter
from statistics import mean, median
from typing import Any, Dict, List, Optional

import faiss  # @manual=//faiss/python:pyfaiss_gpu

import numpy as np

from scipy.optimize import curve_fit

from .descriptors import DatasetDescriptor, IndexDescriptor
from .index import Index, IndexFromCodec, IndexFromFactory

from .utils import dict_merge

logger = logging.getLogger(__name__)


def range_search_pr_curve(
    dist_ann: np.ndarray, metric_score: np.ndarray, gt_rsm: float
):
    assert dist_ann.shape == metric_score.shape
    assert dist_ann.ndim == 1
    l = len(dist_ann)
    if l == 0:
        return {
            "dist_ann": [],
            "metric_score_sample": [],
            "cum_score": [],
            "precision": [],
            "recall": [],
            "unique_key": [],
        }
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
    group_by_dist_max_cum_score = np.empty(l, bool)
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


def optimizer(op, search, cost_metric, perf_metric):
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
        cost, perf, requires = search(
            parameters,
            cost_metric,
            perf_metric,
        )
        if requires is not None:
            return requires
        logger.info(
            f"{cno=:4d} {str(parameters):50}: DONE, {cost=:.3f} {perf=:.3f}"
        )
        op.add_operating_point(key, perf, cost)
    return None


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
        aradius, ascore, aradius_from, aradius_to = [], [], [], []
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
            aradius_from.append(radius_from)
            aradius_to.append(radius_to)

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
            list(zip(aradius, ascore, aradius_from, aradius_to, strict=True)),
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
        return real_range * 2, lambda x: np.where(x < real_range, 1, 0), [], []


@dataclass
class Benchmark:
    num_threads: int
    training_vectors: Optional[DatasetDescriptor] = None
    database_vectors: Optional[DatasetDescriptor] = None
    query_vectors: Optional[DatasetDescriptor] = None
    index_descs: Optional[List[IndexDescriptor]] = None
    range_ref_index_desc: Optional[str] = None
    k: Optional[int] = None
    distance_metric: str = "L2"

    def __post_init__(self):
        if self.distance_metric == "IP":
            self.distance_metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.distance_metric == "L2":
            self.distance_metric_type = faiss.METRIC_L2
        else:
            raise ValueError

    def set_io(self, benchmark_io):
        self.io = benchmark_io
        self.io.distance_metric = self.distance_metric
        self.io.distance_metric_type = self.distance_metric_type

    def get_index_desc(self, factory: str) -> Optional[IndexDescriptor]:
        for desc in self.index_descs:
            if desc.factory == factory:
                return desc
        return None

    def range_search_reference(self, index, parameters, range_metric):
        logger.info("range_search_reference: begin")
        if isinstance(range_metric, list):
            assert len(range_metric) > 0
            m_radius = (
                max(rm[-2] for rm in range_metric)
                if self.distance_metric_type == faiss.METRIC_L2
                else min(rm[-2] for rm in range_metric)
            )
        else:
            m_radius = range_metric

        lims, D, I, R, P, _ = self.range_search(
            False,
            index,
            parameters,
            radius=m_radius,
        )
        flat = index.factory == "Flat"
        (
            gt_radius,
            range_search_metric_function,
            coefficients,
            coefficients_training_data,
        ) = get_range_search_metric_function(
            range_metric,
            D if not flat else None,
            R if not flat else None,
        )
        logger.info("range_search_reference: end")
        return (
            gt_radius,
            range_search_metric_function,
            coefficients,
            coefficients_training_data,
        )

    def estimate_range(self, index, parameters, range_scoring_radius):
        D, I, R, P, _ = index.knn_search(
            False,
            parameters,
            self.query_vectors,
            self.k,
        )
        samples = []
        for i, j in np.argwhere(R < range_scoring_radius):
            samples.append((R[i, j].item(), D[i, j].item()))
        if len(samples) > 0:  # estimate range
            samples.sort(key=itemgetter(0))
            return median(r for _, r in samples[-3:])
        else:  # ensure at least one result
            i, j = np.argwhere(R.min() == R)[0]
            return D[i, j].item()

    def range_search(
        self,
        dry_run,
        index: Index,
        search_parameters: Optional[Dict[str, int]],
        radius: Optional[float] = None,
        gt_radius: Optional[float] = None,
        range_search_metric_function=None,
        gt_rsm=None,
    ):
        logger.info("range_search: begin")
        if radius is None:
            assert gt_radius is not None
            radius = (
                gt_radius
                if index.is_flat()
                else self.estimate_range(
                    index,
                    search_parameters,
                    gt_radius,
                )
            )
        logger.info(f"Radius={radius}")
        lims, D, I, R, P, requires = index.range_search(
            dry_run=dry_run,
            search_parameters=search_parameters,
            query_vectors=self.query_vectors,
            radius=radius,
        )
        if requires is not None:
            return None, None, None, None, None, requires
        if range_search_metric_function is not None:
            range_search_metric = range_search_metric_function(R)
            range_search_pr = range_search_pr_curve(
                D, range_search_metric, gt_rsm
            )
            range_score_sum = np.sum(range_search_metric).item()
            P |= {
                "range_score_sum": range_score_sum,
                "range_score_max_recall": range_score_sum / gt_rsm,
                "range_search_pr": range_search_pr,
            }
        return lims, D, I, R, P, requires

    def range_ground_truth(self, gt_radius, range_search_metric_function):
        logger.info("range_ground_truth: begin")
        flat_desc = self.get_index_desc("Flat")
        lims, D, I, R, P, _ = self.range_search(
            False,
            flat_desc.index,
            search_parameters=None,
            radius=gt_radius,
        )
        gt_rsm = np.sum(range_search_metric_function(R)).item()
        logger.info("range_ground_truth: end")
        return gt_rsm

    def knn_ground_truth(self):
        logger.info("knn_ground_truth: begin")
        flat_desc = self.get_index_desc("Flat")
        self.build_index_wrapper(flat_desc)
        (
            self.gt_knn_D,
            self.gt_knn_I,
            _,
            _,
            requires,
        ) = flat_desc.index.knn_search(
            dry_run=False,
            search_parameters=None,
            query_vectors=self.query_vectors,
            k=self.k,
        )
        assert requires is None
        logger.info("knn_ground_truth: end")

    def search_benchmark(
        self,
        name,
        search_func,
        key_func,
        cost_metrics,
        perf_metrics,
        results: Dict[str, Any],
        index: Index,
    ):
        index_name = index.get_index_name()
        logger.info(f"{name}_benchmark: begin {index_name}")

        def experiment(parameters, cost_metric, perf_metric):
            nonlocal results
            key = key_func(parameters)
            if key in results["experiments"]:
                metrics = results["experiments"][key]
            else:
                metrics, requires = search_func(parameters)
                if requires is not None:
                    return None, None, requires
                results["experiments"][key] = metrics
            return metrics[cost_metric], metrics[perf_metric], None

        for cost_metric in cost_metrics:
            for perf_metric in perf_metrics:
                op = index.get_operating_points()
                requires = optimizer(
                    op,
                    experiment,
                    cost_metric,
                    perf_metric,
                )
                if requires is not None:
                    break
        logger.info(f"{name}_benchmark: end")
        return results, requires

    def knn_search_benchmark(
        self, dry_run, results: Dict[str, Any], index: Index
    ):
        return self.search_benchmark(
            name="knn_search",
            search_func=lambda parameters: index.knn_search(
                dry_run,
                parameters,
                self.query_vectors,
                self.k,
                self.gt_knn_I,
                self.gt_knn_D,
            )[3:],
            key_func=lambda parameters: index.get_knn_search_name(
                search_parameters=parameters,
                query_vectors=self.query_vectors,
                k=self.k,
                reconstruct=False,
            ),
            cost_metrics=["time"],
            perf_metrics=["knn_intersection", "distance_ratio"],
            results=results,
            index=index,
        )

    def reconstruct_benchmark(
        self, dry_run, results: Dict[str, Any], index: Index
    ):
        return self.search_benchmark(
            name="reconstruct",
            search_func=lambda parameters: index.reconstruct(
                dry_run,
                parameters,
                self.query_vectors,
                self.k,
                self.gt_knn_I,
            ),
            key_func=lambda parameters: index.get_knn_search_name(
                search_parameters=parameters,
                query_vectors=self.query_vectors,
                k=self.k,
                reconstruct=True,
            ),
            cost_metrics=["encode_time"],
            perf_metrics=["sym_recall"],
            results=results,
            index=index,
        )

    def range_search_benchmark(
        self,
        dry_run,
        results: Dict[str, Any],
        index: Index,
        metric_key: str,
        radius: float,
        gt_radius: float,
        range_search_metric_function,
        gt_rsm: float,
    ):
        return self.search_benchmark(
            name="range_search",
            search_func=lambda parameters: self.range_search(
                dry_run=dry_run,
                index=index,
                search_parameters=parameters,
                radius=radius,
                gt_radius=gt_radius,
                range_search_metric_function=range_search_metric_function,
                gt_rsm=gt_rsm,
            )[4:],
            key_func=lambda parameters: index.get_range_search_name(
                search_parameters=parameters,
                query_vectors=self.query_vectors,
                radius=radius,
            )
            + metric_key,
            cost_metrics=["time"],
            perf_metrics=["range_score_max_recall"],
            results=results,
            index=index,
        )

    def build_index_wrapper(self, index_desc: IndexDescriptor):
        if hasattr(index_desc, "index"):
            return
        if index_desc.factory is not None:
            training_vectors = copy(self.training_vectors)
            if index_desc.training_size is not None:
                training_vectors.num_vectors = index_desc.training_size
            index = IndexFromFactory(
                num_threads=self.num_threads,
                d=self.d,
                metric=self.distance_metric,
                database_vectors=self.database_vectors,
                search_params=index_desc.search_params,
                construction_params=index_desc.construction_params,
                factory=index_desc.factory,
                training_vectors=training_vectors,
            )
        else:
            index = IndexFromCodec(
                num_threads=self.num_threads,
                d=self.d,
                metric=self.distance_metric,
                database_vectors=self.database_vectors,
                search_params=index_desc.search_params,
                construction_params=index_desc.construction_params,
                path=index_desc.path,
                bucket=index_desc.bucket,
            )
        index.set_io(self.io)
        index_desc.index = index

    def clone_one(self, index_desc):
        benchmark = Benchmark(
            num_threads=self.num_threads,
            training_vectors=self.training_vectors,
            database_vectors=self.database_vectors,
            query_vectors=self.query_vectors,
            index_descs=[self.get_index_desc("Flat"), index_desc],
            range_ref_index_desc=self.range_ref_index_desc,
            k=self.k,
            distance_metric=self.distance_metric,
        )
        benchmark.set_io(self.io.clone())
        return benchmark

    def benchmark_one(
        self,
        dry_run,
        results: Dict[str, Any],
        index_desc: IndexDescriptor,
        train,
        reconstruct,
        knn,
        range,
    ):
        faiss.omp_set_num_threads(self.num_threads)
        if not dry_run:
            self.knn_ground_truth()
        self.build_index_wrapper(index_desc)
        meta, requires = index_desc.index.fetch_meta(dry_run=dry_run)
        if requires is not None:
            return results, (requires if train else None)
        results["indices"][index_desc.index.get_codec_name()] = meta

        # results, requires = self.reconstruct_benchmark(
        #     dry_run=True,
        #     results=results,
        #     index=index_desc.index,
        # )
        # if reconstruct and requires is not None:
        #     if dry_run:
        #         return results, requires
        #     else:
        #         results, requires = self.reconstruct_benchmark(
        #             dry_run=False,
        #             results=results,
        #             index=index_desc.index,
        #         )
        #         assert requires is None

        results, requires = self.knn_search_benchmark(
            dry_run=True,
            results=results,
            index=index_desc.index,
        )
        if knn and requires is not None:
            if dry_run:
                return results, requires
            else:
                results, requires = self.knn_search_benchmark(
                    dry_run=False,
                    results=results,
                    index=index_desc.index,
                )
                assert requires is None

        if (
            self.range_ref_index_desc is None
            or not index_desc.index.supports_range_search()
        ):
            return results, None

        ref_index_desc = self.get_index_desc(self.range_ref_index_desc)
        if ref_index_desc is None:
            raise ValueError(
                f"Unknown range index {self.range_ref_index_desc}"
            )
        if ref_index_desc.range_metrics is None:
            raise ValueError(
                f"Range index {ref_index_desc.factory} has no radius_score"
            )
        for metric_key, range_metric in ref_index_desc.range_metrics.items():
            (
                gt_radius,
                range_search_metric_function,
                coefficients,
                coefficients_training_data,
            ) = self.range_search_reference(
                ref_index_desc.index,
                ref_index_desc.search_params,
                range_metric,
            )
            gt_rsm = self.range_ground_truth(
                gt_radius, range_search_metric_function
            )
            results, requires = self.range_search_benchmark(
                dry_run=True,
                results=results,
                index=index_desc.index,
                metric_key=metric_key,
                radius=index_desc.radius,
                gt_radius=gt_radius,
                range_search_metric_function=range_search_metric_function,
                gt_rsm=gt_rsm,
            )
            if range and requires is not None:
                if dry_run:
                    return results, requires
                else:
                    results, requires = self.range_search_benchmark(
                        dry_run=False,
                        results=results,
                        index=index_desc.index,
                        metric_key=metric_key,
                        radius=index_desc.radius,
                        gt_radius=gt_radius,
                        range_search_metric_function=range_search_metric_function,
                        gt_rsm=gt_rsm,
                    )
                    assert requires is None

        return results, None

    def benchmark(
        self,
        result_file=None,
        local=False,
        train=False,
        reconstruct=False,
        knn=False,
        range=False,
    ):
        logger.info("begin evaluate")

        faiss.omp_set_num_threads(self.num_threads)
        results = {"indices": {}, "experiments": {}}
        xq = self.io.get_dataset(self.query_vectors)
        self.d = xq.shape[1]
        if self.get_index_desc("Flat") is None:
            self.index_descs.append(IndexDescriptor(factory="Flat"))

        self.knn_ground_truth()

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
                    coefficients_training_data,
                ) = self.range_search_reference(
                    index_desc.index, index_desc.search_params, range_metric
                )
                results["metrics"][metric_key] = {
                    "coefficients": coefficients,
                    "training_data": coefficients_training_data,
                }
                gt_rsm = self.range_ground_truth(
                    gt_radius, range_search_metric_function
                )

        self.index_descs = list(dict.fromkeys(self.index_descs))

        todo = self.index_descs
        for index_desc in self.index_descs:
            index_desc.requires = None

        queued = set()
        while todo:
            current_todo = []
            next_todo = []
            for index_desc in todo:
                results, requires = self.benchmark_one(
                    dry_run=True,
                    results=results,
                    index_desc=index_desc,
                    train=train,
                    reconstruct=reconstruct,
                    knn=knn,
                    range=range,
                )
                if requires is None:
                    continue
                if requires in queued:
                    if index_desc.requires != requires:
                        index_desc.requires = requires
                        next_todo.append(index_desc)
                else:
                    queued.add(requires)
                    index_desc.requires = requires
                    current_todo.append(index_desc)

            if current_todo:
                results_one = {"indices": {}, "experiments": {}}
                params = [
                    (
                        index_desc,
                        self.clone_one(index_desc),
                        results_one,
                        train,
                        reconstruct,
                        knn,
                        range,
                    )
                    for index_desc in current_todo
                ]
                for result in self.io.launch_jobs(
                    run_benchmark_one, params, local=local
                ):
                    dict_merge(results, result)

            todo = next_todo

        if result_file is not None:
            self.io.write_json(results, result_file, overwrite=True)
        logger.info("end evaluate")
        return results


def run_benchmark_one(params):
    logger.info(params)
    index_desc, benchmark, results, train, reconstruct, knn, range = params
    results, requires = benchmark.benchmark_one(
        dry_run=False,
        results=results,
        index_desc=index_desc,
        train=train,
        reconstruct=reconstruct,
        knn=knn,
        range=range,
    )
    assert requires is None
    assert results is not None
    return results
