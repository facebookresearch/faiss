# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from operator import itemgetter
from statistics import mean, median
from typing import Any, Dict, List, Optional

import faiss  # @manual=//faiss/python:pyfaiss_gpu

import numpy as np

from scipy.optimize import curve_fit

from .benchmark_io import BenchmarkIO

from .descriptors import (
    CodecDescriptor,
    DatasetDescriptor,
    IndexDescriptor,
    IndexDescriptorClassic,
    KnnDescriptor,
)

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
class IndexOperator:
    num_threads: int
    distance_metric: str

    def __post_init__(self):
        if self.distance_metric == "IP":
            self.distance_metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.distance_metric == "L2":
            self.distance_metric_type = faiss.METRIC_L2
        else:
            raise ValueError

    def set_io(self, benchmark_io: BenchmarkIO):
        self.io = benchmark_io
        self.io.distance_metric = self.distance_metric
        self.io.distance_metric_type = self.distance_metric_type


@dataclass
class TrainOperator(IndexOperator):
    codec_descs: List[CodecDescriptor] = field(default_factory=lambda: [])

    def get_desc(self, name: str) -> Optional[CodecDescriptor]:
        for desc in self.codec_descs:
            if desc.get_name() == name:
                return desc
            elif desc.factory == name:
                return desc
        return None

    def get_flat_desc(self, name=None) -> Optional[CodecDescriptor]:
        for desc in self.codec_descs:
            desc_name = desc.get_name()
            if desc_name == name:
                return desc
            if desc_name.startswith("Flat"):
                return desc
        return None

    def build_index_wrapper(self, codec_desc: CodecDescriptor):
        if hasattr(codec_desc, "index"):
            return

        if codec_desc.factory is not None:
            assert (
                codec_desc.factory == "Flat" or codec_desc.training_vectors is not None
            )
            index = IndexFromFactory(
                num_threads=self.num_threads,
                d=codec_desc.d,
                metric=self.distance_metric,
                construction_params=codec_desc.construction_params,
                factory=codec_desc.factory,
                training_vectors=codec_desc.training_vectors,
                codec_name=codec_desc.get_name(),
            )
            index.set_io(self.io)
            codec_desc.index = index
        else:
            assert codec_desc.is_trained()

    def train(
        self, codec_desc: CodecDescriptor, results: Dict[str, Any], dry_run=False
    ):
        self.build_index_wrapper(codec_desc)
        if codec_desc.is_trained():
            return results, None

        if dry_run:
            meta, requires = codec_desc.index.fetch_meta(dry_run=dry_run)
        else:
            codec_desc.index.get_codec()
            meta, requires = codec_desc.index.fetch_meta(dry_run=dry_run)
            assert requires is None

        if requires is None:
            results["indices"][codec_desc.get_name()] = meta
        return results, requires


@dataclass
class BuildOperator(IndexOperator):
    index_descs: List[IndexDescriptor] = field(default_factory=lambda: [])
    serialize_index: bool = False

    def get_desc(self, name: str) -> Optional[IndexDescriptor]:
        for desc in self.index_descs:
            if desc.get_name() == name:
                return desc
        return None

    def get_flat_desc(self, name=None) -> Optional[IndexDescriptor]:
        for desc in self.index_descs:
            desc_name = desc.get_name()
            if desc_name == name:
                return desc
            if desc_name.startswith("Flat"):
                return desc
        return None

    def build_index_wrapper(self, index_desc: IndexDescriptor):
        if hasattr(index_desc, "index"):
            return

        if hasattr(index_desc.codec_desc, "index"):
            index_desc.index = index_desc.codec_desc.index
            index_desc.index.database_vectors = index_desc.database_desc
            index_desc.index.index_name = index_desc.get_name()
            return

        if index_desc.codec_desc is not None:
            index = IndexFromCodec(
                num_threads=self.num_threads,
                d=index_desc.d,
                metric=self.distance_metric,
                database_vectors=index_desc.database_desc,
                bucket=index_desc.codec_desc.bucket,
                path=index_desc.codec_desc.path,
                index_name=index_desc.get_name(),
                codec_name=index_desc.codec_desc.get_name(),
                serialize_full_index=self.serialize_index,
            )
            index.set_io(self.io)
            index_desc.index = index
        else:
            assert index_desc.is_built()

    def build(self, index_desc: IndexDescriptor, results: Dict[str, Any]):
        self.build_index_wrapper(index_desc)
        if index_desc.is_built():
            return
        index_desc.index.get_index()


@dataclass
class SearchOperator(IndexOperator):
    knn_descs: List[KnnDescriptor] = field(default_factory=lambda: [])
    range: bool = False

    def get_desc(self, name: str) -> Optional[KnnDescriptor]:
        for desc in self.knn_descs:
            if desc.get_name() == name:
                return desc
        return None

    def get_flat_desc(self, name=None) -> Optional[KnnDescriptor]:
        for desc in self.knn_descs:
            if desc.get_name().startswith("Flat"):
                return desc
        return None

    def build_index_wrapper(self, knn_desc: KnnDescriptor):
        if hasattr(knn_desc, "index"):
            return

        assert knn_desc.index_desc is not None
        if hasattr(knn_desc.index_desc, "index"):
            knn_desc.index = knn_desc.index_desc.index
            knn_desc.index.knn_name = knn_desc.get_name()
            knn_desc.index.search_params = knn_desc.search_params
        else:
            index = Index(
                num_threads=self.num_threads,
                d=knn_desc.d,
                metric=self.distance_metric,
                bucket=knn_desc.index_desc.bucket,
                index_path=knn_desc.index_desc.path,
                index_name=knn_desc.index_desc.get_name(),
                # knn_name=knn_desc.get_name(),
                search_params=knn_desc.search_params,
            )
            index.set_io(self.io)
            knn_desc.index = index

        knn_desc.index.get_index()

    def range_search_reference(self, index, parameters, range_metric, query_dataset):
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
            query_dataset=query_dataset,
        )
        flat = index.is_flat_index()
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

    def estimate_range(self, index, parameters, range_scoring_radius, query_dataset):
        D, I, R, P, _ = index.knn_search(
            False,
            parameters,
            query_dataset,
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
        query_dataset: DatasetDescriptor,
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
                    index, search_parameters, gt_radius, query_dataset
                )
            )
        logger.info(f"Radius={radius}")
        lims, D, I, R, P, requires = index.range_search(
            dry_run=dry_run,
            search_parameters=search_parameters,
            query_vectors=query_dataset,
            radius=radius,
        )
        if requires is not None:
            return None, None, None, None, None, requires
        if range_search_metric_function is not None:
            range_search_metric = range_search_metric_function(R)
            range_search_pr = range_search_pr_curve(D, range_search_metric, gt_rsm)
            range_score_sum = np.sum(range_search_metric).item()
            P |= {
                "range_score_sum": range_score_sum,
                "range_score_max_recall": range_score_sum / gt_rsm,
                "range_search_pr": range_search_pr,
            }
        return lims, D, I, R, P, requires

    def range_ground_truth(
        self, gt_radius, range_search_metric_function, flat_desc=None
    ):
        logger.info("range_ground_truth: begin")
        if flat_desc is None:
            flat_desc = self.get_flat_desc()
        lims, D, I, R, P, _ = self.range_search(
            False,
            flat_desc.index,
            search_parameters=None,
            radius=gt_radius,
            query_dataset=flat_desc.query_dataset,
        )
        gt_rsm = np.sum(range_search_metric_function(R)).item()
        logger.info("range_ground_truth: end")
        return gt_rsm

    def knn_ground_truth(self, flat_desc=None):
        logger.info("knn_ground_truth: begin")
        if flat_desc is None:
            flat_desc = self.get_flat_desc()
        self.build_index_wrapper(flat_desc)
        # TODO(kuarora): Consider moving gt results(gt_knn_D, gt_knn_I) to the index as there can be multiple ground truths.
        (
            self.gt_knn_D,
            self.gt_knn_I,
            _,
            _,
            requires,
        ) = flat_desc.index.knn_search(
            dry_run=False,
            search_parameters=None,
            query_vectors=flat_desc.query_dataset,
            k=flat_desc.k,
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

        requires = None
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
        self, dry_run, results: Dict[str, Any], knn_desc: KnnDescriptor
    ):
        gt_knn_D = None
        gt_knn_I = None
        if hasattr(self, "gt_knn_D"):
            gt_knn_D = self.gt_knn_D
            gt_knn_I = self.gt_knn_I

        assert hasattr(knn_desc, "index")
        if not knn_desc.index.is_flat_index() and gt_knn_I is None:
            key = knn_desc.index.get_knn_search_name(
                search_parameters=knn_desc.search_params,
                query_vectors=knn_desc.query_dataset,
                k=knn_desc.k,
                reconstruct=False,
            )
            metrics, requires = knn_desc.index.knn_search(
                dry_run,
                knn_desc.search_params,
                knn_desc.query_dataset,
                knn_desc.k,
            )[3:]
            if requires is not None:
                return results, requires
            results["experiments"][key] = metrics
            return results, requires

        return self.search_benchmark(
            name="knn_search",
            search_func=lambda parameters: knn_desc.index.knn_search(
                dry_run,
                parameters,
                knn_desc.query_dataset,
                knn_desc.k,
                gt_knn_I,
                gt_knn_D,
            )[3:],
            key_func=lambda parameters: knn_desc.index.get_knn_search_name(
                search_parameters=parameters,
                query_vectors=knn_desc.query_dataset,
                k=knn_desc.k,
                reconstruct=False,
            ),
            cost_metrics=["time"],
            perf_metrics=["knn_intersection", "distance_ratio"],
            results=results,
            index=knn_desc.index,
        )

    def reconstruct_benchmark(
        self, dry_run, results: Dict[str, Any], knn_desc: KnnDescriptor
    ):
        return self.search_benchmark(
            name="reconstruct",
            search_func=lambda parameters: knn_desc.index.reconstruct(
                dry_run,
                parameters,
                knn_desc.query_dataset,
                knn_desc.k,
                self.gt_knn_I,
            ),
            key_func=lambda parameters: knn_desc.index.get_knn_search_name(
                search_parameters=parameters,
                query_vectors=knn_desc.query_dataset,
                k=knn_desc.k,
                reconstruct=True,
            ),
            cost_metrics=["encode_time"],
            perf_metrics=["sym_recall"],
            results=results,
            index=knn_desc.index,
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
        query_dataset: DatasetDescriptor,
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
                query_dataset=query_dataset,
            )[4:],
            key_func=lambda parameters: index.get_range_search_name(
                search_parameters=parameters,
                query_vectors=query_dataset,
                radius=radius,
            )
            + metric_key,
            cost_metrics=["time"],
            perf_metrics=["range_score_max_recall"],
            results=results,
            index=index,
        )


@dataclass
class ExecutionOperator:
    distance_metric: str = "L2"
    num_threads: int = 1
    train_op: Optional[TrainOperator] = None
    build_op: Optional[BuildOperator] = None
    search_op: Optional[SearchOperator] = None
    compute_gt: bool = True

    def __post_init__(self):
        if self.distance_metric == "IP":
            self.distance_metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.distance_metric == "L2":
            self.distance_metric_type = faiss.METRIC_L2
        else:
            raise ValueError

    def set_io(self, io: BenchmarkIO):
        self.io = io
        self.io.distance_metric = self.distance_metric
        self.io.distance_metric_type = self.distance_metric_type
        if self.train_op:
            self.train_op.set_io(io)
        if self.build_op:
            self.build_op.set_io(io)
        if self.search_op:
            self.search_op.set_io(io)

    def train_one(self, codec_desc: CodecDescriptor, results: Dict[str, Any], dry_run):
        faiss.omp_set_num_threads(self.num_threads)
        assert self.train_op is not None
        self.train_op.train(codec_desc, results, dry_run)

    def train(self, results, dry_run=False):
        faiss.omp_set_num_threads(self.num_threads)
        if self.train_op is None:
            return

        for codec_desc in self.train_op.codec_descs:
            self.train_one(codec_desc, results, dry_run)

    def build_one(self, results: Dict[str, Any], index_desc: IndexDescriptor):
        faiss.omp_set_num_threads(self.num_threads)
        assert self.build_op is not None
        self.build_op.build(index_desc, results)

    def build(self, results: Dict[str, Any]):
        faiss.omp_set_num_threads(self.num_threads)
        if self.build_op is None:
            return

        for index_desc in self.build_op.index_descs:
            self.build_one(index_desc, results)

    def search(self):
        faiss.omp_set_num_threads(self.num_threads)
        if self.search_op is None:
            return

        for index_desc in self.search_op.knn_descs:
            self.search_one(index_desc)

    def search_one(
        self,
        knn_desc: KnnDescriptor,
        results: Dict[str, Any],
        dry_run=False,
        range=False,
    ):
        faiss.omp_set_num_threads(self.num_threads)
        assert self.search_op is not None

        if not dry_run and self.compute_gt:
            self.create_gt_knn(knn_desc)
            self.create_range_ref_knn(knn_desc)

        self.search_op.build_index_wrapper(knn_desc)

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
        results, requires = self.search_op.knn_search_benchmark(
            dry_run=True,
            results=results,
            knn_desc=knn_desc,
        )
        if requires is not None:
            if dry_run:
                return results, requires
            else:
                results, requires = self.search_op.knn_search_benchmark(
                    dry_run=False,
                    results=results,
                    knn_desc=knn_desc,
                )
                assert requires is None

        if (
            knn_desc.range_ref_index_desc is None or
            not knn_desc.index.supports_range_search()
        ):
            return results, None

        ref_index_desc = self.search_op.get_desc(knn_desc.range_ref_index_desc)
        if ref_index_desc is None:
            raise ValueError(
                f"{knn_desc.get_name()}: Unknown range index {knn_desc.range_ref_index_desc}"
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
            ) = self.search_op.range_search_reference(
                ref_index_desc.index,
                ref_index_desc.search_params,
                range_metric,
            )
            gt_rsm = None
            if self.compute_gt:
                gt_rsm = self.search_op.range_ground_truth(
                    gt_radius, range_search_metric_function
                )
            results, requires = self.search_op.range_search_benchmark(
                dry_run=True,
                results=results,
                index=knn_desc.index,
                metric_key=metric_key,
                radius=knn_desc.radius,
                gt_radius=gt_radius,
                range_search_metric_function=range_search_metric_function,
                gt_rsm=gt_rsm,
                query_vectors=knn_desc.query_dataset,
            )
            if range and requires is not None:
                if dry_run:
                    return results, requires
                else:
                    results, requires = self.search_op.range_search_benchmark(
                        dry_run=False,
                        results=results,
                        index=knn_desc.index,
                        metric_key=metric_key,
                        radius=knn_desc.radius,
                        gt_radius=gt_radius,
                        range_search_metric_function=range_search_metric_function,
                        gt_rsm=gt_rsm,
                        query_vectors=knn_desc.query_dataset,
                    )
                    assert requires is None

        return results, None

    def create_gt_codec(
        self, codec_desc, results, train=True
    ) -> Optional[CodecDescriptor]:
        gt_codec_desc = None
        if self.train_op:
            gt_codec_desc = self.train_op.get_flat_desc(codec_desc.flat_name())
            if gt_codec_desc is None:
                gt_codec_desc = CodecDescriptor(
                    factory="Flat",
                    d=codec_desc.d,
                    metric=codec_desc.metric,
                    num_threads=self.num_threads,
                )
                self.train_op.codec_descs.insert(0, gt_codec_desc)
            if train:
                self.train_op.train(gt_codec_desc, results, dry_run=False)

        return gt_codec_desc

    def create_gt_index(
        self, index_desc: IndexDescriptor, results: Dict[str, Any], build=True
    ) -> Optional[IndexDescriptor]:
        gt_index_desc = None
        if self.build_op:
            gt_index_desc = self.build_op.get_flat_desc(index_desc.flat_name())
            if gt_index_desc is None:
                gt_codec_desc = self.train_op.get_flat_desc(
                    index_desc.codec_desc.flat_name()
                )
                assert gt_codec_desc is not None
                gt_index_desc = IndexDescriptor(
                    d=index_desc.d,
                    metric=index_desc.metric,
                    num_threads=self.num_threads,
                    codec_desc=gt_codec_desc,
                    database_desc=index_desc.database_desc,
                )
                self.build_op.index_descs.insert(0, gt_index_desc)
            if build:
                self.build_op.build(gt_index_desc, results)

        return gt_index_desc

    def create_gt_knn(self, knn_desc, search=True) -> Optional[KnnDescriptor]:
        gt_knn_desc = None
        if self.search_op:
            gt_knn_desc = self.search_op.get_flat_desc(knn_desc.flat_name())
            if gt_knn_desc is None:
                if knn_desc.index_desc is not None:
                    gt_index_desc = knn_desc.gt_index_desc
                else:
                    gt_index_desc = self.build_op.get_flat_desc(
                        knn_desc.index_desc.flat_name()
                    )
                    knn_desc.gt_index_desc = gt_index_desc
                assert gt_index_desc is not None
                gt_knn_desc = KnnDescriptor(
                    d=knn_desc.d,
                    metric=knn_desc.metric,
                    num_threads=self.num_threads,
                    index_desc=gt_index_desc,
                    query_dataset=knn_desc.query_dataset,
                    k=knn_desc.k,
                )
                self.search_op.knn_descs.insert(0, gt_knn_desc)
            if search:
                self.search_op.build_index_wrapper(gt_knn_desc)
                self.search_op.knn_ground_truth(gt_knn_desc)

        return gt_knn_desc

    def create_range_ref_knn(self, knn_desc):
        if (
            knn_desc.range_ref_index_desc is None or
            not knn_desc.index.supports_range_search()
        ):
            return

        if knn_desc.range_ref_index_desc is not None:
            ref_index_desc = self.get_desc(knn_desc.range_ref_index_desc)
            if ref_index_desc is None:
                raise ValueError(f"Unknown range index {knn_desc.range_ref_index_desc}")
            if ref_index_desc.range_metrics is None:
                raise ValueError(
                    f"Range index {knn_desc.get_name()} has no radius_score"
                )
            results["metrics"] = {}
            self.build_index_wrapper(ref_index_desc)
            for metric_key, range_metric in ref_index_desc.range_metrics.items():
                (
                    knn_desc.gt_radius,
                    range_search_metric_function,
                    coefficients,
                    coefficients_training_data,
                ) = self.range_search_reference(
                    knn_desc.index, knn_desc.search_params, range_metric
                )
                results["metrics"][metric_key] = {
                    "coefficients": coefficients,
                    "training_data": coefficients_training_data,
                }
                knn_desc.gt_rsm = self.range_ground_truth(
                    knn_desc.gt_radius, range_search_metric_function
                )

    def create_ground_truths(self, results: Dict[str, Any]):
        # TODO: Create all ground truth descriptors and put them in index descriptor as reference
        if self.train_op is not None:
            for codec_desc in self.train_op.codec_descs:
                self.create_gt_codec(codec_desc, results)

        if self.build_op is not None:
            for index_desc in self.build_op.index_descs:
                self.create_gt_index(
                    index_desc, results
                )  # may need to pass results in future

        if self.search_op is not None:
            for knn_desc in self.search_op.knn_descs:
                self.create_gt_knn(knn_desc, results)
                self.create_range_ref_knn(knn_desc)

    def execute(self, results: Dict[str, Any], dry_run: False):
        if self.train_op is not None:
            for desc in self.train_op.codec_descs:
                results, requires = self.train_op.train(desc, results, dry_run=dry_run)
                if dry_run:
                    if requires is None:
                        continue
                    return results, requires
                assert requires is None

        if self.build_op is not None:
            for desc in self.build_op.index_descs:
                self.build_op.build(desc, results)
        if self.search_op is not None:
            for desc in self.search_op.knn_descs:
                results, requires = self.search_one(
                    knn_desc=desc,
                    results=results,
                    dry_run=dry_run,
                    range=self.search_op.range,
                )
                if dry_run:
                    if requires is None:
                        continue
                    return results, requires

                assert requires is None
        return results, None

    def execute_2(self, result_file=None):
        results = {"indices": {}, "experiments": {}}
        results, requires = self.execute(results=results)
        assert requires is None
        if result_file is not None:
            self.io.write_json(results, result_file, overwrite=True)

    def add_index_descs(self, codec_desc, index_desc, knn_desc):
        if codec_desc is not None:
            self.train_op.codec_descs.append(codec_desc)
        if index_desc is not None:
            self.build_op.index_descs.append(index_desc)
        if knn_desc is not None:
            self.search_op.knn_descs.append(knn_desc)


@dataclass
class Benchmark:
    num_threads: int
    training_vectors: Optional[DatasetDescriptor] = None
    database_vectors: Optional[DatasetDescriptor] = None
    query_vectors: Optional[DatasetDescriptor] = None
    index_descs: Optional[List[IndexDescriptorClassic]] = None
    range_ref_index_desc: Optional[str] = None
    k: int = 1
    distance_metric: str = "L2"

    def set_io(self, benchmark_io):
        self.io = benchmark_io

    def get_embedding_dimension(self):
        if self.training_vectors is not None:
            xt = self.io.get_dataset(self.training_vectors)
            return xt.shape[1]
        if self.database_vectors is not None:
            xb = self.io.get_dataset(self.database_vectors)
            return xb.shape[1]
        if self.query_vectors is not None:
            xq = self.io.get_dataset(self.query_vectors)
            return xq.shape[1]
        raise ValueError("Failed to determine dimension of dataset")

    def create_descriptors(
        self, ci_desc: IndexDescriptorClassic, train, build, knn, reconstruct, range
    ):
        codec_desc = None
        index_desc = None
        knn_desc = None
        dim = self.get_embedding_dimension()
        if train and ci_desc.factory is not None:
            codec_desc = CodecDescriptor(
                d=dim,
                metric=self.distance_metric,
                num_threads=self.num_threads,
                factory=ci_desc.factory,
                construction_params=ci_desc.construction_params,
                training_vectors=self.training_vectors,
            )
        if build:
            if codec_desc is None:
                assert ci_desc.path is not None
                codec_desc = CodecDescriptor(
                    d=dim,
                    metric=self.distance_metric,
                    num_threads=self.num_threads,
                    bucket=ci_desc.bucket,
                    path=ci_desc.path,
                )
            index_desc = IndexDescriptor(
                d=codec_desc.d,
                metric=self.distance_metric,
                num_threads=self.num_threads,
                codec_desc=codec_desc,
                database_desc=self.database_vectors,
            )
        if knn or range:
            if index_desc is None:
                assert ci_desc.path is not None
                index_desc = IndexDescriptor(
                    d=dim,
                    metric=self.distance_metric,
                    num_threads=self.num_threads,
                    bucket=ci_desc.bucket,
                    path=ci_desc.path,
                )
            knn_desc = KnnDescriptor(
                d=dim,
                metric=self.distance_metric,
                num_threads=self.num_threads,
                index_desc=index_desc,
                query_dataset=self.query_vectors,
                search_params=ci_desc.search_params,
                range_metrics=ci_desc.range_metrics,
                radius=ci_desc.radius,
                k=self.k,
            )

        return codec_desc, index_desc, knn_desc

    def create_execution_operator(
        self,
        train,
        build,
        knn,
        reconstruct,
        range,
    ) -> ExecutionOperator:
        # all operators are created, as ground truth are always created in benchmarking
        train_op = TrainOperator(
            num_threads=self.num_threads, distance_metric=self.distance_metric
        )
        build_op = BuildOperator(
            num_threads=self.num_threads, distance_metric=self.distance_metric
        )
        search_op = SearchOperator(
            num_threads=self.num_threads, distance_metric=self.distance_metric
        )
        search_op.range = range

        exec_op = ExecutionOperator(
            train_op=train_op,
            build_op=build_op,
            search_op=search_op,
            num_threads=self.num_threads,
        )
        assert hasattr(self, "io")
        exec_op.set_io(self.io)

        # iterate over classic descriptors
        for ci_desc in self.index_descs:
            codec_desc, index_desc, knn_desc = self.create_descriptors(
                ci_desc, train, build, knn, reconstruct, range
            )
            exec_op.add_index_descs(codec_desc, index_desc, knn_desc)

        return exec_op

    def clone_one(self, index_desc):
        benchmark = Benchmark(
            num_threads=self.num_threads,
            training_vectors=self.training_vectors,
            database_vectors=self.database_vectors,
            query_vectors=self.query_vectors,
            # index_descs=[self.get_flat_desc("Flat"), index_desc],
            index_descs=[index_desc],  # Should automatically find flat descriptors
            range_ref_index_desc=self.range_ref_index_desc,
            k=self.k,
            distance_metric=self.distance_metric,
        )
        benchmark.set_io(self.io.clone())
        return benchmark

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
        results = {"indices": {}, "experiments": {}}
        faiss.omp_set_num_threads(self.num_threads)
        exec_op = self.create_execution_operator(
            train=train,
            build=knn or range,
            knn=knn,
            reconstruct=reconstruct,
            range=range,
        )
        exec_op.create_ground_truths(results)

        todo = self.index_descs
        for index_desc in self.index_descs:
            index_desc.requires = None

        queued = set()
        while todo:
            current_todo = []
            next_todo = []
            for index_desc in todo:
                results, requires = exec_op.execute(results, dry_run=False)
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
    exec_op = benchmark.create_execution_operator(
        train=train,
        build=knn,
        knn=knn,
        reconstruct=reconstruct,
        range=range,
    )
    results, requires = exec_op.execute(results=results, dry_run=False)
    assert requires is None
    assert results is not None
    return results
