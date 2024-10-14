# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss  # @manual=//faiss/python:pyfaiss

# from faiss.contrib.evaluation import (  # @manual=//faiss/contrib:faiss_contrib
#     OperatingPoints,
# )

from .benchmark import Benchmark
from .descriptors import DatasetDescriptor, IndexDescriptorClassic
from .utils import dict_merge, filter_results, ParetoMetric, ParetoMode

logger = logging.getLogger(__name__)


@dataclass
class Optimizer:
    distance_metric: str = "L2"
    num_threads: int = 32
    run_local: bool = True

    def __post_init__(self):
        self.cached_benchmark = None
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

    def benchmark_and_filter_candidates(
        self,
        index_descs,
        training_vectors,
        database_vectors,
        query_vectors,
        result_file,
        include_flat,
        min_accuracy,
        pareto_metric,
    ):
        benchmark = Benchmark(
            num_threads=self.num_threads,
            training_vectors=training_vectors,
            database_vectors=database_vectors,
            query_vectors=query_vectors,
            index_descs=index_descs,
            k=10,
            distance_metric=self.distance_metric,
        )
        benchmark.set_io(self.io)
        results = benchmark.benchmark(
            result_file=result_file, local=self.run_local, train=True, knn=True
        )
        assert results
        filtered = filter_results(
            results=results,
            evaluation="knn",
            accuracy_metric="knn_intersection",
            min_accuracy=min_accuracy,
            name_filter=None
            if include_flat
            else (lambda n: not n.startswith("Flat")),
            pareto_mode=ParetoMode.GLOBAL,
            pareto_metric=pareto_metric,
        )
        assert filtered
        index_descs = [
            IndexDescriptorClassic(
                factory=v["factory"],
                construction_params=v["construction_params"],
                search_params=v["search_params"],
            )
            for _, _, _, _, v in filtered
        ]
        return index_descs, filtered

    def optimize_quantizer(
        self,
        training_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        nlists: List[int],
        min_accuracy: float,
    ):
        quantizer_descs = {}
        for nlist in nlists:
            # cluster
            centroids, _, _ = training_vectors.k_means(
                self.io,
                nlist,
                dry_run=False,
            )

            descs = [IndexDescriptorClassic(factory="Flat"),] + [
                IndexDescriptorClassic(
                    factory="HNSW32",
                    construction_params=[{"efConstruction": 2**i}],
                )
                for i in range(6, 11)
            ]

            descs, _ = self.benchmark_and_filter_candidates(
                descs,
                training_vectors=centroids,
                database_vectors=centroids,
                query_vectors=query_vectors,
                result_file=f"result_{centroids.get_filename()}json",
                include_flat=True,
                min_accuracy=min_accuracy,
                pareto_metric=ParetoMetric.TIME,
            )
            quantizer_descs[nlist] = descs

        return quantizer_descs

    def optimize_ivf(
        self,
        result_file: str,
        training_vectors: DatasetDescriptor,
        database_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        quantizers: Dict[int, List[IndexDescriptorClassic]],
        codecs: List[Tuple[str, str]],
        min_accuracy: float,
    ):
        ivf_descs = []
        for nlist, quantizer_descs in quantizers.items():
            # build IVF index
            for quantizer_desc in quantizer_descs:
                for pretransform, fine_ivf in codecs:
                    if pretransform is None:
                        pretransform = ""
                    else:
                        pretransform = pretransform + ","
                    if quantizer_desc.construction_params is None:
                        construction_params = [
                            None,
                            quantizer_desc.search_params,
                        ]
                    else:
                        construction_params = [
                            None
                        ] + quantizer_desc.construction_params
                        if quantizer_desc.search_params is not None:
                            dict_merge(
                                construction_params[1],
                                quantizer_desc.search_params,
                            )
                    ivf_descs.append(
                        IndexDescriptorClassic(
                            factory=f"{pretransform}IVF{nlist}({quantizer_desc.factory}),{fine_ivf}",
                            construction_params=construction_params,
                        )
                    )
        return self.benchmark_and_filter_candidates(
            ivf_descs,
            training_vectors,
            database_vectors,
            query_vectors,
            result_file,
            include_flat=False,
            min_accuracy=min_accuracy,
            pareto_metric=ParetoMetric.TIME_SPACE,
        )

    # train an IVFFlat index
    # find the nprobe required for the given accuracy
    def ivf_flat_nprobe_required_for_accuracy(
        self,
        result_file: str,
        training_vectors: DatasetDescriptor,
        database_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        nlist,
        accuracy,
    ):
        _, results = self.benchmark_and_filter_candidates(
            index_descs=[
                IndexDescriptorClassic(factory=f"IVF{nlist}(Flat),Flat"),
            ],
            training_vectors=training_vectors,
            database_vectors=database_vectors,
            query_vectors=query_vectors,
            result_file=result_file,
            include_flat=False,
            min_accuracy=accuracy,
            pareto_metric=ParetoMetric.TIME,
        )
        nprobe = nlist // 2
        for _, _, _, k, v in results:
            if (
                ".knn" in k
                and "nprobe" in v["search_params"]
                and v["knn_intersection"] >= accuracy
            ):
                nprobe = min(nprobe, v["search_params"]["nprobe"])
        return nprobe

    # train candidate IVF codecs
    # benchmark them at the same nprobe
    # keep only the space _and_ time Pareto optimal
    def optimize_codec(
        self,
        result_file: str,
        d: int,
        training_vectors: DatasetDescriptor,
        database_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        nlist: int,
        nprobe: int,
        min_accuracy: float,
    ):
        codecs = (
            [
                (None, "Flat"),
                (None, "SQfp16"),
                (None, "SQbf16"),
                (None, "SQ8"),
                (None, "SQ8_direct_signed"),
            ] + [
                (f"OPQ{M}_{M * dim}", f"PQ{M}x{b}")
                for M in [8, 12, 16, 32, 48, 64, 96, 128, 192, 256]
                if d % M == 0
                for dim in range(2, 18, 2)
                if M * dim <= d
                for b in range(4, 14, 2)
                if M * b < d * 8  # smaller than SQ8
            ] + [
                (None, f"PQ{M}x{b}")
                for M in [8, 12, 16, 32, 48, 64, 96, 128, 192, 256]
                if d % M == 0
                for b in range(8, 14, 2)
                if M * b < d * 8  # smaller than SQ8
            ]
        )
        factory = {}
        for opq, pq in codecs:
            factory[
                f"IVF{nlist},{pq}" if opq is None else f"{opq},IVF{nlist},{pq}"
            ] = (
                opq,
                pq,
            )

        _, filtered = self.benchmark_and_filter_candidates(
            index_descs=[
                IndexDescriptorClassic(
                    factory=f"IVF{nlist},{pq}"
                    if opq is None
                    else f"{opq},IVF{nlist},{pq}",
                    search_params={
                        "nprobe": nprobe,
                    },
                )
                for opq, pq in codecs
            ],
            training_vectors=training_vectors,
            database_vectors=database_vectors,
            query_vectors=query_vectors,
            result_file=result_file,
            include_flat=False,
            min_accuracy=min_accuracy,
            pareto_metric=ParetoMetric.TIME_SPACE,
        )
        results = [
            factory[r] for r in set(v["factory"] for _, _, _, k, v in filtered)
        ]
        return results

    def optimize(
        self,
        d: int,
        training_vectors: DatasetDescriptor,
        database_vectors_list: List[DatasetDescriptor],
        query_vectors: DatasetDescriptor,
        min_accuracy: float,
    ):
        # train an IVFFlat index
        # find the nprobe required for near perfect accuracy
        nlist = 4096
        nprobe_at_95 = self.ivf_flat_nprobe_required_for_accuracy(
            result_file=f"result_ivf{nlist}_flat.json",
            training_vectors=training_vectors,
            database_vectors=database_vectors_list[0],
            query_vectors=query_vectors,
            nlist=nlist,
            accuracy=0.95,
        )

        # train candidate IVF codecs
        # benchmark them at the same nprobe
        # keep only the space and time Pareto optima
        codecs = self.optimize_codec(
            result_file=f"result_ivf{nlist}_codec.json",
            d=d,
            training_vectors=training_vectors,
            database_vectors=database_vectors_list[0],
            query_vectors=query_vectors,
            nlist=nlist,
            nprobe=nprobe_at_95,
            min_accuracy=min_accuracy,
        )

        # optimize coarse quantizers
        quantizers = self.optimize_quantizer(
            training_vectors=training_vectors,
            query_vectors=query_vectors,
            nlists=[4096, 8192, 16384, 32768],
            min_accuracy=0.7,
        )

        # combine them with the codecs
        # test them at different scales
        for database_vectors in database_vectors_list:
            self.optimize_ivf(
                result_file=f"result_{database_vectors.get_filename()}json",
                training_vectors=training_vectors,
                database_vectors=database_vectors,
                query_vectors=query_vectors,
                quantizers=quantizers,
                codecs=codecs,
                min_accuracy=min_accuracy,
            )
