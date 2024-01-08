# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
import logging
import math
from dataclasses import dataclass

import faiss  # @manual=//faiss/python:pyfaiss_gpu

from faiss.contrib.evaluation import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    OperatingPoints,
)

from .benchmark import Benchmark
from .descriptors import DatasetDescriptor, IndexDescriptor
from .index import IndexFromFactory

logger = logging.getLogger(__name__)


@dataclass
class Optimizer:
    distance_metric: str = "L2"

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

    def benchmark_candidates(
        self, index_descs, training_vectors, database_vectors, query_vectors, result_file
    ):
        benchmark = Benchmark(
            training_vectors=training_vectors,
            database_vectors=database_vectors,
            query_vectors=query_vectors,
            index_descs=index_descs,
            k=10,
            distance_metric=self.distance_metric,
        )
        benchmark.set_io(self.io)
        results = benchmark.benchmark(result_file)
        # results = set(v['index'] for k, v in results['experiments'].items() if ".knn" in k and v['knn_intersection'] > 0.8)

        op = OperatingPoints()
        for k, v in results["experiments"].items():
            if ".knn" in k:
                op.add_operating_point(v, v["knn_intersection"], v["time"])
                # op.add_operating_point(v, v['knn_intersection'], Cost((v['time'], results['indices'][v['codec']]['code_size'])))

        # results = set(v["factory"] for v, _, _ in op.operating_points)
        results = [
            IndexDescriptor(
                factory=v["factory"], construction_params=v["construction_params"], search_params=v["search_params"]
            )
            for v, _, _ in op.operating_points
        ]
        # breakpoint()
        return results

    def get_fine_factory(
        self, d: int, scale: int, hnsw: bool = True, refine: bool = False
    ):
        options_pq_bits = {
            4: [2, 4],
            6: [2, 4, 6, 8],
            8: [2, 4, 8, 16],
            # 12: [2, 4, 8, 16],
        }
        options_code_size_log2 = range(3, 16)
        options_with_opq = [False]  # [True, False]

        fs = []  # index descriptors
        ps = []  # pretransforms
        cs = []  # code sizes
        for factory, code_size in [
            ("Flat", d * 4),
            ("PQ64x4fs,Refine(SQfp16)", d * 2),
            # ("SQfp16", d * 2),
            # ("SQ8", d),
        ]:
            fs.append(IndexDescriptor(factory=factory))
            ps.append(None)
            cs.append(code_size)

        if hnsw and scale < 1_000_000:
            for factory, code_size in [
                ("HNSW32", d * 4 + 32 * 8),
                # ("HNSW64", d * 4 + 64 * 8),
            ]:
                fs.append(IndexDescriptor(factory=factory))
                ps.append(None)
                cs.append(code_size)
        # if scale < 16384:
        return fs, ps, cs
        for code_size_log2 in options_code_size_log2:
            code_size = 2**code_size_log2
            if code_size >= d:
                continue
            for pq_bits, dimensions_per_pq_bits in options_pq_bits.items():
                if code_size * 8 % pq_bits > 0:
                    continue
                M = int(code_size * 8 / pq_bits)
                for dppb in dimensions_per_pq_bits:
                    d_out = M * dppb
                    if d_out > d:
                        continue
                    for with_opq in options_with_opq:
                        if d_out != d and not with_opq:
                            continue
                        factory = ""
                        pt = None
                        if with_opq:
                            pt = f"OPQ{M}_{d_out}"
                            factory = pt + ","
                        ps.append(pt)
                        factory += f"PQ{M}x{pq_bits}"
                        if pq_bits == 4 and not refine:
                            factory += "fs"
                        # no refinement
                        fs.append(IndexDescriptor(factory=factory))
                        cs.append(code_size)
                        # adding refinement options on top of fastscan
                        if not refine and pq_bits == 4:
                            (
                                refiners,
                                _,
                                refiner_code_sizes,
                            ) = self.get_fine_factory(
                                d, scale, hnsw=False, refine=True
                            )
                            for refiner, refiner_code_size in zip(
                                refiners, refiner_code_sizes
                            ):
                                if refiner_code_size < code_size:
                                    continue
                                fs.append(
                                    IndexDescriptor(
                                        factory=f"{factory},Refine({refiner.factory})"
                                    )
                                )
                                cs.append(code_size + refiner_code_size)

        return fs, ps, cs

    def optimize(
        self,
        d: int,
        scale: int,
        training_vectors: DatasetDescriptor,
        database_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
    ):
        fine_descs, _, _ = [
            IndexDescriptor(factory="Flat"),
            IndexDescriptor(factory="PQ256x4fs,Refine(SQfp16)"),
            IndexDescriptor(factory="HNSW32"),
        ], None, None # self.get_fine_factory(d, scale)
        nlist_log2_min = max(math.ceil(math.log2(math.sqrt(scale))), 10)
        nlist_log2_max = min(math.floor(math.log2(scale / 50)), 19) + 1
        if nlist_log2_min >= nlist_log2_max:
            return self.benchmark_candidates(
                fine_descs, training_vectors, database_vectors, query_vectors, f"result_{d}_{scale}.json"
            )
        ivf_descs = []
        # fine_ivf_descs, pretransforms, _ = [IndexDescriptor(factory="Flat")], [None], None
        # if scale < 1_000_000:
        fine_ivf_descs = [IndexDescriptor(factory="PQ256x4fs,Refine(SQfp16)")]
        pretransforms = [None]
        # self.get_fine_factory(
        #    d, scale, hnsw=False
        #)
        for nlist_log2 in range(nlist_log2_min, nlist_log2_max):
            nlist = 2**nlist_log2
            done = set()
            for pt in pretransforms:
                if pt in done:
                    continue
                done.add(pt)

                # pretransform
                if pt is None:
                    transformed_training_vectors = training_vectors
                    transformed_query_vectors = query_vectors
                else:
                    pretransform_index = IndexFromFactory(pt + ",Flat")
                    pretransform_index.set_io(self.io)
                    transformed_training_vectors = (
                        pretransform_index.transform(training_vectors)
                    )
                    transformed_query_vectors = pretransform_index.transform(
                        query_vectors
                    )

                # cluster
                centroids = transformed_training_vectors.k_means(
                    self.io, nlist
                )
                d = self.io.get_dataset(centroids).shape[1]

                # optimize quantizer
                quantizer_descs = self.optimize(
                    d=d,
                    scale=nlist,
                    training_vectors=centroids,
                    database_vectors=centroids,
                    query_vectors=transformed_query_vectors,
                )

                # build IVF index
                for quantizer_desc in quantizer_descs:
                    for fine_ivf_desc, pretransform in zip(
                        fine_ivf_descs, pretransforms
                    ):
                        if pretransform != pt:
                            continue
                        if pretransform is None:
                            pretransform = ""
                        else:
                            pretransform = pretransform + ","
                        if quantizer_desc.construction_params is None:
                            construction_params = [None, quantizer_desc.search_params]
                        else:
                            construction_params = [None] + copy(quantizer_desc.construction_params)
                            if construction_params[1] is None:
                                construction_params[1] = quantizer_desc.search_params
                            elif quantizer_desc.search_params is not None:
                                construction_params[1] += quantizer_desc.search_params
                            # breakpoint()
                        ivf_descs.append(
                            IndexDescriptor(
                                factory=f"{pretransform}IVF{nlist}({quantizer_desc.factory}),{fine_ivf_desc.factory}",
                                construction_params=construction_params,
                            )
                        )
        return self.benchmark_candidates(
            fine_descs + ivf_descs,
            training_vectors,
            database_vectors,
            query_vectors,
            f"result_{d}_{scale}.json",
        )
