# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from bench_fw.benchmark import Benchmark
from bench_fw.benchmark_io import BenchmarkIO
from bench_fw.descriptors import DatasetDescriptor, IndexDescriptor

logging.basicConfig(level=logging.INFO)

benchmark = Benchmark(
    training_vectors=DatasetDescriptor(
        tablename="training.npy", num_vectors=200000
    ),
    database_vectors=DatasetDescriptor(
        tablename="database.npy", num_vectors=200000
    ),
    query_vectors=DatasetDescriptor(tablename="query.npy", num_vectors=2000),
    index_descs=[
        IndexDescriptor(
            factory="Flat",
            range_metrics={
                "weighted": [
                    [0.1, 0.928],
                    [0.2, 0.865],
                    [0.3, 0.788],
                    [0.4, 0.689],
                    [0.5, 0.49],
                    [0.6, 0.308],
                    [0.7, 0.193],
                    [0.8, 0.0],
                ]
            },
        ),
        IndexDescriptor(
            factory="OPQ32_128,IVF512,PQ32",
        ),
        IndexDescriptor(
            factory="OPQ32_256,IVF512,PQ32",
        ),
        IndexDescriptor(
            factory="HNSW32",
            construction_params=[
                {
                    "efConstruction": 64,
                }
            ],
        ),
    ],
    k=10,
    distance_metric="L2",
    range_ref_index_desc="Flat",
)
io = BenchmarkIO(
    path="/checkpoint",
)
benchmark.set_io(io)
print(benchmark.benchmark("result.json"))
