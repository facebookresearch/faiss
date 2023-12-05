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
        namespace="std_d", tablename="sift1M"
    ),
    database_vectors=DatasetDescriptor(
        namespace="std_d", tablename="sift1M"
    ),
    query_vectors=DatasetDescriptor(
        namespace="std_q", tablename="sift1M"
    ),
    index_descs=[
        IndexDescriptor(
            factory=f"IVF{2 ** nlist},Flat",
        )
        for nlist in range(8, 15)
    ],
    k=1,
    distance_metric="L2",
)
io = BenchmarkIO(
    path="/checkpoint",
)
benchmark.set_io(io)
print(benchmark.benchmark("result.json"))
