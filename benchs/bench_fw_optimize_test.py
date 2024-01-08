# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from bench_fw.benchmark_io import BenchmarkIO
from bench_fw.descriptors import DatasetDescriptor
from bench_fw.optimize import Optimizer

logging.basicConfig(level=logging.INFO)

optimizer = Optimizer(
    distance_metric="L2",
)
io = BenchmarkIO(
    path="/checkpoint/gsz/bench_fw/ivf",
)
optimizer.set_io(io)
training_vectors = DatasetDescriptor(
    namespace="std_t", tablename="bigann1M"
)
xt = io.get_dataset(training_vectors)
scale = 10_000_000
index_descs = optimizer.optimize(
    d=xt.shape[1],
    scale=scale,
    training_vectors=training_vectors,
    database_vectors=DatasetDescriptor(
        namespace="std_d", tablename=f"bigann10M"
    ),
    query_vectors=DatasetDescriptor(
        namespace="std_q", tablename="bigann1M"
    ),
)
print(index_descs)
