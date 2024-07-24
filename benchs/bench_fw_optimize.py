# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

from faiss.benchs.bench_fw.benchmark_io import BenchmarkIO
from faiss.benchs.bench_fw.descriptors import DatasetDescriptor
from faiss.benchs.bench_fw.optimize import Optimizer

logging.basicConfig(level=logging.INFO)


def bigann(bio):
    optimizer = Optimizer(
        distance_metric="L2",
        num_threads=32,
        run_local=False,
    )
    optimizer.set_io(bio)
    query_vectors = DatasetDescriptor(namespace="std_q", tablename="bigann1M")
    xt = bio.get_dataset(query_vectors)
    optimizer.optimize(
        d=xt.shape[1],
        training_vectors=DatasetDescriptor(
            namespace="std_t",
            tablename="bigann1M",
            num_vectors=2_000_000,
        ),
        database_vectors_list=[
            DatasetDescriptor(
                namespace="std_d",
                tablename="bigann1M",
            ),
            DatasetDescriptor(namespace="std_d", tablename="bigann10M"),
        ],
        query_vectors=query_vectors,
        min_accuracy=0.85,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("path")
    args = parser.parse_args()
    assert os.path.exists(args.path)
    path = os.path.join(args.path, args.experiment)
    if not os.path.exists(path):
        os.mkdir(path)
    bio = BenchmarkIO(
        path=path,
    )
    if args.experiment == "bigann":
        bigann(bio)
