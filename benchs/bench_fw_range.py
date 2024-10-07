# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

from faiss.benchs.bench_fw.benchmark import Benchmark
from faiss.benchs.bench_fw.benchmark_io import BenchmarkIO
from faiss.benchs.bench_fw.descriptors import DatasetDescriptor, IndexDescriptorClassic

logging.basicConfig(level=logging.INFO)


def ssnpp(bio):
    benchmark = Benchmark(
        num_threads=32,
        training_vectors=DatasetDescriptor(
            tablename="training.npy",
        ),
        database_vectors=DatasetDescriptor(
            tablename="database.npy",
        ),
        query_vectors=DatasetDescriptor(tablename="query.npy"),
        index_descs=[
            IndexDescriptorClassic(
                factory="Flat",
                range_metrics={
                    "weighted": [
                        [0.05, 0.971],
                        [0.1, 0.956],
                        [0.15, 0.923],
                        [0.2, 0.887],
                        [0.25, 0.801],
                        [0.3, 0.729], 
                        [0.35, 0.651], 
                        [0.4, 0.55], 
                        [0.45, 0.459], 
                        [0.5, 0.372], 
                        [0.55, 0.283], 
                        [0.6, 0.189], 
                        [0.65, 0.143], 
                        [0.7, 0.106], 
                        [0.75, 0.116], 
                        [0.8, 0.088], 
                        [0.85, 0.064],
                        [0.9, 0.05], 
                        [0.95, 0.04], 
                        [1.0, 0.028], 
                        [1.05, 0.02], 
                        [1.1, 0.013],
                        [1.15, 0.007], 
                        [1.2, 0.004], 
                        [1.3, 0],
                    ]
                },
            ),
            IndexDescriptorClassic(
                factory="IVF262144(PQ256x4fs),PQ32",
            ),
        ],
        k=10,
        distance_metric="L2",
        range_ref_index_desc="Flat",
    )
    benchmark.set_io(bio)
    benchmark.benchmark("result.json", local=False, train=True, reconstruct=False, knn=False, range=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('path')
    args = parser.parse_args()
    assert os.path.exists(args.path)
    path = os.path.join(args.path, args.experiment)
    if not os.path.exists(path):
        os.mkdir(path)
    bio = BenchmarkIO(
        path=path,
    )
    if args.experiment == "ssnpp":
        ssnpp(bio)
