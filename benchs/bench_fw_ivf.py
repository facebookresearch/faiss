# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

from faiss.benchs.bench_fw.benchmark import Benchmark
from faiss.benchs.bench_fw.benchmark_io import BenchmarkIO
from faiss.benchs.bench_fw.descriptors import (
    DatasetDescriptor,
    IndexDescriptorClassic,
)

logging.basicConfig(level=logging.INFO)


def sift1M(bio):
    benchmark = Benchmark(
        num_threads=32,
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
            IndexDescriptorClassic(
                factory=f"IVF{2 ** nlist},Flat",
            )
            for nlist in range(8, 15)
        ],
        k=1,
        distance_metric="L2",
    )
    benchmark.io = bio
    benchmark.benchmark(result_file="result.json", local=True, train=True, reconstruct=False, knn=True, range=False)


def bigann(bio):
    for scale in [1, 2, 5, 10, 20, 50]:
        benchmark = Benchmark(
            num_threads=32,
            training_vectors=DatasetDescriptor(
                namespace="std_t", tablename="bigann1M"
            ),
            database_vectors=DatasetDescriptor(
                namespace="std_d", tablename=f"bigann{scale}M"
            ),
            query_vectors=DatasetDescriptor(
                namespace="std_q", tablename="bigann1M"
            ),
            index_descs=[
                IndexDescriptorClassic(
                    factory=f"IVF{2 ** nlist},Flat",
                ) for nlist in range(11, 19)
            ] + [
                IndexDescriptorClassic(
                    factory=f"IVF{2 ** nlist}_HNSW32,Flat",
                    construction_params=[None, {"efConstruction": 200, "efSearch": 40}],
                ) for nlist in range(11, 19)
            ],
            k=1,
            distance_metric="L2",
        )
        benchmark.set_io(bio)
        benchmark.benchmark(f"result{scale}.json", local=False, train=True, reconstruct=False, knn=True, range=False)

def ssnpp(bio):
    benchmark = Benchmark(
        num_threads=32,
        training_vectors=DatasetDescriptor(
            tablename="ssnpp_training_5M.npy"
        ),
        database_vectors=DatasetDescriptor(
            tablename="ssnpp_database_5M.npy"
        ),
        query_vectors=DatasetDescriptor(
            tablename="ssnpp_queries_10K.npy"
        ),
        index_descs=[
            IndexDescriptorClassic(
                factory=f"IVF{2 ** nlist},PQ256x4fs,Refine(SQfp16)",
            ) for nlist in range(9, 16)
        ] + [
            IndexDescriptorClassic(
                factory=f"IVF{2 ** nlist},Flat",
            ) for nlist in range(9, 16)
        ] + [
            IndexDescriptorClassic(
                factory=f"PQ256x4fs,Refine(SQfp16)",
            ),
            IndexDescriptorClassic(
                factory=f"HNSW32",
            ),
        ],
        k=1,
        distance_metric="L2",
    )
    benchmark.set_io(bio)
    benchmark.benchmark("result.json", local=False, train=True, reconstruct=False, knn=True, range=False)

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
    if args.experiment == "sift1M":
        sift1M(bio)
    elif args.experiment == "bigann":
        bigann(bio)
    elif args.experiment == "ssnpp":
        ssnpp(bio)
