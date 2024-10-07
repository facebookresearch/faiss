# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
import os

from faiss.benchs.bench_fw.benchmark import Benchmark
from faiss.benchs.bench_fw.benchmark_io import BenchmarkIO
from faiss.benchs.bench_fw.descriptors import DatasetDescriptor, IndexDescriptorClassic
from faiss.benchs.bench_fw.index import IndexFromFactory

logging.basicConfig(level=logging.INFO)

def factory_factory(d):
    return [
        ("SQ4", None, 256 * (2 ** 10), None),
        ("SQ8", None, 256 * (2 ** 10), None),
        ("SQfp16", None, 256 * (2 ** 10), None),
        ("ITQ64,LSH", None, 256 * (2 ** 10), None),
        ("Pad128,ITQ128,LSH", None, 256 * (2 ** 10), None),
        ("Pad256,ITQ256,LSH", None, 256 * (2 ** 10), None),
    ] + [
        (f"OPQ32_128,Residual2x14,PQ32x{b}", None, 256 * (2 ** 14), None)
        for b in range(8, 16, 2)
    ] + [
        (f"PCAR{2 ** d_out},SQ{b}", None, 256 * (2 ** 10), None)
        for d_out in range(6, 11) 
        if 2 ** d_out <= d
        for b in [4, 8]
    ] + [
        (f"OPQ{M}_{M * dim},PQ{M}x{b}", None, 256 * (2 ** b), None)
        for M in [8, 12, 16, 32, 64, 128]
        for dim in [2, 4, 6, 8, 12, 16]
        if M * dim <= d
        for b in range(8, 16, 2)
    ] + [
        (f"RQ{cs // b}x{b}", [{"max_beam_size": 32}], 256 * (2 ** b), {"max_beam_size": bs, "use_beam_LUT": bl}) 
        for cs in [64, 128, 256, 512]
        for b in [6, 8, 10, 12]
        for bs in [1, 2, 4, 8, 16, 32]
        for bl in [0, 1]
        if cs // b > 1
        if cs // b < 65
        if cs < d * 8 * 2
    ] + [
        (f"LSQ{cs // b}x{b}", [{"encode_ils_iters": 16}], 256 * (2 ** b), {"encode_ils_iters": eii, "lsq_gpu": lg}) 
        for cs in [64, 128, 256, 512]
        for b in [6, 8, 10, 12]
        for eii in [2, 4, 8, 16]
        for lg in [0, 1]
        if cs // b > 1
        if cs // b < 65
        if cs < d * 8 * 2
    ] + [
        (f"PRQ{sub}x{cs // sub // b}x{b}", [{"max_beam_size": 32}], 256 * (2 ** b), {"max_beam_size": bs, "use_beam_LUT": bl})
        for sub in [2, 3, 4, 8, 16, 32]
        for cs in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]
        for b in [6, 8, 10, 12]
        for bs in [1, 2, 4, 8, 16, 32]
        for bl in [0, 1]
        if cs // sub // b > 1
        if cs // sub // b < 65
        if cs < d * 8 * 2
        if d % sub == 0
    ] + [
        (f"PLSQ{sub}x{cs // sub // b}x{b}", [{"encode_ils_iters": 16}], 256 * (2 ** b), {"encode_ils_iters": eii, "lsq_gpu": lg}) 
        for sub in [2, 3, 4, 8, 16, 32]
        for cs in [64, 128, 256, 512, 1024, 2048]
        for b in [6, 8, 10, 12]
        for eii in [2, 4, 8, 16]
        for lg in [0, 1]
        if cs // sub // b > 1
        if cs // sub // b < 65
        if cs < d * 8 * 2
        if d % sub == 0
    ]

def run_local(rp):
    bio, d, tablename, distance_metric = rp
    if tablename == "contriever":
        training_vectors=DatasetDescriptor(
            tablename="training_set.npy"
        )
        database_vectors=DatasetDescriptor(
            tablename="database1M.npy",
        )
        query_vectors=DatasetDescriptor(
            tablename="queries.npy",
        )
    else:
        training_vectors=DatasetDescriptor(
            namespace="std_t", tablename=tablename,
        )
        database_vectors=DatasetDescriptor(
            namespace="std_d", tablename=tablename,
        )
        query_vectors=DatasetDescriptor(
            namespace="std_q", tablename=tablename,
        )

    benchmark = Benchmark(
        num_threads=32,
        training_vectors=training_vectors,
        database_vectors=database_vectors,
        query_vectors=query_vectors,
        index_descs=[
            IndexDescriptorClassic(
                factory=factory,
                construction_params=construction_params,
                training_size=training_size,
                search_params=search_params,
            )
            for factory, construction_params, training_size, search_params in factory_factory(d)
        ],
        k=1,
        distance_metric=distance_metric,
    )
    benchmark.set_io(bio)
    benchmark.benchmark(result_file="result.json", train=True, reconstruct=False, knn=False, range=False)

def run(bio, d, tablename, distance_metric):
    bio.launch_jobs(run_local, [(bio, d, tablename, distance_metric)], local=True)

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
        run(bio, 128, "sift1M", "L2")
    elif args.experiment == "bigann":
        run(bio, 128, "bigann1M", "L2")
    elif args.experiment == "deep1b":
        run(bio, 96, "deep1M", "L2")
    elif args.experiment == "contriever":
        run(bio, 768, "contriever", "IP")
