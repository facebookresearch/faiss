# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark exhaustive evaluation of 100K DINO queries and frozen shards.

Run with torchrun. Centroids are sampled contiguous source ranges, rather
than trained outputs. Timings include mirror preparation and query transfers,
and exclude source centroid loading and the first assignment compilation.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dino_data import DinoQueries, query_mse
from common import source_for
from faiss.contrib.e_means import EmeansShardedCentroids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/tmp/DINO10B")
    parser.add_argument("--n", type=int, default=1_000_000_000)
    parser.add_argument("--k", type=int, default=2_097_152)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1024, 8192, 32768])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fp32-backend", choices=("native", "triton"), default="triton")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank, workers = dist.get_rank(), dist.get_world_size()
    source = source_for(args, rank, workers)
    try:
        if args.n % workers or args.k % workers:
            raise ValueError("n and k must divide by worker count")
        model = EmeansShardedCentroids(
            1024, args.k, group=dist.group.WORLD, assignment_backend=args.fp32_backend,
        )
        model.centroids = torch.from_numpy(source.read(0, args.k // workers)).to(
            device=device, dtype=torch.float32,
        )
        mean, scale = torch.full((1024,), 128., device=device), 64.
        model.centroids.sub_(mean).div_(scale)
        queries = DinoQueries(args.data, 1024)
        reports = []
        configs = [("fp32", args.batch_sizes[0])] + [
            ("bf16", size) for size in args.batch_sizes
        ]
        for precision, batch_size in configs:
            warm = next(queries.batches(mean, scale, batch_size))
            query_mse(model, [warm], precision=precision)
            del warm
            for turn in range(1 if precision == "fp32" else args.repeats):
                dist.barrier()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                mse = query_mse(
                    model, queries.batches(mean, scale, batch_size), precision=precision,
                ) * scale**2
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                timing = torch.tensor(
                    [seconds, torch.cuda.max_memory_allocated(device)],
                    dtype=torch.float64, device=device,
                )
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                report = dict(precision=precision, batch_size=batch_size, turn=turn,
                              seconds=timing[0].item(), mse=mse,
                              peak_gpu_bytes=int(timing[1].item()))
                reports.append(report)
                if rank == 0:
                    print(json.dumps(report), flush=True)
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(
                n=args.n, k=args.k, d=1024, workers=workers,
                queries=queries.n, frozen_source_centroids=True,
                fp32_backend=args.fp32_backend,
                reports=reports,
            ), indent=2) + "\n")
    finally:
        source.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
