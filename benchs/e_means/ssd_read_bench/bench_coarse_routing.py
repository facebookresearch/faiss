# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare reader routing with duplicated owner routing on frozen IVF batches.

Run with torchrun. Only this benchmark overrides the gather hook to recreate
the old query routing. SSD reads, initialization, and EMA updates are excluded.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.distributed as dist

from common import source_for
from faiss.contrib.e_means import EmeansShardedCentroids, EmeansShardedCentroidsIVF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/tmp/DINO10B")
    parser.add_argument("--n", type=int, default=1_000_000_000)
    parser.add_argument("--k", type=int, default=2_097_152)
    parser.add_argument("--batch-size", type=int, default=2_097_152)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank, workers = dist.get_rank(), dist.get_world_size()
    source = source_for(args, rank, workers)
    try:
        local_k, local_b = args.k // workers, args.batch_size // workers
        initial = torch.from_numpy(source.read(0, local_k)).to(device).float()
        initial.sub_(128).div_(64)
        rows = torch.from_numpy(source.read(local_k, local_k + local_b)).to(device).float()
        rows.sub_(128).div_(64)
        model = EmeansShardedCentroidsIVF(
            1024, args.k, group=dist.group.WORLD, niter=1,
            batch_size=args.batch_size, alpha0=.3, nlist=4096, nprobe=4,
            capture_assignments=True, assignment_backend="triton",
        )
        model.begin(args.n, initial)
        del initial
        hooks = {
            "owner": EmeansShardedCentroids._gather_assignment_rows,
            "reader": EmeansShardedCentroidsIVF._gather_assignment_rows,
        }
        for hook in hooks.values():
            model._gather_assignment_rows = hook.__get__(model)
            model.accumulate(rows)
            model._pending_stats = None
        baseline_labels = baseline_counts = None
        reports = []
        gather_name = "all_gather_single" if hasattr(dist, "all_gather_single") else "all_gather_into_tensor"
        real_gather, real_route = getattr(dist, gather_name), model._ivf.route
        for turn in range(args.repeats):
            for routing in ("owner", "reader") if turn % 2 == 0 else ("reader", "owner"):
                model._gather_assignment_rows = hooks[routing].__get__(model)
                events, route_rows = [], []

                def timed_route(x):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = real_route(x)
                    end.record()
                    events.append((start, end))
                    route_rows.append(len(x))
                    return result

                dist.barrier()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                with patch.object(dist, gather_name, wraps=real_gather) as gather, \
                     patch.object(model._ivf, "route", side_effect=timed_route) as route_call:
                    stats = model.accumulate(rows)
                    torch.cuda.synchronize(device)
                    gathers = gather.call_count
                    # Mock call histories hold the full gathered tensor;
                    # clear them so subsequent trials measure live memory.
                    gather.reset_mock()
                    route_call.reset_mock()
                seconds = time.perf_counter() - started
                coarse_s = sum(a.elapsed_time(b) for a, b in events) / 1000
                assert sum(route_rows) == (local_b if routing == "reader" else args.batch_size)
                assert gathers == (2 if routing == "reader" else 1)
                if baseline_labels is None:
                    baseline_labels, baseline_counts = stats["labels"].clone(), stats["counts"].clone()
                else:
                    torch.testing.assert_close(stats["labels"], baseline_labels, rtol=0, atol=0)
                    torch.testing.assert_close(stats["counts"], baseline_counts, rtol=0, atol=0)
                local = dict(rank=rank, routing=routing, turn=turn,
                             seconds=seconds, coarse_seconds=coarse_s,
                             routed_rows=sum(route_rows), gathers=gathers,
                             peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
                ranks = [None] * workers
                dist.all_gather_object(ranks, local)
                report = dict(routing=routing, turn=turn,
                              max_seconds=max(r["seconds"] for r in ranks),
                              max_coarse_seconds=max(r["coarse_seconds"] for r in ranks),
                              ranks=ranks)
                reports.append(report)
                if rank == 0:
                    print(json.dumps(report), flush=True)
                model._pending_stats = None
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(
                n=args.n, k=args.k, workers=workers, batch_size=args.batch_size,
                nlist=4096, nprobe=4, labels_and_counts_identical=True,
                frozen_centroids=True, excludes_ssd=True, reports=reports,
            ), indent=2) + "\n")
    finally:
        source.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
