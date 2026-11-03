"""CPU-only comparison of the real SSD loading functions; launch with torchrun."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ssd_streaming import SSDPrefetch, SSDRowSource, chunk_grouped_order


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000_000)
    parser.add_argument("--data", default="/tmp/DINO10B")
    parser.add_argument("--scratch", default="/scratch/train-cache/ssd-bench-1b-20261103")
    parser.add_argument("--reference", default="/storage/home/matthijs/src/online_k-means")
    parser.add_argument("--mode", choices=("sequential", "shuffled"), default="shuffled")
    parser.add_argument("--window-mib", type=int, default=51200)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--read-threads", type=int, default=1)
    parser.add_argument("--extract-threads", type=int, default=4)
    parser.add_argument("--chunk-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for name in ("n", "window_mib", "chunk_mib", "read_threads", "extract_threads", "chunk_groups"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    return args


def source_for(args, rank, workers):
    paths = sorted((Path(args.data) / "chunked_base_10B").glob("*.bvecs"))
    n = args.n // workers
    return SSDRowSource(paths, 1024, rank * n, (rank + 1) * n, header_bytes=4)


def disk_bytes_read():
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    raise RuntimeError("kernel read counters are unavailable")


def benchmark(implementation):
    args = arguments()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(hours=1))
    rank, workers = dist.get_rank(), dist.get_world_size()
    if args.n % workers:
        raise ValueError("n must divide by worker count")
    started = time.perf_counter()
    source = source_for(args, rank, workers)
    staged = None
    try:
        if not all(source.direct):
            raise RuntimeError("the DINO source must support O_DIRECT")
        n, d = source.n, source.d
        window_rows = min(n, (args.window_mib << 20) // d)
        if implementation == "reference":
            sys.path.insert(0, args.reference)
            import ssd_prefetch as reference
            path = Path(args.scratch) / f"rank-{rank}.u8"
            if path.stat().st_size != n * d:
                raise ValueError(f"incorrect staged shard size: {path}")
            staged = reference.StagedFile(str(path), d)
            if not staged.direct:
                raise RuntimeError("the reference shard must support O_DIRECT")
            rows_per_chunk = (args.chunk_mib << 20) // d
            counters = {}
        else:
            # These functions are CPU-only; bypass CUDA-stream setup in __init__.
            loader = object.__new__(SSDPrefetch)
            loader.source = source
            loader.rows_per_chunk = (args.chunk_mib << 20) // source.row_bytes
            loader.read_threads, loader.extract_threads = args.read_threads, args.extract_threads
            loader.stats = {"bytes_read": 0, "windows": 0}
            rows_per_chunk = loader.rows_per_chunk
            counters = loader.stats
        order = None
        if args.mode == "shuffled":
            # Use the same distribution and window sizes, respecting each file layout.
            order = chunk_grouped_order(n, rows_per_chunk, args.chunk_groups,
                                        window_rows, args.seed + rank)
        out = torch.empty((window_rows, d), dtype=torch.uint8)
        out.zero_()  # Exclude allocation/page faults from the reading benchmark.
        setup_seconds = time.perf_counter() - started
        dist.barrier()
        cache_status = [None]
        if rank == 0 and args.flush_cache:
            result = subprocess.run(["sudo", "-n", "sh", "-c",
                                     "sync; echo 3 > /proc/sys/vm/drop_caches"],
                                    capture_output=True, text=True)
            cache_status[0] = None if result.returncode == 0 else result.stderr
        dist.broadcast_object_list(cache_status, src=0)
        if cache_status[0] is not None:
            raise RuntimeError("cache flushing failed: " + cache_status[0])
        dist.barrier()
        if rank == 0:
            print(json.dumps(dict(implementation=implementation, mode=args.mode,
                                  rows=args.n, workers=workers, window_rows=window_rows,
                                  read_threads=args.read_threads, extract_threads=args.extract_threads,
                                  direct=True, linux_cache_flushed=args.flush_cache)), flush=True)
        window_seconds = []
        kernel_bytes = 0
        wall_start = time.perf_counter()
        for window, begin in enumerate(range(0, n, window_rows), start=1):
            end = min(n, begin + window_rows)
            request = (np.arange(begin, end, dtype=np.int64) if order is None else order[begin:end])
            disk_before = disk_bytes_read()
            before = time.perf_counter()
            if implementation == "reference":
                reference.fill_window([staged], [request], [out[:end - begin]],
                                      chunk_bytes=args.chunk_mib << 20,
                                      read_threads=args.read_threads,
                                      extract_threads=args.extract_threads, stats=counters)
            elif order is None:
                loader.fill_sequential(begin, end, out[:end - begin])
            else:
                loader.fill_window(request, out[:end - begin])
            elapsed = torch.tensor(time.perf_counter() - before, dtype=torch.float64)
            kernel_bytes += disk_bytes_read() - disk_before
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            window_seconds.append(elapsed.item())
            # Validate dispersed rows against the original DINO payloads.
            for position in np.linspace(0, end - begin - 1, 16, dtype=np.int64):
                row = int(request[position])
                np.testing.assert_array_equal(out[position].numpy(), source.read(row, row + 1)[0])
            dist.barrier()
            if rank == 0:
                print(f"window={window} rows_per_rank={end - begin} read_seconds={elapsed.item():.3f}", flush=True)
        wall_seconds = time.perf_counter() - wall_start
        local = dict(rank=rank, setup_seconds=setup_seconds, counters=counters,
                     kernel_bytes_read=kernel_bytes,
                     validated_rows=16 * len(window_seconds))
        ranks = [None] * workers
        dist.all_gather_object(ranks, local)
        if rank == 0:
            read_seconds = sum(window_seconds)
            bytes_read = sum(r["counters"]["bytes_read"] for r in ranks)
            kernel_bytes = sum(r["kernel_bytes_read"] for r in ranks)
            report = dict(implementation=implementation, mode=args.mode,
                          n=args.n, workers=workers, window_rows=window_rows,
                          read_threads=args.read_threads, extract_threads=args.extract_threads,
                          direct=True, linux_cache_flushed=args.flush_cache,
                          read_seconds=read_seconds, wall_seconds=wall_seconds,
                          payload_bytes=args.n * d, bytes_read=bytes_read,
                          kernel_bytes_read=kernel_bytes,
                          kernel_disk_GB_per_second=kernel_bytes / read_seconds / 1e9,
                          disk_GB_per_second=bytes_read / read_seconds / 1e9,
                          payload_GB_per_second=args.n * d / read_seconds / 1e9,
                          window_seconds=window_seconds, ranks=ranks)
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print("RESULT " + json.dumps(report), flush=True)
    finally:
        if staged is not None:
            staged.close()
        source.close()
        dist.destroy_process_group()
