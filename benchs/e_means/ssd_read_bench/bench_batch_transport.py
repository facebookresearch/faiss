"""Time frozen all-gather IVF batch accumulation at production dimensions.

SSD reads/initialization are excluded. No centroid update or full training run
is performed. Uses real DINO vectors and identical fixed centroid shards.
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
from faiss.contrib.e_means import EmeansShardedCentroidsIVF


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='/tmp/DINO10B')
    p.add_argument('--n', type=int, default=1_000_000_000)
    p.add_argument('--k', type=int, default=2_097_152)
    p.add_argument('--batch-size', type=int, default=2_097_152)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
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
            capture_assignments=True,
            assignment_backend="triton",
        )
        model.begin(args.n, initial)
        del initial
        # Warm coarse training and assignment, without EMA updates.
        stats = model.accumulate(rows)
        model._pending_stats = None
        del stats
        reports = []
        gather_name = 'all_gather_single' if hasattr(dist, 'all_gather_single') else 'all_gather_into_tensor'
        real_gather, real_reduce, real_broadcast = getattr(dist, gather_name), dist.all_reduce, dist.broadcast
        for turn in range(args.repeats):
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with patch.object(dist, gather_name, wraps=real_gather) as gather, \
                 patch.object(dist, 'all_reduce', wraps=real_reduce) as reduce, \
                 patch.object(dist, 'broadcast', wraps=real_broadcast) as broadcast:
                stats = model.accumulate(rows)
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            local = dict(rank=rank, transport='all_gather', turn=turn, elapsed_s=elapsed,
                         gathers=gather.call_count, reductions=reduce.call_count,
                         broadcasts=broadcast.call_count,
                         peak_gpu_bytes=torch.cuda.max_memory_allocated())
            assert gather.call_count == 1 and broadcast.call_count == 0
            ranks = [None] * workers
            dist.all_gather_object(ranks, local)
            report = dict(transport='all_gather', turn=turn,
                          max_elapsed_s=max(x['elapsed_s'] for x in ranks), ranks=ranks)
            reports.append(report)
            if rank == 0:
                print(json.dumps(report), flush=True)
            model._pending_stats = None
            del stats
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(n=args.n, k=args.k,
                batch_size=args.batch_size, workers=workers, nlist=4096, nprobe=4,
                assignment_chunk_size=model.recipe['assignment_chunk_size'],
                excludes_ssd=True, frozen_centroids=True, reports=reports), indent=2) + '\n')
    finally:
        source.close()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
