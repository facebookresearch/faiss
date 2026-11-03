"""Time production closing on DINO rows with synthetic saved labels, no training.

This isolates closing throughput. Uniform synthetic codes do not measure the
quality or imbalance of labels produced by a real training run.
"""
import argparse
import json
from pathlib import Path
import time
from contextlib import closing
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from common import source_for, SSDPrefetch, disk_bytes_read
from faiss.contrib.e_means import EmeansShardedCentroidsIVF


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='/tmp/DINO10B')
    p.add_argument('--n', type=int, default=1_000_000_000)
    p.add_argument('--k', type=int, default=2_097_152)
    p.add_argument('--batch-size', type=int, default=2_097_152)
    p.add_argument('--read-threads', type=int, default=4)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    import os
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    rank, workers = dist.get_rank(), dist.get_world_size()
    assert args.n % workers == args.k % workers == args.batch_size % workers == 0
    source = source_for(args, rank, workers)
    try:
        assert all(source.direct)
        codes = torch.from_numpy(np.random.RandomState(19 + rank).randint(
            0, args.k, source.n, dtype=np.int32))
        counts = torch.from_numpy(np.bincount(codes.numpy(), minlength=args.k)).to(device)
        dist.all_reduce(counts)
        model = EmeansShardedCentroidsIVF(
            1024, args.k, group=dist.group.WORLD, niter=1,
            batch_size=args.batch_size, alpha0=.3, nlist=4096, nprobe=4,
        )
        initial = torch.zeros((args.k // workers, 1024), device=device)
        model.begin(args.n, initial)
        del initial
        # Supply a completed lifecycle and exact counts for synthetic saved codes.
        # No assignment or training operation is included in this benchmark.
        model._epoch = len(model.recipe['epoch_rows'])
        model.iteration_stats = [{'obj': 0.}]
        model._pass_counts.copy_(counts[model.centroid_begin:model.centroid_begin + model.k])
        del counts
        def forbidden(*a, **kw):
            raise AssertionError('closing benchmark must not perform search')
        model._prepare_assignment = model._winning_labels = forbidden
        loader = SSDPrefetch(source, device, args.batch_size // workers,
                             ram_budget_bytes=100 << 30,
                             read_threads=args.read_threads, extract_threads=4)
        mean = torch.full((1024,), 128., device=device)
        dist.barrier()
        torch.cuda.synchronize()
        before = disk_bytes_read()
        start = time.perf_counter()
        with closing(loader.batches(sequential=True, mean=mean, scale=64.)) as batches:
            model.final_closing_mean(batches, assignments=codes)
        elapsed = time.perf_counter() - start
        kernel = disk_bytes_read() - before
        assert model.closing_stats['rows'] == args.n
        assert torch.isfinite(model.centroids).all()
        local = dict(rank=rank, elapsed_s=elapsed, kernel_bytes=kernel,
                     closing=model.closing_stats, loader=loader.stats,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
        ranks = [None] * workers
        dist.all_gather_object(ranks, local)
        if rank == 0:
            report = dict(n=args.n, k=args.k, workers=workers, synthetic_codes=True,
                          read_threads=args.read_threads, direct=True,
                          elapsed_s=max(r['elapsed_s'] for r in ranks), ranks=ranks)
            Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
    finally:
        source.close()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
