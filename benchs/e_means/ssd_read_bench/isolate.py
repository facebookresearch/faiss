"""Controlled first-window ablations of the Faiss reader, CPU-only/O_DIRECT."""
import argparse
from datetime import timedelta
import json
import os
import threading
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from common import source_for, SSDRowSource, SSDPrefetch, chunk_grouped_order, disk_bytes_read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='/tmp/DINO10B')
    p.add_argument('--n', type=int, default=1_000_000_000)
    p.add_argument('--scratch', default='/scratch/train-cache/ssd-bench-1b-20261103')
    p.add_argument('--reference', default='/storage/home/matthijs/src/online_k-means')
    p.add_argument('--output', required=True)
    p.add_argument('--windows', type=int, default=1)
    p.add_argument('--bind-cpus', action='store_true')
    p.add_argument('--variants', default='baseline,reference-plan,torch32,raw-numpy,raw-torch64,read-only,baseline-repeat')
    args = p.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(hours=1))
    rank, workers = dist.get_rank(), dist.get_world_size()
    affinity = sorted(os.sched_getaffinity(0))
    if args.bind_cpus:
        def parse_cpus(text):
            cpus = set()
            for part in text.strip().split(','):
                if '-' in part:
                    first, last = map(int, part.split('-'))
                    cpus.update(range(first, last + 1))
                else:
                    cpus.add(int(part))
            return cpus
        nodes = sorted(Path('/sys/devices/system/node').glob('node[0-9]*'))
        nodes = [node for node in nodes
                 if parse_cpus((node / 'cpulist').read_text()) & set(affinity)]
        node = rank % len(nodes)
        local = sorted(parse_cpus((nodes[node] / 'cpulist').read_text()) & set(affinity))
        # Keep SMT siblings together so ranks do not share physical cores.
        def core_key(cpu):
            topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            return (int((topology / 'physical_package_id').read_text()),
                    int((topology / 'core_id').read_text()), cpu)
        local.sort(key=core_key)
        peers = list(range(node, workers, len(nodes)))
        affinity = np.array_split(local, len(peers))[peers.index(rank)].tolist()
        os.sched_setaffinity(0, affinity)
    sys.path.insert(0, args.reference)
    from ssd_prefetch import _plan_file
    original = source_for(args, rank, workers)
    raw = SSDRowSource([Path(args.scratch) / f'rank-{rank}.u8'], 1024, 0, original.n)
    assert all(original.direct) and all(raw.direct)
    chunk_rows = (64 << 20) // original.row_bytes
    window_rows = (50 << 30) // original.d
    order = chunk_grouped_order(original.n, chunk_rows, 4, window_rows, 1 + rank)
    out = torch.empty((window_rows, 1024), dtype=torch.uint8)
    out.zero_()
    all_reports = []
    try:
        for variant in args.variants.split(','):
            reports = []
            for window in range(args.windows):
                start = window * window_rows
                request = order[start:min(original.n, start + window_rows)]
                if not len(request):
                    break
                source = raw if variant.startswith('raw-') else original
                loader = object.__new__(SSDPrefetch)
                loader.source, loader.rows_per_chunk = source, chunk_rows
                loader.read_threads, loader.extract_threads = 1, 4
                loader.stats = {'bytes_read': 0, 'windows': 0}
                sums = dict(read_s=0., extract_s=0., gather_s=0., scatter_s=0., reads=0)
                lock = threading.Lock()
                actual_read = source.read_views
                def timed_read(*a):
                    t = time.perf_counter()
                    views = actual_read(*a)
                    dt = time.perf_counter() - t
                    with lock:
                        sums['read_s'] += dt
                        sums['reads'] += 1
                    return views
                source.read_views = timed_read
                dist.barrier()
                kernel_before = disk_bytes_read()
                t = time.perf_counter()
                if variant.startswith('reference-plan'):
                    rows, dest, chunks, bounds = _plan_file(request, source.n, chunk_rows)
                else:
                    dest = np.argsort(request, kind='stable')
                    rows = request[dest]
                    chunk_ids = rows // chunk_rows
                    bounds = np.r_[0, np.flatnonzero(np.diff(chunk_ids)) + 1, len(rows)]
                    chunks = chunk_ids[bounds[:-1]]
                plan_s = time.perf_counter() - t
                output = out.numpy()
                dest_torch = torch.from_numpy(dest)
                def tasks():
                    for chunk, lo, hi in zip(chunks, bounds[:-1], bounds[1:]):
                        begin = int(chunk) * chunk_rows
                        yield begin, min(source.n, begin + chunk_rows), int(lo), int(hi)
                def extract(task, views):
                    if variant == 'read-only':
                        return
                    t0 = time.perf_counter()
                    gather_s, scatter_s = 0., 0.
                    _, _, lo, hi = task
                    for begin, data in views:
                        a, b = np.searchsorted(rows[lo:hi], [begin, begin + len(data)])
                        a, b = lo + int(a), lo + int(b)
                        local = rows[a:b] - begin
                        tg = time.perf_counter()
                        if variant in ('torch32', 'reference-plan-torch32', 'raw-torch64'):
                            dtype = torch.int64 if variant == 'raw-torch64' else torch.int32
                            src = torch.from_numpy(data).view(dtype)
                            selected = src.index_select(0, torch.from_numpy(local))
                            ts = time.perf_counter()
                            out.view(dtype).index_copy_(0, dest_torch[a:b], selected)
                        else:
                            selected = data[local]
                            ts = time.perf_counter()
                            output[dest[a:b]] = selected
                        gather_s += ts - tg
                        scatter_s += time.perf_counter() - ts
                        del selected
                    with lock:
                        sums['extract_s'] += time.perf_counter() - t0
                        sums['gather_s'] += gather_s
                        sums['scatter_s'] += scatter_s
                scan_start = time.perf_counter()
                loader._scan_chunks(tasks(), extract)
                scan_s = time.perf_counter() - scan_start
                total_s = time.perf_counter() - t
                kernel = disk_bytes_read() - kernel_before
                source.read_views = actual_read
                if variant != 'read-only':
                    for position in np.linspace(0, len(request) - 1, 16, dtype=np.int64):
                        row = int(request[position])
                        np.testing.assert_array_equal(output[position], original.read(row, row + 1)[0])
                local_report = dict(rank=rank, affinity=affinity, plan_s=plan_s, scan_s=scan_s, total_s=total_s,
                                    kernel_bytes=kernel, **sums)
                ranks = [None] * workers
                dist.all_gather_object(ranks, local_report)
                report = dict(variant=variant, window=window + 1, rows_per_rank=len(request),
                              max_plan_s=max(r['plan_s'] for r in ranks),
                              max_scan_s=max(r['scan_s'] for r in ranks),
                              max_total_s=max(r['total_s'] for r in ranks),
                              kernel_bytes=sum(r['kernel_bytes'] for r in ranks), ranks=ranks)
                reports.append(report)
                if rank == 0:
                    print(json.dumps({k:v for k,v in report.items() if k != 'ranks'}), flush=True)
                del rows, dest, chunks, bounds, dest_torch
            all_reports.extend(reports)
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(n=args.n, workers=workers,
                direct=True, linux_cache_flushed=False, chunk_rows=chunk_rows,
                window_rows=window_rows, reports=all_reports), indent=2) + '\n')
    finally:
        original.close()
        raw.close()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
