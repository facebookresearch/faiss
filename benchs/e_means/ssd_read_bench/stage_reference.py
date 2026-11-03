"""Create flat reference shards from local bvecs, without holding the corpus."""
import os
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from common import arguments, source_for

if __name__ == "__main__":
    args = arguments()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, workers = dist.get_rank(), dist.get_world_size()
    if args.n % workers:
        raise ValueError("n must divide by worker count")
    source = source_for(args, rank, workers)
    directory = Path(args.scratch)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"rank-{rank}.u8"
    partial = directory / f"rank-{rank}.partial"
    started = time.perf_counter()
    try:
        if final.exists():
            raise FileExistsError(f"refusing to overwrite {final}")
        with partial.open("xb", buffering=0) as output:
            for begin in range(0, source.n, 1 << 20):
                end = min(source.n, begin + (1 << 20))
                data = source.read(begin, end)
                data.tofile(output)
                if end == source.n or end // 25_000_000 != begin // 25_000_000:
                    print(f"rank={rank} staged={end}/{source.n} elapsed={time.perf_counter() - started:.1f}s", flush=True)
            os.fsync(output.fileno())
        if partial.stat().st_size != source.n * source.d:
            raise ValueError("incorrect staged file size")
        mapping = np.memmap(partial, dtype=np.uint8, mode="r", shape=(source.n, source.d))
        for row in np.linspace(0, source.n - 1, 32, dtype=np.int64):
            np.testing.assert_array_equal(mapping[row], source.read(int(row), int(row) + 1)[0])
        del mapping
        partial.rename(final)
        dist.barrier()
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(
                n=args.n, workers=workers, scratch=str(directory),
                seconds=time.perf_counter() - started,
            ), indent=2) + "\n")
            print(f"PASS: {args.n} DINO rows staged and validated across {workers} files", flush=True)
    finally:
        source.close()
        dist.destroy_process_group()
