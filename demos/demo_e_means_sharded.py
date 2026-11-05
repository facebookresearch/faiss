# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Small caller-driven CUDA centroid sharding, with resident or generated rows.

Run: torchrun --standalone --nproc-per-node=2 demos/demo_e_means_sharded.py
Add --stream to generate each bounded contribution instead of keeping local rows.
No disk loader or performance measurement is involved.
"""
import argparse
import datetime
import os

import torch
import torch.distributed as dist

from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    try:
        rank, ranks = dist.get_rank(), dist.get_world_size()
        if ranks != 2:
            raise ValueError("this small example requires exactly two CUDA ranks")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        n, k, d, batch, supplied_chunk = 100, 4, 8, 16, 3
        offset = torch.arange(d, dtype=torch.float32, device=device) / 64
        initial = torch.arange(k, dtype=torch.float32, device=device)[:, None] + offset

        def rows(begin, end):
            ids = torch.arange(begin, end, device=device)
            return (ids % k).float()[:, None] + offset

        resident = None if args.stream else rows(rank * (n // ranks), (rank + 1) * (n // ranks))
        model = EmeansShardedCentroids(
            d,
            k,
            group=dist.group.WORLD,
            niter=2,
            batch_size=batch,
            alpha0=0.5,
        )
        model.begin(n, initial[rank * (k // ranks):(rank + 1) * (k // ranks)])
        for _ in range(model.niter):
            for begin in range(0, n // ranks, batch // ranks):
                end = min(begin + batch // ranks, n // ranks)
                for a in range(begin, end, supplied_chunk):
                    b = min(a + supplied_chunk, end)
                    contribution = (rows(rank * (n // ranks) + a, rank * (n // ranks) + b)
                                    if args.stream else resident[a:b])
                    stats = model.accumulate(contribution)
                model.update(stats)  # Exactly one update for the completed local share.
            model.end_epoch()
        active_snapshot = model.current_centroids().clone()
        closed = model.finalize()
        print("rank=%d centroid_begin=%d local_shape=%s objective=%g mode=%s"
              % (rank, model.centroid_begin, tuple(closed.shape), model.obj[-1].item(),
                 "generated chunks" if args.stream else "resident rows"))
        assert active_snapshot.dtype == closed.dtype == torch.float32
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
