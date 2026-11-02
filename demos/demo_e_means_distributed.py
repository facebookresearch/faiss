# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Row-distributed E-means on generated, rank-local FP32 batches.

Run: torchrun --standalone --nproc-per-node=2 demos/demo_e_means_distributed.py
Uses CUDA/NCCL by default, or CPU/Gloo with --device cpu.
Each rank keeps all centroids; the population is generated
one equal local contribution at a time. Global batches and the final tail must
divide evenly by rank count. This demonstrates loading/collective ordering,
not disk-loading or training performance.

For CPU/Gloo, add --device cpu; add --numpy for NumPy/Faiss CPU assignment.
"""

import argparse
import os

import torch
import torch.distributed as dist

from faiss.contrib.e_means import DistributedEmeans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--transport", choices=("auto", "dense", "sparse"), default="auto")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--numpy", action="store_true", help="NumPy state/rows on CPU")
    args = parser.parse_args()
    if args.numpy and args.device != "cpu":
        parser.error("--numpy requires --device cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    try:
        rank, ranks = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
        offset = torch.arange(args.d, device=device, dtype=torch.float32) / 64
        initial = torch.arange(args.k, device=device, dtype=torch.float32)[:, None] + offset
        if args.numpy:
            initial = initial.numpy()
        model = DistributedEmeans(args.d, args.k, niter=args.epochs,
                                  group=dist.group.WORLD, transport=args.transport)
        model.begin(args.n, initial)
        batch_size = model.recipe["batch_size"]
        for _epoch in range(args.epochs):
            for begin in range(0, args.n, batch_size):
                n = min(batch_size, args.n - begin)
                local_n = n // ranks
                local_begin = begin + local_n * rank
                local_end = local_begin + local_n
                ids = torch.arange(local_begin, local_end, device=device)
                rows = (ids % args.k).float()[:, None] + offset
                if args.numpy:
                    rows = rows.numpy()
                stats = model.accumulate(rows)
                model.update(stats)
            model.end_epoch()
        centroids = model.finalize()
        if rank == 0:
            print("rows=%d ranks=%d batch_size=%d objective=%.6g centroids=%s"
                  % (args.n, ranks, batch_size, model.obj[-1].item(), tuple(centroids.shape)))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
