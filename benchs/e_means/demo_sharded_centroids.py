# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Resident DINO training on two GPUs with exact sharded assignment.

python demo_sharded_centroids.py --launch 2
In fbcode: buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_sharded_centroids -- --launch 2
"""

import argparse
from datetime import timedelta
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

if __package__:
    from .dino_data import (
        DinoBvecs, DinoQueries, DinoTrainset, compute_mse_per_batches,
    )
else:
    from dino_data import (
        DinoBvecs, DinoQueries, DinoTrainset, compute_mse_per_batches,
    )

from faiss.contrib.e_means import (
    EmeansShardedCentroids,
    EmeansShardedCentroidsIVF,
)
from faiss.contrib.e_means.base import _epoch_rows, _epoch_seed
from faiss.contrib.e_means.e_means_torch import _torch_generator


def launch(args, worker):
    if min(_epoch_rows(args.n, args.epochs)) < 1:
        raise ValueError("--epochs must process at least one training row")
    if args.launch:
        if not 1 <= args.launch <= torch.cuda.device_count():
            raise ValueError("--launch exceeds visible GPU count")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=args.launch,
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            run_id="e-means-sharded",
            max_restarts=0,
            start_method="spawn",
        )
        elastic_launch(config, worker)(args)
    else:
        worker(args)


def run(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        rank, ranks = dist.get_rank(), dist.get_world_size()
        if args.n % ranks:
            raise ValueError("--n must be divisible by the GPU count")
        if args.query_batch_size < 1:
            raise ValueError("query batch size must be positive")
        dataset = DinoBvecs(args.data, args.n)
        test_vectors = (
            DinoTrainset(args.data, min(args.n, 50_000))
            if args.test_on_xt else DinoQueries(args.data, dataset.d)
        )
        if rank == 0:
            print(
                f"Query evaluation: exhaustive {args.query_precision}, "
                f"FP32 distances, batch_size={args.query_batch_size}, "
                f"test_vectors={test_vectors.n}, "
                f"source={'train_queries_99M.bvecs prefix' if args.test_on_xt else 'query test split (seed=1234)'}",
                flush=True,
            )
        start = time.perf_counter()
        x = dataset.resident(
            args.n * rank // ranks, args.n * (rank + 1) // ranks, device
        )
        mean = x.mean(dim=0)
        dist.all_reduce(mean)
        mean /= ranks
        x.sub_(mean)
        scale = 1.0
        if args.ivf:
            variance = x.square().mean()
            dist.all_reduce(variance)
            scale = (variance / ranks).sqrt().item()
            if scale <= 0:
                raise ValueError("DINO corpus has zero variance")
            x.div_(scale)
        model_type = (
            EmeansShardedCentroidsIVF if args.ivf else EmeansShardedCentroids
        )
        extra = {"nlist": args.nlist, "nprobe": args.nprobe} if args.ivf else {}
        model = model_type(
            dataset.d,
            args.k,
            nepochs=args.epochs,
            batch_law="quality",
            seed=args.seed,
            assignment_backend=args.assignment_backend,
            group=dist.group.WORLD,
            verbose=True,
            **extra,
        )
        initial_ids = torch.randperm(
            len(x),
            device=device,
            generator=_torch_generator(device, args.seed + rank),
        )[: model.k]
        model.begin(args.n, x[initial_ids].float())
        del initial_ids
        batch = model.recipe["batch_size"] // ranks
        for epoch, global_rows in enumerate(model.recipe["epoch_rows"]):
            local_rows = global_rows // ranks
            preceding_rows = sum(model.recipe["epoch_rows"][:epoch])
            order = torch.randperm(
                len(x),
                device=device,
                generator=_torch_generator(
                    device, _epoch_seed(args.seed, epoch)
                ),
            )
            for begin in range(0, local_rows, batch):
                end = min(local_rows, begin + batch)
                previous_revivals = model.n_splits
                model.update(model.accumulate(x[order[begin:end]]))
                revived = model.n_splits - previous_revivals
                if rank == 0 and revived:
                    progress = (preceding_rows + end * ranks) / args.n
                    print(
                        f"epoch={progress:g} step={begin // batch + 1} "
                        f"centroid_revivals={revived} "
                        f"revivals_total={model.n_splits}",
                        flush=True,
                    )
            model.end_epoch()
            test_mse = compute_mse_per_batches(
                model, test_vectors.batches(mean, scale, args.query_batch_size),
                precision=args.query_precision,
            ) * scale**2
            model.iteration_stats[-1]["test_mse"] = test_mse
            if rank == 0:
                passes = sum(model.recipe["epoch_rows"][:epoch + 1]) / args.n
                print(
                    f"epoch={passes:g} test_mse={test_mse:.6f} "
                    f"centroid_revivals={model.iteration_stats[-1]['nsplit']} "
                    f"revivals_total={model.n_splits}",
                    flush=True,
                )
            del order
        model.finalize()
        test_mse = compute_mse_per_batches(
            model, test_vectors.batches(mean, scale, args.query_batch_size),
            precision=args.query_precision,
        ) * scale**2
        assert torch.isfinite(model.centroids).all()
        if args.output:
            torch.save(
                {
                    "centroids": model.centroids.cpu(),
                    "mean": mean.cpu(),
                    "scale": scale,
                    "centroid_begin": model.centroid_begin,
                    "recipe": model.recipe,
                    "iteration_stats": model.iteration_stats,
                    "test_mse": test_mse,
                    "test_query_count": test_vectors.n,
                    "test_on_xt": args.test_on_xt,
                    "query_precision": args.query_precision,
                },
                f"{args.output}.rank{rank}.pt",
            )
        torch.cuda.synchronize()
        if rank == 0:
            moving_mse = (
                model.obj[-1].item() / model.recipe["epoch_rows"][-1] * scale**2
            )
            print(
                f"PASS: n={args.n} k={args.k} ranks={ranks} "
                f"elapsed={time.perf_counter() - start:.1f}s "
                f"MSE={moving_mse:.6f} "
                f"test_mse={test_mse:.6f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def positive_epochs(value):
    epochs = float(value)
    if not math.isfinite(epochs) or epochs <= 0:
        raise argparse.ArgumentTypeError("epochs must be finite and positive")
    return epochs


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--launch", type=int, nargs="?", const=2)
    result.add_argument("--data", default="/tmp/DINO10B")
    result.add_argument("--n", type=int, default=10_000_000)
    result.add_argument("--k", type=int, default=10_000)
    result.add_argument(
        "--epochs", type=positive_epochs, default=2.0,
        help="training passes, including fractional budgets such as 0.6 or 2.6",
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--assignment-backend", choices=("native", "triton"), default="triton"
    )
    result.add_argument("--ivf", action="store_true")
    result.add_argument("--nlist", type=int, default=256)
    result.add_argument("--nprobe", type=int, default=4)
    result.add_argument("--output")
    result.add_argument("--query-batch-size", type=int, default=1024)
    result.add_argument(
        "--test-on-xt", action="store_true",
        help="evaluate MSE on the first min(n, 50000) vectors from train_queries_99M.bvecs",
    )
    result.add_argument(
        "--query-precision", choices=("bf16", "fp32"), default="bf16",
        help="exhaustive query assignment precision; reported distances use FP32",
    )
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    launch(args, run)
