# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Resident DINO training on two GPUs with exact sharded assignment.

python demo_sharded_centroids.py --launch 2
In fbcode: buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_sharded_centroids -- --launch 2
Use --self-test for comparison with the existing unsharded EMA lifecycle.
"""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

if __package__:
    from .dino_data import DinoBvecs, DinoQueries, query_mse
else:
    from dino_data import DinoBvecs, DinoQueries, query_mse

from faiss.contrib.e_means import (
    Emeans,
    EmeansShardedCentroids,
    EmeansShardedCentroidsIVF,
)
from faiss.contrib.e_means.e_means import _epoch_seed, _torch_generator


def launch(args, worker):
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


class ExhaustiveIVFReference(Emeans):
    """Independent exhaustive oracle for the IVF index's FP16 operands."""

    def _assign_rows(self, rows, centroids, centroid_norm=None):
        represented = centroids.half()
        score = (
            represented.float().square().sum(1)[None, :]
            - 2 * torch.mm(rows.half(), represented.t(), out_dtype=torch.float32)
        )
        labels = score.argmin(1)
        distance = (rows.float() - centroids[labels].float()).square().sum(1)
        return distance, labels

    def _apply_splits(self, source, target):
        dtype = self._input_dtype
        self._input_dtype = torch.float16
        try:
            return super()._apply_splits(source, target)
        finally:
            self._input_dtype = dtype


def self_test_queries(model, device):
    rank, ranks = dist.get_rank(), dist.get_world_size()
    centroids = (
        model.centroids if model.centroids is not None else model._centroids
    )
    shards = [torch.empty_like(centroids) for _ in range(ranks)]
    dist.all_gather(shards, centroids)
    full = torch.cat(shards)
    state = {
        key: value.clone()
        for key, value in vars(model).items()
        if isinstance(value, torch.Tensor)
    }
    ivf = getattr(model, "_ivf", None)
    data = np.random.RandomState(27).randint(
        0, 8, (17, model.d), dtype=np.uint8
    )
    raw = np.empty((17, model.d + 4), dtype=np.uint8)
    raw[:, :4] = np.frombuffer(np.int32(model.d).tobytes(), dtype=np.uint8)
    raw[:, 4:] = data
    mean = torch.full((model.d,), 3.0, device=device)
    scale = 2.0
    original = torch.from_numpy(data).to(device).double()
    expected = (
        (original[:, None, :] - (full.double() * scale + mean)[None, :, :])
        .square()
        .sum(2)
        .min(1)
        .values.mean()
        .item()
    )
    with tempfile.TemporaryDirectory(prefix="dino-query-test-") as directory:
        path = Path(directory) / "queries_clean.bvecs"
        raw.tofile(path)
        queries = DinoQueries(directory, model.d, n=17)
        for batch_size in (4, 17, 32):
            actual = query_mse(
                model, queries.batches(mean, scale, batch_size)
            ) * scale**2
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
        del queries
        raw[-1, 0] = 0
        raw.tofile(path)
        try:
            DinoQueries(directory, model.d, n=17)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid query header must fail")
    for key, value in state.items():
        torch.testing.assert_close(getattr(model, key), value, rtol=0, atol=0)
    assert getattr(model, "_ivf", None) is ivf
    if rank == 0:
        print(
            "PASS: held-out MSE, query tails, scaling and unchanged state",
            flush=True,
        )


def self_test(device, ivf=False, assignment_backend="native"):
    if ivf:
        from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

        generator = _torch_generator(device, 11)
        vectors = torch.randn((128, 8), generator=generator, device=device)
        vectors[1] = vectors[0]
        queries = torch.cat(
            (
                vectors[:11],
                torch.randn((25, 8), generator=generator, device=device),
            )
        )
        index = IndexIVFFlat(
            torch.zeros((4, 8), device=device),
            nprobe=4,
            chunk_budget=32,
            search_workspace_bytes=4096,
        )
        index.add(vectors)
        scores, labels = index.search_1nn(queries)
        represented = vectors.half()
        oracle = (
            represented.float().square().sum(1)[None, :]
            - 2 * torch.mm(queries.half(), represented.t(), out_dtype=torch.float32)
        )
        torch.testing.assert_close(labels.long(), oracle.argmin(1))
        torch.testing.assert_close(scores, oracle.min(1).values)

    rank, ranks = dist.get_rank(), dist.get_world_size()
    generator = _torch_generator(device, 7)
    x = torch.randn((129 * ranks, 8), generator=generator, device=device)
    initial = x[:32].clone()
    begin, end = 32 * rank // ranks, 32 * (rank + 1) // ranks
    model_type = EmeansShardedCentroidsIVF if ivf else EmeansShardedCentroids
    extra = {"nlist": 4, "nprobe": 4} if ivf else {}
    model = model_type(
        8,
        32,
        niter=2,
        batch_size=34,
        alpha0=0.3,
        seed=19,
        group=dist.group.WORLD,
        assignment_chunk_size=7,
        assignment_backend=assignment_backend,
        **extra,
    )
    reference_type = ExhaustiveIVFReference if ivf else Emeans
    reference = reference_type(
        8,
        32,
        niter=2,
        batch_size=34,
        alpha0=0.3,
        seed=19,
        assignment_chunk_size=7,
    )
    local = x[rank * 129 : (rank + 1) * 129]
    model.begin(len(x), initial[begin:end])
    reference.begin(len(x), initial)
    for epoch in range(2):
        order = torch.randperm(
            129,
            device=device,
            generator=_torch_generator(device, _epoch_seed(19, epoch)),
        )
        for start in range(0, len(x), 34):
            stop = min(start + 34, len(x))
            parts = []
            for r in range(ranks):
                a = start // ranks
                b = stop // ranks
                parts.append(x[r * 129 + order[a:b]])
            model.update(model.accumulate(local[order[a:b]]))
            reference.update(reference.accumulate(torch.cat(parts)))
        model.end_epoch()
        reference.end_epoch()
        self_test_queries(model, device)
    model.finalize()
    reference.finalize()
    self_test_queries(model, device)
    torch.testing.assert_close(
        model.centroids,
        reference.centroids[begin:end],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(model.obj, reference.obj, atol=1e-3, rtol=2e-5)
    assert model.n_splits == reference.n_splits
    if rank == 0:
        print(
            "PASS: centroid shards, ragged batches, EMA, revival "
            "and closing means",
            flush=True,
        )


def run(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        if args.self_test:
            self_test(device, args.ivf, args.assignment_backend)
            return
        rank, ranks = dist.get_rank(), dist.get_world_size()
        if args.n % ranks:
            raise ValueError("--n must be divisible by the GPU count")
        if args.query_batch_size < 1:
            raise ValueError("query batch size must be positive")
        dataset = DinoBvecs(args.data, args.n)
        queries = DinoQueries(args.data, dataset.d)
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
            niter=args.epochs,
            operating_point="quality",
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
        for epoch in range(args.epochs):
            order = torch.randperm(
                len(x),
                device=device,
                generator=_torch_generator(
                    device, _epoch_seed(args.seed, epoch)
                ),
            )
            for begin in range(0, len(x), batch):
                model.update(model.accumulate(x[order[begin : begin + batch]]))
            model.end_epoch()
            test_mse = query_mse(
                model, queries.batches(mean, scale, args.query_batch_size)
            ) * scale**2
            model.iteration_stats[-1]["test_mse"] = test_mse
            if rank == 0:
                print(f"epoch={epoch + 1} test_mse={test_mse:.6f}", flush=True)
            del order
        model.finalize()
        test_mse = query_mse(
            model, queries.batches(mean, scale, args.query_batch_size)
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
                    "test_query_count": queries.n,
                },
                f"{args.output}.rank{rank}.pt",
            )
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"PASS: n={args.n} k={args.k} ranks={ranks} "
                f"elapsed={time.perf_counter() - start:.1f}s "
                f"MSE={model.obj[-1].item() / args.n * scale**2:.6f} "
                f"test_mse={test_mse:.6f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--launch", type=int, nargs="?", const=2)
    result.add_argument("--data", default="/tmp/DINO10B")
    result.add_argument("--n", type=int, default=10_000_000)
    result.add_argument("--k", type=int, default=10_000)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--assignment-backend", choices=("native", "triton"), default="triton"
    )
    result.add_argument("--ivf", action="store_true")
    result.add_argument("--nlist", type=int, default=256)
    result.add_argument("--nprobe", type=int, default=4)
    result.add_argument("--output")
    result.add_argument("--query-batch-size", type=int, default=1024)
    result.add_argument("--self-test", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    launch(args, run)
