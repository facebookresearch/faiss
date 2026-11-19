# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train sharded e-means or Lloyd k-means from SSD windows.

buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_training_from_ssd -- --launch 2
The default case is 200M DINO points and 1M centroids. Every rank scans its
own row range. Prefetch is automatic; --ssd-prefetch-chunk-groups defaults
to 4 and groups random windows by scan chunks. --final-closing-mean makes
a sequential no-search pass using the final training epoch's saved labels.
Assignment uses IVF by default; --exact searches all centroid shards instead.
--kmeans freezes centroids for each epoch and takes their means afterward.
It requires --batch-size and skips saved-assignment closing.
"""

from contextlib import closing as _closing
from datetime import timedelta
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

if __package__:
    from .demo_sharded_centroids import launch, parser as resident_parser
    from .dino_data import (
        DinoBvecs, DinoQueries, DinoTrainset, compute_mse_per_batches,
    )
    from .ssd_streaming import SSDPrefetch, SSDRowSource
    from .lloyd import LloydShardedCentroids, LloydShardedCentroidsIVF
else:
    from demo_sharded_centroids import launch, parser as resident_parser
    from dino_data import (
        DinoBvecs, DinoQueries, DinoTrainset, compute_mse_per_batches,
    )
    from ssd_streaming import SSDPrefetch, SSDRowSource
    from lloyd import LloydShardedCentroids, LloydShardedCentroidsIVF

from faiss.contrib.e_means import EmeansShardedCentroids, EmeansShardedCentroidsIVF
from faiss.contrib.e_means.base import _ALPHA_LAW, _epoch_seed, _resolve_recipe


def training_recipe(args, ranks):
    if args.kmeans and args.batch_size is None:
        raise ValueError("--kmeans requires --batch-size (processing rows per batch)")
    if args.batch_size is None:
        return _resolve_recipe(args.n, args.k, args.epochs, "quality", None, None)
    batch = args.batch_size
    if not 1 <= batch <= args.n or batch % ranks:
        raise ValueError("batch size must be in [1, n] and divide by worker count")
    if args.kmeans:
        # Explicit settings bypass the e-means recipe in the shared lifecycle.
        # Lloyd updates do not use alpha.
        return batch, 1.0
    coefficient, k_exp, batch_exp, n_exp, iter_exp = _ALPHA_LAW
    alpha = (
        coefficient
        * args.k**k_exp
        * batch**batch_exp
        * args.n**n_exp
        * args.epochs**iter_exp
    )
    return batch, min(max(alpha, 1e-4), 1.0)


def run(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    source = None
    try:
        rank, ranks = dist.get_rank(), dist.get_world_size()
        final_closing_mean = args.final_closing_mean and not args.kmeans
        if args.n % ranks or args.k % ranks:
            raise ValueError("n and k must divide by worker count")
        if final_closing_mean and args.k > 2**31:
            raise ValueError("CPU int32 closing codes require k <= 2**31")
        if args.query_batch_size < 1:
            raise ValueError("query batch size must be positive")
        if args.verbose_every < 1:
            raise ValueError("verbose interval must be positive")
        batch, alpha = training_recipe(args, ranks)
        dataset = DinoBvecs(args.data, args.n)
        test_vectors = (
            None
            if args.initialization_only
            else (
                DinoTrainset(args.data, min(args.n, 50_000))
                if args.test_on_xt else DinoQueries(args.data, dataset.d)
            )
        )
        source = SSDRowSource(
            dataset.paths,
            dataset.d,
            args.n * rank // ranks,
            args.n * (rank + 1) // ranks,
            header_bytes=4,
        )
        local_k = args.k // ranks
        model_type = EmeansShardedCentroids if args.exact else EmeansShardedCentroidsIVF
        if args.kmeans:
            model_type = LloydShardedCentroids if args.exact else LloydShardedCentroidsIVF
        extra = {} if args.exact else {"nlist": args.nlist, "nprobe": args.nprobe}
        model = model_type(
            dataset.d,
            args.k,
            nepochs=args.epochs,
            batch_law="quality",
            batch_size=batch if args.batch_size is not None else None,
            alpha0=alpha if args.batch_size is not None else None,
            seed=args.seed,
            assignment_backend=args.assignment_backend,
            group=dist.group.WORLD,
            verbose=True,
            capture_assignments=final_closing_mean,
            **extra,
        )
        loader = SSDPrefetch(
            source,
            device,
            batch // ranks,
            chunk_groups=args.ssd_prefetch_chunk_groups,
            chunk_bytes=args.ssd_chunk_mib << 20,
            ram_budget_bytes=int(args.ssd_ram_gib * (1 << 30)),
            read_threads=args.ssd_read_threads,
            extract_threads=args.ssd_extract_threads,
        )
        if rank == 0:
            host_bytes = 2 * loader.window_rows * dataset.d
            gpu_batch_bytes = loader.batch_size * dataset.d * 4
            batch_source = "quality recipe" if args.batch_size is None else "override"
            print(
                f"Query evaluation: exhaustive {args.query_precision}, "
                f"FP32 distances, batch_size={args.query_batch_size}, "
                f"test_vectors={test_vectors.n if test_vectors is not None else 0}, "
                f"source={'train_queries_99M.bvecs prefix' if args.test_on_xt else 'query test split (seed=1234)'}",
                flush=True,
            )
            if args.exact:
                print("Training assignment: BF16 brute force over all centroid shards", flush=True)
            else:
                route_bytes = batch * args.nprobe * (
                    4 if args.nlist <= torch.iinfo(torch.int32).max else 8
                )
                print(
                    f"Coarse routing: reader-local rows, gathered list IDs="
                    f"{route_bytes / (1 << 20):.3f} MiB per rank", flush=True,
                )
            rounded_rows = args.exact and not args.kmeans
            batch_dtype = "BF16" if rounded_rows else "FP32"
            gathered_item_bytes = 2 if rounded_rows else 4
            update_description = (
                "Lloyd: frozen centroids per epoch, epoch mean updates"
                if args.kmeans else f"e-means: alpha0={alpha:.6g}"
            )
            print(
                f"SSD training: global_batch={batch} rows "
                f"per_rank_batch={loader.batch_size} rows ({batch_source}) "
                f"{update_description}\n"
                f"Batch communication: all-gather, "
                f"gathered {batch_dtype} batch buffer="
                f"{batch * dataset.d * gathered_item_bytes / (1 << 30):.3f} "
                f"GiB per rank\n"
                f"SSD prefetch: {loader.n_prefetch} batches per window, "
                f"2 host windows per rank, window_rows={loader.window_rows}, "
                f"chunk_groups={loader.chunk_groups}, direct={all(source.direct)}\n"
                f"SSD memory: pinned host buffers={host_bytes / (1 << 30):.3f} GiB "
                f"per rank, {host_bytes * ranks / (1 << 30):.3f} GiB total; "
                f"each FP32 CUDA batch={gpu_batch_bytes / (1 << 30):.3f} GiB per rank",
                flush=True,
            )
            if final_closing_mean:
                print(
                    f"Closing codes: CPU int32 in source-row order, "
                    f"{4 * source.n / (1 << 30):.3f} GiB per rank, "
                    f"{4 * args.n / (1 << 30):.3f} GiB total",
                    flush=True,
                )
        start = time.perf_counter()
        rng = np.random.RandomState(args.seed + rank)
        ids = rng.choice(source.n, local_k, replace=False)
        initial = torch.empty(
            (local_k, dataset.d), dtype=torch.uint8, pin_memory=True
        )
        if rank == 0:
            print("Queued SSD reads for random initial centroids", flush=True)
        initial_stats = source.gather(ids, initial, depth=args.ssd_io_depth)
        initial_stats["time_s"] = time.perf_counter() - start
        loader.stats["bytes_read"] += initial_stats["bytes_read"]
        if args.initialization_only:
            check = np.linspace(
                0, local_k - 1, min(local_k, 1024), dtype=np.int64
            )
            absolute = source.begin + ids[check]
            expected = np.empty((len(check), dataset.d), dtype=np.uint8)
            for mapping, hi, lo in zip(
                dataset.maps, dataset.ends, np.r_[0, dataset.ends[:-1]]
            ):
                selected = (absolute >= lo) & (absolute < hi)
                expected[selected] = mapping[absolute[selected] - lo, 4:]
            np.testing.assert_array_equal(initial[check].numpy(), expected)
            if rank == 0:
                print(
                    f"PASS initialization rank={rank}: "
                    + json.dumps(
                        {
                            "n": args.n,
                            "k": args.k,
                            "checked": len(check),
                            "initialization": initial_stats,
                        }
                    ),
                    flush=True,
                )
            return
        assert test_vectors is not None
        sample_n = min(local_k, max(1, 10_000 // ranks))
        sample = initial[:sample_n].to(device).double()
        total = sample.sum(0)
        total_sq = sample.square().sum()
        dist.all_reduce(total)
        dist.all_reduce(total_sq)
        mean = total / (sample_n * ranks)
        scale = (
            (total_sq / (sample_n * ranks * dataset.d) - mean.square().mean())
            .clamp_min(0)
            .sqrt()
            .item()
        )
        if scale <= 0:
            raise ValueError("DINO centering sample has zero variance")
        mean = mean.float()
        del sample
        model.begin(args.n, (initial.to(device).float() - mean) / scale)
        del initial, ids
        if rank == 0:
            print(f"SSD center_scale={scale:.6g}", flush=True)
        closing_codes = None
        for epoch, global_rows in enumerate(model.recipe["epoch_rows"]):
            passes = sum(model.recipe["epoch_rows"][:epoch + 1]) / args.n
            epoch_steps = model.recipe["epoch_steps"][epoch]
            batch_objectives, batch_rows = [], []
            order = loader.order(_epoch_seed(args.seed + rank, epoch))
            order = order[:global_rows // ranks]
            capture = (
                final_closing_mean
                and epoch == len(model.recipe["epoch_rows"]) - 1
            )
            if capture:
                closing_codes = torch.empty(source.n, dtype=torch.int32)
                if global_rows < args.n:
                    closing_codes.fill_(-1)
            at = 0
            with _closing(
                loader.batches(order, mean=mean, scale=scale)
            ) as batches:
                for step, rows in enumerate(batches, start=1):
                    stats = model.accumulate(rows)
                    if capture:
                        closing_codes.numpy()[order[at:at + len(rows)]] = (
                            stats["labels"]
                            .to(device="cpu", dtype=torch.int32).numpy()
                        )
                    at += len(rows)
                    batch_objectives.append(stats["objective"].detach().clone())
                    batch_rows.append(stats["n"])
                    model.update(stats)
                    if (
                        (step - 1) % args.verbose_every == 0
                        or step == epoch_steps
                    ):
                        # Each owner contributes only its winning rows' SSE.
                        objective = batch_objectives[-1].clone()
                        dist.all_reduce(objective)
                        # Report the largest cumulative wait across workers.
                        ssd_wait = objective.new_tensor(loader.stats["wait_s"])
                        dist.all_reduce(ssd_wait, op=dist.ReduceOp.MAX)
                        if rank == 0:
                            batch_mse = (
                                objective.item() / batch_rows[-1] * scale**2
                            )
                            print(
                                f"epoch={passes:g} steps={step}/{epoch_steps} "
                                f"batch_mse={batch_mse:.6f} "
                                f"revivals_total={model.n_splits} "
                                f"elapsed={time.perf_counter() - start:.1f}s "
                                f"ssd_wait={ssd_wait.item():.1f}s",
                                flush=True,
                            )
            model.end_epoch()
            if args.kmeans:
                model.iteration_stats[-1]["assignment_mse"] *= scale**2
            # Collect all batches with one reduction and host transfer per epoch.
            objectives = torch.stack(batch_objectives)
            dist.all_reduce(objectives)
            model.iteration_stats[-1]["batch_mse"] = [
                value / count * scale**2
                for value, count in zip(objectives.tolist(), batch_rows)
            ]
            test_mse = compute_mse_per_batches(
                model, test_vectors.batches(mean, scale, args.query_batch_size),
                precision=args.query_precision,
            ) * scale**2
            model.iteration_stats[-1]["test_mse"] = test_mse
            if rank == 0:
                training_mse = (
                    f"train_assignment_mse={model.iteration_stats[-1]['assignment_mse']:.6f} "
                    if args.kmeans else ""
                )
                print(
                    f"epoch={passes:g} {training_mse}test_mse={test_mse:.6f} "
                    f"revivals_total={model.n_splits}",
                    flush=True,
                )
            del order
        if final_closing_mean:
            if rank == 0:
                host_bytes = 2 * loader.batch_size * dataset.d
                print(
                    f"Sequential saved-assignment final-closing-mean scan (no search): "
                    f"global_batch={batch} rows, per_rank_batch={loader.batch_size} rows, "
                    f"1 prefetched batch per window, 2 host windows per rank, "
                    f"pinned host buffers={host_bytes / (1 << 30):.3f} GiB per rank, "
                    f"{host_bytes * ranks / (1 << 30):.3f} GiB total",
                    flush=True,
                )
            with _closing(
                loader.batches(sequential=True, mean=mean, scale=scale)
            ) as batches:
                model.final_closing_mean(batches, assignments=closing_codes)
            del closing_codes
            if rank == 0:
                print(
                    f"final_closing_mean time={model.closing_stats['time_s']:.1f}s "
                    f"rows={model.closing_stats['rows']}",
                    flush=True,
                )
        else:
            model.finalize()
        test_mse = compute_mse_per_batches(
            model, test_vectors.batches(mean, scale, args.query_batch_size),
            precision=args.query_precision,
        ) * scale**2
        assert torch.isfinite(model.centroids).all()
        torch.cuda.synchronize()
        closing = getattr(model, "closing_stats", None)
        if closing is not None:
            closing = {
                **closing,
                "assignment_mse": closing["assignment_mse"] * scale**2,
            }
        report = {
            "algorithm": "kmeans" if args.kmeans else "emeans",
            "n": args.n,
            "k": args.k,
            "ranks": ranks,
            "elapsed_s": time.perf_counter() - start,
            "moving_pass_mse": (
                model.obj[-1].item() / model.recipe["epoch_rows"][-1] * scale**2
            ),
            "test_mse": test_mse,
            "test_query_count": test_vectors.n,
            "test_on_xt": args.test_on_xt,
            "query_precision": args.query_precision,
            "scale": scale,
            "closing": closing,
            "ssd": loader.stats,
            "initialization": initial_stats,
            "recipe": model.recipe,
            "iteration_stats": model.iteration_stats,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        }
        if args.kmeans:
            report["pre_update_train_mse"] = report.pop("moving_pass_mse")
            report["train_assignment_mse"] = model.iteration_stats[-1]["assignment_mse"]
        if rank == 0:
            print(f"PASS rank={rank}: {json.dumps(report)}", flush=True)
        if args.output:
            torch.save(
                {
                    "centroids": model.centroids.cpu(),
                    "mean": mean.cpu(),
                    "scale": scale,
                    "centroid_begin": model.centroid_begin,
                    "report": report,
                },
                f"{args.output}.rank{rank}.pt",
            )
    finally:
        if source is not None:
            source.close()
        dist.destroy_process_group()


def parser():
    result = resident_parser()
    result.description = __doc__
    result.set_defaults(n=200_000_000, k=1_000_000, nlist=16384, ivf=True)
    result.add_argument(
        "--exact", action="store_true",
        help="use brute-force centroid-sharded assignment; ignore --nlist and --nprobe",
    )
    result.add_argument(
        "--kmeans", action="store_true",
        help="run Lloyd updates once per epoch; requires --batch-size and skips final closing",
    )
    result.add_argument(
        "--batch-size",
        type=int,
        help="global processing batch size; refits the EMA step for e-means",
    )
    result.add_argument("--ssd-prefetch-chunk-groups", type=int, default=4)
    result.add_argument("--ssd-chunk-mib", type=int, default=64)
    result.add_argument("--ssd-ram-gib", type=float, default=16)
    result.add_argument("--ssd-io-depth", type=int, default=256)
    result.add_argument(
        "--initialization-only",
        action="store_true",
        help="verify sampled rows, report queued initialization I/O and exit",
    )
    result.add_argument("--ssd-read-threads", type=int, default=1)
    result.add_argument("--ssd-extract-threads", type=int, default=4)
    result.add_argument("--verbose-every", type=int, default=10)
    result.add_argument(
        "--final-closing-mean", action="store_true", default=True
    )
    result.add_argument(
        "--no-final-closing-mean",
        dest="final_closing_mean",
        action="store_false",
    )
    return result


if __name__ == "__main__":
    launch(parser().parse_args(), run)
