# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train sharded IVF e-means from SSD windows, with saved-assignment closing.

buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_training_from_ssd -- --launch 2
The default case is 200M DINO points and 1M centroids. Every rank scans its
own row range. Prefetch is automatic; --ssd-prefetch-chunk-groups defaults
to 4 and groups random windows by scan chunks. --final-closing-mean makes
a sequential no-search pass using the final training epoch's saved labels.
"""

from contextlib import closing as _closing
from datetime import timedelta
import errno
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

if __package__:
    from .demo_sharded_centroids import (
        launch,
        parser as resident_parser,
        self_test_queries,
    )
    from .dino_data import DinoBvecs, DinoQueries, query_mse
    from .ssd_streaming import SSDPrefetch, SSDRowSource, chunk_grouped_order
else:
    from demo_sharded_centroids import (
        launch,
        parser as resident_parser,
        self_test_queries,
    )
    from dino_data import DinoBvecs, DinoQueries, query_mse
    from ssd_streaming import SSDPrefetch, SSDRowSource, chunk_grouped_order

from faiss.contrib.e_means import EmeansShardedCentroidsIVF
from faiss.contrib.e_means.base import _ALPHA_LAW, _epoch_seed, _resolve_recipe


def training_recipe(args, ranks):
    if args.batch_size is None:
        return _resolve_recipe(args.n, args.k, args.epochs, "quality", None, None)
    batch = args.batch_size
    if not 1 <= batch <= args.n or batch % ranks:
        raise ValueError("batch size must be in [1, n] and divide by worker count")
    coefficient, k_exp, batch_exp, n_exp, iter_exp = _ALPHA_LAW
    alpha = (
        coefficient
        * args.k**k_exp
        * batch**batch_exp
        * args.n**n_exp
        * args.epochs**iter_exp
    )
    return batch, min(max(alpha, 1e-4), 1.0)


def self_test(device, nepochs=2):
    # Different-sized files force a range across chunk-file boundaries.
    rank, ranks = dist.get_rank(), dist.get_world_size()
    data = np.random.RandomState(1).randint(0, 256, (258, 8), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="e-means-ssd-") as directory:
        paths = [Path(directory) / "first.bin", Path(directory) / "second.bin"]
        data[:111].tofile(paths[0])
        data[111:].tofile(paths[1])
        source = SSDRowSource(paths, 8, 0, 258)
        try:
            np.testing.assert_array_equal(source.read(99, 129), data[99:129])
            loader = SSDPrefetch(
                source,
                device,
                34,
                n_prefetch=2,
                chunk_bytes=8 * 17,
                chunk_groups=4,
                ram_budget_bytes=4096,
                read_threads=2,
            )
            order = loader.order(9)
            np.testing.assert_array_equal(np.sort(order), np.arange(258))
            request = np.array([99, 99, 127, 1, 250])
            out = torch.empty((len(request), 8), dtype=torch.uint8)
            loader.fill_window(request, out)
            np.testing.assert_array_equal(out.numpy(), data[request])
            # Empty requests perform no reads; invalid indices fail planning.
            before = loader.stats["bytes_read"]
            loader.fill_window(np.empty(0, dtype=np.int64), out[:0])
            assert loader.stats["bytes_read"] == before
            for invalid in (np.array([-1]), np.array([source.n])):
                try:
                    loader.fill_window(invalid, out[:1])
                except IndexError:
                    pass
                else:
                    raise AssertionError("invalid prefetch request must fail")
            batches = torch.cat(list(loader.batches(order)))
            torch.testing.assert_close(
                batches.cpu(), torch.from_numpy(data[order]).float()
            )
            before = loader.stats["bytes_read"]
            sequential = torch.cat(list(loader.batches(sequential=True)))
            torch.testing.assert_close(
                sequential.cpu(), torch.from_numpy(data).float()
            )
            assert loader.stats["bytes_read"] - before == data.nbytes
            for groups in (1, 4, 100):
                order = chunk_grouped_order(258, 17, groups, 68, 3)
                np.testing.assert_array_equal(np.sort(order), np.arange(258))
        finally:
            source.close()
        bvecs = np.random.RandomState(2).randint(
            0, 256, (258, 1024), dtype=np.uint8
        )
        raw = np.empty((258, 1028), dtype=np.uint8)
        raw[:, :4] = np.frombuffer(np.int32(1024).tobytes(), dtype=np.uint8)
        raw[:, 4:] = bvecs
        bpaths = [
            Path(directory) / "first.bvecs",
            Path(directory) / "second.bvecs",
        ]
        raw[:111].tofile(bpaths[0])
        raw[111:].tofile(bpaths[1])
        source = SSDRowSource(bpaths, 1024, 1, 257, header_bytes=4)
        try:
            request = np.r_[
                np.random.RandomState(3).permutation(256), [2, 2, 255]
            ]
            out = torch.empty(
                (len(request), 1024), dtype=torch.uint8, pin_memory=True
            )
            io = source.gather(request, out, depth=8)
            np.testing.assert_array_equal(out.numpy(), bvecs[request + 1])
            assert io["requests"] > 8
            source.gather(np.array([0, 1, 2]), out[:3], depth=8)
            np.testing.assert_array_equal(out[:3].numpy(), bvecs[1:4])
            loader = SSDPrefetch(
                source, device, 17, n_prefetch=2, chunk_bytes=17 * 1028,
                ram_budget_bytes=128 << 10, read_threads=2, extract_threads=2,
            )
            loader.fill_window(request, out)
            np.testing.assert_array_equal(out.numpy(), bvecs[request + 1])
            sequential_out = torch.empty((source.n, 1024), dtype=torch.uint8)
            before = loader.stats["bytes_read"]
            loader.fill_sequential(0, source.n, sequential_out)
            np.testing.assert_array_equal(sequential_out.numpy(), bvecs[1:257])
            assert loader.stats["bytes_read"] - before == source.n * 1028
            original = source.fds[0]
            source.fds[0] = -1
            try:
                try:
                    source.gather(np.array([0]), out[:1], depth=8)
                except OSError as error:
                    assert error.errno == errno.EBADF
                else:
                    raise AssertionError("invalid FD must fail random I/O")
            finally:
                source.fds[0] = original
        finally:
            source.close()
        local_n = 120
        source = SSDRowSource(paths, 8, rank * local_n, (rank + 1) * local_n)
        try:
            loader = SSDPrefetch(
                source,
                device,
                16,
                n_prefetch=2,
                chunk_bytes=8 * 17,
                chunk_groups=4,
                ram_budget_bytes=4096,
                read_threads=2,
            )
            model = EmeansShardedCentroidsIVF(
                8,
                16 * ranks,
                nepochs=nepochs,
                batch_size=16 * ranks,
                alpha0=0.3,
                nlist=4,
                nprobe=4,
                coarse_niter=2,
                group=dist.group.WORLD,
            )
            initial = (
                torch.from_numpy(data[rank * 16 : (rank + 1) * 16])
                .to(device)
                .float()
            )
            model.begin(local_n * ranks, initial)
            for epoch, global_rows in enumerate(model.recipe["epoch_rows"]):
                order = loader.order(_epoch_seed(5, epoch))
                order = order[:global_rows // ranks]
                codes = torch.full((local_n,), -1, dtype=torch.int32)
                at = 0
                with _closing(loader.batches(order)) as batches:
                    for rows in batches:
                        stats = model.accumulate(rows)
                        codes.numpy()[order[at:at + len(rows)]] = (
                            stats["labels"].cpu().numpy()
                        )
                        at += len(rows)
                        model.update(stats)
                assert at == global_rows // ranks
                model.end_epoch()
            self_test_queries(model, device)
            sums = torch.zeros_like(model._centroids)
            counts = sums.new_zeros(model.k)
            parts = [torch.empty(local_n, dtype=torch.int32, device=device) for _ in range(ranks)]
            dist.all_gather(parts, codes.to(device))
            all_codes = torch.cat(parts).long()
            # Independent means from saved labels in original source-row order.
            for a in range(0, local_n * ranks, 16):
                rows = torch.from_numpy(data[a : a + 16]).to(device).float()
                winner = all_codes[a:a + len(rows)]
                owned = (winner >= model.centroid_begin) & (
                    winner < model.centroid_begin + model.k
                )
                ids = winner[owned] - model.centroid_begin
                sums.index_add_(0, ids, rows[owned])
                counts.index_add_(0, ids, rows.new_ones(int(owned.sum())))
            expected = torch.where(
                (counts > 0)[:, None],
                sums / counts.clamp_min(1)[:, None],
                model._centroids,
            )

            def no_search(*args, **kwargs):
                raise AssertionError("saved-assignment closing must not search")
            model._prepare_assignment = model._winning_labels = no_search
            with _closing(loader.batches(sequential=True)) as batches:
                model.final_closing_mean(batches, assignments=codes, chunk_rows=7)
            assert model.closing_stats["rows"] == model.recipe["epoch_rows"][-1]
            self_test_queries(model, device)
            torch.testing.assert_close(
                model.centroids, expected, atol=2e-5, rtol=2e-5
            )
        finally:
            source.close()
    if rank == 0:
        print(
            "PASS: queued bvecs random reads, direct SSD reads, "
            "duplicate requests, chunk groups, "
            "async batches, sequential scan and saved-assignment closing means",
            flush=True,
        )


def run(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    source = None
    try:
        if args.self_test:
            self_test(device, args.epochs)
            return
        rank, ranks = dist.get_rank(), dist.get_world_size()
        if args.n % ranks or args.k % ranks:
            raise ValueError("n and k must divide by worker count")
        if args.final_closing_mean and args.k > 2**31:
            raise ValueError("CPU int32 closing codes require k <= 2**31")
        if args.query_batch_size < 1:
            raise ValueError("query batch size must be positive")
        if args.verbose_every < 1:
            raise ValueError("verbose interval must be positive")
        batch, alpha = training_recipe(args, ranks)
        dataset = DinoBvecs(args.data, args.n)
        queries = (
            None
            if args.initialization_only
            else DinoQueries(args.data, dataset.d)
        )
        source = SSDRowSource(
            dataset.paths,
            dataset.d,
            args.n * rank // ranks,
            args.n * (rank + 1) // ranks,
            header_bytes=4,
        )
        local_k = args.k // ranks
        model = EmeansShardedCentroidsIVF(
            dataset.d,
            args.k,
            nepochs=args.epochs,
            batch_law="quality",
            batch_size=batch if args.batch_size is not None else None,
            alpha0=alpha if args.batch_size is not None else None,
            seed=args.seed,
            nlist=args.nlist,
            nprobe=args.nprobe,
            assignment_backend=args.assignment_backend,
            group=dist.group.WORLD,
            verbose=True,
            capture_assignments=args.final_closing_mean,
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
                f"test_queries={queries.n if queries is not None else 0}, "
                f"split_seed=1234", flush=True,
            )
            route_bytes = batch * args.nprobe * (
                4 if args.nlist <= torch.iinfo(torch.int32).max else 8
            )
            print(
                f"Coarse routing: reader-local rows, gathered list IDs="
                f"{route_bytes / (1 << 20):.3f} MiB per rank", flush=True,
            )
            print(
                f"SSD training: global_batch={batch} rows "
                f"per_rank_batch={loader.batch_size} rows ({batch_source}) "
                f"alpha0={alpha:.6g}\n"
                f"Batch communication: all-gather, "
                f"gathered FP32 batch buffer="
                f"{batch * dataset.d * 4 / (1 << 30):.3f} "
                f"GiB per rank\n"
                f"SSD prefetch: {loader.n_prefetch} batches per window, "
                f"2 host windows per rank, window_rows={loader.window_rows}, "
                f"chunk_groups={loader.chunk_groups}, direct={all(source.direct)}\n"
                f"SSD memory: pinned host buffers={host_bytes / (1 << 30):.3f} GiB "
                f"per rank, {host_bytes * ranks / (1 << 30):.3f} GiB total; "
                f"each FP32 CUDA batch={gpu_batch_bytes / (1 << 30):.3f} GiB per rank",
                flush=True,
            )
            if args.final_closing_mean:
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
        assert queries is not None
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
                args.final_closing_mean
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
            # Collect all batches with one reduction and host transfer per epoch.
            objectives = torch.stack(batch_objectives)
            dist.all_reduce(objectives)
            model.iteration_stats[-1]["batch_mse"] = [
                value / count * scale**2
                for value, count in zip(objectives.tolist(), batch_rows)
            ]
            test_mse = query_mse(
                model, queries.batches(mean, scale, args.query_batch_size),
                precision=args.query_precision,
            ) * scale**2
            model.iteration_stats[-1]["test_mse"] = test_mse
            if rank == 0:
                print(
                    f"epoch={passes:g} test_mse={test_mse:.6f} "
                    f"revivals_total={model.n_splits}",
                    flush=True,
                )
            del order
        if args.final_closing_mean:
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
        test_mse = query_mse(
            model, queries.batches(mean, scale, args.query_batch_size),
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
            "n": args.n,
            "k": args.k,
            "ranks": ranks,
            "elapsed_s": time.perf_counter() - start,
            "moving_pass_mse": (
                model.obj[-1].item() / model.recipe["epoch_rows"][-1] * scale**2
            ),
            "test_mse": test_mse,
            "test_query_count": queries.n,
            "query_precision": args.query_precision,
            "scale": scale,
            "closing": closing,
            "ssd": loader.stats,
            "initialization": initial_stats,
            "recipe": model.recipe,
            "iteration_stats": model.iteration_stats,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        }
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
        "--batch-size",
        type=int,
        help="global training batch size across all workers; refits the EMA step",
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
