# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train sharded IVF e-means from SSD windows, with a frozen closing scan.

buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_training_from_ssd -- --launch 2
The default case is 200M DINO points and 1M centroids. Every rank scans its
own row range. Prefetch is automatic; --ssd-prefetch-chunk-groups defaults
to 4 and groups random windows by scan chunks. --final-closing-mean makes
a sequential pass against frozen centroids.
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
from faiss.contrib.e_means.e_means import _epoch_seed


def self_test(device):
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
        local_n = 128
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
                niter=1,
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
            for rows in loader.batches(loader.order(5)):
                model.update(model.accumulate(rows))
            model.end_epoch()
            self_test_queries(model, device)
            sums = torch.zeros_like(model._centroids)
            counts = sums.new_zeros(model.k)
            prepared = model._prepare_assignment()
            # Independent iteration over the original host corpus.
            for a in range(0, local_n * ranks, 16):
                rows = torch.from_numpy(data[a : a + 16]).to(device).float()
                winner = model._winning_labels(rows, prepared)
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
            with _closing(loader.batches(sequential=True)) as batches:
                model.final_closing_mean(batches)
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
            "async batches, sequential scan and frozen closing means",
            flush=True,
        )


def run(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    source = None
    try:
        if args.self_test:
            self_test(device)
            return
        rank, ranks = dist.get_rank(), dist.get_world_size()
        if args.n % ranks or args.k % ranks:
            raise ValueError("n and k must divide by worker count")
        if args.query_batch_size < 1:
            raise ValueError("query batch size must be positive")
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
            niter=args.epochs,
            operating_point="quality",
            seed=args.seed,
            nlist=args.nlist,
            nprobe=args.nprobe,
            assignment_backend=args.assignment_backend,
            group=dist.group.WORLD,
            verbose=True,
        )
        from faiss.contrib.e_means.e_means import _resolve_recipe

        batch, _ = _resolve_recipe(
            args.n, args.k, args.epochs, "quality", None, None
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
            print(
                f"SSD batch={batch} local_batch={batch // ranks} "
                f"prefetch={loader.n_prefetch} "
                f"window_rows={loader.window_rows} "
                f"chunk_groups={loader.chunk_groups} "
                f"direct={all(source.direct)} center_scale={scale:.6g}",
                flush=True,
            )
        for epoch in range(args.epochs):
            order = loader.order(_epoch_seed(args.seed + rank, epoch))
            for step, rows in enumerate(
                loader.batches(order, mean=mean, scale=scale)
            ):
                model.update(model.accumulate(rows))
                if rank == 0 and step % args.verbose_every == 0:
                    print(
                        f"epoch={epoch + 1} step={step} "
                        f"elapsed={time.perf_counter() - start:.1f}s",
                        flush=True,
                    )
            model.end_epoch()
            test_mse = query_mse(
                model, queries.batches(mean, scale, args.query_batch_size)
            ) * scale**2
            model.iteration_stats[-1]["test_mse"] = test_mse
            if rank == 0:
                print(f"epoch={epoch + 1} test_mse={test_mse:.6f}", flush=True)
            del order
        if args.final_closing_mean:
            if rank == 0:
                print("Sequential frozen final-closing-mean scan", flush=True)
            with _closing(
                loader.batches(sequential=True, mean=mean, scale=scale)
            ) as batches:
                model.final_closing_mean(batches)
        else:
            model.finalize()
        test_mse = query_mse(
            model, queries.batches(mean, scale, args.query_batch_size)
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
            "moving_pass_mse": model.obj[-1].item() / args.n * scale**2,
            "test_mse": test_mse,
            "test_query_count": queries.n,
            "scale": scale,
            "closing": closing,
            "ssd": loader.stats,
            "initialization": initial_stats,
            "recipe": model.recipe,
            "iteration_stats": model.iteration_stats,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        }
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
