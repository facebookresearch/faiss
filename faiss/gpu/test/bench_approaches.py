#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark 4 approaches for multi-GPU CAGRA -> HNSW:
  A. IndexShards (independent HNSW per shard, no stitching)
  B. Unified + GPU brute-force sampled stitching (exact cross-shard NNs)
  C. Unified + CPU HNSW stitching (approximate cross-shard NNs)
  D. all_neighbors + optimize (multi-GPU overlapping clusters, no stitching)

Usage:
  buck run @//mode/opt fbcode//faiss/gpu/test:bench_approaches -- \\
      --data /path/to/vectors.npy --approaches D --multi-gpu-optimize
"""

import argparse
import os
import sys
import tempfile
import time

_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_SENTINEL = "/tmp/bench_done"
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

if _local_rank == 0 and os.path.exists(_SENTINEL):
    os.remove(_SENTINEL)
elif _local_rank != 0:
    while not os.path.exists(_SENTINEL):
        time.sleep(5)
    sys.exit(0)

sys.stdout = sys.stderr

import faiss
import numpy as np


_t0 = time.time()


def compute_recall(I_test, I_gt, k):
    nq = I_test.shape[0]
    return np.mean([len(set(I_test[i]) & set(I_gt[i])) / k for i in range(nq)])


def load_from_hive(n, table, namespace, ds, ts, oncall="faiss", batch_size=4096):
    """Stream `n` real `prefilter_embedding` vectors from a Hive table into a
    preallocated (n, d) float32 array using the in-process Koski reader.

    This is the no-duplicate path: it reads genuine rows from the warehouse on
    the MAST host (no Manifold .npy, no tiling). `batch_iterator()` yields a list
    of rows; each row is a tuple of columns; row[0] is the embedding (list of d
    floats). dim is inferred from the first row.
    """
    import koski.dataframes as kd

    print(
        f"Loading {n:,} vectors from Hive {namespace}/{table} "
        f"(ds={ds}, ts={ts}) via Koski...",
        flush=True,
    )
    t0 = time.time()
    ctx = kd.create_ctx(
        oncall=oncall,
        use_case=kd.UseCase.TEST,
        description="multi-GPU CAGRA benchmark data load",
    )
    df = kd.data_warehouse(namespace=namespace, table=table, session_ctx=ctx)
    df = df.filter(f"ds = '{ds}' AND ts = '{ts}'").map(["prefilter_embedding"])
    df = df.limit(n).rebatch(batch_size=batch_size)

    xb = None
    i = 0
    for batch in df.batch_iterator():
        rows = [row[0] for row in batch]
        arr = np.asarray(rows, dtype=np.float32)
        if xb is None:
            xb = np.empty((n, arr.shape[1]), dtype=np.float32)
        m = min(arr.shape[0], n - i)
        xb[i : i + m] = arr[:m]
        i += m
        if i % (batch_size * 50) == 0:
            elapsed = time.time() - t0
            print(f"  {i:,}/{n:,} ({elapsed:.0f}s)", flush=True)
        if i >= n:
            break
    if xb is None or i == 0:
        print("ERROR: Hive returned no rows", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {i:,} vectors from Hive in {time.time() - t0:.1f}s")
    return xb[:i]


def approach_a_indexshards(xb, d, num_gpus, graph_degree=32, save_dir=None):
    """Build N independent CAGRA->HNSW, wrap in IndexShards."""
    n = xb.shape[0]
    shard_size = (n + num_gpus - 1) // num_gpus
    timings = {}

    t0 = time.time()
    shards = []
    for g in range(num_gpus):
        start = g * shard_size
        end = min(start + shard_size, n)
        shard_data = xb[start:end]
        if len(shard_data) == 0:
            continue

        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexCagraConfig()
        config.graph_degree = graph_degree
        config.intermediate_graph_degree = graph_degree * 2
        config.build_algo = faiss.graph_build_algo_NN_DESCENT
        idx = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2, config)
        idx.train(shard_data)

        cpu_idx = faiss.IndexHNSWCagra()
        idx.copyTo(cpu_idx)
        shards.append(cpu_idx)
    timings["build"] = time.time() - t0

    t0 = time.time()
    index = faiss.IndexShards(d, True, True)
    for s in shards:
        index.add_shard(s)
    timings["assemble"] = time.time() - t0

    out_dir = save_dir or tempfile.mkdtemp()
    t0 = time.time()
    total_bytes = 0
    for i, s in enumerate(shards):
        path = os.path.join(out_dir, f"A_shard_{i}_{n // 1_000_000}M.faiss")
        faiss.write_index(s, path)
        total_bytes += os.path.getsize(path)
    timings["serialize"] = time.time() - t0
    timings["file_size_gb"] = total_bytes / 1e9

    return index, shards, timings


def approach_b_unified_gpu_stitch(
    xb,
    d,
    num_gpus,
    graph_degree=32,
    stitch_per_shard=100000,
    stitch_k=16,
    save_dir=None,
):
    """Unified SNMG build + GPU brute-force sampled stitching (exact NNs)."""
    n = xb.shape[0]
    timings = {}

    resources = faiss.GpuResourcesVector()
    devices = faiss.Int32Vector()
    res_list = []
    for i in range(num_gpus):
        res = faiss.StandardGpuResources()
        res_list.append(res)
        resources.push_back(res)
        devices.push_back(i)

    config = faiss.GpuIndexCagraConfig()
    config.graph_degree = graph_degree
    config.intermediate_graph_degree = graph_degree * 2
    config.build_algo = faiss.graph_build_algo_NN_DESCENT
    index = faiss.GpuIndexCagra(res_list[0], d, faiss.METRIC_L2, config)

    t0 = time.time()
    index.trainMultiGpu(
        n,
        faiss.swig_ptr(xb),
        resources,
        devices,
        stitch_per_shard,
        stitch_k,
        1,
    )
    timings["build"] = time.time() - t0

    t0 = time.time()
    cpu_index = faiss.IndexHNSWCagra()
    index.copyTo(cpu_index)
    timings["copyTo"] = time.time() - t0

    out_dir = save_dir or tempfile.mkdtemp()
    path = os.path.join(out_dir, f"B_unified_{n // 1_000_000}M.faiss")
    t0 = time.time()
    faiss.write_index(cpu_index, path)
    timings["file_size_gb"] = os.path.getsize(path) / 1e9
    timings["serialize"] = time.time() - t0

    return cpu_index, timings


def approach_c_unified_cpu_stitch(
    xb,
    d,
    num_gpus,
    graph_degree=32,
    stitch_per_shard=0,
    stitch_k=16,
    save_dir=None,
):
    """Unified SNMG build + CPU HNSW stitching (approximate NNs)."""
    n = xb.shape[0]
    timings = {}

    resources = faiss.GpuResourcesVector()
    devices = faiss.Int32Vector()
    res_list = []
    for i in range(num_gpus):
        res = faiss.StandardGpuResources()
        res_list.append(res)
        resources.push_back(res)
        devices.push_back(i)

    config = faiss.GpuIndexCagraConfig()
    config.graph_degree = graph_degree
    config.intermediate_graph_degree = graph_degree * 2
    config.build_algo = faiss.graph_build_algo_NN_DESCENT
    index = faiss.GpuIndexCagra(res_list[0], d, faiss.METRIC_L2, config)

    t0 = time.time()
    index.trainMultiGpu(
        n,
        faiss.swig_ptr(xb),
        resources,
        devices,
        stitch_per_shard,
        stitch_k,
        0,
    )
    timings["build"] = time.time() - t0

    t0 = time.time()
    cpu_index = faiss.IndexHNSWCagra()
    index.copyTo(cpu_index)
    timings["copyTo"] = time.time() - t0

    out_dir = save_dir or tempfile.mkdtemp()
    path = os.path.join(out_dir, f"C_unified_{n // 1_000_000}M.faiss")
    t0 = time.time()
    faiss.write_index(cpu_index, path)
    timings["file_size_gb"] = os.path.getsize(path) / 1e9
    timings["serialize"] = time.time() - t0

    return cpu_index, timings


def approach_d_all_neighbors(
    xb,
    d,
    num_gpus,
    graph_degree=32,
    save_dir=None,
    n_clusters=0,
    overlap_factor=0,
    multi_gpu_optimize=False,
    build_algo=0,
    base_level_only=False,
    intermediate_graph_degree=48,
    refinement_rate=2.0,
    ivfpq_search_batch=0,
):
    """all_neighbors + cagra::optimize → unified CAGRA graph. No stitching."""
    n = xb.shape[0]
    timings = {}

    devices = faiss.Int32Vector()
    res_list = []
    for i in range(num_gpus):
        res = faiss.StandardGpuResources()
        res_list.append(res)
        devices.push_back(i)

    config = faiss.GpuIndexCagraConfig()
    config.graph_degree = graph_degree
    config.intermediate_graph_degree = intermediate_graph_degree
    config.build_algo = faiss.graph_build_algo_NN_DESCENT
    index = faiss.GpuIndexCagra(res_list[0], d, faiss.METRIC_L2, config)

    t0 = time.time()
    index.trainAllNeighbors(
        n,
        faiss.swig_ptr(xb),
        devices,
        n_clusters,
        overlap_factor,
        multi_gpu_optimize,
        build_algo,
        refinement_rate,
        ivfpq_search_batch,
    )
    timings["build"] = time.time() - t0

    t0 = time.time()
    cpu_index = faiss.IndexHNSWCagra()
    cpu_index.base_level_only = base_level_only
    index.copyTo(cpu_index)
    timings["copyTo"] = time.time() - t0

    out_dir = save_dir or tempfile.mkdtemp()
    path = os.path.join(out_dir, f"D_allneighbors_{n // 1_000_000}M.faiss")
    t0 = time.time()
    faiss.write_index(cpu_index, path)
    timings["file_size_gb"] = os.path.getsize(path) / 1e9
    timings["serialize"] = time.time() - t0

    return cpu_index, timings


def _set_ef(index, ef):
    if isinstance(index, faiss.IndexShards):
        for i in range(index.count()):
            shard = faiss.downcast_index(index.at(i))
            if hasattr(shard, "hnsw"):
                shard.hnsw.efSearch = ef
    elif hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef


def eval_recall(index, xq, Igt, k=10, ef_values=None):
    if ef_values is None:
        ef_values = [64, 128, 256]
    results = {}

    for ef in ef_values:
        _set_ef(index, ef)
        D, I = index.search(xq, k)
        results[ef] = compute_recall(I, Igt, k)

    return results


def eval_kcycles(index, xq, k=10, ef_values=None, warmup=3, repeat=5):
    """Measure kcycles/query using vench CycleCounter."""
    if ef_values is None:
        ef_values = [64, 128, 256]

    try:
        from vector_search.vench.perf_cycles import CycleCounter
    except ImportError:
        print("  [kcycles] vench not available, skipping")
        return {}

    cc = CycleCounter()
    if not cc.available:
        print("  [kcycles] perf_event_open not available, skipping")
        return {}

    nq = xq.shape[0]
    results = {}
    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)

    measure_index = index
    if isinstance(index, faiss.IndexShards) and index.count() > 0:
        measure_index = faiss.IndexShards(index.d, False, True)
        for i in range(index.count()):
            measure_index.add_shard(index.at(i))

    for ef in ef_values:
        _set_ef(index, ef)
        _set_ef(measure_index, ef)
        for _ in range(warmup):
            measure_index.search(xq, k)

        best_kcycles = float("inf")
        for _ in range(repeat):
            c0 = cc.read()
            measure_index.search(xq, k)
            c1 = cc.read()
            kc = (c1 - c0) / 1000.0 / nq
            best_kcycles = min(best_kcycles, kc)
        results[ef] = best_kcycles

    faiss.omp_set_num_threads(prev_threads)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-GPU CAGRA->HNSW approaches"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to .npy file with vectors (n, d) float32 "
        "(not needed when --hive-table is set)",
    )
    parser.add_argument("--n", type=int, default=0, help="Use first N vectors (0=all)")
    parser.add_argument("--nq", type=int, default=1000, help="Number of queries")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--stitch-k", type=int, default=16)
    parser.add_argument("--stitch-per-shard", type=int, default=100000)
    parser.add_argument(
        "--n-clusters", type=int, default=0, help="all_neighbors n_clusters (0=auto)"
    )
    parser.add_argument(
        "--overlap-factor",
        type=int,
        default=0,
        help="all_neighbors overlap_factor (0=default 2)",
    )
    parser.add_argument(
        "--multi-gpu-optimize",
        action="store_true",
        help="Partition detour counting across GPUs",
    )
    parser.add_argument(
        "--build-algo",
        type=int,
        default=0,
        help="0=nn_descent, 1=brute_force, 2=ivf_pq",
    )
    parser.add_argument(
        "--base-level-only",
        action="store_true",
        help="Skip HNSW upper-level construction in copyTo",
    )
    parser.add_argument(
        "--intermediate-graph-degree",
        type=int,
        default=48,
        help="kNN graph degree before pruning (use 32 for 100M+)",
    )
    parser.add_argument(
        "--refinement-rate",
        type=float,
        default=2.0,
        help="IVF-PQ refinement multiplier (build-algo=2 only); "
        "higher trades build time for recall (cuVS default 2.0)",
    )
    parser.add_argument(
        "--ivfpq-search-batch",
        type=int,
        default=0,
        help="Cap IVF-PQ search max_internal_batch_size in the all_neighbors "
        "build (build-algo=2). 0=cuVS default (131072); smaller (e.g. 8192) "
        "bounds GPU search workspace to avoid OOM at 100M (recall-neutral)",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default="A,B,C,D",
        help="A (IndexShards), B (GPU stitch), " "C (CPU stitch), D (all_neighbors)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="/tmp/cagra_bench/indices",
        help="Directory to persist serialized indices for later experiments",
    )
    parser.add_argument(
        "--kcycles-only",
        action="store_true",
        help="Load persisted indices and measure kcycles only",
    )
    parser.add_argument(
        "--manifold-data",
        type=str,
        default="",
        help="If data file doesn't exist, download from this Manifold path",
    )
    parser.add_argument(
        "--hive-table",
        type=str,
        default="",
        help="If set, stream real vectors from this Hive table via Koski "
        "(no Manifold/tiling). Requires --n for the row count.",
    )
    parser.add_argument("--hive-namespace", type=str, default="feed_fblearner")
    parser.add_argument("--hive-ds", type=str, default="2026-06-07")
    parser.add_argument("--hive-ts", type=str, default="2026-06-07+19:00:99")
    parser.add_argument("--hive-oncall", type=str, default="faiss")
    args = parser.parse_args()

    if "CUVS" not in faiss.get_compile_options():
        print("ERROR: faiss not compiled with cuVS support", file=sys.stderr)
        sys.exit(1)

    if args.num_gpus > 0:
        num_gpus = args.num_gpus
    else:
        num_gpus = min(faiss.get_num_gpus(), 8)
    if num_gpus < 2:
        print(
            f"ERROR: need >= 2 GPUs, found {faiss.get_num_gpus()}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.hive_table:
        # Preferred path: stream real (non-duplicated) vectors from Hive.
        if args.n <= 0:
            print("ERROR: --hive-table requires --n", file=sys.stderr)
            sys.exit(1)
        xb = load_from_hive(
            args.n,
            args.hive_table,
            args.hive_namespace,
            args.hive_ds,
            args.hive_ts,
            oncall=args.hive_oncall,
        )
        n, d = xb.shape
        print(f"  Loaded: {n:,} vectors, dim={d}")
    else:
        if not args.data:
            print("ERROR: provide --data (.npy) or --hive-table", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.data) and args.manifold_data:
            os.makedirs(os.path.dirname(args.data), exist_ok=True)
            try:
                from manifold.clients.python import ManifoldClient
            except ImportError:
                print(
                    "ERROR: manifold client not available and "
                    f"data file {args.data} not found",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"Downloading from Manifold: {args.manifold_data}")
            parts = args.manifold_data.split("/", 1)
            with ManifoldClient.get_client(bucket=parts[0]) as mc:
                mc.sync_get(path=parts[1], output=args.data)
            print(f"  Downloaded to {args.data}")
        elif not os.path.exists(args.data):
            print(f"ERROR: data file {args.data} not found", file=sys.stderr)
            sys.exit(1)

        print(f"Loading data from {args.data}...")
        xb = np.load(args.data).astype(np.float32)
        if args.n > 0 and args.n > xb.shape[0]:
            reps = (args.n + xb.shape[0] - 1) // xb.shape[0]
            print(
                f"  WARNING: tiling {reps}x to reach {args.n:,} vectors "
                "(duplicates inflate recall; prefer --hive-table)..."
            )
            xb = np.tile(xb, (reps, 1))[: args.n]
        elif args.n > 0:
            xb = xb[: args.n]
        n, d = xb.shape
        print(f"  Loaded: {n:,} vectors, dim={d}")

    xq = xb[: args.nq].copy()
    k = 10

    print(
        f"Computing ground truth (brute-force on {n:,} vectors)...",
        flush=True,
    )
    t0 = time.time()
    gt = faiss.IndexFlatL2(d)
    gt.add(xb)
    Dgt, Igt = gt.search(xq, k)
    del gt
    print(f"  Done in {time.time() - t0:.1f}s")

    print(
        f"\nConfig: n={n:,}, d={d}, GPUs={num_gpus}, "
        f"graph_degree={args.graph_degree}"
    )
    print(
        f"        stitch_k={args.stitch_k}, "
        f"stitch_per_shard={args.stitch_per_shard}"
    )
    print(
        f"        [D] n_clusters={args.n_clusters or 'auto'}, "
        f"overlap_factor={args.overlap_factor or 'auto'}, "
        f"intermediate_graph_degree={args.intermediate_graph_degree}, "
        f"build_algo={args.build_algo}, refinement_rate={args.refinement_rate}"
    )
    print()

    approaches_to_run = [a.strip().upper() for a in args.approaches.split(",")]
    results_table = []
    save_dir = args.index_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.kcycles_only:
        print(f"Loading persisted indices from {save_dir} (kcycles-only mode)")
        n_tag = f"{n // 1_000_000}M"
        for approach in approaches_to_run:
            if approach == "A":
                pre = "A_shard_"
                suf = f"_{n_tag}.faiss"
                shard_files = sorted(
                    f
                    for f in os.listdir(save_dir)
                    if f.startswith(pre) and f.endswith(suf)
                )
                if not shard_files:
                    print(f"  No A files for {n_tag}")
                    continue
                shards = [
                    faiss.read_index(os.path.join(save_dir, f)) for f in shard_files
                ]
                idx = faiss.IndexShards(d, True, True)
                for s in shards:
                    idx.add_shard(s)
                label = f"A: IndexShards ({len(shard_files)})"
            else:
                prefixes = {
                    "B": "B_unified",
                    "C": "C_unified",
                    "D": "D_allneighbors",
                }
                prefix = prefixes.get(approach)
                if not prefix:
                    prefix = f"{approach}_unified"
                path = os.path.join(save_dir, f"{prefix}_{n_tag}.faiss")
                if not os.path.exists(path):
                    print(f"  No {prefix} for {n_tag}")
                    continue
                idx = faiss.read_index(path)
                label = f"{approach}: unified"

            recall = eval_recall(idx, xq, Igt, k)
            kcycles = eval_kcycles(idx, xq, k)
            for ef in sorted(recall.keys()):
                kc = kcycles.get(ef, 0)
                print(
                    f"  {label:25s} ef={ef:>3d}  recall={recall[ef]:.4f}  "
                    f"kcyc/q={kc:.0f}"
                )
            del idx
        return

    print(f"Indices will be saved to: {save_dir}")

    if "A" in approaches_to_run:
        print("=" * 60)
        print("APPROACH A: IndexShards (independent HNSW per shard)")
        print("=" * 60)
        index_a, shards_a, timings_a = approach_a_indexshards(
            xb,
            d,
            num_gpus,
            args.graph_degree,
            save_dir=save_dir,
        )
        recall_a = eval_recall(index_a, xq, Igt, k)
        kcycles_a = eval_kcycles(index_a, xq, k)
        total_build_a = timings_a["build"] + timings_a.get("assemble", 0)
        print(f"  Build:     {timings_a['build']:.1f}s")
        print(
            f"  Serialize: {timings_a['serialize']:.1f}s "
            f"({timings_a['file_size_gb']:.2f} GB, {num_gpus} files)"
        )
        for ef in sorted(recall_a.keys()):
            kc = kcycles_a.get(ef, 0)
            print(
                f"  efSearch={ef}: recall@{k}={recall_a[ef]:.4f}  "
                f"kcycles/q={kc:.0f}"
            )
        results_table.append(
            (
                "A: IndexShards",
                total_build_a,
                timings_a["serialize"],
                timings_a["file_size_gb"],
                recall_a,
                kcycles_a,
            )
        )
        del index_a, shards_a
        print()

    if "B" in approaches_to_run:
        print("=" * 60)
        print(
            f"APPROACH B: Unified + GPU brute-force stitch "
            f"(sps={args.stitch_per_shard}, k={args.stitch_k})"
        )
        print("=" * 60)
        index_b, timings_b = approach_b_unified_gpu_stitch(
            xb,
            d,
            num_gpus,
            args.graph_degree,
            args.stitch_per_shard,
            args.stitch_k,
            save_dir=save_dir,
        )
        recall_b = eval_recall(index_b, xq, Igt, k)
        kcycles_b = eval_kcycles(index_b, xq, k)
        total_build_b = timings_b["build"] + timings_b["copyTo"]
        print(f"  Build (SNMG+GPU-stitch): {timings_b['build']:.1f}s")
        print(f"  copyTo:                  {timings_b['copyTo']:.1f}s")
        print(
            f"  Serialize:               {timings_b['serialize']:.1f}s "
            f"({timings_b['file_size_gb']:.2f} GB)"
        )
        for ef in sorted(recall_b.keys()):
            kc = kcycles_b.get(ef, 0)
            print(
                f"  efSearch={ef}: recall@{k}={recall_b[ef]:.4f}  "
                f"kcycles/q={kc:.0f}"
            )
        results_table.append(
            (
                "B: GPU-brute",
                total_build_b,
                timings_b["serialize"],
                timings_b["file_size_gb"],
                recall_b,
                kcycles_b,
            )
        )
        del index_b
        print()

    if "C" in approaches_to_run:
        print("=" * 60)
        print(f"APPROACH C: Unified + CPU stitch (sps=0, k={args.stitch_k})")
        print("=" * 60)
        index_c, timings_c = approach_c_unified_cpu_stitch(
            xb,
            d,
            num_gpus,
            args.graph_degree,
            0,
            args.stitch_k,
            save_dir=save_dir,
        )
        recall_c = eval_recall(index_c, xq, Igt, k)
        kcycles_c = eval_kcycles(index_c, xq, k)
        total_build_c = timings_c["build"] + timings_c["copyTo"]
        print(f"  Build (SNMG+CPU-stitch): {timings_c['build']:.1f}s")
        print(f"  copyTo:                  {timings_c['copyTo']:.1f}s")
        print(
            f"  Serialize:               {timings_c['serialize']:.1f}s "
            f"({timings_c['file_size_gb']:.2f} GB)"
        )
        for ef in sorted(recall_c.keys()):
            kc = kcycles_c.get(ef, 0)
            print(
                f"  efSearch={ef}: recall@{k}={recall_c[ef]:.4f}  "
                f"kcycles/q={kc:.0f}"
            )
        results_table.append(
            (
                "C: CPU-stitch",
                total_build_c,
                timings_c["serialize"],
                timings_c["file_size_gb"],
                recall_c,
                kcycles_c,
            )
        )
        del index_c
        print()

    if "D" in approaches_to_run:
        print("=" * 60)
        print("APPROACH D: all_neighbors + cagra::optimize (no stitching)")
        print("=" * 60)
        index_d, timings_d = approach_d_all_neighbors(
            xb,
            d,
            num_gpus,
            args.graph_degree,
            save_dir=save_dir,
            n_clusters=args.n_clusters,
            overlap_factor=args.overlap_factor,
            multi_gpu_optimize=args.multi_gpu_optimize,
            build_algo=args.build_algo,
            base_level_only=args.base_level_only,
            intermediate_graph_degree=args.intermediate_graph_degree,
            refinement_rate=args.refinement_rate,
            ivfpq_search_batch=args.ivfpq_search_batch,
        )
        recall_d = eval_recall(index_d, xq, Igt, k)
        kcycles_d = eval_kcycles(index_d, xq, k)
        total_build_d = timings_d["build"] + timings_d["copyTo"]
        print(f"  Build (allneighbors+optimize): {timings_d['build']:.1f}s")
        print(f"  copyTo:                        {timings_d['copyTo']:.1f}s")
        ser = timings_d["serialize"]
        gb = timings_d["file_size_gb"]
        print(f"  Serialize:                     {ser:.1f}s " f"({gb:.2f} GB)")
        # Index-build wall-clock, isolated from data load (Koski/Manifold) and
        # ground-truth: this is the headline build->serialize number.
        index_total_d = (
            timings_d["build"] + timings_d["copyTo"] + timings_d["serialize"]
        )
        print(
            f"  >>> INDEX build->serialize total "
            f"(excl. data load + ground truth): "
            f"{index_total_d:.1f}s ({index_total_d / 60:.1f} min)"
        )
        for ef in sorted(recall_d.keys()):
            kc = kcycles_d.get(ef, 0)
            print(
                f"  efSearch={ef}: recall@{k}=" f"{recall_d[ef]:.4f}  kcyc/q={kc:.0f}"
            )
        results_table.append(
            (
                "D: all_neighbors",
                total_build_d,
                timings_d["serialize"],
                timings_d["file_size_gb"],
                recall_d,
                kcycles_d,
            )
        )
        del index_d
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = (
        f"{'Approach':<20s} {'Build':>7s} {'Ser':>5s} {'GB':>5s}"
        f" {'ef':>4s} {'recall':>7s} {'kcyc/q':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, build, ser, size, recalls, kcycles in results_table:
        for ef in [64, 128, 256]:
            r = recalls.get(ef, 0)
            kc = kcycles.get(ef, 0)
            bld = f"{build:.0f}" if ef == 64 else ""
            sr = f"{ser:.0f}" if ef == 64 else ""
            sz = f"{size:.1f}" if ef == 64 else ""
            nm = name if ef == 64 else ""
            print(
                f"{nm:<20s} {bld:>7s} {sr:>5s} {sz:>5s}"
                f" {ef:>4d} {r:>7.4f} {kc:>7.0f}"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        open(_SENTINEL, "w").close()
