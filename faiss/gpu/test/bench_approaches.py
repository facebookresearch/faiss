#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark the multi-GPU CAGRA -> HNSW build: cuVS all_neighbors builds a knn
graph over overlapping clusters across all GPUs, cagra::optimize prunes it into
a single unified graph, and copyTo produces a CPU IndexHNSWCagra.

Measures build/copyTo/serialize wall-clock, index size, and recall@10 vs
brute-force ground truth at several efSearch values.

Usage:
  buck run @//mode/opt fbcode//faiss/gpu/test:bench_approaches -- \\
      --data /path/to/vectors.npy
"""

import argparse
import os
import sys
import tempfile
import time
import faiss
import numpy as np

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


_t0 = time.time()


EF_VALUES = [16, 32, 64, 128, 256, 512]


def compute_recall(I_test, I_gt, k):
    nq = I_test.shape[0]
    return np.mean([len(set(I_test[i]) & set(I_gt[i])) / k for i in range(nq)])


def load_from_hive(
    n, table, namespace, ds, ts, oncall="faiss", batch_size=4096
):
    """Stream `n` real `prefilter_embedding` vectors from a Hive table into a
    preallocated (n, d) float32 array using the in-process Koski reader.

    This is the no-duplicate path: it reads genuine rows from the warehouse on
    the MAST host (no Manifold .npy, no tiling).
    `batch_iterator()` yields a list
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


BUILD_ALGOS = ("ivf_pq", "nn_descent", "brute_force")


def cagra_build_algo(name):
    return getattr(faiss, f"graph_build_algo_{name.upper()}")


def index_filename(n):
    return f"cagra_hnsw_{n // 1_000_000}M.faiss"


def build_cagra_hnsw(
    xb,
    d,
    num_gpus,
    graph_degree=32,
    save_dir=None,
    n_clusters=0,
    overlap_factor=0,
    build_algo="ivf_pq",
    base_level_only=False,
    intermediate_graph_degree=48,
    refinement_rate=1.0,
    ivfpq_search_batch=0,
    guarantee_connectivity=False,
    ivfpq_size_from_cluster=True,
    gpu_hnsw_upper_levels=False,
    gpu_hnsw_igd=0,
    gpu_hnsw_guarantee_connectivity=True,
    ef_construction=0,
    entrypoints=0,
):
    """Multi-GPU all_neighbors build -> optimize -> CPU IndexHNSWCagra."""
    # The multi-GPU build does not copy the dataset, so xb must stay alive and
    # contiguous until copyTo() below has run.
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    n = xb.shape[0]
    timings = {}

    res_list = [faiss.StandardGpuResources() for _ in range(num_gpus)]

    devices = faiss.Int32Vector()
    for i in range(num_gpus):
        devices.push_back(i)

    all_neighbors = faiss.AllNeighborsCagraConfig()
    all_neighbors.n_clusters = n_clusters
    all_neighbors.overlap_factor = overlap_factor
    all_neighbors.ivf_pq_search_batch_size = ivfpq_search_batch
    all_neighbors.refinement_rate = refinement_rate
    all_neighbors.ivf_pq_size_from_cluster = ivfpq_size_from_cluster

    config = faiss.GpuIndexCagraConfig()
    config.graph_degree = graph_degree
    config.intermediate_graph_degree = intermediate_graph_degree
    config.build_algo = cagra_build_algo(build_algo)
    config.guarantee_connectivity = guarantee_connectivity
    config.gpu_hnsw_upper_levels = gpu_hnsw_upper_levels
    config.gpu_hnsw_intermediate_degree = gpu_hnsw_igd
    config.gpu_hnsw_guarantee_connectivity = gpu_hnsw_guarantee_connectivity
    config.devices = devices
    config.all_neighbors_params = all_neighbors

    index = faiss.GpuIndexCagra(res_list[0], d, faiss.METRIC_L2, config)

    t0 = time.time()
    index.train(xb)
    timings["build"] = time.time() - t0

    t0 = time.time()
    cpu_index = faiss.IndexHNSWCagra()
    cpu_index.base_level_only = base_level_only
    if ef_construction > 0:
        cpu_index.hnsw.efConstruction = ef_construction
    if entrypoints > 0:
        cpu_index.num_base_level_search_entrypoints = entrypoints
    index.copyTo(cpu_index)
    timings["copyTo"] = time.time() - t0

    out_dir = save_dir or tempfile.mkdtemp()
    path = os.path.join(out_dir, index_filename(n))
    t0 = time.time()
    faiss.write_index(cpu_index, path)
    timings["file_size_gb"] = os.path.getsize(path) / 1e9
    timings["serialize"] = time.time() - t0

    return cpu_index, timings


def _set_ef(index, ef):
    index.hnsw.efSearch = ef


def eval_recall(index, xq, Igt, k=10, ef_values=None):
    if ef_values is None:
        ef_values = EF_VALUES
    results = {}

    for ef in ef_values:
        _set_ef(index, ef)
        D, I = index.search(xq, k)
        results[ef] = compute_recall(I, Igt, k)

    return results


def eval_qps(index, xq, k=10, ef_values=None, repeat=5):
    """Full-batch throughput at each efSearch, best of `repeat` runs."""
    if ef_values is None:
        ef_values = EF_VALUES
    nq = xq.shape[0]
    results = {}

    for ef in ef_values:
        _set_ef(index, ef)
        index.search(xq, k)  # warmup
        best = float("inf")
        for _ in range(repeat):
            t0 = time.time()
            index.search(xq, k)
            best = min(best, time.time() - t0)
        results[ef] = nq / best

    return results


def eval_kcycles(index, xq, k=10, ef_values=None, warmup=3, repeat=5):
    """Measure kcycles/query using vench CycleCounter."""
    if ef_values is None:
        ef_values = EF_VALUES

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

    for ef in ef_values:
        _set_ef(index, ef)
        for _ in range(warmup):
            index.search(xq, k)

        best_kcycles = float("inf")
        for _ in range(repeat):
            c0 = cc.read()
            index.search(xq, k)
            c1 = cc.read()
            kc = (c1 - c0) / 1000.0 / nq
            best_kcycles = min(best_kcycles, kc)
        results[ef] = best_kcycles

    faiss.omp_set_num_threads(prev_threads)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the multi-GPU CAGRA->HNSW build"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to .npy file with vectors (n, d) float32 "
        "(not needed when --hive-table is set)",
    )
    parser.add_argument(
        "--n", type=int, default=0, help="Use first N vectors (0=all)"
    )
    parser.add_argument(
        "--nq", type=int, default=1000, help="Number of queries"
    )
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="all_neighbors n_clusters (0=auto)",
    )
    parser.add_argument(
        "--overlap-factor",
        type=int,
        default=0,
        help="all_neighbors overlap_factor (0=default 2)",
    )
    parser.add_argument(
        "--build-algo",
        choices=BUILD_ALGOS,
        default="ivf_pq",
        help="graph_build_algo used for the kNN graph (brute_force is "
        "multi-GPU only and O(N^2 D) per cluster)",
    )
    parser.add_argument(
        "--ivfpq-size-from-cluster",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive IVF-PQ params from the per-cluster subproblem. "
        "--no-ivfpq-size-from-cluster sizes them from the full dataset, "
        "which over-partitions every cluster",
    )
    parser.add_argument(
        "--guarantee-connectivity",
        action="store_true",
        help="Run the MST pass in cagra::optimize so the pruned graph is "
        "guaranteed connected (cuVS default is on; costs build time)",
    )
    parser.add_argument(
        "--entrypoints-sweep",
        type=str,
        default="",
        help="Comma-separated num_base_level_search_entrypoints values to "
        "sweep at search time (base-level-only only), e.g. 32,256,1024",
    )
    parser.add_argument(
        "--gpu-hnsw-upper-levels",
        action="store_true",
        help="Build HNSW levels >=1 as GPU CAGRA subgraphs in copyTo instead "
        "of incremental CPU insertion (base-level-only off only)",
    )
    parser.add_argument(
        "--gpu-hnsw-igd",
        type=int,
        default=0,
        help="intermediate_graph_degree for the per-level GPU subgraphs "
        "(0 = 2x the upper-level degree)",
    )
    parser.add_argument(
        "--gpu-hnsw-guarantee-connectivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MST connectivity pass on the per-level GPU subgraphs "
        "(on by default; greedy descent needs connectivity)",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=0,
        help="Override hnsw.efConstruction used when building upper levels "
        "(0 = faiss default of 40)",
    )
    parser.add_argument(
        "--entrypoints",
        type=int,
        default=0,
        help="Override num_base_level_search_entrypoints (0 = faiss default "
        "of 256); only affects base-level-only search",
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
        default=1.0,
        help="IVF-PQ refinement multiplier (build-algo=2 only). Sets "
        "candidate_k = k * rate; the refine pass runs on the host so cost is "
        "linear in this. Measured best at 1.0 (default)",
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
            print(
                "ERROR: provide --data (.npy) or --hive-table",
                file=sys.stderr,
            )
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
        f"        n_clusters={args.n_clusters or 'auto'}, "
        f"overlap_factor={args.overlap_factor or 'auto'}, "
        f"intermediate_graph_degree={args.intermediate_graph_degree}, "
        f"build_algo={args.build_algo}, refinement_rate={args.refinement_rate}"
    )
    print()

    save_dir = args.index_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.kcycles_only:
        print(f"Loading persisted index from {save_dir} (kcycles-only mode)")
        path = os.path.join(save_dir, index_filename(n))
        if not os.path.exists(path):
            print(f"ERROR: no index at {path}", file=sys.stderr)
            sys.exit(1)
        idx = faiss.read_index(path)
        recall = eval_recall(idx, xq, Igt, k)
        qps = eval_qps(idx, xq, k)
        kcycles = eval_kcycles(idx, xq, k)
        for ef in sorted(recall.keys()):
            print(
                f"  ef={ef:>4d}  recall@{k}={recall[ef]:.4f}  "
                f"qps={qps[ef]:,.1f}  kcyc/q={kcycles.get(ef, 0):.0f}"
            )
        return

    print(f"Index will be saved to: {save_dir}")

    index, timings = build_cagra_hnsw(
        xb,
        d,
        num_gpus,
        args.graph_degree,
        save_dir=save_dir,
        n_clusters=args.n_clusters,
        overlap_factor=args.overlap_factor,
        build_algo=args.build_algo,
        base_level_only=args.base_level_only,
        intermediate_graph_degree=args.intermediate_graph_degree,
        refinement_rate=args.refinement_rate,
        ivfpq_search_batch=args.ivfpq_search_batch,
        guarantee_connectivity=args.guarantee_connectivity,
        ivfpq_size_from_cluster=args.ivfpq_size_from_cluster,
        gpu_hnsw_upper_levels=args.gpu_hnsw_upper_levels,
        gpu_hnsw_igd=args.gpu_hnsw_igd,
        gpu_hnsw_guarantee_connectivity=args.gpu_hnsw_guarantee_connectivity,
        ef_construction=args.ef_construction,
        entrypoints=args.entrypoints,
    )
    sweep_eps = [
        int(v) for v in args.entrypoints_sweep.split(",") if v.strip()
    ]
    built_ep = index.num_base_level_search_entrypoints
    for ep in sweep_eps:
        index.num_base_level_search_entrypoints = ep
        r = eval_recall(index, xq, Igt, k)
        q = eval_qps(index, xq, k)
        print(f"\n  --- num_base_level_search_entrypoints={ep} ---")
        print(f"  {'efSearch':>8s} {'recall@' + str(k):>10s} {'QPS':>12s}")
        for ef in sorted(r.keys()):
            print(f"  {ef:>8d} {r[ef]:>10.4f} {q[ef]:>12,.1f}")
    index.num_base_level_search_entrypoints = built_ep

    recall = eval_recall(index, xq, Igt, k)
    qps = eval_qps(index, xq, k)
    kcycles = eval_kcycles(index, xq, k)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Build (all_neighbors+optimize): {timings['build']:.1f}s")
    print(f"  copyTo:                         {timings['copyTo']:.1f}s")
    print(
        f"  Serialize:                      {timings['serialize']:.1f}s "
        f"({timings['file_size_gb']:.2f} GB)"
    )
    # Index-build wall-clock, isolated from data load (Koski/Manifold) and
    # ground-truth: this is the headline build->serialize number.
    index_total = (
        timings["build"] + timings["copyTo"] + timings["serialize"]
    )
    print(
        f"  >>> INDEX build->serialize total "
        f"(excl. data load + ground truth): "
        f"{index_total:.1f}s ({index_total / 60:.1f} min)"
    )
    print(f"  {'efSearch':>8s} {'recall@' + str(k):>10s} {'QPS':>12s} "
          f"{'us/query':>10s} {'kcyc/q':>9s}")
    for ef in sorted(recall.keys()):
        q = qps[ef]
        print(
            f"  {ef:>8d} {recall[ef]:>10.4f} {q:>12,.1f} "
            f"{1e6 / q:>10.1f} {kcycles.get(ef, 0):>9.0f}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        open(_SENTINEL, "w").close()
