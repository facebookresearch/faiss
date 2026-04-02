# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark FAISS wheel quality: recall@10, QPS, and training time.

Compares different FAISS installations (our wheel, conda, current pip)
across IVF index types on synthetic data.

Usage:
    python bench_wheels.py --output results.json
    python bench_wheels.py --compare wheel.json conda.json pip.json
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
D = 128
N_TRAIN = 100_000
N_DB = 500_000
N_QUERY = 10_000
K = 10

INDEXES = [
    # (name, factory_string, {search_param: value})
    ("IVFFlat", "IVF1024,Flat", {"nprobe": 32}),
    ("IVFPQ", "IVF1024,PQ16", {"nprobe": 32}),
    ("IVFSQ8", "IVF1024,SQ8", {"nprobe": 32}),
]

N_WARMUP = 3
N_SEARCH_RUNS = 10


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def get_system_info(label):
    import faiss

    info = {
        "label": label,
        "faiss_version": faiss.__version__,
        "compile_options": faiss.get_compile_options(),
        "omp_threads": faiss.omp_get_max_threads(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "arch": platform.machine(),
    }

    # SIMD level
    opts = faiss.get_compile_options().upper()
    for level in ("AVX512_SPR", "AVX512", "AVX2", "NEON", "SVE", "DD"):
        if level in opts:
            info.setdefault("simd_levels", [])
            info["simd_levels"].append(level)
    if "simd_levels" not in info:
        info["simd_levels"] = ["GENERIC"]

    # BLAS detection
    try:
        with open(f"/proc/{os.getpid()}/maps") as f:
            maps = f.read()
        if "libmkl" in maps:
            info["blas"] = "MKL"
        elif "libopenblas" in maps:
            info["blas"] = "OpenBLAS"
        else:
            info["blas"] = "unknown"
    except (FileNotFoundError, OSError):
        info["blas"] = "system"

    # OS
    system = platform.system()
    if system == "Darwin":
        info["os"] = f"macOS {platform.mac_ver()[0]}"
        info["blas"] = "Accelerate"  # macOS always uses Accelerate or OpenBLAS
        try:
            import subprocess
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if brand:
                info["cpu"] = brand
        except Exception:
            info["cpu"] = platform.processor() or platform.machine()
    elif system == "Linux":
        info["os"] = "Linux"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            info["cpu"] = platform.processor() or platform.machine()
    elif system == "Windows":
        info["os"] = f"Windows {platform.version()}"
        info["cpu"] = platform.processor() or platform.machine()
    else:
        info["os"] = system
        info["cpu"] = platform.processor() or platform.machine()

    info["cores"] = os.cpu_count()

    return info


# ---------------------------------------------------------------------------
# Data generation and ground truth
# ---------------------------------------------------------------------------
def make_dataset():
    """Create a SyntheticDataset with a non-trivial distribution."""
    from faiss.contrib.datasets import SyntheticDataset

    return SyntheticDataset(D, N_TRAIN, N_DB, N_QUERY)


def recall_at_k(gt, results, k):
    """Compute recall@k: fraction of true k-NN found in results."""
    assert gt.shape[0] == results.shape[0]
    n = gt.shape[0]
    hits = 0
    for i in range(n):
        hits += len(set(gt[i, :k].tolist()) & set(results[i, :k].tolist()))
    return hits / (n * k)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark_index(faiss, xt, xb, xq, gt, name, factory, search_params):
    """Benchmark a single index configuration."""
    # Train
    t0 = time.perf_counter()
    index = faiss.index_factory(D, factory)
    index.train(xt)
    train_time = time.perf_counter() - t0

    # Add
    t0 = time.perf_counter()
    index.add(xb)
    add_time = time.perf_counter() - t0

    # Set search params
    params = faiss.ParameterSpace()
    for param, value in search_params.items():
        params.set_index_parameter(index, param, value)

    # Warmup
    for _ in range(N_WARMUP):
        index.search(xq, K)

    # Search (multiple runs, take median)
    times = []
    results = None
    for _ in range(N_SEARCH_RUNS):
        t0 = time.perf_counter()
        D_out, I_out = index.search(xq, K)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if results is None:
            results = I_out

    median_search = float(np.median(times))
    qps = N_QUERY / median_search

    # Recall
    r10 = recall_at_k(gt, results, K)

    return {
        "index": name,
        "factory": factory,
        "search_params": search_params,
        "train_time_s": round(train_time, 4),
        "add_time_s": round(add_time, 4),
        "search_time_s": round(median_search, 4),
        "qps": round(qps, 1),
        "recall10": round(r10, 4),
    }


def run_benchmarks(label, output_path):
    import faiss

    info = get_system_info(label)

    print(f"Machine:  {info.get('cpu', 'unknown')}")
    print(f"OS:       {info['os']}")
    print(f"Cores:    {info['cores']}")
    print(f"FAISS:    {info['faiss_version']}")
    print(f"BLAS:     {info.get('blas', 'unknown')}")
    print(f"SIMD:     {', '.join(info['simd_levels'])}")
    print(f"Threads:  {info['omp_threads']}")
    print(f"Label:    {label}")
    print()

    print("Generating dataset...")
    ds = make_dataset()
    xt = ds.get_train()
    xb = ds.get_database()
    xq = ds.get_queries()
    print(f"Data: {N_DB:,} db, {N_TRAIN:,} train, {N_QUERY:,} queries, d={D}, k={K}")

    print("Computing ground truth (FlatL2)...")
    gt = ds.get_groundtruth(K)

    results = []
    for name, factory, search_params in INDEXES:
        print(f"\nBenchmarking {name} ({factory})...")
        r = benchmark_index(faiss, xt, xb, xq, gt, name, factory, search_params)
        results.append(r)
        print(f"  train: {r['train_time_s']:.2f}s  "
              f"search: {r['search_time_s']:.4f}s  "
              f"QPS: {r['qps']:,.0f}  "
              f"recall@10: {r['recall10']:.4f}")

    # Summary table
    print(f"\n{'Index':<10} {'Train(s)':>10} {'QPS':>10} {'Recall@10':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['index']:<10} {r['train_time_s']:>10.2f} "
              f"{r['qps']:>10,.0f} {r['recall10']:>10.4f}")

    data = {"system": info, "params": {"n_db": N_DB, "n_train": N_TRAIN,
            "n_query": N_QUERY, "d": D, "k": K}, "results": results}

    if output_path:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return data


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_results(files):
    datasets = []
    for path in files:
        with open(path) as f:
            datasets.append(json.load(f))

    # Print system info for each
    print("## System Info\n")
    print(f"| | " + " | ".join(d["system"]["label"] for d in datasets) + " |")
    print(f"|---|" + "|".join("---" for _ in datasets) + "|")
    for key in ("faiss_version", "blas", "simd_levels", "omp_threads"):
        vals = []
        for d in datasets:
            v = d["system"].get(key, "—")
            if isinstance(v, list):
                v = ", ".join(v)
            vals.append(str(v))
        print(f"| {key} | " + " | ".join(vals) + " |")

    # Print machine info (from first dataset)
    sys_info = datasets[0]["system"]
    print(f"\n**Machine:** {sys_info.get('cpu', 'unknown')}")
    print(f"**OS:** {sys_info.get('os', 'unknown')}")
    print(f"**Cores:** {sys_info.get('cores', 'unknown')}")

    # Build index maps
    labels = [d["system"]["label"] for d in datasets]
    index_maps = []
    for d in datasets:
        index_maps.append({r["index"]: r for r in d["results"]})

    # Get all index names (preserve order from first dataset)
    index_names = [r["index"] for r in datasets[0]["results"]]

    # Recall table
    print("\n## Recall@10\n")
    print(f"| Index | " + " | ".join(labels) + " |")
    print(f"|---|" + "|".join("---:" for _ in labels) + "|")
    for idx in index_names:
        vals = []
        for im in index_maps:
            r = im.get(idx)
            vals.append(f"{r['recall10']:.4f}" if r else "—")
        print(f"| {idx} | " + " | ".join(vals) + " |")

    # QPS table
    print("\n## QPS (queries/sec)\n")
    print(f"| Index | " + " | ".join(labels) + " |")
    print(f"|---|" + "|".join("---:" for _ in labels) + "|")
    for idx in index_names:
        vals = []
        for im in index_maps:
            r = im.get(idx)
            vals.append(f"{r['qps']:,.0f}" if r else "—")
        print(f"| {idx} | " + " | ".join(vals) + " |")

    # Training time table
    print("\n## Training Time (seconds)\n")
    print(f"| Index | " + " | ".join(labels) + " |")
    print(f"|---|" + "|".join("---:" for _ in labels) + "|")
    for idx in index_names:
        vals = []
        for im in index_maps:
            r = im.get(idx)
            vals.append(f"{r['train_time_s']:.2f}" if r else "—")
        print(f"| {idx} | " + " | ".join(vals) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FAISS wheel quality")
    parser.add_argument("--label", default="unknown",
                        help="Label for this run (e.g., 'wheel', 'conda', 'pip')")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--compare", nargs="+", metavar="FILE",
                        help="Compare multiple result JSON files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare)
    else:
        run_benchmarks(args.label, args.output)


if __name__ == "__main__":
    main()
