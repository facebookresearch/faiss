# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark `faiss.SuperKMeans` against vanilla `faiss.Clustering` on the same
input. Inputs may be a .fvecs file or a synthetic Gaussian mixture.

Examples:
    # Synthetic data
    python bench_super_kmeans.py --n 10000 --d 128 --k 64 --niter 10

    # Real .fvecs (e.g., SIFT base)
    python bench_super_kmeans.py --fvecs /path/to/sift_base.fvecs --k 1024 \\
        --niter 10
"""

import argparse
import sys
import time

import faiss
import numpy as np
from datasets import fvecs_read


def gaussian_mixture(n, d, k, seed):
    """k centers in [-1, 1]^d, ~n/k samples per center from N(center, 0.1)."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-1.0, 1.0, size=(k, d)).astype("float32")
    cluster = np.arange(n) % k
    noise = rng.normal(0.0, 0.1, size=(n, d)).astype("float32")
    return centers[cluster] + noise


def run_vanilla(x, d, k, niter, seed):
    quant = faiss.IndexFlatL2(d)
    cl = faiss.Clustering(d, k)
    cl.seed = seed
    cl.niter = niter
    cl.verbose = False
    t0 = time.perf_counter()
    cl.train(x, quant)
    dt = time.perf_counter() - t0
    final_obj = cl.iteration_stats.at(cl.iteration_stats.size() - 1).obj
    return dt, final_obj, cl.iteration_stats.size()


def run_super(x, d, k, niter, seed):
    p = faiss.SuperKMeansParameters()
    p.seed = seed
    p.niter = niter
    p.verbose = False
    sc = faiss.SuperKMeans(d, k, p)
    t0 = time.perf_counter()
    sc.train(x)
    dt = time.perf_counter() - t0
    final_obj = sc.iteration_stats.at(sc.iteration_stats.size() - 1).obj
    iter_sum = sum(
        sc.iteration_stats.at(i).time for i in range(sc.iteration_stats.size())
    )
    setup = dt - iter_sum
    rates = faiss.vector_to_array(sc.gemm_pruning_rates)
    return dt, final_obj, sc.iteration_stats.size(), iter_sum, setup, rates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--fvecs", help="path to a .fvecs file")
    ap.add_argument("--n", type=int, default=10000, help="synthetic n")
    ap.add_argument("--d", type=int, default=128, help="synthetic d")
    ap.add_argument("--k", type=int, default=64, help="number of centroids")
    ap.add_argument("--niter", type=int, default=10, help="iterations")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    if args.fvecs:
        x = fvecs_read(args.fvecs)
        n, d = x.shape
        print("=== bench_super_kmeans ===")
        print(
            f"fvecs={args.fvecs} n={n} d={d} k={args.k} "
            f"niter={args.niter} seed={args.seed}"
        )
    else:
        n, d = args.n, args.d
        x = gaussian_mixture(n, d, args.k, args.seed)
        print("=== bench_super_kmeans ===")
        print(
            f"n={n} d={d} k={args.k} niter={args.niter} seed={args.seed}"
        )

    print("\n--- faiss.Clustering ---")
    v_dt, v_obj, v_iters = run_vanilla(x, d, args.k, args.niter, args.seed)
    print(f"  wall_time={v_dt:.3f}s final_obj={v_obj:g} iters={v_iters}")

    print("\n--- faiss.SuperKMeans ---")
    s_dt, s_obj, s_iters, iter_sum, setup, rates = run_super(
        x, d, args.k, args.niter, args.seed
    )
    print(
        f"  wall_time={s_dt:.3f}s final_obj={s_obj:g} iters={s_iters} "
        f"iter_sum={iter_sum:.3f}s setup={setup:.3f}s"
    )
    print("  gemm_pruning_rates:", " ".join(f"{r:.3f}" for r in rates))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
