# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for testing HNSW prune_headroom recall impact.

Compares recall and build time between different prune_headroom values.
Default comparison is between:
- Baseline: prune_headroom = 0.0 (original behavior, no headroom)
- With headroom: prune_headroom = 0.2 (proposed default)

Usage:
    python bench_hnsw_prune_headroom.py
    python bench_hnsw_prune_headroom.py --nb 100000 --d 256
    python bench_hnsw_prune_headroom.py --headroom_values 0.0 0.1 0.2 0.3
"""

import argparse
import time

import faiss

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M

from faiss.contrib.datasets import SyntheticDataset


def compute_recall(I, gt, k):
    """Compute recall@k given search results I and ground truth gt."""
    nq = gt.shape[0]
    return faiss.eval_intersection(I[:, :k], gt[:, :k]) / (nq * k)


def build_hnsw_index(d, m, xb, ef_construction, prune_headroom):
    """Build an HNSW index with the specified configuration."""
    index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_4bit, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.prune_headroom = prune_headroom

    index.train(xb)
    start_time = time.time()
    index.add(xb)
    build_time = time.time() - start_time

    return index, build_time


def run_benchmark(
    d=384,
    m=32,
    nb=50000,
    nq=1000,
    reps=3,
    ef_construction=40,
    ef_search_values=None,
    k_values=None,
    headroom_values=None,
    use_sift1m=False,
):
    """
    Run the prune_headroom recall benchmark.

    Args:
        d: Dimension of vectors
        nb: Number of base vectors
        nq: Number of query vectors
        ef_construction: efConstruction parameter for HNSW
        ef_search_values: List of efSearch values to test
        k_values: List of k values for recall@k
        headroom_values: List of prune_headroom values to compare
        use_sift1m: Use SIFT1M dataset instead of synthetic

    Returns:
        Dictionary containing benchmark results
    """
    if ef_search_values is None:
        ef_search_values = [16, 32, 64, 128, 256]
    if k_values is None:
        k_values = [1, 10]
    if headroom_values is None:
        headroom_values = [0.0, 0.2]

    if use_sift1m:
        print("Loading SIFT1M dataset")
        ds = DatasetSIFT1M()
        xb = ds.get_database()
        xq = ds.get_queries()
        d = xb.shape[1]
        nb = xb.shape[0]
        nq = xq.shape[0]
    else:
        print(f"Generating synthetic dataset: d={d}, nb={nb}, nq={nq}")
        ds = SyntheticDataset(d=d, nt=0, nb=nb, nq=nq)
        xb = ds.get_database()
        xq = ds.get_queries()

    max_k = max(k_values)
    print(f"Computing ground truth for k={max_k}")
    gt = ds.get_groundtruth(k=max_k)

    results = {"build_times": {}, "ndis_search": {}, "recalls": {}}

    for headroom in headroom_values:
        for rep in range(reps):
            index, build_time = build_hnsw_index(
                d, m, xb, ef_construction, headroom)
            results["build_times"][headroom] = build_time

            faiss.cvar.hnsw_stats.reset()
            row = {}
            results["recalls"][(headroom, rep)] = row
            for ef_search in ef_search_values:
                index.hnsw.efSearch = ef_search
                _, I = index.search(xq, max_k)

                col = {}
                row[ef_search] = col
                for k in k_values:
                    recall = compute_recall(I, gt, k)
                    col[k] = recall
            ndis_search = faiss.cvar.hnsw_stats.ndis / nq
            results["ndis_search"][headroom] = ndis_search
            print(
                f"HNSW{m}(prune_headroom={headroom:4.2f}): "
                f"{build_time=:4.2f}s, {ndis_search=:5.1f}"
            )

    print_results_table(results, ef_search_values, k_values, headroom_values)
    return results


def print_results_table(results, ef_search_values, k_values, headroom_values):

    for k in k_values:
        header_parts = [f"{k=:2}  "]
        for ef_search in ef_search_values:
            header_parts.append(f"ef={ef_search:3}")
        header = " | ".join(header_parts)

        print(f"\n{header}")
        print("-" * len(header))
        for (h, _), row in results["recalls"].items():
            row_parts = [f"h={h:4.2f}"]
            for ef_search in ef_search_values:
                recall = row[ef_search][k]
                row_parts.append(f"{recall:6.4f}")
            print(" | ".join(row_parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HNSW prune_headroom recall and build time benchmark"
    )
    parser.add_argument(
        "--d",
        type=int,
        default=128,
        help="Dimension of vectors (default: 128)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=32,
        help="Node degree (M, default: 32)",
    )
    parser.add_argument(
        "--nb",
        type=int,
        default=50000,
        help="Number of base vectors (default: 50000)",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=10000,
        help="Number of query vectors (default: 10000)",
    )
    parser.add_argument(
        "--ef_construction",
        type=int,
        default=40,
        help="efConstruction parameter (default: 40)",
    )
    parser.add_argument(
        "--ef_search",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="efSearch values to test (default: 16 32 64 128 256)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 10],
        help="k values for recall@k (default: 1 10)",
    )
    parser.add_argument(
        "--headroom_values",
        type=float,
        nargs="+",
        default=[0.0, 0.04, 0.08, 0.12, 0.16, 0.20],
        help="prune_headroom values to compare (default: 0.0 0.2)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions (default: 3)",
    )
    parser.add_argument(
        "--sift1m",
        action="store_true",
        help="Use SIFT1M dataset instead of synthetic",
    )
    args = parser.parse_args()

    run_benchmark(
        d=args.d,
        m=args.m,
        nb=args.nb,
        nq=args.nq,
        reps=args.reps,
        ef_construction=args.ef_construction,
        ef_search_values=args.ef_search,
        k_values=args.k,
        headroom_values=args.headroom_values,
        use_sift1m=args.sift1m,
    )
