# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark of the HNSW Adaptive Beam Search (adaptive_beam_gamma) against
the default beam search (efSearch).

For each target recall, the script finds the smallest efSearch and the
smallest gamma that reach it, without and with a bound on the number of
explored nodes (adaptive_beam_max_hops), and compares the operating points
on:
- the number of distance computations per query (mean and 99th percentile)
- the single-thread QPS

Usage:
    python bench_hnsw_adaptive_beam.py
    python bench_hnsw_adaptive_beam.py --sift1m --k 100
    python bench_hnsw_adaptive_beam.py --nb 200000 --d 64 --M 16
"""

import argparse
import time

import faiss
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M

from faiss.contrib.datasets import SyntheticDataset


class Evaluator:
    """Evaluates search parameters of an index, with a cache."""

    def __init__(self, index, xq, gt, k):
        self.index = index
        self.xq = xq
        self.gt = gt[:, :k]
        self.k = k
        self.cache = {}
        self.nthread = faiss.omp_get_max_threads()

    def params(self, method, param):
        if method == "beam":
            return faiss.SearchParametersHNSW(efSearch=int(param))
        if method == "capped":
            gamma, max_hops = param
            return faiss.SearchParametersHNSW(
                adaptive_beam_gamma=float(gamma),
                adaptive_beam_max_hops=int(max_hops))
        return faiss.SearchParametersHNSW(adaptive_beam_gamma=float(param))

    def evaluate(self, method, param):
        """Returns (recall, mean ndis), computed with all threads."""
        key = (method, param)
        if key not in self.cache:
            faiss.omp_set_num_threads(self.nthread)
            faiss.cvar.hnsw_stats.reset()
            _, I = self.index.search(
                self.xq, self.k, params=self.params(method, param))
            nq = len(self.xq)
            ndis = faiss.cvar.hnsw_stats.ndis / nq
            recall = faiss.eval_intersection(I, self.gt) / (nq * self.k)
            self.cache[key] = (recall, ndis)
        return self.cache[key]

    def smallest_reaching(self, method, target, grid, max_ndis):
        """Smallest parameter of the sorted grid that reaches the target
        recall, None if it needs more than max_ndis distances."""
        lo = -1
        i = 0
        step = 1
        while True:
            recall, ndis = self.evaluate(method, grid[i])
            if recall >= target:
                hi = i
                break
            lo = i
            if ndis > max_ndis or i == len(grid) - 1:
                return None
            i = min(i + step, len(grid) - 1)
            step *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.evaluate(method, grid[mid])[0] >= target:
                hi = mid
            else:
                lo = mid
        return grid[hi]

    def ndis_percentile(self, method, param, percentile):
        faiss.omp_set_num_threads(1)
        p = self.params(method, param)
        stats = faiss.cvar.hnsw_stats
        ndis = np.zeros(len(self.xq))
        for i in range(len(self.xq)):
            stats.reset()
            self.index.search(self.xq[i:i + 1], self.k, params=p)
            ndis[i] = stats.ndis
        return np.percentile(ndis, percentile)

    def qps(self, method, param):
        faiss.omp_set_num_threads(1)
        p = self.params(method, param)
        t0 = time.perf_counter()
        self.index.search(self.xq, self.k, params=p)
        return len(self.xq) / (time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the HNSW adaptive beam search")
    aa = parser.add_argument
    aa("--sift1m", action="store_true", help="use SIFT1M, not synthetic")
    aa("--d", type=int, default=64, help="dimension (synthetic)")
    aa("--nb", type=int, default=100000, help="database size (synthetic)")
    aa("--nq", type=int, default=2000, help="nb of queries (synthetic)")
    aa("--k", type=int, default=10)
    aa("--M", type=int, default=32)
    aa("--efConstruction", type=int, default=40)
    aa("--targets", type=float, nargs="+",
        default=[0.8, 0.9, 0.95, 0.98, 0.99, 0.995])
    aa("--max_ef", type=int, default=2048)
    aa("--max_ndis", type=float, default=100000)
    aa("--reps", type=int, default=3, help="nb of timing runs")
    aa("--cap_factor", type=float, default=1.25,
        help="adaptive_beam_max_hops is this times the efSearch of the "
        "beam search")
    args = parser.parse_args()

    if args.sift1m:
        ds = DatasetSIFT1M()
    else:
        ds = SyntheticDataset(args.d, 0, args.nb, args.nq)
    xb = ds.get_database()
    xq = ds.get_queries()
    gt = ds.get_groundtruth(args.k)

    index = faiss.IndexHNSWFlat(ds.d, args.M)
    index.hnsw.efConstruction = args.efConstruction
    t0 = time.time()
    index.add(xb)
    print(f"nb={ds.nb} d={ds.d} M={args.M} k={args.k} "
          f"build time {time.time() - t0:.1f} s")

    faiss.set_search_stats_enabled(True)
    ev = Evaluator(index, xq, gt, args.k)
    ef_grid = list(range(args.k, args.max_ef + 1))
    gamma_grid = [round(i * 0.001, 3) for i in range(1001)]

    print(f"{'target':>7} {'method':>9} {'param':>11} {'recall':>7} "
          f"{'ndis':>8} {'ndis p99':>9} {'QPS':>8}")
    for target in args.targets:
        ops = {
            "beam": ev.smallest_reaching(
                "beam", target, ef_grid, args.max_ndis),
            "adaptive": ev.smallest_reaching(
                "adaptive", target, gamma_grid, args.max_ndis),
        }
        if None in ops.values():
            print(f"{target:7.3f} not reachable")
            break
        max_hops = int(np.ceil(args.cap_factor * ops["beam"]))
        capped = ev.smallest_reaching(
            "capped", target, [(g, max_hops) for g in gamma_grid],
            args.max_ndis)
        if capped is not None:
            ops["capped"] = capped
        qps = {method: 0 for method in ops}
        for _ in range(args.reps):
            # interleave the timings of the two methods
            for method in ops:
                qps[method] = max(qps[method], ev.qps(method, ops[method]))
        for method in ops:
            recall, ndis = ev.evaluate(method, ops[method])
            p99 = ev.ndis_percentile(method, ops[method], 99)
            param = ops[method]
            if method == "capped":
                param = f"{param[0]:g},{param[1]}"
            print(f"{target:7.3f} {method:>9} {param:>11} "
                  f"{recall:7.4f} {ndis:8.1f} {p99:9.0f} "
                  f"{qps[method]:8.0f}")


if __name__ == "__main__":
    main()
