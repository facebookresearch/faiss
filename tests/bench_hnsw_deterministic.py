# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Microbenchmark for the HNSW graph build (deterministic, lock-free). Two modes:
#
#   summary (default): build the index FAISS_DET_NBUILDS times, reporting build
#     time (min/mean) + peak RSS, graph reproducibility (blake2b of
#     neighbors+offsets across builds), and the recall/QPS tradeoff vs
#     brute-force ground truth.
#
#   timing study (FAISS_DET_ROUNDS > 0): time the build FAISS_DET_ROUNDS times,
#     reporting per-round times and min/mean/median/max/std.
#
# Memory: a single HNSWFlat index at large nb is tens of GB, so only one index
# is alive at a time, the dataset is serialized once and memory-mapped
# (file-backed pages are evictable instead of hitting swap), and ground truth
# uses faiss.knn (no second full-size index copy).
#
# Env (all optional): FAISS_DET_NB (default 40_000_000), FAISS_DET_NBUILDS (3),
#   FAISS_DET_ROUNDS (0), FAISS_DET_M (32), FAISS_DET_EFC (64), FAISS_DET_NQ
#   (1_000), FAISS_DET_SEED (1234), FAISS_DET_DATA_DIR (a tempdir, reused across
#   runs), FAISS_DET_GRAPH_OUT (dump last graph), FAISS_DET_TAG.
#
# Run with:
#   buck2 run fbcode//faiss/tests:bench_hnsw_deterministic

from __future__ import annotations

import gc
import hashlib
import os
import tempfile
import threading
import time
import unittest

import faiss
import numpy as np

from faiss.contrib.datasets import SyntheticDataset


def _envint(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


class RSSPeak:
    # Samples VmRSS in a daemon thread; reset() rebases the peak to "now" so the
    # peak during a single build interval can be captured.
    def __init__(self, interval: float = 0.25) -> None:
        self._peak = 0
        self._stop = False
        self._lock = threading.Lock()
        self._interval = interval
        self._t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _rss_kb() -> int:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        return 0

    def _run(self) -> None:
        while not self._stop:
            r = self._rss_kb()
            with self._lock:
                self._peak = max(self._peak, r)
            time.sleep(self._interval)

    def start(self) -> None:
        self._t.start()

    def reset(self) -> None:
        with self._lock:
            self._peak = self._rss_kb()

    def peak_gb(self) -> float:
        with self._lock:
            return self._peak / 1048576.0

    def stop(self) -> None:
        self._stop = True
        # Join the sampler so no in-flight `self._peak = max(...)` update can
        # race a subsequent peak_gb() read.
        self._t.join()


def dataset_paths(d: int, nb: int, nq: int, seed: int) -> tuple[str, str]:
    base = os.environ.get("FAISS_DET_DATA_DIR") or os.path.join(
        tempfile.gettempdir(), "faiss_det_bench"
    )
    os.makedirs(base, exist_ok=True)
    tag = f"d{d}_nb{nb}_nq{nq}_s{seed}"
    return (
        os.path.join(base, f"xb_{tag}.npy"),
        os.path.join(base, f"xq_{tag}.npy"),
    )


def make_dataset(d: int, nb: int, nq: int, seed: int):
    # SyntheticDataset: a low-intrinsic-dimensional ellipsoid so recall is
    # meaningful; HNSWFlat needs no training (nt=0). Serialized once and
    # returned as a copy-on-write memmap so its pages stay file-backed (a build
    # still makes its own resident copy, but the source can be evicted under
    # pressure rather than swapped). Queries are tiny and kept in memory.
    xb_path, xq_path = dataset_paths(d, nb, nq, seed)
    if not (os.path.exists(xb_path) and os.path.exists(xq_path)):
        ds = SyntheticDataset(d, 0, nb, nq, seed=seed)
        np.save(xb_path, ds.get_database())
        np.save(xq_path, ds.get_queries())
        del ds
        gc.collect()
    xb = np.load(xb_path, mmap_mode="c")
    xq = np.load(xq_path)
    return xb, xq


def ground_truth(xb, xq, k):
    # Block-wise brute force: streams over xb without a second full-size copy.
    _, gt = faiss.knn(xq, xb, k, metric=faiss.METRIC_L2)
    return gt


def dump_graph(index, path):
    neighbors = faiss.vector_to_array(index.hnsw.neighbors)
    offsets = faiss.vector_to_array(index.hnsw.offsets)
    with open(path, "wb") as f:
        f.write(neighbors.tobytes())
        f.write(offsets.tobytes())
    return neighbors.nbytes + offsets.nbytes


def build_index(xb, d, M, efc, rss):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efc
    rss.reset()
    t0 = time.perf_counter()
    index.add(xb)
    return index, time.perf_counter() - t0, rss.peak_gb()


def recall_at_k(index, xq, gt, k, efSearch):
    index.hnsw.efSearch = efSearch
    t0 = time.perf_counter()
    _, ids = index.search(xq, k)
    dt = time.perf_counter() - t0
    nq = xq.shape[0]
    hits = 0
    for i in range(nq):
        hits += len(set(ids[i]).intersection(set(gt[i])))
    return hits / (nq * k), nq / dt


def graph_hash(index) -> str:
    # blake2b of neighbors+offsets, no copy beyond vector_to_array.
    h = hashlib.blake2b()
    neighbors = faiss.vector_to_array(index.hnsw.neighbors)
    h.update(memoryview(neighbors))
    del neighbors
    offsets = faiss.vector_to_array(index.hnsw.offsets)
    h.update(memoryview(offsets))
    return h.hexdigest()


def _stats(xs):
    xs2 = sorted(xs)
    n = len(xs2)
    mean = sum(xs2) / n
    median = xs2[n // 2] if n % 2 else (xs2[n // 2 - 1] + xs2[n // 2]) / 2
    std = (sum((x - mean) ** 2 for x in xs2) / n) ** 0.5
    return min(xs2), mean, median, max(xs2), std


EF_SEARCH = (16, 32, 64, 128)


def summary(xb, xq, gt, d, nb, M, efc, k, nbuilds, rss, threads):
    print(
        f"\n=== d={d} nb={nb:,} M={M} efC={efc} k={k} nbuilds={nbuilds} "
        f"(threads={threads}) ==="
    )
    times, peaks, hashes = [], [], []
    idx = None
    for _ in range(nbuilds):
        if idx is not None:
            del idx
            gc.collect()
        idx, t, peak = build_index(xb, d, M, efc, rss)
        times.append(t)
        peaks.append(peak)
        hashes.append(graph_hash(idx))
    out = os.environ.get("FAISS_DET_GRAPH_OUT")
    if out:
        dump_graph(idx, out)

    tmin, tmean = min(times), sum(times) / len(times)
    print(
        f"build time: min={tmin:7.2f}s mean={tmean:7.2f}s "
        f"peak={max(peaks):.1f}GB"
    )
    print(
        f"graph reproducible across {nbuilds} builds: "
        f"{'YES' if len(set(hashes)) == 1 else 'NO'}"
    )
    print(f"{'efSearch':>9} | {'recall':>8} {'QPS':>9}")
    for efs in EF_SEARCH:
        r, q = recall_at_k(idx, xq, gt, k, efs)
        print(f"{efs:>9} | {r:>8.4f} {q:>9.0f}")
    del idx
    gc.collect()


def timing_study(xb, d, M, efc, rounds, rss, tag, threads):
    print(
        f"\n=== timing study: {rounds} rounds, threads={threads}, "
        f"tag={tag!r} ==="
    )
    ts = []
    for r in range(rounds):
        idx, t, peak = build_index(xb, d, M, efc, rss)
        ts.append(t)
        del idx
        gc.collect()
        print(f"round {r + 1:>2}/{rounds}  {t:7.2f}s  peak={peak:.1f}GB")
    mn, mean, med, mx, std = _stats(ts)
    print("build s: " + " ".join(f"{x:.2f}" for x in ts))
    print(
        f"build: min={mn:.2f} mean={mean:.2f} median={med:.2f} "
        f"max={mx:.2f} std={std:.2f}"
    )


def main() -> None:
    d = 128
    nb = _envint("FAISS_DET_NB", 40_000_000)
    nq = _envint("FAISS_DET_NQ", 1_000)
    M = _envint("FAISS_DET_M", 32)
    efc = _envint("FAISS_DET_EFC", 64)
    seed = _envint("FAISS_DET_SEED", 1234)
    nbuilds = _envint("FAISS_DET_NBUILDS", 3)
    rounds = _envint("FAISS_DET_ROUNDS", 0)
    k = 10
    tag = os.environ.get("FAISS_DET_TAG", "")

    # Fixed thread count: the build guarantees reproducibility at a fixed
    # thread count.
    faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    threads = faiss.omp_get_max_threads()
    faiss.cvar.hnsw_deterministic_build = True
    print("faiss HNSW deterministic-build benchmark")

    rss = RSSPeak()
    rss.start()
    xb, xq = make_dataset(d, nb, nq, seed)
    if rounds > 0:
        timing_study(xb, d, M, efc, rounds, rss, tag, threads)
    else:
        gt = ground_truth(xb, xq, k)
        summary(xb, xq, gt, d, nb, M, efc, k, nbuilds, rss, threads)
    rss.stop()


class BenchHNSWDeterministic(unittest.TestCase):
    def test_bench(self) -> None:
        main()


if __name__ == "__main__":
    main()
