# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark IndexSQFastScan / IndexIVFSQFastScan vs RaBitQFastScan / IVFRaBitQFastScan.

Head-to-head speed/recall comparison at matched bit widths (4, 6, 8 bits).

Usage:
    python benchs/bench_sqfs_vs_rabitqfs.py [--dataset sift1m|gist1m]
                                            [--threads N]
                                            [--rerank-factor F]
                                            [--k K]
"""

from __future__ import print_function
import argparse
import multiprocessing as mp
import time

import numpy as np
import faiss

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M, DatasetGIST1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M, DatasetGIST1M


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="SQFastScan vs RaBitQFastScan speed/recall comparison")
parser.add_argument("--dataset", default="sift1m",
                    choices=["sift1m", "gist1m"])
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--repeat", type=int, default=3,
                    help="timed repetitions, report fastest")
parser.add_argument("--rerank-factor", type=float, default=1.0,
                    help="overselection factor for SQfs reranking (6/8 bit)")
parser.add_argument("--k", type=int, default=10,
                    help="number of nearest neighbors")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if args.dataset == "sift1m":
    ds = DatasetSIFT1M()
else:
    ds = DatasetGIST1M()

xb = ds.get_database()
xq = ds.get_queries()
xt = ds.get_train()
gt = ds.get_groundtruth()

nb, d = xb.shape
nq = xq.shape[0]
k = args.k
gt_k = gt[:, :k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timed_search(index, xq, k, params=None, repeat=3):
    """Return best ms/query over `repeat` runs."""
    # warmup
    index.search(xq[:4], k) if params is None else index.search(xq[:4], k, params=params)

    best = np.inf
    for _ in range(repeat):
        t0 = time.time()
        if params is not None:
            D, I = index.search(xq, k, params=params)
        else:
            D, I = index.search(xq, k)
        t = time.time() - t0
        best = min(best, t)
    ms_per_q = best * 1000.0 / xq.shape[0]
    return ms_per_q, I


def recall_at(I, gt_k, r):
    """Recall@r: fraction of queries whose true top-r results appear."""
    n = gt_k.shape[0]
    nmatches = sum(
        len(np.intersect1d(I[i, :r], gt_k[i, :r]))
        for i in range(n)
    )
    return nmatches / float(n * r)


def recall_at_1(I, gt_k):
    """Recall@1: fraction of queries whose true top-1 is in results."""
    return (I[:, :1] == gt_k[:, :1]).sum() / float(I.shape[0])


def report(label, ms, I, mem_bytes=None):
    r1 = recall_at_1(I, gt_k)
    r10 = recall_at(I, gt_k, k)
    mem_str = ""
    if mem_bytes is not None:
        mem_str = " %7.1f" % (mem_bytes / 1e6)
    print("%-50s %7.3f %7.4f %7.4f%s" % (label, ms, r1, r10, mem_str))


def build_index(factory_str):
    faiss.omp_set_num_threads(mp.cpu_count())
    index = faiss.index_factory(d, factory_str)
    index.train(xt)
    index.add(xb)
    faiss.omp_set_num_threads(args.threads)
    return index


def index_memory(index):
    """Rough memory estimate in bytes."""
    try:
        return index.sa_code_size() * index.ntotal
    except Exception:
        return None


def set_rerank_factor(index, factor):
    """Set rerank_factor on IndexIVFSQFastScan."""
    # Try direct access first, then downcast
    if hasattr(index, 'rerank_factor'):
        index.rerank_factor = factor
        return
    try:
        ivf = faiss.downcast_index(index)
        if hasattr(ivf, 'rerank_factor'):
            ivf.rerank_factor = factor
            return
    except Exception:
        pass
    # Try extracting the IVF sub-index from a pre-transform chain
    try:
        vec_transform = faiss.downcast_VectorTransform(index)
        sub = faiss.downcast_index(vec_transform.index)
        if hasattr(sub, 'rerank_factor'):
            sub.rerank_factor = factor
            return
    except Exception:
        pass
    print("  WARNING: could not set rerank_factor on %s" % type(index))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

nlist = 256
nprobes = [1, 4, 8, 16, 32, 64]

# RaBitQ qb values to sweep (higher qb = more accurate query quantization)
rabitq_qbs = [4, 8]

print()
print("Configuration:")
print("  dataset=%s  d=%d  nb=%d  nq=%d  k=%d  threads=%d" % (
    args.dataset, d, nb, nq, k, args.threads))
print("  rerank_factor=%.1f (for SQfs 6/8-bit IVF)" % args.rerank_factor)
print()

# ===========================================================================
# FLAT INDEX COMPARISON (4-bit only — SQFastScan flat only supports 4-bit)
# ===========================================================================

print("=" * 100)
print("FLAT INDEX COMPARISON — %s (4-bit)" % args.dataset.upper())
print("=" * 100)
print("%-50s %7s %7s %7s %7s" % ("index", "ms/q", "R@1", "R@10", "MB"))
print("-" * 100)

flat_configs = [
    ("Flat (brute-force baseline)", "Flat"),
    ("SQ4fs (4-bit SQFastScan)", "SQ4fs"),
    ("RaBitQfs4 (4-bit RaBitQ, qb=4)", "RaBitQfs4"),
    ("RaBitQfs4 (4-bit RaBitQ, qb=8)", "RaBitQfs4"),
]

for label, factory in flat_configs:
    try:
        index = build_index(factory)
    except Exception as e:
        print("%-50s  SKIP (%s)" % (label, e))
        continue

    mem = index_memory(index)

    if "RaBitQ" in factory and "qb=8" in label:
        index.qb = 8
    elif "RaBitQ" in factory and "qb=4" in label:
        index.qb = 4

    ms, I = timed_search(index, xq, k, repeat=args.repeat)
    report(label, ms, I, mem)
    del index

print()

# ===========================================================================
# IVF COMPARISON — 4-bit
# ===========================================================================

print("=" * 100)
print("IVF COMPARISON — %s — 4-bit (nlist=%d)" % (args.dataset.upper(), nlist))
print("=" * 100)
print("%-50s %7s %7s %7s %7s %7s" % (
    "index", "nprobe", "ms/q", "R@1", "R@10", "MB"))
print("-" * 105)

ivf4_configs = [
    ("IVF%d,Flat (baseline)" % nlist, "IVF%d,Flat" % nlist),
    ("IVF%d,SQ4fs" % nlist, "IVF%d,SQ4fs" % nlist),
    ("IVF%d,RaBitQfs4" % nlist, "IVF%d,RaBitQfs4" % nlist),
]

for label, factory in ivf4_configs:
    try:
        index = build_index(factory)
    except Exception as e:
        print("%-50s  SKIP (%s)" % (label, e))
        continue

    mem = index_memory(index)
    is_rabitq = "RaBitQ" in factory

    for nprobe in nprobes:
        if is_rabitq:
            params = faiss.IVFRaBitQSearchParameters()
            params.qb = 8
            params.nprobe = nprobe
            ms, I = timed_search(index, xq, k, params=params, repeat=args.repeat)
        else:
            index.nprobe = nprobe
            ms, I = timed_search(index, xq, k, repeat=args.repeat)

        r1 = recall_at_1(I, gt_k)
        r10 = recall_at(I, gt_k, k)
        mem_str = "%7.1f" % (mem / 1e6) if mem and nprobe == nprobes[0] else "       "
        print("%-50s %7d %7.3f %7.4f %7.4f %s" % (
            label, nprobe, ms, r1, r10, mem_str))

    print()
    del index

# ===========================================================================
# IVF COMPARISON — 6-bit
# ===========================================================================

print("=" * 100)
print("IVF COMPARISON — %s — 6-bit (nlist=%d)" % (args.dataset.upper(), nlist))
print("=" * 100)
print("%-50s %7s %7s %7s %7s %7s" % (
    "index", "nprobe", "ms/q", "R@1", "R@10", "MB"))
print("-" * 105)

ivf6_configs = [
    ("IVF%d,SQ6fs (rerank=%.0f)" % (nlist, args.rerank_factor),
     "IVF%d,SQ6fs" % nlist),
    ("IVF%d,RaBitQfs6" % nlist, "IVF%d,RaBitQfs6" % nlist),
]

for label, factory in ivf6_configs:
    try:
        index = build_index(factory)
    except Exception as e:
        print("%-50s  SKIP (%s)" % (label, e))
        continue

    if "SQ6fs" in factory:
        set_rerank_factor(index, args.rerank_factor)

    mem = index_memory(index)
    is_rabitq = "RaBitQ" in factory

    for nprobe in nprobes:
        if is_rabitq:
            params = faiss.IVFRaBitQSearchParameters()
            params.qb = 8
            params.nprobe = nprobe
            ms, I = timed_search(index, xq, k, params=params, repeat=args.repeat)
        else:
            index.nprobe = nprobe
            ms, I = timed_search(index, xq, k, repeat=args.repeat)

        r1 = recall_at_1(I, gt_k)
        r10 = recall_at(I, gt_k, k)
        mem_str = "%7.1f" % (mem / 1e6) if mem and nprobe == nprobes[0] else "       "
        print("%-50s %7d %7.3f %7.4f %7.4f %s" % (
            label, nprobe, ms, r1, r10, mem_str))

    print()
    del index

# ===========================================================================
# IVF COMPARISON — 8-bit
# ===========================================================================

print("=" * 100)
print("IVF COMPARISON — %s — 8-bit (nlist=%d)" % (args.dataset.upper(), nlist))
print("=" * 100)
print("%-50s %7s %7s %7s %7s %7s" % (
    "index", "nprobe", "ms/q", "R@1", "R@10", "MB"))
print("-" * 105)

ivf8_configs = [
    ("IVF%d,SQ8fs (rerank=%.0f)" % (nlist, args.rerank_factor),
     "IVF%d,SQ8fs" % nlist),
    ("IVF%d,RaBitQfs8" % nlist, "IVF%d,RaBitQfs8" % nlist),
]

for label, factory in ivf8_configs:
    try:
        index = build_index(factory)
    except Exception as e:
        print("%-50s  SKIP (%s)" % (label, e))
        continue

    if "SQ8fs" in factory:
        set_rerank_factor(index, args.rerank_factor)

    mem = index_memory(index)
    is_rabitq = "RaBitQ" in factory

    for nprobe in nprobes:
        if is_rabitq:
            params = faiss.IVFRaBitQSearchParameters()
            params.qb = 8
            params.nprobe = nprobe
            ms, I = timed_search(index, xq, k, params=params, repeat=args.repeat)
        else:
            index.nprobe = nprobe
            ms, I = timed_search(index, xq, k, repeat=args.repeat)

        r1 = recall_at_1(I, gt_k)
        r10 = recall_at(I, gt_k, k)
        mem_str = "%7.1f" % (mem / 1e6) if mem and nprobe == nprobes[0] else "       "
        print("%-50s %7d %7.3f %7.4f %7.4f %s" % (
            label, nprobe, ms, r1, r10, mem_str))

    print()
    del index

# ===========================================================================
# Summary
# ===========================================================================

print("=" * 100)
print("Done.")
print()
print("Notes:")
print("  - SQ4fs: native 4-bit scalar quantizer on FastScan SIMD path")
print("  - SQ6fs/SQ8fs: 4-bit fast-scan pre-filter + rerank against original codes")
print("  - RaBitQfs{N}: RaBitQ with N-bit codes on FastScan SIMD path")
print("  - RaBitQ uses qb=8 (query quantization bits) for all IVF runs")
print("  - rerank_factor=%.1f for SQ6fs/SQ8fs" % args.rerank_factor)
print("  - Memory reported is approximate (code bytes only, excludes metadata)")
