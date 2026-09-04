# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark IndexIVFSQFastScan vs IndexIVFScalarQuantizer on SIFT1M.

Compares across all supported ScalarQuantizer types:

  4-bit types                — vpshufb SIMD path (16-level LUT)
  6-bit / 8-bit / 8bit_direct — 4-bit fast scan + rerank with original codes
  fp16 / bf16                — fallback to SQ's own SIMD scanner (expect ~1.0x)
  TurboQuant (tqmse / tq)   — fallback (expect ~1.0x)

Sweeps nprobe values to show the recall/speed tradeoff.
"""

from __future__ import print_function
import numpy as np
import faiss
from datasets import load_sift1M, evaluate

xb, xq, xt, gt = load_sift1M()
nq, d = xq.shape

k = 32
nlist = 256

SQ = faiss.ScalarQuantizer

# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------

native_4bit = {SQ.QT_4bit, SQ.QT_4bit_uniform}

rerank_types = {SQ.QT_6bit, SQ.QT_8bit, SQ.QT_8bit_uniform,
                SQ.QT_8bit_direct, SQ.QT_8bit_direct_signed}

# Types that need L2-normalized input
normalized_types = {SQ.QT_1bit_tqmse, SQ.QT_2bit_tqmse,
                    SQ.QT_3bit_tqmse, SQ.QT_4bit_tqmse, SQ.QT_8bit_tqmse,
                    SQ.QT_2bit_tq, SQ.QT_3bit_tq,
                    SQ.QT_4bit_tq, SQ.QT_5bit_tq}

# Types that need data in [0,255] (uint8 direct) or [-128,127] (signed)
direct_types = {SQ.QT_8bit_direct, SQ.QT_8bit_direct_signed}

# Rerank factors to test for rerank types
rerank_factors = [1, 2]

# nprobe values to sweep
nprobes = [1, 4, 16, 64, 256]

# ---------------------------------------------------------------------------
# All QuantizerTypes to test
# ---------------------------------------------------------------------------

qtypes = [
    # Native 4-bit
    ("QT_4bit",                SQ.QT_4bit),
    ("QT_4bit_uniform",        SQ.QT_4bit_uniform),
    # Rerank
    ("QT_6bit",                SQ.QT_6bit),
    ("QT_8bit",                SQ.QT_8bit),
    ("QT_8bit_uniform",        SQ.QT_8bit_uniform),
    ("QT_8bit_direct",         SQ.QT_8bit_direct),
    ("QT_8bit_direct_signed",  SQ.QT_8bit_direct_signed),
    # Fallback — expect ~1.0x, no regression
    ("QT_fp16",                SQ.QT_fp16),
    ("QT_bf16",                SQ.QT_bf16),
    # TurboQuant MSE — fallback, needs normalized data
    ("QT_2bit_tqmse",          SQ.QT_2bit_tqmse),
    ("QT_3bit_tqmse",          SQ.QT_3bit_tqmse),
    ("QT_4bit_tqmse",          SQ.QT_4bit_tqmse),
    # TurboQuant full — fallback, needs normalized data
    ("QT_2bit_tq",             SQ.QT_2bit_tq),
    ("QT_3bit_tq",             SQ.QT_3bit_tq),
    ("QT_4bit_tq",             SQ.QT_4bit_tq),
    ("QT_5bit_tq",             SQ.QT_5bit_tq),
]

# ---------------------------------------------------------------------------
# Prepare normalized copies for TurboQuant types
# ---------------------------------------------------------------------------

xb_norm = xb.copy()
xq_norm = xq.copy()
xt_norm = xt.copy()
faiss.normalize_L2(xb_norm)
faiss.normalize_L2(xq_norm)
faiss.normalize_L2(xt_norm)

# Ground truth for normalized data (TurboQuant uses inner product)
index_flat_norm = faiss.IndexFlatIP(d)
index_flat_norm.add(xb_norm)
_, gt_norm = index_flat_norm.search(xq_norm, 100)
del index_flat_norm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ivfsq(d, nlist, qtype, db, train, metric=faiss.METRIC_L2):
    """Build and populate an IndexIVFScalarQuantizer."""
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, qtype, metric)
    index.train(train)
    index.add(db)
    return index


def make_ivfsqfs(d, nlist, qtype, db, train, metric=faiss.METRIC_L2,
                 rerank_factor=4):
    """Build and populate an IndexIVFSQFastScan."""
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFSQFastScan(quantizer, d, nlist, qtype, metric)
    index.rerank_factor = rerank_factor
    index.train(train)
    index.add(db)
    return index


def evaluate_at_nprobes(index, xq, gt, k, nprobes):
    """Evaluate index at multiple nprobe values, return list of (nprobe, ms, recalls)."""
    results = []
    for nprobe in nprobes:
        index.nprobe = nprobe
        t, r = evaluate(index, xq, gt, k)
        results.append((nprobe, t, r))
    return results


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print("%-30s %-22s %7s %7s %7s %7s" % (
    "index", "qtype", "nprobe", "ms/q", "R@1", "R@10"))
print("-" * 100)

for qname, qtype in qtypes:

    # Select appropriate data and ground truth
    if qtype in normalized_types:
        db, query, train, ground_truth = xb_norm, xq_norm, xt_norm, gt_norm
        metric = faiss.METRIC_INNER_PRODUCT
    elif qtype in direct_types:
        db, query, train, ground_truth = xb, xq, xt, gt
        metric = faiss.METRIC_L2
    else:
        db, query, train, ground_truth = xb, xq, xt, gt
        metric = faiss.METRIC_L2

    # --- Baseline: IndexIVFScalarQuantizer ---
    try:
        index_sq = make_ivfsq(d, nlist, qtype, db, train, metric)
    except Exception as e:
        print("%-30s %-22s  SKIP (%s)" % ("IndexIVFScalarQuantizer", qname, e))
        print()
        continue

    results_sq = evaluate_at_nprobes(index_sq, query, ground_truth, k, nprobes)

    for nprobe, t_sq, r_sq in results_sq:
        print("%-30s %-22s %7d %7.3f %7.4f %7.4f" % (
            "IndexIVFScalarQuantizer", qname, nprobe, t_sq, r_sq[1], r_sq[10]))
    print()

    # --- IndexIVFSQFastScan ---
    if qtype in rerank_types:
        # Test multiple rerank factors at all nprobe values
        for rf in rerank_factors:
            try:
                index_fs = make_ivfsqfs(d, nlist, qtype, db, train, metric, rf)
            except Exception as e:
                print("%-30s %-22s  SKIP (%s)" % (
                    "IndexIVFSQFastScan(rf=%d)" % rf, qname, e))
                continue

            results_fs = evaluate_at_nprobes(index_fs, query, ground_truth, k, nprobes)

            for (nprobe, t_fs, r_fs), (_, t_sq, r_sq) in zip(
                    results_fs, results_sq):
                speedup = t_sq / t_fs if t_fs > 0 else float('inf')
                label = "IVFSQFastScan(rf=%d)" % rf
                print("%-30s %-22s %7d %7.3f %7.4f %7.4f  %.1fx" % (
                    label, qname, nprobe, t_fs, r_fs[1], r_fs[10], speedup))

            del index_fs
            print()
    else:
        # Native 4-bit or fallback: single config
        try:
            index_fs = make_ivfsqfs(d, nlist, qtype, db, train, metric)
        except Exception as e:
            print("%-30s %-22s  SKIP (%s)" % (
                "IndexIVFSQFastScan", qname, e))
            print()
            del index_sq
            continue

        results_fs = evaluate_at_nprobes(index_fs, query, ground_truth, k, nprobes)

        is_fallback = qtype not in native_4bit and qtype not in rerank_types
        tag = " (fallback)" if is_fallback else ""

        for (nprobe, t_fs, r_fs), (_, t_sq, r_sq) in zip(
                results_fs, results_sq):
            speedup = t_sq / t_fs if t_fs > 0 else float('inf')
            label = "IVFSQFastScan" + tag
            print("%-30s %-22s %7d %7.3f %7.4f %7.4f  %.1fx" % (
                label, qname, nprobe, t_fs, r_fs[1], r_fs[10], speedup))

        # For fallback types, verify no regression in recall
        if is_fallback:
            worst_r1_delta = max(
                abs(r_sq[1] - r_fs[1])
                for (_, _, r_fs), (_, _, r_sq) in zip(results_fs, results_sq))
            worst_r10_delta = max(
                abs(r_sq[10] - r_fs[10])
                for (_, _, r_fs), (_, _, r_sq) in zip(results_fs, results_sq))
            if worst_r1_delta > 0.001 or worst_r10_delta > 0.001:
                print("  ** WARNING: recall mismatch! "
                      "worst R@1 delta=%.4f, worst R@10 delta=%.4f" % (
                          worst_r1_delta, worst_r10_delta))
            else:
                print("  -> recall matches IVF-SQ (fallback OK)")

        del index_fs
        print()

    del index_sq

print("=" * 100)
print("Done.")
