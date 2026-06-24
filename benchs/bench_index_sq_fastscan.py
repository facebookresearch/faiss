# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark all ScalarQuantizer types on SIFT1M.

Compares IndexScalarQuantizer (baseline) with IndexSQFastScan:

  4-bit types                — vpshufb SIMD path (16-level LUT)
  6-bit / 8-bit / 8bit_direct — 4-bit fast scan + rerank with original codes
  fp16 / bf16                — fallback to SQ's own SIMD scanner (expect ~1.0x)
  TurboQuant (tqmse / tq)   — fallback (expect ~1.0x)

QT_0bit is skipped: zero code size, meaningful only inside IVF.
"""

from __future__ import print_function
import numpy as np
import faiss
from datasets import load_sift1M, evaluate

xb, xq, xt, gt = load_sift1M()
nq, d = xq.shape

k = 32

SQ = faiss.ScalarQuantizer

# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------

# Native 4-bit: vpshufb SIMD path
native_4bit = {SQ.QT_4bit, SQ.QT_4bit_uniform}

# Rerank: 4-bit fast scan + exact rerank
rerank_types = {SQ.QT_6bit, SQ.QT_8bit, SQ.QT_8bit_uniform,
                SQ.QT_8bit_direct, SQ.QT_8bit_direct_signed}

# Everything else is fallback (SQ scanner, expect ~1.0x)

# Rerank factors to test
rerank_factors = [2, 4, 8]

# Types that need L2-normalized input
normalized_types = {SQ.QT_1bit_tqmse, SQ.QT_2bit_tqmse,
                    SQ.QT_3bit_tqmse, SQ.QT_4bit_tqmse, SQ.QT_8bit_tqmse,
                    SQ.QT_2bit_tq, SQ.QT_3bit_tq,
                    SQ.QT_4bit_tq, SQ.QT_5bit_tq}

# Types that need data in [0,255] (uint8 direct) or [-128,127] (signed)
direct_types = {SQ.QT_8bit_direct, SQ.QT_8bit_direct_signed}

# ---------------------------------------------------------------------------
# All QuantizerTypes to test (everything except QT_0bit)
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
    ("QT_1bit_tqmse",          SQ.QT_1bit_tqmse),
    ("QT_2bit_tqmse",          SQ.QT_2bit_tqmse),
    ("QT_3bit_tqmse",          SQ.QT_3bit_tqmse),
    ("QT_4bit_tqmse",          SQ.QT_4bit_tqmse),
    ("QT_8bit_tqmse",          SQ.QT_8bit_tqmse),
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

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print("Loading sift1M...done")
print("%-30s %-22s %7s %7s %7s" % (
    "index", "qtype", "ms/q", "R@1", "R@10"))
print("-" * 86)

for qname, qtype in qtypes:

    # Select appropriate data
    if qtype in normalized_types:
        db, query, train = xb_norm, xq_norm, xt_norm
    elif qtype in direct_types:
        # SIFT data is already uint8 histograms in [0,128]; fine for direct
        db, query, train = xb, xq, xt
    else:
        db, query, train = xb, xq, xt

    # --- Baseline: IndexScalarQuantizer ---
    index_sq = faiss.IndexScalarQuantizer(d, qtype)
    index_sq.train(train)
    index_sq.add(db)
    t_sq, r_sq = evaluate(index_sq, query, gt, k)
    print("%-30s %-22s %7.3f %7.4f %7.4f" % (
        "IndexScalarQuantizer", qname, t_sq, r_sq[1], r_sq[10]))

    # --- IndexSQFastScan ---
    if qtype in rerank_types:
        # Test multiple rerank factors
        for rf in rerank_factors:
            index_fs = faiss.IndexSQFastScan(d, qtype)
            index_fs.rerank_factor = rf
            index_fs.train(train)
            index_fs.add(db)
            t_fs, r_fs = evaluate(index_fs, query, gt, k)
            label = "IndexSQFastScan(rf=%d)" % rf
            print("%-30s %-22s %7.3f %7.4f %7.4f" % (
                label, qname, t_fs, r_fs[1], r_fs[10]))
            speedup = t_sq / t_fs if t_fs > 0 else float('inf')
            print("  -> speedup: %.1fx" % speedup)
            del index_fs
    else:
        index_fs = faiss.IndexSQFastScan(d, qtype)
        index_fs.train(train)
        index_fs.add(db)
        t_fs, r_fs = evaluate(index_fs, query, gt, k)

        if qtype in native_4bit:
            tag = ""
        else:
            tag = " (fallback)"
        print("%-30s %-22s %7.3f %7.4f %7.4f" % (
            "IndexSQFastScan" + tag, qname, t_fs, r_fs[1], r_fs[10]))
        speedup = t_sq / t_fs if t_fs > 0 else float('inf')
        print("  -> speedup: %.1fx" % speedup)

        # For fallback types, verify no regression in recall
        if qtype not in native_4bit:
            r1_delta = abs(r_sq[1] - r_fs[1])
            r10_delta = abs(r_sq[10] - r_fs[10])
            if r1_delta > 0.001 or r10_delta > 0.001:
                print("  ** WARNING: recall mismatch! "
                      "R@1 delta=%.4f, R@10 delta=%.4f" % (
                          r1_delta, r10_delta))
            else:
                print("  -> recall matches SQ (fallback OK)")

        del index_fs

    print()
    del index_sq
