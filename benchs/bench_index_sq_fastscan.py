# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark native 4-bit ScalarQuantizer types on SIFT1M.

Compares IndexScalarQuantizer (baseline) with IndexSQFastScan:
  4-bit types -- vpshufb SIMD path (16-level LUT)

For higher-precision types with reranking, use
IndexRefine(IndexSQFastScan(QT_4bit), IndexScalarQuantizer(QT_8bit)).
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
# Types to benchmark
# ---------------------------------------------------------------------------

qtypes = [
    ("QT_4bit",         SQ.QT_4bit),
    ("QT_4bit_uniform", SQ.QT_4bit_uniform),
]

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print("Loading sift1M...done")
print("%-30s %-22s %7s %7s %7s" % (
    "index", "qtype", "ms/q", "R@1", "R@10"))
print("-" * 86)

for qname, qtype in qtypes:
    db, query, train, ground_truth = xb, xq, xt, gt
    metric = faiss.METRIC_L2

    # --- Baseline: IndexScalarQuantizer ---
    index_sq = faiss.IndexScalarQuantizer(d, qtype, metric)
    index_sq.train(train)
    index_sq.add(db)
    t_sq, r_sq = evaluate(index_sq, query, ground_truth, k)
    print("%-30s %-22s %7.3f %7.4f %7.4f" % (
        "IndexScalarQuantizer", qname, t_sq, r_sq[1], r_sq[10]))

    # --- IndexSQFastScan ---
    index_fs = faiss.IndexSQFastScan(d, qtype, metric)
    index_fs.train(train)
    index_fs.add(db)
    t_fs, r_fs = evaluate(index_fs, query, ground_truth, k)

    print("%-30s %-22s %7.3f %7.4f %7.4f" % (
        "IndexSQFastScan", qname, t_fs, r_fs[1], r_fs[10]))
    speedup = t_sq / t_fs if t_fs > 0 else float('inf')
    print("  -> speedup: %.1fx" % speedup)

    print()
    del index_sq, index_fs
