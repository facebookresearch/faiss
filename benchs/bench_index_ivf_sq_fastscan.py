# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark IndexIVFSQFastScan vs IndexIVFScalarQuantizer on SIFT1M.

IndexIVFSQFastScan supports native 4-bit scalar quantizers only (QT_4bit /
QT_4bit_uniform), packed onto the PQ4 FastScan SIMD path.  For higher-precision
quantizers, wrap a 4-bit IndexIVFSQFastScan with IndexRefine, or use
IndexIVFScalarQuantizer directly.

Sweeps nprobe to show the recall/speed tradeoff of the SIMD fast-scan path
against the scalar IndexIVFScalarQuantizer baseline at the same 4-bit width.
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

qtypes = [
    ("QT_4bit", SQ.QT_4bit),
    ("QT_4bit_uniform", SQ.QT_4bit_uniform),
]

nprobes = [1, 4, 16, 64, 256]


def make_ivfsq(qtype):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, qtype)
    index.train(xt)
    index.add(xb)
    return index


def make_ivfsqfs(qtype):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFSQFastScan(quantizer, d, nlist, qtype)
    index.train(xt)
    index.add(xb)
    return index


def sweep(index):
    results = []
    for nprobe in nprobes:
        index.nprobe = nprobe
        t, r = evaluate(index, xq, gt, k)
        results.append((nprobe, t, r))
    return results


print("%-30s %-18s %7s %7s %7s %7s" % (
    "index", "qtype", "nprobe", "ms/q", "R@1", "R@10"))
print("-" * 90)

for qname, qtype in qtypes:
    ref = make_ivfsq(qtype)
    res_ref = sweep(ref)
    for nprobe, t, r in res_ref:
        print("%-30s %-18s %7d %7.3f %7.4f %7.4f" % (
            "IndexIVFScalarQuantizer", qname, nprobe, t, r[1], r[10]))
    del ref
    print()

    fs = make_ivfsqfs(qtype)
    res_fs = sweep(fs)
    for (nprobe, t_fs, r_fs), (_, t_ref, _) in zip(res_fs, res_ref):
        speedup = t_ref / t_fs if t_fs > 0 else float("inf")
        print("%-30s %-18s %7d %7.3f %7.4f %7.4f  %.1fx" % (
            "IndexIVFSQFastScan", qname, nprobe, t_fs, r_fs[1], r_fs[10],
            speedup))
    del fs
    print()

print("=" * 90)
print("Done.")
