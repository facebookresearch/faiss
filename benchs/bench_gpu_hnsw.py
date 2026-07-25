# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU HNSW benchmark on SIFT1M (open-source, reproducible).

Builds a CPU faiss.IndexHNSW on SIFT1M, moves it to the GPU with
index_cpu_to_gpu (the GpuCloner routes IndexHNSWFlat / IndexHNSWSQ to
GpuIndexHNSW), and compares CPU vs GPU search over an efSearch sweep.
Both traverse the identical HNSW graph, so recall tracks closely; the GPU
runs the beam search on device.

Usage:
    python bench_gpu_hnsw.py <k> [hnsw hnsw_sq]

Example:
    python bench_gpu_hnsw.py 10 hnsw
"""

import sys
import time

import faiss
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M


k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
todo = sys.argv[2:]
if todo == []:
    todo = ["hnsw", "hnsw_sq"]

print("load data")
ds = DatasetSIFT1M()
xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()
xt = ds.get_train()

nq, d = xq.shape

res = faiss.StandardGpuResources()

EFSEARCH = [16, 32, 64, 128, 256]


def evaluate(search_fn):
    # warm-up (first GPU launch pays kernel-load / allocation cost)
    search_fn(xq[:32], k)

    t0 = time.time()
    D, I = search_fn(xq, k)
    t1 = time.time()

    ms_per_query = (t1 - t0) * 1000.0 / nq
    qps = nq / (t1 - t0)
    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)
    recall_at_k = np.mean([
        len(set(I[i]) & set(gt[i, :k])) / k for i in range(nq)
    ])
    return ms_per_query, qps, recall_at_1, recall_at_k, D, I


def run(name, cpu_index):
    print("\n=== %s ===" % name)
    print("add %d vectors" % xb.shape[0])
    cpu_index.add(xb)

    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    assert isinstance(gpu_index, faiss.GpuIndexHNSW), type(gpu_index)

    print("%-6s %-4s | %-9s %-9s %-7s %-7s | %-9s %-9s %-7s %-7s | %-6s"
          % ("dev", "ef", "ms/q", "qps", "R@1", "R@%d" % k,
             "", "", "", "", "gpu/cpu"))
    for ef in EFSEARCH:
        cpu_index.hnsw.efSearch = ef

        def cpu_search(x, kk, idx=cpu_index):
            return idx.search(x, kk)

        cms, cqps, cr1, crk, _, _ = evaluate(cpu_search)

        params = faiss.SearchParametersGpuHNSW()
        params.ef = ef

        def gpu_search(x, kk, idx=gpu_index, p=params):
            return idx.search(x, kk, params=p)

        gms, gqps, gr1, grk, _, _ = evaluate(gpu_search)

        speedup = gqps / cqps if cqps > 0 else float("nan")
        print("CPU    %-4d | %9.4f %9.0f %7.4f %7.4f" %
              (ef, cms, cqps, cr1, crk))
        print("GPU    %-4d | %9.4f %9.0f %7.4f %7.4f | %5.2fx"
              % (ef, gms, gqps, gr1, grk, speedup))


if "hnsw" in todo:
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 40
    index.verbose = True
    run("HNSW Flat (fp32)", index)

if "hnsw_sq" in todo:
    index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 32)
    index.hnsw.efConstruction = 40
    print("\ntrain SQ8")
    index.train(xt)
    index.verbose = True
    run("HNSW SQ8", index)
