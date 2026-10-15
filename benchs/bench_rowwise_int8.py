# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Exact search on row-wise int8 vectors (GpuIndexRowwiseInt8) vs GpuIndexFlat
in float32 and float16 on the same data: search time, storage per vector,
and recall@k against the float32 result.

    python bench_rowwise_int8.py [--nb 300000] [--nq 100000] [--d 768]
"""

import argparse
import time

import faiss
import numpy as np


def timed_search(index, xq, k, repeats):
    index.search(xq[:1000], k)  # warm-up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.time()
        D, I = index.search(xq, k)
        best = min(best, time.time() - t0)
    return best, D, I


def recall(I, Iref):
    k = I.shape[1]
    hits = sum(len(np.intersect1d(a, b)) for a, b in zip(I, Iref))
    return hits / (len(I) * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=300000)
    ap.add_argument("--nq", type=int, default=100000)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    rs = np.random.RandomState(0)
    # vectors with varying norms, like unnormalized embeddings
    xb = (rs.randn(a.nb, a.d)
          * rs.uniform(0.5, 2, (a.nb, 1))).astype("float32")
    xq = (rs.randn(a.nq, a.d)
          * rs.uniform(0.5, 2, (a.nq, 1))).astype("float32")
    res = faiss.StandardGpuResources()

    rows = []
    ref = None
    for name, fp16 in [("GpuIndexFlatL2 float32", False),
                       ("GpuIndexFlatL2 float16", True)]:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = fp16
        index = faiss.GpuIndexFlatL2(res, a.d, cfg)
        index.add(xb)
        t, D, I = timed_search(index, xq, a.k, a.repeats)
        if ref is None:
            ref = I
        rows.append((name, a.d * (2 if fp16 else 4), t, recall(I, ref)))
        del index

    cpu = faiss.IndexRowwiseInt8(a.d)
    cpu.add(xb)
    index = faiss.GpuIndexRowwiseInt8(res, cpu)
    t, D, I = timed_search(index, xq, a.k, a.repeats)
    rows.append(("GpuIndexRowwiseInt8", cpu.code_size, t, recall(I, ref)))

    print(f"nb={a.nb} nq={a.nq} d={a.d} k={a.k} "
          f"GPU={faiss.get_num_gpus()} (best of {a.repeats})")
    print(f"{'index':28s} {'bytes/vec':>10s} {'search s':>9s} "
          f"{'recall@k vs fp32':>17s}")
    for name, b, t, r in rows:
        print(f"{name:28s} {b:10d} {t:9.2f} {r:17.4f}")


if __name__ == "__main__":
    main()
