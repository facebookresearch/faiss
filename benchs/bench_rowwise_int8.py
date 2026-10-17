# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Exact search on row-wise int8 vectors (GpuIndexRowwiseInt8) vs GpuIndexFlat
in float32 and float16 on the same data: search time, storage per vector,
recall@k against the float32 result, and the mean error of the k neighbour
distances against float32 (L2 rounding makes int8 distances larger on
average; SearchParametersRowwiseInt8.rounding_bias = 1 removes that).

For comparison, int8 with one scale for all vectors (raw int8 codes without a
per-vector scale, s = max |x| over the database / 127) is searched with the
same kernels: every vector is given that scale through add_int8 / search_int8,
so the row only measures the effect of the per-vector scale.

    python bench_rowwise_int8.py [--nb 300000] [--nq 100000] [--d 768]
"""

import argparse
import time

import faiss
import numpy as np


def timed_search(index, xq, k, repeats, params=None):
    index.search(xq[:1000], k, params=params)  # warm-up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.time()
        D, I = index.search(xq, k, params=params)
        best = min(best, time.time() - t0)
    return best, D, I


def i8ptr(a):
    return faiss.cast_integer_to_int8_ptr(a.ctypes.data)


def global_scale_int8(res, xb, xq, k):
    """int8 with one scale for all vectors, on the same index type"""
    s = np.float32(np.abs(xb).max() / 127)
    cb = np.clip(np.rint(xb / s), -127, 127).astype("int8")
    cq = np.clip(np.rint(xq / s), -127, 127).astype("int8")
    sb = np.full(len(cb), s, dtype="float32")   # swig_ptr keeps no reference
    cpu = faiss.IndexRowwiseInt8(xb.shape[1])
    cpu.add_int8(len(cb), i8ptr(cb), faiss.swig_ptr(sb))
    index = faiss.GpuIndexRowwiseInt8(res, cpu)
    sq = np.full(len(cq), s, dtype="float32")
    D = np.empty((len(cq), k), dtype="float32")
    I = np.empty((len(cq), k), dtype="int64")
    index.search_int8(len(cq), i8ptr(cq), faiss.swig_ptr(sq), k,
                      faiss.swig_ptr(D), faiss.swig_ptr(I))
    return D, I


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
    ref = Dref = None
    for name, fp16 in [("GpuIndexFlatL2 float32", False),
                       ("GpuIndexFlatL2 float16", True)]:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = fp16
        index = faiss.GpuIndexFlatL2(res, a.d, cfg)
        index.add(xb)
        t, D, I = timed_search(index, xq, a.k, a.repeats)
        if ref is None:
            ref, Dref = I, D
        rows.append((name, a.d * (2 if fp16 else 4), t, recall(I, ref),
                     (D - Dref).mean()))
        del index

    cpu = faiss.IndexRowwiseInt8(a.d)
    cpu.add(xb)
    index = faiss.GpuIndexRowwiseInt8(res, cpu)
    p = faiss.SearchParametersRowwiseInt8()
    p.rounding_bias = 1.0
    for name, params in [("GpuIndexRowwiseInt8", None),
                         ("  rounding_bias = 1", p)]:
        t, D, I = timed_search(index, xq, a.k, a.repeats, params)
        rows.append((name, cpu.code_size, t, recall(I, ref),
                     (D - Dref).mean()))
    del index
    D, I = global_scale_int8(res, xb, xq, a.k)
    rows.append(("int8, one global scale", a.d, float("nan"), recall(I, ref),
                 (D - Dref).mean()))

    print(f"nb={a.nb} nq={a.nq} d={a.d} k={a.k} "
          f"GPU={faiss.get_num_gpus()} (best of {a.repeats})")
    print(f"{'index':28s} {'bytes/vec':>10s} {'search s':>9s} "
          f"{'recall@k vs fp32':>17s} {'mean dist err':>14s}")
    for name, b, t, r, e in rows:
        ts = f"{t:9.2f}" if t == t else f"{'(same)':>9s}"   # same kernels
        print(f"{name:28s} {b:10d} {ts} {r:17.4f} {e:14.4f}")


if __name__ == "__main__":
    main()
