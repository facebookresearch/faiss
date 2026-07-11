# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_kern_distances.cpp — distance computation
# through the public API (faiss.knn, faiss.pairwise_distances). The raw
# fvec_* kernels are C++-only; these benchmarks exercise the same kernels
# via their public entry points.
#
# Design notes:
# - One query vector per iteration (cross-query cache effects are not
#   exercised).
# - Data is uniform [0, 1) (rand_mat), chosen so JensenShannon/BrayCurtis
#   stay valid.

import faiss
import numpy as np
import pytest

from bench_utils import params, rand_mat, require_attr


@params(d=[16, 32, 64, 128, 256, 512, 768, 1536], n=[1000, 10000, 100000])
def bench_knn_L2sqr(benchmark, d, n):
    xb = rand_mat(n, d)
    xq = rand_mat(1, d, seed=54321)
    benchmark(faiss.knn, xq, xb, 1)


@params(d=[16, 32, 64, 128, 256, 512, 768, 1536], n=[1000, 10000, 100000])
def bench_knn_inner_product(benchmark, d, n):
    xb = rand_mat(n, d)
    xq = rand_mat(1, d, seed=54321)
    benchmark(faiss.knn, xq, xb, 1, metric=faiss.METRIC_INNER_PRODUCT)


@params(d=[128, 256, 768], n=[100, 1000, 10000])
def bench_pairwise_L2sqr(benchmark, d, n):
    xb = rand_mat(n, d)
    xq = rand_mat(100, d, seed=54321)
    benchmark(faiss.pairwise_distances, xq, xb)


# SIMD implementations of the additional metrics, exercised through
# faiss.pairwise_distances. Sizes are d=64, nq=4096, nb=16384, declared as
# single-value sweeps so they stay CLI-overridable.
@params(
        metric=["L1", "Linf", "Canberra", "BrayCurtis", "JensenShannon"],
        d=[64],
        nq=[4096],
        nb=[16384])
def bench_pairwise_extra_metrics(benchmark, metric, d, nq, nb):
    metric_type = getattr(faiss, "METRIC_" + metric, None)
    if metric_type is None:
        pytest.skip(f"faiss build has no METRIC_{metric}")
    # rand_mat is uniform [0, 1) rather than randn — strictly
    # positive inputs keep JensenShannon well-defined for all builds.
    xq = rand_mat(nq, d, seed=54321)
    xb = rand_mat(nb, d)
    benchmark(faiss.pairwise_distances, xq, xb, metric_type)


# One-query-vs-many low-level kernels called through their SWIG bindings.
# d sweep including d=3 (the kernels have no alignment requirement —
# verified); n (=ny) sweeps 1e5 (quick point) and 1e6.


@params(d=[3, 4, 12, 36, 64], n=[100000, 1000000])
def bench_fvec_L2sqr_ny(benchmark, d, n):
    require_attr(faiss, "fvec_L2sqr_ny")
    x = rand_mat(1, d, seed=54321)[0]
    ys = rand_mat(n, d)
    dis = np.empty(n, dtype=np.float32)  # preallocated, outside the timing
    benchmark(
            faiss.fvec_L2sqr_ny,
            faiss.swig_ptr(dis), faiss.swig_ptr(x), faiss.swig_ptr(ys), d, n)


@params(d=[3, 4, 12, 36, 64], n=[100000, 1000000])
def bench_fvec_inner_products_ny(benchmark, d, n):
    require_attr(faiss, "fvec_inner_products_ny")
    x = rand_mat(1, d, seed=54321)[0]
    ys = rand_mat(n, d)
    dis = np.empty(n, dtype=np.float32)  # preallocated, outside the timing
    benchmark(
            faiss.fvec_inner_products_ny,
            faiss.swig_ptr(dis), faiss.swig_ptr(x), faiss.swig_ptr(ys), d, n)
