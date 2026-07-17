# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python benchmark for HadamardRotation (fast Walsh-Hadamard transform) vs
# RandomRotationMatrix (BLAS sgemm). Covers:
#   * apply() throughput (bench_*_apply). The dim sweep includes
#     non-power-of-two dims: {384,768,1536,3072,6144} (HadamardRotation pads
#     to the next power of two internally, so these are valid).
#   * recall@1 with an IVF index built behind an HR/RR/none pre-transform
#     prefix (bench_fwht_ivf_recall): dims {64,128,256,768,1024,2048,4096},
#     nlist=64, nprobe=8, k=1, n=10000, nq=200.
# Transforms are built/trained once per dimension (module-local caches); only
# .apply(x) is timed for the speed benchmarks.
#
# Design notes: the apply-throughput benchmarks use uniform seed-12345 data
# (rand_mat); the recall benchmarks use Gaussian data with seed 42.

import functools

import faiss
import numpy as np

from bench_utils import (
        built_dataset_index, compute_recall, dataset_or_skip, params,
        rand_mat, require_attr)

# Speed-sweep dims: powers of two plus the non-power-of-two dims that
# HadamardRotation pads internally. n=10000 vectors per apply (matching the
# C++ port).
DIMS = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


@functools.lru_cache(maxsize=None)
def hadamard_rotation(d, seed=42):
    return faiss.HadamardRotation(d, seed)


@functools.lru_cache(maxsize=None)
def random_rotation(d):
    rr = faiss.RandomRotationMatrix(d, d)
    rr.train(rand_mat(100, d))  # RRM ignores the data; train() seeds it
    return rr


@params(d=DIMS, n=[10000])
def bench_hadamard_rotation_apply(benchmark, d, n):
    require_attr(faiss, "HadamardRotation")
    x = rand_mat(n, d)
    transform = hadamard_rotation(d)
    benchmark(transform.apply, x)


@params(d=DIMS, n=[10000])
def bench_random_rotation_apply(benchmark, d, n):
    x = rand_mat(n, d)
    transform = random_rotation(d)
    benchmark(transform.apply, x)


# Recall@1: build an IVF index behind an HR (HadamardRotation) / RR
# (RandomRotation) / none pre-transform prefix and measure recall@1 vs
# brute-force ground truth. Uses Gaussian data (np.random.randn, seed 42),
# n=10000, nq=200, nlist=64, nprobe=8, k=1.
RECALL_DIMS = [64, 128, 256, 768, 1024, 2048, 4096]


@functools.lru_cache(maxsize=None)
def _fwht_recall_data(d, n, nq, seed=42):
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((nq, d), dtype=np.float32)
    gt_index = faiss.IndexFlatL2(d)
    gt_index.add(xb)
    _, gt_I = gt_index.search(xq, 1)
    return xb, xq, gt_I


@params(
        d=RECALL_DIMS,
        n=[10000],
        nq=[200],
        nlist=[64],
        nprobe=[8],
        prefix=["HR", "RR", "none"])
def bench_fwht_ivf_recall(benchmark, d, n, nq, nlist, nprobe, prefix):
    require_attr(faiss, "HadamardRotation")
    xb, xq, gt_I = _fwht_recall_data(d, n, nq)
    if prefix == "none":
        factory_str = f"IVF{nlist},Flat"
    else:
        factory_str = f"{prefix},IVF{nlist},Flat"
    index = faiss.index_factory(d, factory_str)
    index.train(xb)
    index.add(xb)
    faiss.extract_index_ivf(index).nprobe = nprobe
    benchmark(index.search, xq, 1)

    _, I = index.search(xq, 1)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# SIFT1M dataset variant of the recall benchmark: HR/RR/none pre-transform
# prefix in front of an IVF index trained on the learn set and populated with
# the full 1M base set (d=128), recall@1 against the dataset ground truth.
# The apply()-throughput benchmarks above stay synthetic. Registered only
# when --data_dir points at a SIFT1M-layout directory.
@params(nlist=[1024], nprobe=[8, 64], nq=[200], prefix=["HR", "RR", "none"])
def bench_fwht_ivf_recall_sift1m(
        benchmark, data_dir, nlist, nprobe, nq, prefix):
    require_attr(faiss, "HadamardRotation")
    ds = dataset_or_skip(data_dir)
    if prefix == "none":
        factory_str = f"IVF{nlist},Flat"
    else:
        factory_str = f"{prefix},IVF{nlist},Flat"
    index = built_dataset_index(factory_str, data_dir)
    faiss.extract_index_ivf(index).nprobe = nprobe
    xq = ds["xq"][:nq]
    _, I = benchmark(index.search, xq, 1)
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])
