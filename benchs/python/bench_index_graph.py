# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_graph.cpp — HNSW build/search and
# NSG search.
#
# Covers:
#   * ivf_hnsw_quantizer — IndexIVFFlat whose coarse quantizer is an
#     IndexHNSWFlat (nlist=16384, quantizer efSearch=64,
#     quantizer_trains_alone=2), build + nprobe search sweep.
#   * hnsw_locks — batched add() throughput with IndexHNSW.retain_locks toggled
#     (use --threads=N for multiple threads).
#   * HNSWSQ with QT_8bit and M=16 — the exact IndexHNSWSQ(d, QT_8bit, 16)
#     config (distinct from the QT_4bit/M=32 prune_headroom case below).
# The HNSW efSearch default sweep and NSG search_L default sweep are
# {16,32,64,128,256} and {-1,16,32,64,128,256} respectively
# (search_L=-1 means "default/full search" — NSG clamps it to k).
#
# Design notes:
#   * SIFT1M HNSW (bench_hnsw_search_sift1m) builds over the first 100k of
#     the 1M base vectors (a full-base HNSW build takes minutes), so recall
#     against the full-base ground truth is not reported.
#   * IVFFlat search lives in bench_index_ivf.py and k-means
#     training/assignment in bench_codec_kmeans.py.

import functools

import faiss
import pytest

from bench_utils import (
        compute_recall, dataset_or_skip, ground_truth, load_dataset, params,
        rand_mat, require_attr)

K = 10  # fixed in the C++ file


@functools.lru_cache(maxsize=4)
def _hnsw_index(d, nb, M, efConstruction):
    """Cached HNSW index so efSearch/nq sweeps reuse one (expensive) build."""
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.verbose = False
    index.add(rand_mat(nb, d))
    return index


@functools.lru_cache(maxsize=4)
def _nsg_index(d, nb, R):
    """Cached NSG index so search_L/nq sweeps reuse one (expensive) build."""
    index = faiss.IndexNSGFlat(d, R)
    index.build_type = 1  # NNDescent build, matching the cpp file
    index.verbose = False
    index.add(rand_mat(nb, d))
    return index


@params(d=[128], nb=[100000], M=[16, 32, 64], efConstruction=[40, 128])
def bench_hnsw_build(benchmark, d, nb, M, efConstruction):
    xb = rand_mat(nb, d)

    def fresh_index():
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = efConstruction
        index.verbose = False
        return (index,), {}

    # rounds=1: a single-threaded HNSW build at nb=100000 takes tens of
    # seconds at M=64/efConstruction=128; each round builds a fresh index.
    benchmark.pedantic(
            lambda index: index.add(xb), setup=fresh_index, rounds=1)


@params(
        d=[128],
        nb=[100000],
        M=[16, 32, 64],
        efSearch=[16, 32, 64, 128, 256],
        nq=[1, 10, 100],
        bounded=[1, 0])
def bench_hnsw_search(benchmark, d, nb, M, efSearch, nq, bounded):
    # efConstruction fixed at 40 for search benchmarks, matching the cpp file.
    index = _hnsw_index(d, nb, M, 40)
    sp = faiss.SearchParametersHNSW()
    sp.efSearch = efSearch
    sp.bounded_queue = bool(bounded)
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K, params=sp)

    # Report recall
    _, I = index.search(xq, K, params=sp)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(
        d=[128],
        nb=[100000],
        R=[32, 64],
        search_L=[-1, 16, 32, 64, 128, 256],
        nq=[1, 10, 100])
def bench_nsg_search(benchmark, d, nb, R, search_L, nq):
    require_attr(faiss, "IndexNSGFlat")
    index = _nsg_index(d, nb, R)
    # search_L=-1 means "default/full search"; NSG internally
    # clamps it to k (see IndexNSG.cpp: L = max(nsg.search_L, k)).
    index.nsg.search_L = int(search_L)
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# HNSWSQ case: IndexHNSWSQ(d, QT_8bit, 16). This is a distinct configuration
# from the QT_4bit/M=32 prune_headroom case below. efConstruction fixed at 40.


@functools.lru_cache(maxsize=4)
def _hnsw_sq8_index(d, nb, M):
    index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M)
    index.hnsw.efConstruction = 40
    index.verbose = False
    xb = rand_mat(nb, d)
    index.train(xb)
    index.add(xb)
    return index


@params(
        d=[128],
        nb=[50000],
        M=[16],
        efSearch=[16, 32, 64, 128, 256],
        nq=[1, 10, 100])
def bench_hnsw_sq8_search(benchmark, d, nb, M, efSearch, nq):
    require_attr(faiss, "IndexHNSWSQ")
    index = _hnsw_sq8_index(d, nb, M)
    index.hnsw.efSearch = efSearch
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# IVF HNSW quantizer case: IndexIVFFlat whose coarse quantizer is an
# IndexHNSWFlat (nlist=16384, quantizer M=32, efSearch=64,
# quantizer_trains_alone=2). nlist stays CLI-overridable.


@functools.lru_cache(maxsize=4)
def _ivf_hnsw_quantizer_index(d, nb, nlist, quant_M, quant_efSearch):
    quantizer = faiss.IndexHNSWFlat(d, quant_M)
    quantizer.hnsw.efSearch = quant_efSearch
    # The faiss SWIG wrapper appends constructor args to referenced_objects,
    # so the quantizer stays alive without an explicit reference here.
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.cp.min_points_per_centroid = 5  # quiet warning
    index.quantizer_trains_alone = 2
    index.verbose = False
    xb = rand_mat(nb, d)
    index.train(xb)
    index.add(xb)
    return index


@params(
        d=[128],
        nb=[100000],
        nlist=[16384],
        M=[32],
        efSearch=[64],
        nprobe=[1, 4, 16, 64, 256])
def bench_ivf_hnsw_quantizer_search(
        benchmark, d, nb, nlist, M, efSearch, nprobe):
    index = _ivf_hnsw_quantizer_index(d, nb, nlist, M, efSearch)
    index.nprobe = nprobe
    xq = rand_mat(100, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, 100, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# HNSW locks case: batched add() throughput with IndexHNSW.retain_locks toggled.
# Use --threads=N for multiple threads; retain_locks keeps per-node locks
# across add() calls so concurrent batched adds avoid re-locking. Timed
# region is the batched add loop over the whole database.


@params(
        d=[128],
        nb=[100000],
        M=[32],
        retain_locks=[0, 1])
def bench_hnsw_locks_add(benchmark, d, nb, M, retain_locks):
    require_attr(faiss, "IndexHNSWFlat")
    probe = faiss.IndexHNSWFlat(d, M)
    if not hasattr(probe, "retain_locks"):
        pytest.skip("faiss build has no IndexHNSW.retain_locks")
    xb = rand_mat(nb, d)
    batch_size = max(1, nb // 100)

    def fresh_index():
        index = faiss.IndexHNSWFlat(d, M)
        index.retain_locks = bool(int(retain_locks))
        index.verbose = False
        return (index,), {}

    def batched_add(index):
        for i in range(0, nb, batch_size):
            index.add(xb[i:i + batch_size])

    # rounds=1: each round rebuilds the full graph via batched adds.
    benchmark.pedantic(batched_add, setup=fresh_index, rounds=1)


# IndexHNSWSQ (QT_4bit) build and search with a prune_headroom sweep.
# Defaults: d=128, nb=50000, M=32, efConstruction=40, headroom in
# {0.0, 0.04, 0.08, 0.12, 0.16, 0.20}. headroom values are floats,
# so CLI overrides arrive as strings — coerce with float().


def _skip_unless_prune_headroom(index):
    if not hasattr(index.hnsw, "prune_headroom"):
        pytest.skip("faiss build has no hnsw.prune_headroom")


@params(
        d=[128],
        nb=[50000],
        M=[32],
        efConstruction=[40],
        headroom=[0.0, 0.04, 0.08, 0.12, 0.16, 0.20])
def bench_hnsw_sq_build(benchmark, d, nb, M, efConstruction, headroom):
    require_attr(faiss, "IndexHNSWSQ")
    xb = rand_mat(nb, d)

    def fresh_trained_index():
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_4bit, M)
        _skip_unless_prune_headroom(index)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.prune_headroom = float(headroom)
        index.verbose = False
        # SQ training is untimed setup; the timed region is only add()
        index.train(xb)
        return (index,), {}

    benchmark.pedantic(
            lambda index: index.add(xb), setup=fresh_trained_index, rounds=1)


# maxsize=8: the search sweep caches one index per headroom value (6).
@functools.lru_cache(maxsize=8)
def _hnsw_sq_index(d, nb, M, headroom):
    index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_4bit, M)
    index.hnsw.efConstruction = 40
    index.hnsw.prune_headroom = headroom
    index.verbose = False
    xb = rand_mat(nb, d)
    index.train(xb)
    index.add(xb)
    return index


@params(
        d=[128],
        nb=[50000],
        M=[32],
        efSearch=[16, 32, 64, 128, 256],
        nq=[1, 10, 100],
        headroom=[0.0, 0.04, 0.08, 0.12, 0.16, 0.20])
def bench_hnsw_sq_search(benchmark, d, nb, M, efSearch, nq, headroom):
    require_attr(faiss, "IndexHNSWSQ")
    probe = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_4bit, M)
    _skip_unless_prune_headroom(probe)
    index = _hnsw_sq_index(d, nb, M, float(headroom))
    index.hnsw.efSearch = efSearch
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# SIFT1M HNSW search, mirroring the cpp dataset section (M=32,
# efConstruction=40, efSearch and bounded_queue sweeps). Building HNSW over
# the full 1M base takes minutes, so the index is built over the first 100k
# base vectors; ds["gt"] is only valid for the full base, so recall is not
# reported (see the header note).

SIFT1M_HNSW_NB = 100000


@functools.lru_cache(maxsize=2)
def _hnsw_sift1m_index(data_dir, M, nb):
    """Cached HNSW index over the first `nb` dataset base vectors."""
    ds = load_dataset(data_dir)
    index = faiss.IndexHNSWFlat(ds["xb"].shape[1], M)
    index.hnsw.efConstruction = 40
    index.verbose = False
    index.add(ds["xb"][:nb])
    return index


@params(
        M=[32],
        efSearch=[16, 32, 64, 128, 256],
        nq=[1, 10, 100],
        bounded=[1, 0])
def bench_hnsw_search_sift1m(benchmark, data_dir, M, efSearch, nq, bounded):
    ds = dataset_or_skip(data_dir)
    # Subsampled base (100k of 1M): no recall reported (see comment above).
    index = _hnsw_sift1m_index(data_dir, M, SIFT1M_HNSW_NB)
    sp = faiss.SearchParametersHNSW()
    sp.efSearch = efSearch
    sp.bounded_queue = bool(bounded)
    benchmark(index.search, ds["xq"][:nq], K, params=sp)
