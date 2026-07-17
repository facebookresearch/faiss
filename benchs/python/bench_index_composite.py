# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_composite.cpp (OPQ + IVFPQFastScan
# and IndexRefineFlat-over-IVFPQ search) and cpp/bench_index_rcq.cpp
# (ResidualCoarseQuantizer search over batch/nprobe/beam-factor sweeps).

import functools

import faiss
import numpy as np

from bench_utils import (
        built_dataset_index, compute_recall, dataset_or_skip, ground_truth,
        params, rand_mat, require_attr)


@functools.lru_cache(maxsize=4)
def _composite_index(factory, d, nb):
    """Trained+populated composite index.

    Like the C++ benchmark, trains on min(nb, 50000) vectors (which is why
    bench_utils.built_index, which trains on all nb, is not used here).
    """
    index = faiss.index_factory(d, factory)
    nt = min(nb, 50000)
    index.train(rand_mat(nt, d))
    index.add(rand_mat(nb, d, seed=54321))
    return index


# nlist=1024, M=16, 4-bit PQ and nprobe=8 fixed as in the C++ benchmark.
@params(d=[128], nb=[100000], nq=[1, 10, 100], k=[10], nprobe=[8])
def bench_opq_ivfpqfs_search(benchmark, d, nb, nq, k, nprobe):
    index = _composite_index("OPQ16,IVF1024,PQ16x4fs", d, nb)
    faiss.extract_index_ivf(index).nprobe = nprobe
    xq = rand_mat(nq, d, seed=67890)
    benchmark(index.search, xq, k)

    # Report recall
    _, I = index.search(xq, k)
    gt_I = ground_truth(d, nb, nq, k, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# nlist=256, M=16, 8-bit PQ, nprobe=8 and k_factor=4 fixed as in the C++
# benchmark.
@params(d=[128], nb=[100000], nq=[1, 10, 100], k=[10], nprobe=[8])
def bench_refine_ivfpq_search(benchmark, d, nb, nq, k, nprobe):
    index = _composite_index("IVF256,PQ16,RFlat", d, nb)
    index.k_factor = 4.0
    faiss.extract_index_ivf(index).nprobe = nprobe
    xq = rand_mat(nq, d, seed=67890)
    benchmark(index.search, xq, k)

    # Report recall
    _, I = index.search(xq, k)
    gt_I = ground_truth(d, nb, nq, k, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# SIFT1M dataset variants: same composite configurations, trained on the
# dataset's learn set and populated with the full 1M base set (so recall
# against the dataset ground truth is valid).


@params(nq=[1, 10, 100], k=[10], nprobe=[8])
def bench_opq_ivfpqfs_search_sift1m(benchmark, data_dir, nq, k, nprobe):
    ds = dataset_or_skip(data_dir)
    index = built_dataset_index("OPQ16,IVF1024,PQ16x4fs", data_dir)
    faiss.extract_index_ivf(index).nprobe = nprobe
    _, I = benchmark(index.search, ds["xq"][:nq], k)
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])


@params(nq=[1, 10, 100], k=[10], nprobe=[8])
def bench_refine_ivfpq_search_sift1m(benchmark, data_dir, nq, k, nprobe):
    ds = dataset_or_skip(data_dir)
    index = built_dataset_index("IVF256,PQ16,RFlat", data_dir)
    index.k_factor = 4.0
    faiss.extract_index_ivf(index).nprobe = nprobe
    _, I = benchmark(index.search, ds["xq"][:nq], k)
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])


# ---------------------------------------------------------------------------
# ResidualCoarseQuantizer search (cpp/bench_index_rcq.cpp). d is fixed at
# 512 as in the C++ benchmark; the C++ `batch` sweep maps to nq here.
# ---------------------------------------------------------------------------

RCQ_D = 512


@functools.lru_cache(maxsize=1)
def _trained_rcq(d):
    """RCQ with per-level nbits {16, 8}, as in the C++ benchmark.

    Trains on 2 << 15 = 65536 vectors (the k-means minimum for the 16-bit
    first level), matching the C++ benchmark.
    """
    nbits = faiss.UInt64Vector()
    faiss.copy_array_to_vector(np.array([16, 8], dtype="uint64"), nbits)
    rcq = faiss.ResidualCoarseQuantizer(d, nbits)
    rcq.train(rand_mat(2 << 15, d))
    return rcq


@params(nq=[1, 4, 16, 64], nprobe=[1, 4, 16], beam=[1, 4])
def bench_rcq_search(benchmark, nq, nprobe, beam):
    require_attr(faiss, "ResidualCoarseQuantizer")
    rcq = _trained_rcq(RCQ_D)
    # The C++ benchmark passes beam_factor via
    # SearchParametersResidualCoarseQuantizer; set_beam_factor applies the
    # same knob index-wide from Python.
    rcq.set_beam_factor(float(beam))
    xq = rand_mat(nq, RCQ_D, seed=54321)
    benchmark(rcq.search, xq, nprobe)
