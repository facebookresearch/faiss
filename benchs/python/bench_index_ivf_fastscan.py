# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_ivf_fastscan.cpp — IVFPQFastScan
# (SIMD-accelerated IVFPQ with 4-bit codes) search.
#
# Sweeps six index_factory variants on SIFT1M (k=1, M/nlist=32/1024,
# nprobe curve {1,2,4,6,8,12,16,24,32,48,64,128}):
#   IVF1024,PQ32x4fs                  -> bench_ivfpq_fastscan_search
#   IVF1024,PQ32x4fsr  (residual PQ)  -> bench_ivfpq_fastscan_residual_search
#   IVF1024,LSQ30x4fsr_Nlsq2x4        -> bench_ivf_lsq_fastscan_search
#   IVF1024,RQ30x4fsr_Nrq2x4          -> bench_ivf_rq_fastscan_search
#   IVF1024,LSQ30x4fs_Nlsq2x4  (AQ,   -> bench_ivf_lsq_fastscan_nonresidual_search
#     no by_residual)
#   IVF1024,RQ30x4fs_Nrq2x4    (AQ,   -> bench_ivf_rq_fastscan_nonresidual_search
#     no by_residual)
# The AQ variants are benchmarked with rescale_norm True and False via the
# `rescale` axis. k is swept in {1, 10}, and the default nprobe sweep uses
# the 12-point curve.
#
# Design notes:
#   * AQ (LSQ/RQ) builders train on 40000 vectors (enough for nlist=1024)
#     rather than the full 100k — LSQ training is slow and roughly linear
#     in nt (the C++ synthetic PQ group similarly trains on 50k).

import functools

import faiss
import pytest

from bench_utils import built_index, compute_recall, ground_truth, params, rand_mat, require_attr
from bench_utils import built_dataset_index, dataset_or_skip

# Default 12-point nprobe sweep shared by all search benchmarks below.
NPROBE_SWEEP = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128]


def _ivfpq_fastscan_search(benchmark, factory, nlist, M, nprobe, nq, implem, k):
    d, nb = 128, 100000
    require_attr(faiss, "IndexIVFPQFastScan")
    if M > d:
        pytest.skip(f"M {M} > d {d}")
    if nprobe > nlist:
        pytest.skip(f"nprobe {nprobe} > nlist {nlist}")
    index = built_index(factory, d, nb)
    if not hasattr(index, "implem"):
        pytest.skip("implem is not settable from Python in this faiss build")
    index.nprobe = nprobe
    index.implem = implem
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, k)

    # Report recall
    _, I = index.search(xq, k)
    gt_I = ground_truth(d, nb, nq, k)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(
        nlist=[256, 1024],
        M=[16, 32, 64],
        nprobe=NPROBE_SWEEP,
        nq=[1, 10, 100],
        implem=[0, 12, 14],
        k=[1, 10])
def bench_ivfpq_fastscan_search(benchmark, nlist, M, nprobe, nq, implem, k):
    _ivfpq_fastscan_search(
            benchmark, f"IVF{nlist},PQ{M}x4fs", nlist, M, nprobe, nq, implem, k)


# SIFT1M search (skipped when --data-dir is missing), mirroring the
# "sift1m/ivfpq_fastscan" group of cpp/bench_index_ivf_fastscan.cpp: M sweep
# {16,32} over the full 12-point nprobe curve, k in {1,10}, default implem.
# The full 10000-query set is searched (matching the C++ group);
# --nprobe/--nq can trim the sweep for quicker runs.
@params(nlist=[1024], M=[16, 32], nprobe=NPROBE_SWEEP, nq=[10000], k=[1, 10])
def bench_ivfpq_fastscan_search_sift1m(
        benchmark, data_dir, nlist, M, nprobe, nq, k):
    ds = dataset_or_skip(data_dir)
    require_attr(faiss, "IndexIVFPQFastScan")
    if nprobe > nlist:
        pytest.skip(f"nprobe {nprobe} > nlist {nlist}")
    index = built_dataset_index(f"IVF{nlist},PQ{M}x4fs", data_dir)
    index.nprobe = nprobe
    _, I = benchmark(index.search, ds["xq"][:nq], k)

    # Report recall against the dataset ground truth
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])


# "IVF1024,PQ32x4fsr" — residual PQ fastscan (by_residual=True, the
# trailing "r"). Same IndexIVFPQFastScan class as above with residual codes.
@params(
        nlist=[1024],
        M=[32],
        nprobe=NPROBE_SWEEP,
        nq=[1, 10, 100],
        implem=[0, 12, 14],
        k=[1, 10])
def bench_ivfpq_fastscan_residual_search(
        benchmark, nlist, M, nprobe, nq, implem, k):
    _ivfpq_fastscan_search(
            benchmark, f"IVF{nlist},PQ{M}x4fsr", nlist, M, nprobe, nq, implem, k)


# Additive-quantizer fastscan
# variants (LSQ / RQ), both the "x4fsr" residual encoding and the "x4fs"
# non-residual encoding, each with a 2x4-bit norm quantizer. The AQ builders
# train on 40000 vectors (enough for nlist=1024: 1024*39=39936) rather than
# the usual 50000: LSQ training is slow and roughly linear in nt.
AQ_TRAIN_SIZE = 40000


# The suite trains four distinct AQ factories (LSQ/RQ x residual/non-residual);
# keep them all cached so a full sweep does not retrain (~100 s each).
@functools.lru_cache(maxsize=8)
def _aq_fastscan_index(factory, d, nb):
    index = faiss.index_factory(d, factory)
    xb = rand_mat(nb, d)
    index.train(xb[:AQ_TRAIN_SIZE])
    index.add(xb)
    return index


def _aq_fastscan_search(benchmark, factory_tpl, nlist, M, nprobe, nq, rescale, k):
    d, nb = 128, 100000
    if nprobe > nlist:
        pytest.skip(f"nprobe {nprobe} > nlist {nlist}")
    # M-2 codebooks for the vector + 2x4-bit norm codec, like the
    # "LSQ{M-2}x4fsr_Nlsq2x4" naming logic.
    try:
        index = _aq_fastscan_index(
                factory_tpl.format(nlist=nlist, M=M - 2), d, nb)
    except RuntimeError as e:
        pytest.skip(f"factory not supported by this faiss build: {e}")
    if not hasattr(index, "rescale_norm"):
        pytest.skip("rescale_norm is not settable from Python in this build")
    index.nprobe = nprobe
    index.rescale_norm = bool(int(rescale))
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, k)

    # Report recall
    _, I = index.search(xq, k)
    gt_I = ground_truth(d, nb, nq, k)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(nlist=[1024], M=[32], nprobe=NPROBE_SWEEP, nq=[1, 10, 100],
        rescale=[0, 1], k=[1, 10])
def bench_ivf_lsq_fastscan_search(benchmark, nlist, M, nprobe, nq, rescale, k):
    require_attr(faiss, "IndexIVFLocalSearchQuantizerFastScan")
    _aq_fastscan_search(
            benchmark, "IVF{nlist},LSQ{M}x4fsr_Nlsq2x4",
            nlist, M, nprobe, nq, rescale, k)


@params(nlist=[1024], M=[32], nprobe=NPROBE_SWEEP, nq=[1, 10, 100],
        rescale=[0, 1], k=[1, 10])
def bench_ivf_rq_fastscan_search(benchmark, nlist, M, nprobe, nq, rescale, k):
    require_attr(faiss, "IndexIVFResidualQuantizerFastScan")
    _aq_fastscan_search(
            benchmark, "IVF{nlist},RQ{M}x4fsr_Nrq2x4",
            nlist, M, nprobe, nq, rescale, k)


# "IVF1024,LSQ30x4fs_Nlsq2x4" — non-residual (no by_residual) AQ
# fastscan variants.
@params(nlist=[1024], M=[32], nprobe=NPROBE_SWEEP, nq=[1, 10, 100],
        rescale=[0, 1], k=[1, 10])
def bench_ivf_lsq_fastscan_nonresidual_search(
        benchmark, nlist, M, nprobe, nq, rescale, k):
    require_attr(faiss, "IndexIVFLocalSearchQuantizerFastScan")
    _aq_fastscan_search(
            benchmark, "IVF{nlist},LSQ{M}x4fs_Nlsq2x4",
            nlist, M, nprobe, nq, rescale, k)


@params(nlist=[1024], M=[32], nprobe=NPROBE_SWEEP, nq=[1, 10, 100],
        rescale=[0, 1], k=[1, 10])
def bench_ivf_rq_fastscan_nonresidual_search(
        benchmark, nlist, M, nprobe, nq, rescale, k):
    require_attr(faiss, "IndexIVFResidualQuantizerFastScan")
    _aq_fastscan_search(
            benchmark, "IVF{nlist},RQ{M}x4fs_Nrq2x4",
            nlist, M, nprobe, nq, rescale, k)
