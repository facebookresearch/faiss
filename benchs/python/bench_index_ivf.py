# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_ivf.cpp, cpp/bench_index_ivf_parallel_mode.cpp
# and cpp/bench_index_ivfpq_add.cpp — IVFFlat / IVFPQ / IVFScalarQuantizer search,
# IVF parallel_mode search, and IVFPQ add throughput.
#
# Design notes:
#   * the default bench_ivfsq_rangestat_search uses QT_8bit with one
#     representative arg per rangestat on synthetic d=128/nb=100k data.
#     The full grids remain available via the --sq_type, --rangestat and
#     --rangestat_arg pytest options.

import functools

import faiss
import pytest

from bench_utils import built_index, compute_recall, ground_truth, params, rand_mat
from bench_utils import built_dataset_index, dataset_or_skip

K = 10  # matches the fixed k=10 in the C++ benchmarks


def skip_if_nprobe_gt_nlist(nprobe, nlist):
    if nprobe > nlist:
        pytest.skip(f"nprobe {nprobe} > nlist {nlist}")


@params(nlist=[256, 1024], nprobe=[1, 8, 32], nq=[1, 10, 100])
def bench_ivfflat_search(benchmark, nlist, nprobe, nq):
    d, nb = 128, 100000
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    index = built_index(f"IVF{nlist},Flat", d, nb)
    index.nprobe = nprobe
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(
        nlist=[256, 1024],
        M=[8, 16, 32],
        nprobe=[1, 8, 32],
        nq=[1, 10, 100],
        precomp=[0, 1])
def bench_ivfpq_search(benchmark, nlist, M, nprobe, nq, precomp):
    d, nb = 128, 100000
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    index = built_index(f"IVF{nlist},PQ{M}", d, nb)
    index.nprobe = nprobe
    index.use_precomputed_table = precomp
    if precomp:
        # make sure the tables exist when forcing them on (search-time only,
        # idempotent — safe on the shared cached index)
        index.precompute_table()
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# index_factory names for the scalar quantizer types benchmarked in C++.
SQ_FACTORY = {
    "QT_8bit": "SQ8",
    "QT_6bit": "SQ6",
    "QT_4bit": "SQ4",
    "QT_fp16": "SQfp16",
    "QT_bf16": "SQbf16",
}


@params(
        nlist=[256, 1024],
        sq_type=["QT_8bit", "QT_fp16", "QT_4bit", "QT_bf16"],
        nprobe=[1, 8, 32],
        nq=[1, 10, 100])
def bench_ivfsq_search(benchmark, nlist, sq_type, nprobe, nq):
    d, nb = 128, 100000
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    if sq_type not in SQ_FACTORY:
        pytest.fail(f"unknown sq_type {sq_type!r}; "
                    f"choose from {sorted(SQ_FACTORY)}")
    if not hasattr(faiss.ScalarQuantizer, sq_type):
        pytest.skip(f"faiss build has no ScalarQuantizer.{sq_type}")
    index = built_index(f"IVF{nlist},{SQ_FACTORY[sq_type]}", d, nb)
    index.nprobe = nprobe
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# SIFT1M search (skipped when --data-dir is missing), mirroring the
# "sift1m/ivfflat" and "sift1m/ivfpq" groups of cpp/bench_index_ivf.cpp
# (nprobe sweeps at k=10; the C++ groups search the full query set, here
# nq defaults to 100 and is overridable with --nq).
@params(nlist=[1024], nprobe=[1, 16, 64], nq=[100])
def bench_ivfflat_search_sift1m(benchmark, data_dir, nlist, nprobe, nq):
    ds = dataset_or_skip(data_dir)
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    index = built_dataset_index(f"IVF{nlist},Flat", data_dir)
    index.nprobe = nprobe
    _, I = benchmark(index.search, ds["xq"][:nq], K)

    # Report recall against the dataset ground truth
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])


@params(nlist=[1024], M=[8, 16], nprobe=[1, 16], nq=[100])
def bench_ivfpq_search_sift1m(benchmark, data_dir, nlist, M, nprobe, nq):
    ds = dataset_or_skip(data_dir)
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    index = built_dataset_index(f"IVF{nlist},PQ{M}", data_dir)
    index.nprobe = nprobe
    _, I = benchmark(index.search, ds["xq"][:nq], K)

    # Report recall against the dataset ground truth
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])


@functools.lru_cache(maxsize=2)
def _ivfflat_subset_trained(d, nb, nlist):
    """IVFFlat trained on a subset, as in the C++ benchmarks.

    built_index would train on all nb vectors, which is pathologically slow
    through a single thread at nb=500000/nlist=4096; the C++ parallel_mode
    benchmark also trains on a subset. The training set must be at least
    39*nlist to avoid a k-means warning.
    """
    nt = max(min(nb, 50000), 39 * nlist)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.own_fields = True
    quantizer.this.disown()
    xb = rand_mat(nb, d)
    index.train(xb[:nt])
    index.add(xb)
    return index


# NOTE: parallel_mode only matters with --threads > 1 (the C++ benchmark
# defaults to 4 threads; this suite defaults to 1).

@params(nlist=[4096], nprobe=[64, 256], pmode=[0, 2])
def bench_ivf_parallel_mode_single_query(benchmark, nlist, nprobe, pmode):
    # single-query, high-nprobe scenario (where pmode=2 shines)
    d, nb = 128, 500000
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    index = _ivfflat_subset_trained(d, nb, nlist)
    index.nprobe = nprobe
    index.parallel_mode = pmode
    xq = rand_mat(1, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, 1, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(nlist=[4096], nq=[10, 100], nprobe=[16, 64], pmode=[0, 2])
def bench_ivf_parallel_mode(benchmark, nlist, nq, nprobe, pmode):
    # multi-query scenario (where pmode=0 should be better)
    d, nb = 128, 500000
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    index = _ivfflat_subset_trained(d, nb, nlist)
    index.nprobe = nprobe
    index.parallel_mode = pmode
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@functools.lru_cache(maxsize=4)
def _trained_empty_ivfpq(d, nlist, M, nt):
    """Trained but empty IVFPQ, cloned per round by bench_ivfpq_add."""
    index = faiss.index_factory(d, f"IVF{nlist},PQ{M}")
    index.train(rand_mat(nt, d))
    return index


@params(nlist=[1024], M=[8, 16, 32])
def bench_ivfpq_add(benchmark, nlist, M):
    d, nb = 64, 100000  # d=64 as in cpp/bench_index_ivfpq_add.cpp
    if M > d:
        pytest.skip(f"M {M} > d {d}")
    xb = rand_mat(nb, d)
    trained = _trained_empty_ivfpq(d, nlist, M, min(nb, 50000))

    def fresh_index():
        # fresh pre-trained index per round: measure pure add cost
        return (faiss.clone_index(trained),), {}

    benchmark.pedantic(
            lambda index: index.add(xb), setup=fresh_index, rounds=3)


# Rangestat calibration sweep for IVFScalarQuantizer. Sweeps every QT_*
# quantizer type, and for each rangestat a full grid of rangestat_arg values;
# ncent=256, nprobe=16, k=100. All of that is reproduced here and exposed via
# CLI overrides. Because the grid is large, the default keeps a
# representative rangestat_arg per rangestat (RANGESTAT_ARG_SWEEP holds the
# full grid; override --rangestat_arg to walk the whole calibration
# curve, or --sq_type to restrict the quantizer types).
RANGESTAT_K = 100  # searches k=100

# Full rangestat_arg grids, per rangestat.
RANGESTAT_ARG_SWEEP = {
    "RS_minmax": [-0.4, -0.2, -0.1, -0.05, 0.0, 0.1, 0.5],
    "RS_meanstd": [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    "RS_quantiles": [0.02, 0.05, 0.1, 0.15],
    "RS_optim": [0.0],
}
# Default representative arg per rangestat (used when --rangestat_arg is not
# given); a single point keeps the default run tractable.
RANGESTAT_ARG = {
    "RS_minmax": 0.0,
    "RS_meanstd": 2.0,
    "RS_quantiles": 0.1,
    "RS_optim": 0.0,
}

# All QT_* scalar quantizer types sweeps.
RANGESTAT_SQ_TYPES = [
    "QT_8bit", "QT_4bit", "QT_6bit", "QT_fp16", "QT_bf16",
    "QT_8bit_uniform", "QT_4bit_uniform", "QT_8bit_direct",
    "QT_8bit_direct_signed",
]


@functools.lru_cache(maxsize=32)
def _ivfsq_rangestat_trained(d, nb, nlist, sq_type, rangestat, rangestat_arg):
    if not hasattr(faiss.ScalarQuantizer, sq_type):
        pytest.skip(f"faiss build has no ScalarQuantizer.{sq_type}")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist, getattr(faiss.ScalarQuantizer, sq_type))
    index.own_fields = True
    quantizer.this.disown()
    # rangestat affects training, so it is part of the cache key and is set
    # before train
    index.sq.rangestat = getattr(faiss.ScalarQuantizer, rangestat)
    index.sq.rangestat_arg = rangestat_arg
    xb = rand_mat(nb, d)
    index.train(xb[:min(nb, 50000)])
    index.add(xb)
    return index


@params(
        sq_type=["QT_8bit"],
        rangestat=["RS_minmax", "RS_meanstd", "RS_quantiles", "RS_optim"],
        rangestat_arg=[None],
        nlist=[256],
        nprobe=[16],
        nq=[1, 10, 100])
def bench_ivfsq_rangestat_search(
        benchmark, sq_type, rangestat, rangestat_arg, nlist, nprobe, nq):
    d, nb = 128, 100000
    skip_if_nprobe_gt_nlist(nprobe, nlist)
    if rangestat not in RANGESTAT_ARG:
        pytest.fail(f"unknown rangestat {rangestat!r}; "
                    f"choose from {sorted(RANGESTAT_ARG)}")
    if not hasattr(faiss.ScalarQuantizer, rangestat):
        pytest.skip(f"faiss build has no ScalarQuantizer.{rangestat}")
    # rangestat_arg=None (the default) selects the representative arg; an
    # explicit --rangestat_arg walks the full calibration curve.
    arg = RANGESTAT_ARG[rangestat] if rangestat_arg is None \
        else float(rangestat_arg)
    index = _ivfsq_rangestat_trained(d, nb, nlist, sq_type, rangestat, arg)
    index.nprobe = nprobe
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, RANGESTAT_K)

    # Report recall
    _, I = index.search(xq, RANGESTAT_K)
    gt_I = ground_truth(d, nb, nq, RANGESTAT_K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)
