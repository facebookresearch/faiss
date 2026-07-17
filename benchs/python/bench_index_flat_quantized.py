# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_flat_quantized.cpp and cpp/bench_index_pq_polysemous.cpp —
# IndexPQ search, IndexPQFastScan search and polysemous PQ search. (The
# IndexScalarQuantizer part of bench_index_flat_quantized.cpp is covered by the quantizer
# benchmarks; the raw kernels have no public Python entry point.)
#
# Covers:
#   - IndexPQ(d, M=8, nbits) with the nbits sweep {4,6,8,10,12}, k=32,
#     reporting recall@k (the sift1m benches below also report recall_1)
#     -> bench_pq_nbits_search.
#   - IndexPQ(d, M=16, nbits=8) with do_polysemous_training=True: polysemous
#     training time (bench_polysemous_train), the ST_PQ baseline search
#     (bench_polysemous_st_pq_baseline) and the polysemous_ht sweep
#     {64,62,58,54,50,46,42,38,34,30} with k=1
#     (bench_polysemous_ht_search).
#
# SIFT1M polysemous benchmarks exist only in the C++ suite
# (cpp/bench_index_pq_polysemous.cpp); the polysemous benches here run on
# synthetic data.

import functools

import faiss
import pytest

from bench_utils import built_index, compute_recall, ground_truth, params, rand_mat, require_attr
from bench_utils import built_dataset_index, dataset_or_skip

K = 10  # matches the fixed k=10 in cpp/bench_index_flat_quantized.cpp

@params(M=[8, 16, 32], nq=[1, 10, 100])
def bench_pq_search(benchmark, M, nq):
    d, nb = 128, 100000
    index = built_index(f"PQ{M}", d, nb)
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@functools.lru_cache(maxsize=8)
def _pq_nbits_index(d, nb, M, nbits):
    """IndexPQ with enough training data for high-nbits sweeps.

    PQ trains 2^nbits centroids per subquantizer; k-means needs >= 39 *
    2^nbits training points. At nbits=12 that is 159744, exceeding nb.
    """
    min_train = 39 * (1 << nbits)
    nt = max(nb, min_train)
    index = faiss.IndexPQ(d, M, nbits)
    index.train(rand_mat(nt, d))
    # Default seed so the database matches ground_truth's xb_seed default.
    index.add(rand_mat(nb, d))
    return index


@params(nbits=[4, 6, 8, 10, 12], nq=[1, 10, 100])
def bench_pq_nbits_search(benchmark, nbits, nq):
    # IndexPQ(d, M=8, nbits), k=32, reporting recall@k (the sift1m variant
    # below also reports recall_1).
    d, nb, M, k = 128, 100000, 8, 32
    index = _pq_nbits_index(d, nb, M, nbits)
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, k)

    # Report recall
    _, I = index.search(xq, k)
    gt_I = ground_truth(d, nb, nq, k)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(M=[16, 32, 64], nq=[1, 10, 100], implem=[0, 12, 14])
def bench_pqfastscan_search(benchmark, M, nq, implem):
    d, nb = 128, 100000
    require_attr(faiss, "IndexPQFastScan")
    if M > d:
        pytest.skip(f"M {M} > d {d}")
    index = built_index(f"PQ{M}x4fs", d, nb)
    if not hasattr(index, "implem"):
        pytest.skip("implem is not settable from Python in this faiss build")
    index.implem = implem
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# SIFT1M search (skipped when --data-dir is missing), mirroring the
# "sift1m/pq" and "sift1m/pq_nbits" groups of cpp/bench_index_flat_quantized.cpp:
# an M sweep with nbits=8 and an nbits sweep with M=8, both at k=32.
# nq defaults to the full query set (matching the C++ groups); use --nq to
# trim it.
@params(M=[8, 16], nq=[10000])
def bench_pq_search_sift1m(benchmark, data_dir, M, nq):
    k = 32
    ds = dataset_or_skip(data_dir)
    index = built_dataset_index(f"PQ{M}", data_dir)
    _, I = benchmark(index.search, ds["xq"][:nq], k)

    # Report recall against the dataset ground truth, plus R@1 (fraction of
    # queries whose true 1-NN is returned at rank 0).
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])
    benchmark.extra_info["recall_1"] = float(
            (I[:, 0] == ds["gt"][:nq, 0]).mean())


@params(nbits=[4, 6, 8, 10, 12], nq=[10000])
def bench_pq_nbits_search_sift1m(benchmark, data_dir, nbits, nq):
    M, k = 8, 32
    ds = dataset_or_skip(data_dir)
    index = built_dataset_index(f"PQ{M}x{nbits}", data_dir)
    _, I = benchmark(index.search, ds["xq"][:nq], k)

    # Report recall against the dataset ground truth, plus R@1 (fraction of
    # queries whose true 1-NN is returned at rank 0).
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])
    benchmark.extra_info["recall_1"] = float(
            (I[:, 0] == ds["gt"][:nq, 0]).mean())


@functools.lru_cache(maxsize=2)
def _polysemous_index(d, nb, M):
    """IndexPQ with polysemous training + ST_polysemous search.

    Module-local builder: do_polysemous_training must be set before train,
    which an index_factory string (and hence built_index) cannot express.
    Trains on a 50k subset like cpp/bench_index_pq_polysemous.cpp.
    """
    index = faiss.IndexPQ(d, M, 8)
    index.do_polysemous_training = True
    xb = rand_mat(nb, d)
    index.train(xb[:min(nb, 50000)])
    index.add(xb)
    index.search_type = faiss.IndexPQ.ST_polysemous
    return index


@params(M=[8, 16], nq=[1, 10, 100])
def bench_polysemous_search(benchmark, M, nq):
    d, nb = 128, 100000
    index = _polysemous_index(d, nb, M)
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, 32)  # k=32 as in cpp/bench_index_pq_polysemous.cpp

    # Report recall
    _, I = index.search(xq, 32)
    gt_I = ground_truth(d, nb, nq, 32)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(M=[16])
def bench_polysemous_train(benchmark, M):
    """Polysemous PQ training time.

    IndexPQ(d, M=16, nbits=8) with do_polysemous_training=True;
    times index.train() over NUM_TRAIN_RUNS=5. Uses benchmark.pedantic
    with a fresh untrained index rebuilt in setup so each measured round
    times only train() on a clean index (analogous to the C++
    bench_pq_polysemous_train).
    """
    d, nb = 128, 100000
    xb = rand_mat(nb, d)
    xt = xb[:min(nb, 50000)]

    def setup():
        index = faiss.IndexPQ(d, M, 8)
        index.do_polysemous_training = True
        index.verbose = False
        return (index,), {}

    benchmark.pedantic(
            lambda index: index.train(xt),
            setup=setup,
            rounds=5,
            iterations=1,
            warmup_rounds=0,
    )


@params(M=[16], nq=[1, 10, 100])
def bench_polysemous_st_pq_baseline(benchmark, M, nq):
    """ST_PQ baseline search on a polysemous-trained index (k=1)."""
    d, nb, k = 128, 100000, 1
    index = _polysemous_index(d, nb, M)
    old_search_type = index.search_type
    index.search_type = faiss.IndexPQ.ST_PQ
    xq = rand_mat(nq, d, seed=54321)
    try:
        benchmark(index.search, xq, k)

        # Report recall
        _, I = index.search(xq, k)
        gt_I = ground_truth(d, nb, nq, k)
        benchmark.extra_info["recall"] = compute_recall(I, gt_I)
    finally:
        index.search_type = old_search_type  # the trained index is a shared cache


@params(M=[16], polysemous_ht=[64, 62, 58, 54, 50, 46, 42, 38, 34, 30],
        nq=[1, 10, 100])
def bench_polysemous_ht_search(benchmark, M, polysemous_ht, nq):
    """Polysemous search sweeping polysemous_ht (k=1)."""
    d, nb, k = 128, 100000, 1
    index = _polysemous_index(d, nb, M)
    old_ht = index.polysemous_ht
    index.search_type = faiss.IndexPQ.ST_polysemous
    index.polysemous_ht = polysemous_ht
    xq = rand_mat(nq, d, seed=54321)
    try:
        benchmark(index.search, xq, k)

        # Report recall
        _, I = index.search(xq, k)
        gt_I = ground_truth(d, nb, nq, k)
        benchmark.extra_info["recall"] = compute_recall(I, gt_I)
    finally:
        index.polysemous_ht = old_ht  # the trained index is a shared cache
