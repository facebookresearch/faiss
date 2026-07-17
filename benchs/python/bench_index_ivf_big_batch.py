# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python benchmark for faiss.contrib.big_batch_search over a trained IVF
# index, comparing the per-invlist computation methods it supports.
#
# Defaults: d=32, nlist=100, nprobe=5, k=10, and the S/M/L SyntheticDataset
# working points (nb/nq of 4000/1000, 40000/10000, 400000/100000, nt=nb/2).
# The default method is "pairwise_distances"; the extra "knn_function" method
# is also kept. The `factory` @params axis (default "IVF{nlist},Flat") allows
# an arbitrary index override; a factory string may contain a literal
# "{nlist}" placeholder (substituted) or be a fixed key like "IVF100,PQ8"
# (used as-is, with the resulting index.nlist reported).

import functools

import faiss
import pytest

from bench_utils import (
        built_dataset_index, compute_recall, dataset_or_skip, ground_truth,
        params, rand_mat)

D = 32  # SyntheticDataset dimensionality
K = 10


@functools.lru_cache(maxsize=4)
def _ivf_index(factory, d, nb, nlist):
    """Trained+populated IVF index; trains on max(nb//2, 39*nlist) vectors."""
    nt = max(nb // 2, 39 * nlist)
    factory_string = factory.format(nlist=nlist)
    index = faiss.index_factory(d, factory_string)
    index.train(rand_mat(nt, d))
    index.add(rand_mat(nb, d, seed=54321))
    return index


# nlist=100, nprobe=5 and k=10 are the defaults; "index" (its third method)
# is omitted as it rebuilds a flat index per invlist and dominates runtime
# without exercising a distinct kernel. nq sweeps the S/M/L query counts;
# nb = 4*nq and nt = nb/2 reproduce the S/M/L SyntheticDataset working
# points (nb/nq of 4000/1000, 40000/10000, 400000/100000).
@params(
        nlist=[100],
        nprobe=[5],
        nq=[1000, 10000, 100000],
        method=["pairwise_distances", "knn_function"],
        factory=["IVF{nlist},Flat"])
def bench_big_batch_ivf_search(benchmark, nlist, nprobe, nq, method, factory):
    bbs = pytest.importorskip(
            "faiss.contrib.big_batch_search",
            reason="faiss.contrib.big_batch_search not available")
    nb = 4 * nq
    try:
        index = _ivf_index(factory, D, nb, nlist)
    except RuntimeError as e:
        pytest.skip(f"factory not supported by this faiss build: {e}")
    index.nprobe = nprobe
    xq = rand_mat(nq, D, seed=67890)
    # A single big-batch call sweeps every inverted list; rounds=1 keeps
    # the benchmark affordable.
    benchmark.pedantic(
            bbs.big_batch_search,
            args=(index, xq, K),
            kwargs={"method": method},
            rounds=1)
    benchmark.extra_info["nlist"] = index.nlist

    # Report recall
    _, I = bbs.big_batch_search(index, xq, K, method=method)
    gt_I = ground_truth(D, nb, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# SIFT1M dataset variant: big-batch search over an IVF index trained on the
# learn set and populated with the full 1M base set (d=128, nlist=1024). The
# full 10k query set forms the big batch; recall against the dataset ground
# truth is valid since the whole base set is searched. Registered only when
# --data_dir points at a SIFT1M-layout directory.
@params(
        nlist=[1024],
        nprobe=[1, 16],
        method=["pairwise_distances", "knn_function"],
        factory=["IVF{nlist},Flat"])
def bench_big_batch_ivf_search_sift1m(
        benchmark, data_dir, nlist, nprobe, method, factory):
    bbs = pytest.importorskip(
            "faiss.contrib.big_batch_search",
            reason="faiss.contrib.big_batch_search not available")
    ds = dataset_or_skip(data_dir)
    try:
        index = built_dataset_index(factory.format(nlist=nlist), data_dir)
    except RuntimeError as e:
        pytest.skip(f"factory not supported by this faiss build: {e}")
    index.nprobe = nprobe
    xq = ds["xq"]
    # A single big-batch call sweeps every inverted list; rounds=1 keeps
    # the benchmark affordable. The timed call's own results feed the
    # recall report (re-running the search would double the runtime).
    _, I = benchmark.pedantic(
            bbs.big_batch_search,
            args=(index, xq, K),
            kwargs={"method": method},
            rounds=1)
    benchmark.extra_info["nlist"] = index.nlist
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"])
