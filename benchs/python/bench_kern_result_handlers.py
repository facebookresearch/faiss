# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_kern_result_handlers.cpp — top-k collection inside
# IndexIVF.search over a k / nprobe / index-type grid. The interesting axis is
# how search cost grows with k (result-handler work), nprobe (candidates
# flowing through it), and the index/quantizer type (--factory).
#
# Design notes:
#   * the AVX512 custom reservoir result-handler comparison is excluded: that
#     handler is not part of libfaiss.
#   * data is drawn from one shared uniform rand_mat (the C++ port uses
#     rand_smooth_vectors with separate train/base seeds 1234/4567) —
#     timing-comparable but not bit-identical workloads.

import functools

import faiss

from bench_utils import (
        built_dataset_index, compute_recall, dataset_or_skip, params,
        rand_mat)

D = 64
NB = 1_000_000
NLIST = 1024  # IVF1024,Flat

# Default is Flat only; also supports SQ4/RaBitQ/SQfp16.
# The {nlist} placeholder is substituted. Override with e.g.
# --factory='IVF{nlist},SQ4;IVF{nlist},SQfp16'.
FACTORIES = ["IVF{nlist},Flat"]


@functools.lru_cache(maxsize=8)
def _ivf_index(factory, d, nb, nlist):
    """Trained+populated index for the given factory, shared across sweep."""
    index = faiss.index_factory(
            d, factory.replace("{nlist}", str(nlist)), faiss.METRIC_L2)
    xb = rand_mat(nb, d)
    index.train(xb[: min(nb, 50000)])
    index.add(xb)
    return index


@params(
        factory=FACTORIES,
        nprobe=[1, 2, 4, 8, 16, 64],
        k=[1, 4, 10, 16, 20, 50, 100, 200, 500, 1000])
def bench_ivf_topk(benchmark, factory, nprobe, k):
    d, nb, nlist = D, NB, NLIST
    if nprobe > nlist:
        import pytest

        pytest.skip(f"nprobe {nprobe} > nlist {nlist}")
    index = _ivf_index(factory, d, nb, nlist)
    faiss.extract_index_ivf(index).nprobe = nprobe
    nq = 100
    xq = rand_mat(nq, d, seed=7890)
    benchmark(index.search, xq, k)


# SIFT1M dataset variant: the same top-k collection measurement over an IVF
# index trained on the learn set and populated with the full 1M base set
# (d=128, nlist=1024, nq=100 queries as in the synthetic sweep). Recall
# against the dataset ground truth is valid since the whole base set is
# searched. Registered only when --data_dir points at a SIFT1M-layout
# directory.
@params(factory=FACTORIES, nprobe=[1, 16], k=[10, 100, 1000])
def bench_ivf_topk_sift1m(benchmark, data_dir, factory, nprobe, k):
    ds = dataset_or_skip(data_dir)
    index = built_dataset_index(
            factory.replace("{nlist}", str(NLIST)), data_dir)
    faiss.extract_index_ivf(index).nprobe = nprobe
    nq = 100
    xq = ds["xq"][:nq]
    _, I = benchmark(index.search, xq, k)
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])
