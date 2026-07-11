# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python benchmark for IndexRaBitQ and IndexIVFRaBitQ search over d/nq/nprobe
# sweeps on synthetic data.
#
# Design notes: non-RaBitQ baselines (SQ, PQ, HNSW) and memory-footprint
# reporting are covered by their respective benchmark files.

import functools

import faiss
import pytest

from bench_utils import (
        compute_recall, dataset_or_skip, ground_truth, load_dataset, params,
        rand_mat, require_attr)

NB = 200000  # database size
K = 100  # measures everything at k=100
QB_FLAT = 8  # qb=8 on the flat RaBitQ index before training


@functools.lru_cache(maxsize=4)
def _rabitq_index(factory, d, nb, qb=None):
    """Trained+populated RaBitQ index.

    Trains on min(nb, 100000) vectors. `qb` (query quantization bits) is set
    on the index before training when given for the non-IVF variant.
    """
    index = faiss.index_factory(d, factory)
    if qb is not None:
        index.qb = qb
    index.train(rand_mat(min(nb, 100000), d))
    index.add(rand_mat(nb, d, seed=54321))
    return index


# Dimensions chosen to hit distinct SIMD-width tiers of the RaBitQ kernels.
@params(d=[256, 512, 768, 1024], nq=[1, 10, 100, 1000])
def bench_rabitq_flat_search(benchmark, d, nq):
    require_attr(faiss, "IndexRaBitQ")
    index = _rabitq_index("RaBitQ", d, NB, qb=QB_FLAT)
    xq = rand_mat(nq, d, seed=67890)
    benchmark(index.search, xq, K)

    # Report recall
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, NB, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@params(d=[256, 512, 768, 1024], nprobe=[4, 16, 32], nq=[1, 10, 100, 1000],
        qb=[8])
def bench_ivf_rabitq_search(benchmark, d, nprobe, nq, qb):
    require_attr(faiss, "IndexIVFRaBitQ")
    require_attr(faiss, "IVFRaBitQSearchParameters")
    # nlist=1000 fixed.
    index = _rabitq_index("IVF1000,RaBitQ", d, NB)
    index.nprobe = nprobe
    # When search parameters are passed, params.nprobe overrides the
    # index-level nprobe (which defaults to 1 on the params object), so it
    # must be set on the params as well.
    search_params = faiss.IVFRaBitQSearchParameters(qb=qb, nprobe=nprobe)
    xq = rand_mat(nq, d, seed=67890)
    benchmark(index.search, xq, K, params=search_params)

    # Report recall
    _, I = index.search(xq, K, params=search_params)
    gt_I = ground_truth(d, NB, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


@functools.lru_cache(maxsize=4)
def _rabitq_rrot_index(d, nb):
    """IVF1000,RaBitQ behind an IndexPreTransform random rotation.

    A RandomRotationMatrix(d, d) seeded with 123 in front of the IVF RaBitQ
    index. Trains on min(nb, 100000) vectors.
    """
    index = faiss.index_factory(d, "IVF1000,RaBitQ")
    rrot = faiss.RandomRotationMatrix(d, d)
    rrot.init(123)
    index_pt = faiss.IndexPreTransform(rrot, index)
    index_pt.train(rand_mat(min(nb, 100000), d))
    index_pt.add(rand_mat(nb, d, seed=54321))
    return index_pt


@params(d=[256, 512, 768, 1024], nprobe=[4, 16, 32], nq=[1, 10, 100, 1000])
def bench_ivf_rabitq_search_rrot(benchmark, d, nprobe, nq):
    # IVF RaBitQ behind a random rotation ("RROT" variant); qb is fixed at
    # the default 8 to keep the sweep modest.
    require_attr(faiss, "IndexIVFRaBitQ")
    require_attr(faiss, "IVFRaBitQSearchParameters")
    index = _rabitq_rrot_index(d, NB)
    # IndexPreTransform forwards the search parameters to the IVF sub-index.
    search_params = faiss.IVFRaBitQSearchParameters(qb=8, nprobe=nprobe)
    xq = rand_mat(nq, d, seed=67890)
    benchmark(index.search, xq, K, params=search_params)

    # Report recall (the rotation is orthonormal, so the raw-vector ground
    # truth still applies)
    _, I = index.search(xq, K, params=search_params)
    gt_I = ground_truth(d, NB, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)


# SIFT1M dataset variants: same flat/IVF RaBitQ configurations, trained on
# the dataset's learn set and populated with the full 1M base set (so recall
# against the dataset ground truth is valid).


@functools.lru_cache(maxsize=4)
def _rabitq_dataset_index(factory, data_dir, qb=None):
    """Trained+populated RaBitQ index over the on-disk dataset.

    Trains on the learn set and adds the full base set. `qb` (query
    quantization bits) is set on the index before training when given for
    the non-IVF variant.
    """
    ds = load_dataset(data_dir)
    index = faiss.index_factory(ds["xb"].shape[1], factory)
    if qb is not None:
        index.qb = qb
    index.train(ds["xt"])
    index.add(ds["xb"])
    return index


@params(nq=[1, 10, 100])
def bench_rabitq_flat_search_sift1m(benchmark, data_dir, nq):
    require_attr(faiss, "IndexRaBitQ")
    ds = dataset_or_skip(data_dir)
    index = _rabitq_dataset_index("RaBitQ", data_dir, qb=QB_FLAT)
    _, I = benchmark(index.search, ds["xq"][:nq], K)
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])


@params(nprobe=[4, 16, 32], nq=[100], qb=[8])
def bench_ivf_rabitq_search_sift1m(benchmark, data_dir, nprobe, nq, qb):
    require_attr(faiss, "IndexIVFRaBitQ")
    require_attr(faiss, "IVFRaBitQSearchParameters")
    ds = dataset_or_skip(data_dir)
    # nlist=1000 fixed, as in the synthetic sweep.
    index = _rabitq_dataset_index("IVF1000,RaBitQ", data_dir)
    index.nprobe = nprobe
    search_params = faiss.IVFRaBitQSearchParameters(qb=qb, nprobe=nprobe)
    _, I = benchmark(index.search, ds["xq"][:nq], K, params=search_params)
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])
