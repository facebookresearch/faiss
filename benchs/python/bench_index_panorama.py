# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python benchmark for Panorama (level-pruned) flat / IVF-flat / HNSW-flat /
# refine search.
#
# A PCA{d} pre-transform concentrates energy in the early dimensions (which
# improves level pruning). That path is reproduced here via the `pca` sweep
# axis (pca=1 prepends "PCA{d}," to the factory; pca=0 keeps the raw no-PCA
# path for comparison). Data is synthetic (rand_mat) rather than SIFT1M/GIST1M
# so the suite stays dependency-free; point --nb/--d at larger values to
# approximate realistic working points.
#
# Design notes:
# - Plain (non-Panorama) baselines (Flat, IVFFlat, HNSWFlat, IndexRefineFlat)
#   live in bench_index_flat / bench_index_ivf / bench_index_graph; they are
#   not duplicated here.
# - Uses d=128 synthetic data plus optional SIFT1M variants; sweeps cap nq
#   at 100 (SIFT1M variants included).

import functools

import faiss
import pytest

from bench_utils import (
        compute_recall, dataset_or_skip, ground_truth, load_dataset, params,
        rand_mat)

D = 128  # SIFT-like dimensionality
NB_IVF = 100000  # database size for the IVF and refine benchmarks
K = 10  # recall@10

# Per-script defaults for a 128d dataset: nlevels=8; batch sizes are
# per-script: 512 for the flat index, 1024 for the IVF variant, 1 for the
# refine stage.
FLAT_LEVELS, FLAT_BS = 8, 512
IVF_LEVELS, IVF_BS = 8, 1024
REFINE_LEVELS, REFINE_BS = 8, 1
HNSW_LEVELS, HNSW_M = 8, 32


def _pca_prefix(pca, d):
    return f"PCA{d}," if pca else ""


def _try_factory(d, factory):
    try:
        return faiss.index_factory(d, factory)
    except RuntimeError as e:
        pytest.skip(f"this faiss build lacks Panorama ({factory!r}: {e})")


@functools.lru_cache(maxsize=8)
def _built_panorama_index(factory, d, nb):
    """Trained+populated Panorama index; trains on min(nb, 50000)."""
    index = _try_factory(d, factory)
    index.train(rand_mat(min(nb, 50000), d))
    index.add(rand_mat(nb, d, seed=54321))
    return index


@functools.lru_cache(maxsize=4)
def _built_refine_panorama(base_factory, refine_factory, d, nb):
    """(base, refine, wrapper) triple; the tuple keeps base and refine
    alive for the lifetime of the (non-owning) wrapper."""
    refine = _built_panorama_index(refine_factory, d, nb)
    base = faiss.index_factory(d, base_factory)
    nt = min(nb, 50000)
    base.train(rand_mat(nt, d))
    base.add(rand_mat(nb, d, seed=54321))
    try:
        wrapper = faiss.IndexRefinePanorama(base, refine)
    except AttributeError:
        pytest.skip("this faiss build lacks IndexRefinePanorama")
    return base, refine, wrapper


def _reset_panorama_stats():
    try:
        faiss.cvar.indexPanorama_stats.reset()
    except AttributeError:
        pass


def _record_panorama_dims(benchmark):
    try:
        benchmark.extra_info["dims_scanned_ratio"] = (
                faiss.cvar.indexPanorama_stats.ratio_dims_scanned)
    except AttributeError:
        pass


@params(d=[128], nb=[100000], nq=[1, 10, 100], pca=[0, 1])
def bench_flat_panorama_search(benchmark, d, nb, nq, pca):
    index = _built_panorama_index(
            f"{_pca_prefix(pca, d)}FlatL2Panorama{FLAT_LEVELS}_{FLAT_BS}",
            d, nb)
    xq = rand_mat(nq, d, seed=67890)
    _reset_panorama_stats()
    benchmark(index.search, xq, K)

    # Report recall + fraction of dimensions scanned
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)
    _record_panorama_dims(benchmark)


@params(nlist=[128, 256], nprobe=[1, 2, 4, 8, 16, 32, 64],
        nq=[1, 10, 100], pca=[0, 1])
def bench_ivf_flat_panorama_search(benchmark, nlist, nprobe, nq, pca):
    index = _built_panorama_index(
            f"{_pca_prefix(pca, D)}IVF{nlist},FlatPanorama{IVF_LEVELS}_{IVF_BS}",
            D, NB_IVF)
    # nprobe lives on the underlying IVF index (unwrap the PCA transform).
    faiss.extract_index_ivf(index).nprobe = nprobe
    xq = rand_mat(nq, D, seed=67890)
    _reset_panorama_stats()
    benchmark(index.search, xq, K)

    # Report recall + fraction of dimensions scanned
    _, I = index.search(xq, K)
    gt_I = ground_truth(D, NB_IVF, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)
    _record_panorama_dims(benchmark)


@params(M=[32], efSearch=[16, 32, 64, 128, 256, 512],
        nq=[1, 10, 100], pca=[0, 1])
def bench_hnsw_flat_panorama_search(benchmark, M, efSearch, nq, pca):
    # HNSW with Panorama storage: a "HNSW{M},FlatPanorama{nlevels}" factory
    # index (pca=1 prepends a "PCA{d}," transform).
    index = _built_panorama_index(
            f"{_pca_prefix(pca, D)}HNSW{M},FlatPanorama{HNSW_LEVELS}",
            D, NB_IVF)
    hnsw_index = faiss.downcast_index(
            index.index if isinstance(index, faiss.IndexPreTransform)
            else index)
    hnsw_index.hnsw.efSearch = efSearch
    xq = rand_mat(nq, D, seed=67890)
    _reset_panorama_stats()
    benchmark(index.search, xq, K)

    # Report recall + fraction of dimensions scanned
    _, I = index.search(xq, K)
    gt_I = ground_truth(D, NB_IVF, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)
    _record_panorama_dims(benchmark)


@params(k_factor=[1, 8, 64, 256, 1024], nprobe=[4, 16, 64, 256],
        nq=[1, 10, 100], pca=[0, 1])
def bench_refine_panorama_search(benchmark, k_factor, nprobe, nq, pca):
    # Base is IVF256,PQ16x4fs (PQ60x4fs targets 960d GIST1M;
    # 16 subquantizers fit d=128). pca=1 PCA-prefixes the refine stage,
    # giving a "PCA{d},FlatL2Panorama{n_levels}_1" refine factory.
    base, _, index = _built_refine_panorama(
            "IVF256,PQ16x4fs",
            f"{_pca_prefix(pca, D)}FlatL2Panorama{REFINE_LEVELS}_{REFINE_BS}",
            D, NB_IVF)
    base.nprobe = nprobe
    index.k_factor = k_factor
    xq = rand_mat(nq, D, seed=67890)
    _reset_panorama_stats()
    benchmark(index.search, xq, K)

    # Report recall + fraction of dimensions scanned
    _, I = index.search(xq, K)
    gt_I = ground_truth(D, NB_IVF, nq, K, xb_seed=54321, xq_seed=67890)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)
    _record_panorama_dims(benchmark)


# SIFT1M dataset variants: flat and IVF Panorama over the full 1M base set
# (trained on the learn set), with the same PCA-vs-raw toggle; recall against
# the dataset ground truth is valid since the full base is indexed.


@functools.lru_cache(maxsize=8)
def _built_panorama_dataset_index(factory, data_dir):
    """Trained+populated Panorama index over the on-disk dataset."""
    ds = load_dataset(data_dir)
    index = _try_factory(ds["xb"].shape[1], factory)
    index.train(ds["xt"])
    index.add(ds["xb"])
    return index


@params(nq=[1, 10], pca=[0, 1])
def bench_flat_panorama_search_sift1m(benchmark, data_dir, nq, pca):
    ds = dataset_or_skip(data_dir)
    d = ds["xb"].shape[1]
    index = _built_panorama_dataset_index(
            f"{_pca_prefix(pca, d)}FlatL2Panorama{FLAT_LEVELS}_{FLAT_BS}",
            data_dir)
    _reset_panorama_stats()
    _, I = benchmark(index.search, ds["xq"][:nq], K)

    # Report recall + fraction of dimensions scanned
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])
    _record_panorama_dims(benchmark)


# nlist=128 for the 1M dataset variants.
@params(nlist=[128], nprobe=[4, 16, 64], nq=[100], pca=[0, 1])
def bench_ivf_flat_panorama_search_sift1m(
        benchmark, data_dir, nlist, nprobe, nq, pca):
    ds = dataset_or_skip(data_dir)
    d = ds["xb"].shape[1]
    index = _built_panorama_dataset_index(
            f"{_pca_prefix(pca, d)}IVF{nlist},FlatPanorama"
            f"{IVF_LEVELS}_{IVF_BS}",
            data_dir)
    # nprobe lives on the underlying IVF index (unwrap the PCA transform).
    faiss.extract_index_ivf(index).nprobe = nprobe
    _reset_panorama_stats()
    _, I = benchmark(index.search, ds["xq"][:nq], K)

    # Report recall + fraction of dimensions scanned
    benchmark.extra_info["recall"] = compute_recall(I, ds["gt"][:nq])
    _record_panorama_dims(benchmark)
