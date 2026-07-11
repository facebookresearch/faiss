# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_binary.cpp — IndexBinaryFlat,
# IndexBinaryIVF and IndexBinaryHNSW search. `d` is the binary vector
# dimension in bits (code size d//8 bytes).

import functools

import faiss
import pytest

from bench_utils import params, rand_codes, require_attr

K = 10  # fixed in the C++ file


@functools.lru_cache(maxsize=8)
def _built_binary_index(factory, d, nb):
    """Cached trained+populated binary index (bench_utils.built_index is
    float-only). Search-only — callers must not mutate the returned index
    beyond search-time parameters (nprobe, efSearch)."""
    index = faiss.index_binary_factory(d, factory)
    index.verbose = False
    xb = rand_codes(nb, d // 8)
    if not index.is_trained:
        index.train(xb)
    index.add(xb)
    return index


@params(d=[128, 256], nb=[100000], nq=[1, 10, 100])
def bench_binary_flat_search(benchmark, d, nb, nq):
    index = _built_binary_index("BFlat", d, nb)
    xq = rand_codes(nq, d // 8, seed=54321)
    benchmark(index.search, xq, K)


@params(
        d=[128, 256],
        nb=[100000],
        nlist=[256, 1024],
        nprobe=[1, 8, 32],
        nq=[1, 10, 100])
def bench_binary_ivf_search(benchmark, d, nb, nlist, nprobe, nq):
    if nprobe > nlist:
        pytest.skip(f"nprobe={nprobe} > nlist={nlist}")
    index = _built_binary_index(f"BIVF{nlist}", d, nb)
    index.nprobe = nprobe
    xq = rand_codes(nq, d // 8, seed=54321)
    benchmark(index.search, xq, K)


@params(
        d=[128, 256],
        nb=[100000],
        M=[16, 32],
        efSearch=[16, 64],
        nq=[1, 10, 100])
def bench_binary_hnsw_search(benchmark, d, nb, M, efSearch, nq):
    require_attr(faiss, "IndexBinaryHNSW")
    index = _built_binary_index(f"BHNSW{M}", d, nb)
    index.hnsw.efSearch = efSearch
    xq = rand_codes(nq, d // 8, seed=54321)
    benchmark(index.search, xq, K)


# Standalone faiss.knn_hamming function (no index), hashtable ("hc") and
# multi-index ("mc") variants. Bit widths listed below are all byte-aligned.
# nq=10000, nb=30000, k in {1,4,16,64,256}.
HAMMING_DIMS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
HAMMING_K = [1, 4, 16, 64, 256]


@params(d=HAMMING_DIMS, nq=[10000], nb=[30000], k=HAMMING_K)
def bench_knn_hamming_hc(benchmark, d, nq, nb, k):
    require_attr(faiss, "knn_hamming")
    xq = rand_codes(nq, d // 8, seed=54321)
    xb = rand_codes(nb, d // 8)
    benchmark(faiss.knn_hamming, xq, xb, k, variant="hc")


@params(d=HAMMING_DIMS, nq=[10000], nb=[30000], k=HAMMING_K)
def bench_knn_hamming_mc(benchmark, d, nq, nb, k):
    require_attr(faiss, "knn_hamming")
    xq = rand_codes(nq, d // 8, seed=54321)
    xb = rand_codes(nb, d // 8)
    benchmark(faiss.knn_hamming, xq, xb, k, variant="mc")
