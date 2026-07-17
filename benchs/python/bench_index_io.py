# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_io.cpp — write_index/read_index
# timing for a populated IVFPQ index (nlist=256, PQ16x8) and an HNSW32
# index. Reads also cover the mmap path when the build supports it.

import functools

import faiss

from bench_utils import (
        built_dataset_index, dataset_or_skip, params, rand_mat, require_attr)

# nlist=256, M=16, 8-bit PQ and HNSW M=32, as in the C++ benchmark.
IVFPQ_FACTORY = "IVF256,PQ16"
HNSW_FACTORY = "HNSW32"


@functools.lru_cache(maxsize=4)
def _populated_index(factory, d, nb):
    """Trained+populated index for serialization benchmarks.

    Like the C++ benchmark, trains on min(nb, 50000) vectors (which is why
    bench_utils.built_index, which trains on all nb, is not used here).
    """
    index = faiss.index_factory(d, factory)
    if not index.is_trained:
        index.train(rand_mat(min(nb, 50000), d))
    index.add(rand_mat(nb, d, seed=54321))
    return index


def _bench_write(benchmark, tmp_path, factory, name, d, nb):
    index = _populated_index(factory, d, nb)
    fname = str(tmp_path / f"bench_io_{name}.faissindex")
    benchmark.pedantic(faiss.write_index, args=(index, fname), rounds=3)


def _bench_read(benchmark, tmp_path, factory, name, d, nb, io_flags=0):
    index = _populated_index(factory, d, nb)
    fname = str(tmp_path / f"bench_io_{name}.faissindex")
    # Written once in setup; only the repeated reads are timed.
    faiss.write_index(index, fname)
    benchmark.pedantic(faiss.read_index, args=(fname, io_flags), rounds=3)


@params(d=[128], nb=[100000])
def bench_write_index_ivfpq(benchmark, tmp_path, d, nb):
    _bench_write(benchmark, tmp_path, IVFPQ_FACTORY, "ivfpq", d, nb)


@params(d=[128], nb=[100000])
def bench_write_index_hnsw(benchmark, tmp_path, d, nb):
    _bench_write(benchmark, tmp_path, HNSW_FACTORY, "hnsw", d, nb)


@params(d=[128], nb=[100000])
def bench_read_index_ivfpq(benchmark, tmp_path, d, nb):
    _bench_read(benchmark, tmp_path, IVFPQ_FACTORY, "ivfpq", d, nb)


@params(d=[128], nb=[100000])
def bench_read_index_hnsw(benchmark, tmp_path, d, nb):
    _bench_read(benchmark, tmp_path, HNSW_FACTORY, "hnsw", d, nb)


@params(d=[128], nb=[100000])
def bench_read_index_ivfpq_mmap(benchmark, tmp_path, d, nb):
    require_attr(faiss, "IO_FLAG_MMAP")
    _bench_read(
            benchmark, tmp_path, IVFPQ_FACTORY, "ivfpq", d, nb,
            io_flags=faiss.IO_FLAG_MMAP)


# SIFT1M dataset variants: write/read of an IVFPQ index trained on the
# dataset's learn set and populated with the full 1M base set.


def bench_write_index_ivfpq_sift1m(benchmark, tmp_path, data_dir):
    dataset_or_skip(data_dir)
    index = built_dataset_index(IVFPQ_FACTORY, data_dir)
    fname = str(tmp_path / "bench_io_ivfpq_sift1m.faissindex")
    benchmark.pedantic(faiss.write_index, args=(index, fname), rounds=3)


def bench_read_index_ivfpq_sift1m(benchmark, tmp_path, data_dir):
    dataset_or_skip(data_dir)
    index = built_dataset_index(IVFPQ_FACTORY, data_dir)
    fname = str(tmp_path / "bench_io_ivfpq_sift1m.faissindex")
    # Written once in setup; only the repeated reads are timed.
    faiss.write_index(index, fname)
    benchmark.pedantic(faiss.read_index, args=(fname, 0), rounds=3)
