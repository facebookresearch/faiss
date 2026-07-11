# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_codec_pq.cpp (ProductQuantizer
# train/compute_codes/decode), cpp/bench_codec_sq.cpp
# (ScalarQuantizer compute_codes/decode over quantizer types) and
# cpp/bench_codec_rq.cpp (ResidualQuantizer encode/decode). Also covers
# PQ distance-table construction, LSQ/PRQ/PLSQ encoding and the RQ beam-size
# sweep. The SDC / DistanceComputer benchmarks of the C++ files use internal
# APIs with no public Python entry point.

import functools

import faiss
import numpy as np
import pytest

from bench_utils import (
        dataset_or_skip, load_dataset, params, rand_mat, require_attr)

# ---------------------------------------------------------------------------
# ProductQuantizer (cpp/bench_codec_pq.cpp): d=128, nbits=8,
# nt = n = 10000.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _trained_pq(d, M):
    pq = faiss.ProductQuantizer(d, M, 8)
    pq.train(rand_mat(10000, d))
    return pq


@params(d=[128], M=[8, 16, 32, 64])
def bench_pq_train(benchmark, d, M):
    xt = rand_mat(10000, d)

    def fresh_pq():
        return (faiss.ProductQuantizer(d, M, 8),), {}

    benchmark.pedantic(lambda pq: pq.train(xt), setup=fresh_pq, rounds=3)


@params(d=[128], M=[8, 16, 32, 64])
def bench_pq_compute_codes(benchmark, d, M):
    pq = _trained_pq(d, M)
    x = rand_mat(10000, d, seed=54321)
    benchmark(pq.compute_codes, x)


@params(d=[128], M=[8, 16, 32, 64])
def bench_pq_decode(benchmark, d, M):
    pq = _trained_pq(d, M)
    codes = pq.compute_codes(rand_mat(10000, d, seed=54321))
    benchmark(pq.decode, codes)


# ---------------------------------------------------------------------------
# ScalarQuantizer (cpp/bench_codec_sq.cpp): encode and decode of
# n = 2000 vectors, over all quantizer types.
# ---------------------------------------------------------------------------

SQ_TYPES = [
    "QT_8bit",
    "QT_4bit",
    "QT_6bit",
    "QT_8bit_uniform",
    "QT_4bit_uniform",
    "QT_fp16",
    "QT_bf16",
    "QT_8bit_direct",
    "QT_8bit_direct_signed",
]


def _sq_qtype(name):
    qtype = getattr(faiss.ScalarQuantizer, name, None)
    if qtype is None:
        pytest.skip(f"faiss build has no ScalarQuantizer.{name}")
    return qtype


@functools.lru_cache(maxsize=32)
def _trained_sq(d, type_name):
    sq = faiss.ScalarQuantizer(d, getattr(faiss.ScalarQuantizer, type_name))
    sq.train(rand_mat(10000, d))
    return sq


@params(sq_type=SQ_TYPES, d=[128, 768])
def bench_sq_compute_codes(benchmark, sq_type, d):
    _sq_qtype(sq_type)  # skip on builds lacking this quantizer type
    sq = _trained_sq(d, sq_type)
    x = rand_mat(2000, d)
    benchmark(sq.compute_codes, x)


@params(sq_type=SQ_TYPES, d=[128, 768])
def bench_sq_decode(benchmark, sq_type, d):
    _sq_qtype(sq_type)
    sq = _trained_sq(d, sq_type)
    codes = sq.compute_codes(rand_mat(2000, d))
    benchmark(sq.decode, codes)


# ---------------------------------------------------------------------------
# ResidualQuantizer (cpp/bench_codec_rq.cpp): nt = 10000, n = 1000,
# M x nbits sweep.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _trained_rq(d, M, nbits):
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.max_beam_size = 30
    rq.train(rand_mat(10000, d))
    return rq


@params(d=[128, 256], M=[8, 16], nbits=[8, 4])
def bench_rq_encode(benchmark, d, M, nbits):
    rq = _trained_rq(d, M, nbits)
    xb = rand_mat(1000, d, seed=54321)
    benchmark(rq.compute_codes, xb)


@params(d=[128, 256], M=[8, 16], nbits=[8, 4])
def bench_rq_decode(benchmark, d, M, nbits):
    rq = _trained_rq(d, M, nbits)
    codes = rq.compute_codes(rand_mat(1000, d, seed=54321))
    benchmark(rq.decode, codes)


# Sweeps max_beam_size over the full {1,2,4,8,16,32} grid (the RQ is trained
# with the beam pinned to 30 for compatibility). Overridable with --beam.
@params(d=[128], M=[8, 16], beam=[1, 2, 4, 8, 16, 32])
def bench_rq_encode_beam(benchmark, d, M, beam):
    rq = _trained_rq(d, M, 8)
    old_beam = rq.max_beam_size
    rq.max_beam_size = beam
    xb = rand_mat(1000, d, seed=54321)
    try:
        benchmark(rq.compute_codes, xb)
    finally:
        rq.max_beam_size = old_beam  # the trained RQ is a shared cache


# "rq_lut" variant: RQ with use_beam_LUT=1 and a beam sweep extended to
# 64. Trained RQ is separate from the non-LUT cache (LUT is a train-time knob).
@functools.lru_cache(maxsize=8)
def _trained_rq_lut(d, M, nbits):
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.max_beam_size = 30
    rq.use_beam_LUT = 1
    rq.train(rand_mat(10000, d))
    return rq


@params(d=[128], M=[8, 16], beam=[1, 2, 4, 8, 16, 32, 64])
def bench_rq_lut_encode_beam(benchmark, d, M, beam):
    rq = _trained_rq_lut(d, M, 8)
    old_beam = rq.max_beam_size
    rq.max_beam_size = beam
    xb = rand_mat(1000, d, seed=54321)
    try:
        benchmark(rq.compute_codes, xb)
    finally:
        rq.max_beam_size = old_beam


# "opq" variant: OPQMatrix rotation trained + applied, then PQ codec.
# Times the PQ encode over the OPQ-rotated data.
@functools.lru_cache(maxsize=8)
def _trained_opq_pq(d, M, nbits):
    d2 = ((d + M - 1) // M) * M
    opq = faiss.OPQMatrix(d, M, d2)
    xt = rand_mat(10000, d)
    opq.train(xt)
    pq = faiss.ProductQuantizer(d2, M, nbits)
    pq.train(opq.apply(xt))
    return opq, pq


@params(d=[128], M=[8, 16], nbits=[8])
def bench_opq_pq_encode(benchmark, d, M, nbits):
    require_attr(faiss, "OPQMatrix")
    opq, pq = _trained_opq_pq(d, M, nbits)
    xb = opq.apply(rand_mat(1000, d, seed=54321))
    benchmark(pq.compute_codes, xb)


# ---------------------------------------------------------------------------
# PQ distance-table construction: dsub=4 fixed (d = 4*M), table built for nx
# queries; only compute_distance_tables / compute_inner_prod_tables is timed.
# ---------------------------------------------------------------------------


# Sweeps dsub in {2,4,8} (with d = M*dsub), nbit in {4,8}, M in {8,20},
# nx in {1,10,100}, for both metrics. dsub is exposed as a sweep axis here
# (d = M*dsub).
@functools.lru_cache(maxsize=32)
def _trained_tables_pq(M, nbits, dsub):
    d = dsub * M
    pq = faiss.ProductQuantizer(d, M, nbits)
    # trains on max(1000, ksub * 50) points
    pq.train(rand_mat(max(1000, pq.ksub * 50), d))
    return pq


def _pq_tables_bench(benchmark, M, nbits, dsub, nq, method_name):
    pq = _trained_tables_pq(M, nbits, dsub)
    x = rand_mat(nq, pq.d, seed=54321)
    tab = np.zeros((nq, M, pq.ksub), dtype=np.float32)
    method = getattr(pq, method_name)
    benchmark(
            method, nq, faiss.swig_ptr(x), faiss.swig_ptr(tab))


@params(M=[8, 20], nbits=[4, 8], dsub=[2, 4, 8], nq=[1, 10, 100])
def bench_pq_compute_distance_tables(benchmark, M, nbits, dsub, nq):
    _pq_tables_bench(benchmark, M, nbits, dsub, nq, "compute_distance_tables")


@params(M=[8, 20], nbits=[4, 8], dsub=[2, 4, 8], nq=[1, 10, 100])
def bench_pq_compute_inner_prod_tables(benchmark, M, nbits, dsub, nq):
    _pq_tables_bench(benchmark, M, nbits, dsub, nq, "compute_inner_prod_tables")


# ---------------------------------------------------------------------------
# Additive-quantizer encoders: LSQ, and the 2-split product variants
# PRQ / PLSQ; nt = 10000, encode n = 1000.
# LSQ / PLSQ default to nbits=4: at nbits=8 their training takes 63-200 s
# single-threaded (measured), so 8 is available only via --nbits=8.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _trained_aq(kind, d, M, nbits):
    if kind == "lsq":
        aq = faiss.LocalSearchQuantizer(d, M, nbits)
    elif kind == "prq":
        aq = faiss.ProductResidualQuantizer(d, 2, M // 2, nbits)
    else:  # plsq
        aq = faiss.ProductLocalSearchQuantizer(d, 2, M // 2, nbits)
    aq.train(rand_mat(10000, d))
    return aq


# LSQ / PLSQ default to nbits=4 (nbits=8 training takes 63-200 s
# single-threaded, so 8 is available only via --nbits=8). ils_iters sweeps
# encode_ils_iters values {2,3,4,8,16}.
@params(d=[128], M=[8, 16], nbits=[4], ils_iters=[2, 3, 4, 8, 16])
def bench_lsq_encode(benchmark, d, M, nbits, ils_iters):
    require_attr(faiss, "LocalSearchQuantizer")
    aq = _trained_aq("lsq", d, M, nbits)
    old = aq.encode_ils_iters
    aq.encode_ils_iters = ils_iters
    xb = rand_mat(1000, d, seed=54321)
    try:
        benchmark(aq.compute_codes, xb)
    finally:
        aq.encode_ils_iters = old  # shared cache


@params(d=[128], M=[8, 16], nbits=[8, 4], beam=[1, 2, 4, 8, 16, 32])
def bench_prq_encode(benchmark, d, M, nbits, beam):
    require_attr(faiss, "ProductResidualQuantizer")
    aq = _trained_aq("prq", d, M, nbits)
    # sets max_beam_size on each subquantizer
    olds = []
    for i in range(aq.nsplits):
        subq = faiss.downcast_Quantizer(aq.subquantizer(i))
        olds.append(subq.max_beam_size)
        subq.max_beam_size = beam
    xb = rand_mat(1000, d, seed=54321)
    try:
        benchmark(aq.compute_codes, xb)
    finally:
        for i in range(aq.nsplits):
            faiss.downcast_Quantizer(aq.subquantizer(i)).max_beam_size = olds[i]


@params(d=[128], M=[8, 16], nbits=[4], ils_iters=[2, 3, 4, 8, 16])
def bench_plsq_encode(benchmark, d, M, nbits, ils_iters):
    require_attr(faiss, "ProductLocalSearchQuantizer")
    aq = _trained_aq("plsq", d, M, nbits)
    olds = []
    for i in range(aq.nsplits):
        subq = faiss.downcast_Quantizer(aq.subquantizer(i))
        olds.append(subq.encode_ils_iters)
        subq.encode_ils_iters = ils_iters
    xb = rand_mat(1000, d, seed=54321)
    try:
        benchmark(aq.compute_codes, xb)
    finally:
        for i in range(aq.nsplits):
            faiss.downcast_Quantizer(
                    aq.subquantizer(i)).encode_ils_iters = olds[i]


# ---------------------------------------------------------------------------
# SIFT1M dataset variants: PQ train on the real learn set (nt=100000, d=128)
# and encode of a subset of the base set. Registered only when --data_dir
# points at a SIFT1M-layout directory.
# ---------------------------------------------------------------------------

# Encodes a 100k subset of the 1M base vectors: the full 1M encode would
# dominate the run without changing the per-vector cost.
SIFT1M_ENCODE_N = 100000


@functools.lru_cache(maxsize=4)
def _trained_pq_sift1m(data_dir, M):
    ds = load_dataset(data_dir)
    pq = faiss.ProductQuantizer(ds["xt"].shape[1], M, 8)
    pq.train(ds["xt"])
    return pq


@params(M=[8, 16])
def bench_pq_train_sift1m(benchmark, data_dir, M):
    ds = dataset_or_skip(data_dir)
    xt = ds["xt"]
    d = xt.shape[1]

    def fresh_pq():
        return (faiss.ProductQuantizer(d, M, 8),), {}

    # rounds=1: training on the 100k learn set is expensive.
    benchmark.pedantic(lambda pq: pq.train(xt), setup=fresh_pq, rounds=1)


@params(M=[8, 16])
def bench_pq_compute_codes_sift1m(benchmark, data_dir, M):
    ds = dataset_or_skip(data_dir)
    pq = _trained_pq_sift1m(data_dir, M)
    xb = ds["xb"][:SIFT1M_ENCODE_N]
    benchmark(pq.compute_codes, xb)
