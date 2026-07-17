/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// RaBitQ Kernel Benchmarks
//
// Bitwise AND/XOR dot-product (plus the fused AND dot-product + popcount
// workload) and popcount kernels plus rearrange_bit_planes (scalar
// SIMDLevel::NONE and the AVX2 path), swept over dims
// {64,100,256,512,1000,1024,3072} and nb_bits (qb) values {1,2,4,8}.
//
// The dot-product/popcount kernels stream over a shared 10 MiB buffer
// of vectors per timed iteration
// (n = bufsize / vector_size distinct vectors), measuring cache-cold
// throughput rather than a single cache-resident vector.

#include <memory>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;
using namespace faiss::rabitq;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        d,
        "",
        "comma-separated vector dimensions (d/8 rounded up = bytes) "
        "(default: 64,100,256,512,1000,1024,3072)");
DEFINE_string(
        qb,
        "",
        "comma-separated nb_bits (quantization bits per dim) values "
        "(default: 1,2,4,8)");

/// Shared 10 MiB buffer of random database vectors. The dot-product and
/// popcount benchmarks iterate over all n = size / vector_size vectors in
/// the buffer per timed iteration so each kernel call sees a distinct
/// (cache-cold) vector.
static const AlignedTable<uint8_t, 64>& random_data() {
    static AlignedTable<uint8_t, 64> buf = [] {
        AlignedTable<uint8_t, 64> x(10 << 20); // 10 MiB
        byte_rand(x.data(), x.size(), 456);
        return x;
    }();
    return buf;
}

static void bench_bitwise_and_dot_product(
        benchmark::State& state,
        int d,
        int qb) {
    size_t nbytes = (d + 7) / 8;
    // Query holds qb bit-planes, each nbytes long.
    AlignedTable<uint8_t, 64> query(qb * nbytes);
    std::vector<float> tmp(qb * nbytes);
    float_rand(tmp.data(), qb * nbytes, 12345);
    for (size_t i = 0; i < query.size(); i++) {
        query[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    const AlignedTable<uint8_t, 64>& data = random_data();
    size_t n = data.size() / nbytes;

    omp_set_num_threads(1);
    uint64_t result = 0;
    for (auto _ : state) {
        for (size_t i = 0; i < n; i++) {
            result += bitwise_and_dot_product<SIMDLevel::NONE>(
                    query.data(), data.data() + i * nbytes, nbytes, qb);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * nbytes);
    state.counters["d"] = d;
    state.counters["qb"] = qb;
}

static void bench_bitwise_xor_dot_product(
        benchmark::State& state,
        int d,
        int qb) {
    size_t nbytes = (d + 7) / 8;
    AlignedTable<uint8_t, 64> query(qb * nbytes);
    std::vector<float> tmp(qb * nbytes);
    float_rand(tmp.data(), qb * nbytes, 12345);
    for (size_t i = 0; i < query.size(); i++) {
        query[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    const AlignedTable<uint8_t, 64>& data = random_data();
    size_t n = data.size() / nbytes;

    omp_set_num_threads(1);
    uint64_t result = 0;
    for (auto _ : state) {
        for (size_t i = 0; i < n; i++) {
            result += bitwise_xor_dot_product<SIMDLevel::NONE>(
                    query.data(), data.data() + i * nbytes, nbytes, qb);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * nbytes);
    state.counters["d"] = d;
    state.counters["qb"] = qb;
}

// Fused workload: popcount of the data vector plus the AND dot-product --
// the pair of kernels issued per code by the RaBitQ distance computation.
// There is no fused faiss API, so the two kernels are composed per vector.
static void bench_bitwise_and_dot_product_with_sum(
        benchmark::State& state,
        int d,
        int qb) {
    size_t nbytes = (d + 7) / 8;
    // Query holds qb bit-planes, each nbytes long.
    AlignedTable<uint8_t, 64> query(qb * nbytes);
    std::vector<float> tmp(qb * nbytes);
    float_rand(tmp.data(), qb * nbytes, 12345);
    for (size_t i = 0; i < query.size(); i++) {
        query[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    const AlignedTable<uint8_t, 64>& data = random_data();
    size_t n = data.size() / nbytes;

    omp_set_num_threads(1);
    uint64_t result = 0;
    for (auto _ : state) {
        for (size_t i = 0; i < n; i++) {
            const uint8_t* x = data.data() + i * nbytes;
            result += popcount<SIMDLevel::NONE>(x, nbytes);
            result += bitwise_and_dot_product<SIMDLevel::NONE>(
                    query.data(), x, nbytes, qb);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * nbytes);
    state.counters["d"] = d;
    state.counters["qb"] = qb;
}

static void bench_popcount(benchmark::State& state, int d) {
    size_t nbytes = (d + 7) / 8;
    const AlignedTable<uint8_t, 64>& data = random_data();
    size_t n = data.size() / nbytes;

    omp_set_num_threads(1);
    uint64_t result = 0;
    for (auto _ : state) {
        for (size_t i = 0; i < n; i++) {
            result +=
                    popcount<SIMDLevel::NONE>(data.data() + i * nbytes, nbytes);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * nbytes);
    state.counters["d"] = d;
}

template <SIMDLevel SL>
static void bench_rearrange_bit_planes(benchmark::State& state, int d, int qb) {
    // rotated_qq holds one qb-bit code per dimension (one byte each).
    AlignedTable<uint8_t, 64> rotated_qq(d);
    byte_rand(rotated_qq.data(), rotated_qq.size(), 10996);
    const uint8_t code_mask = static_cast<uint8_t>((1u << qb) - 1);
    for (int i = 0; i < d; i++) {
        rotated_qq[i] &= code_mask;
    }
    size_t offset = (d + 7) / 8;
    AlignedTable<uint8_t, 64> out(offset * qb);

    omp_set_num_threads(1);
    for (auto _ : state) {
        rearrange_bit_planes<SL>(rotated_qq.data(), d, qb, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * d);
    state.SetBytesProcessed(state.iterations() * offset * qb);
    state.counters["d"] = d;
    state.counters["qb"] = qb;
}

/// Mean of the train vectors, used as the RaBitQ centroid (mirrors
/// IndexRaBitQ::train).
static std::vector<float> train_mean(const float* xt, int nt, int d) {
    std::vector<float> centroid(d, 0.0f);
    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < d; j++) {
            centroid[j] += xt[(size_t)i * d + j];
        }
    }
    for (int j = 0; j < d; j++) {
        centroid[j] /= (float)nt;
    }
    return centroid;
}

// SIFT1M variant: RaBitQ encoding of real base vectors. Here qb is the
// number of bits per dimension of the database code (nb_bits).
static void bench_rabitq_encode_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        int d,
        int qb) {
    std::vector<float> centroid = train_mean(xt, nt, d);

    RaBitQuantizer quantizer(d, METRIC_L2, qb);
    quantizer.centroid = centroid.data();

    std::vector<uint8_t> codes((size_t)n * quantizer.code_size);

    omp_set_num_threads(1);
    for (auto _ : state) {
        quantizer.compute_codes(xb, codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)n * quantizer.code_size);
    state.counters["d"] = d;
    state.counters["qb"] = qb;
    state.counters["n"] = n;
    state.counters["dataset"] = 1;
}

// SIFT1M variant: RaBitQ distance estimation over 1-bit codes of real base
// vectors against a real query quantized to qb bits — the bitwise
// dot-product kernels above, exercised end-to-end through the distance
// computer.
static void bench_rabitq_distance_estimate_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        const float* xq,
        int d,
        int qb) {
    std::vector<float> centroid = train_mean(xt, nt, d);

    RaBitQuantizer quantizer(d, METRIC_L2, 1);
    quantizer.centroid = centroid.data();

    std::vector<uint8_t> codes((size_t)n * quantizer.code_size);
    quantizer.compute_codes(xb, codes.data(), n);

    std::unique_ptr<FlatCodesDistanceComputer> dc(
            quantizer.get_distance_computer(
                    (uint8_t)qb, centroid.data(), false));
    dc->codes = codes.data();
    dc->code_size = quantizer.code_size;
    dc->set_query(xq);

    omp_set_num_threads(1);
    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            result += (*dc)(i);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)n * quantizer.code_size);
    state.counters["d"] = d;
    state.counters["qb"] = qb;
    state.counters["n"] = n;
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "RaBitQ bitwise AND/XOR dot-product (incl. fused AND+popcount), "
            "popcount and rearrange_bit_planes (scalar + AVX2) kernels",
            "--d=128,256 --qb=1,4 --benchmark_filter='popcount/.*'");

    std::vector<int> dims = benchmarks::int_list(
            FLAGS_d, {64, 100, 256, 512, 1000, 1024, 3072});
    std::vector<int> qbs = benchmarks::int_list(FLAGS_qb, {1, 2, 4, 8});

    auto set_iters = [](auto* b) {
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    };

    for (int d : dims) {
        for (int qb : qbs) {
            std::string suffix =
                    "/qb:" + std::to_string(qb) + "/d:" + std::to_string(d);

            set_iters(
                    benchmark::RegisterBenchmark(
                            ("bitwise_and_dot_product" + suffix).c_str(),
                            bench_bitwise_and_dot_product,
                            d,
                            qb));

            set_iters(
                    benchmark::RegisterBenchmark(
                            ("bitwise_xor_dot_product" + suffix).c_str(),
                            bench_bitwise_xor_dot_product,
                            d,
                            qb));

            set_iters(
                    benchmark::RegisterBenchmark(
                            ("bitwise_and_dot_product_with_sum" + suffix)
                                    .c_str(),
                            bench_bitwise_and_dot_product_with_sum,
                            d,
                            qb));

            set_iters(
                    benchmark::RegisterBenchmark(
                            ("rearrange_bit_planes_scalar" + suffix).c_str(),
                            bench_rearrange_bit_planes<SIMDLevel::NONE>,
                            d,
                            qb));

            // The AVX2 specialization is only linked in when faiss was built
            // with a SIMD-enabled opt level (avx2/avx512/avx512_spr/dd); the
            // build system defines BENCH_FAISS_HAS_AVX2 in that case. At
            // generic/sse the symbol is absent, so skip registration.
#if defined(BENCH_FAISS_HAS_AVX2)
            set_iters(
                    benchmark::RegisterBenchmark(
                            ("rearrange_bit_planes_avx2" + suffix).c_str(),
                            bench_rearrange_bit_planes<SIMDLevel::AVX2>,
                            d,
                            qb));
#endif
        }

        // popcount does not depend on qb.
        set_iters(
                benchmark::RegisterBenchmark(
                        ("popcount/d:" + std::to_string(d)).c_str(),
                        bench_popcount,
                        d));
    }

    // SIFT1M-based benchmarks (if dataset available)
    static benchmarks::DatasetSIFT1M sift;
    if (benchmarks::dataset_available(FLAGS_data_dir) &&
        sift.load(
                FLAGS_data_dir,
                FLAGS_train_file,
                FLAGS_base_file,
                FLAGS_query_file,
                FLAGS_gt_file)) {
        const std::string ds = benchmarks::dataset_label(FLAGS_base_file);
        // Subsample the base set (SIFT1M nb=1M) to keep the kernel-level
        // dataset variants in line with the synthetic sweep runtimes.
        const int n_enc = sift.nb < 10000 ? (int)sift.nb : 10000;
        for (int qb : qbs) {
            // RaBitQuantizer supports 1..9 bits per dimension.
            if (qb < 1 || qb > 9)
                continue;
            std::string suffix = "/qb:" + std::to_string(qb) +
                    "/d:" + std::to_string(sift.d);

            set_iters(
                    benchmark::RegisterBenchmark(
                            (ds + "/rabitq_encode" + suffix).c_str(),
                            bench_rabitq_encode_dataset,
                            sift.xt.data(),
                            (int)sift.nt,
                            sift.xb.data(),
                            n_enc,
                            (int)sift.d,
                            qb));

            set_iters(
                    benchmark::RegisterBenchmark(
                            (ds + "/rabitq_distance_estimate" + suffix).c_str(),
                            bench_rabitq_distance_estimate_dataset,
                            sift.xt.data(),
                            (int)sift.nt,
                            sift.xb.data(),
                            n_enc,
                            sift.xq.data(),
                            (int)sift.d,
                            qb));
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
