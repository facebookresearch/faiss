/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Hamming Distance Benchmarks
//
// Width-specific Hamming computers (HammingComputer{4,8,16,20,32,64}),
// GenHammingComputer{8,16,32} and the hamming<N>() free-function templates,
// in addition to the HammingComputerDefault inner loop and hammings_knn
// heap/counting variants.
//
// Design notes:
// - --threads defaults to 1 for the width-specific / hamming<N> /
//   GenHammingComputer groups.

#include <gflags/gflags.h>
#include <omp.h>

#include <cstdint>

#include <benchmark/benchmark.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/hamming_distance/common.h>
#include <faiss/utils/hamming_distance/hamming_computer-generic.h>
#include <faiss/utils/hamming_distance/hamming_computer.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
DEFINE_string(
        code_size,
        "",
        "comma-separated binary code sizes in bytes "
        "(default: 4,8,16,32,64,128,256,512,1000; knn group: 16,32,64)");
DEFINE_string(k, "", "comma-separated knn k values (default: 1,4,16,64,256)");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

static void bench_hamming_computer(
        benchmark::State& state,
        int code_size,
        int n) {
    std::vector<uint8_t> a(code_size);
    std::vector<uint8_t> b((size_t)code_size * n);

    // Fill with pseudo-random data
    byte_rand(a.data(), a.size(), 12345);
    byte_rand(b.data(), b.size(), 23456);

    omp_set_num_threads(1);
    int result = 0;
    HammingComputerDefault_tpl<SIMDLevel::NONE> hc(a.data(), code_size);
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            int dis = hc.hamming(b.data() + (size_t)i * code_size);
            result += dis;
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * (int64_t)n * code_size * 2);
    state.counters["code_size"] = code_size;
    state.counters["n"] = n;
}

// Width-specific HammingComputer / GenHammingComputer inner loop over n1
// query x n2 database codes (n1=65536, n2=16384). The computer type fixes
// its own natural code_size (in bytes). OpenMP-parallel over queries with
// --threads threads.
template <class HammingComputerT>
static void bench_hamming_computer_width(
        benchmark::State& state,
        int code_size,
        int n1,
        int n2) {
    std::vector<uint8_t> a((size_t)code_size * n1);
    std::vector<uint8_t> b((size_t)code_size * n2);

    byte_rand(a.data(), a.size(), 12345);
    byte_rand(b.data(), b.size(), 23456);

    omp_set_num_threads(FLAGS_threads);
    uint64_t result = 0;
    for (auto _ : state) {
        uint64_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
        for (int i = 0; i < n1; i++) {
            HammingComputerT hc(a.data() + (size_t)i * code_size, code_size);
            uint64_t local_sum = 0;
            for (int j = 0; j < n2; j++) {
                local_sum += hc.hamming(b.data() + (size_t)j * code_size);
            }
            sum += local_sum;
        }
        result += sum;
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n1 * n2);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)n1 * n2 * code_size * 2);
    state.counters["code_size"] = code_size;
    state.counters["n1"] = n1;
    state.counters["n2"] = n2;
    state.counters["threads"] = FLAGS_threads;
}

// hamming<CODE_SIZE_IN_BITS>() free-function template inner loop over n1
// query x n2 database codes (n1=65536, n2=16384). OpenMP-parallel over
// queries with --threads threads.
template <int CODE_SIZE_IN_BITS>
static void bench_hamming_func(benchmark::State& state, int n1, int n2) {
    constexpr int code_size = CODE_SIZE_IN_BITS / 8;
    std::vector<uint8_t> a((size_t)code_size * n1);
    std::vector<uint8_t> b((size_t)code_size * n2);

    byte_rand(a.data(), a.size(), 12345);
    byte_rand(b.data(), b.size(), 23456);

    omp_set_num_threads(FLAGS_threads);
    uint64_t result = 0;
    for (auto _ : state) {
        uint64_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
        for (int i = 0; i < n1; i++) {
            const uint64_t* a64 = reinterpret_cast<const uint64_t*>(
                    a.data() + (size_t)i * code_size);
            uint64_t local_sum = 0;
            for (int j = 0; j < n2; j++) {
                const uint64_t* b64 = reinterpret_cast<const uint64_t*>(
                        b.data() + (size_t)j * code_size);
                local_sum += hamming<CODE_SIZE_IN_BITS>(a64, b64);
            }
            sum += local_sum;
        }
        result += sum;
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n1 * n2);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)n1 * n2 * code_size * 2);
    state.counters["code_size"] = code_size;
    state.counters["n1"] = n1;
    state.counters["n2"] = n2;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_hammings_knn_hc(
        benchmark::State& state,
        int code_size,
        int nb,
        int nq,
        int k) {
    std::vector<uint8_t> a(code_size * nq);
    std::vector<uint8_t> b(code_size * nb);

    std::vector<float> tmp(code_size * (nq + nb));
    float_rand(tmp.data(), code_size * (nq + nb), 12345);
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    for (size_t i = 0; i < b.size(); i++) {
        b[i] = (uint8_t)(((int)(tmp[a.size() + i] * 256)) & 0xFF);
    }

    std::vector<int32_t> distances(nq * k);
    std::vector<int64_t> labels(nq * k);
    int_maxheap_array_t res = {
            (size_t)nq, (size_t)k, labels.data(), distances.data()};

    omp_set_num_threads(1);
    for (auto _ : state) {
        hammings_knn_hc(&res, a.data(), b.data(), nb, code_size, 0);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["code_size"] = code_size;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
}

static void bench_hammings_knn_mc(
        benchmark::State& state,
        int code_size,
        int nb,
        int nq,
        int k) {
    std::vector<uint8_t> a(code_size * nq);
    std::vector<uint8_t> b(code_size * nb);

    std::vector<float> tmp(code_size * (nq + nb));
    float_rand(tmp.data(), code_size * (nq + nb), 12345);
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    for (size_t i = 0; i < b.size(); i++) {
        b[i] = (uint8_t)(((int)(tmp[a.size() + i] * 256)) & 0xFF);
    }

    std::vector<int32_t> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    omp_set_num_threads(1);
    for (auto _ : state) {
        hammings_knn_mc(
                a.data(),
                b.data(),
                nq,
                nb,
                k,
                code_size,
                distances.data(),
                labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["code_size"] = code_size;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "Hamming-distance kernels (HammingComputer inner loop, "
            "hammings_knn heap and counting variants)",
            "--code_size=32 --k=10 --benchmark_filter='hammings_knn_hc/.*'");

    auto set_iters = [](auto* b) {
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    };

    // Hamming computer (HammingComputerDefault): various code sizes, one
    // query scanned against n=4M codes.
    std::vector<int> code_sizes = benchmarks::int_list(
            FLAGS_code_size, {4, 8, 16, 32, 64, 128, 256, 512, 1000});
    int n = 4000000;
    for (int cs : code_sizes) {
        std::string name = "hamming_computer/code_size:" + std::to_string(cs) +
                "/n:" + std::to_string(n);
        set_iters(
                benchmark::RegisterBenchmark(
                        name.c_str(), bench_hamming_computer, cs, n));
    }

    // Width-specific HammingComputer{4,8,16,20,32,64}: each at its natural
    // code_size in bytes, over the n1 x n2 query/database shape.
    {
        int n1 = 65536;
        int n2 = 16384;
        std::string sfx =
                "/n1:" + std::to_string(n1) + "/n2:" + std::to_string(n2);
        set_iters(
                benchmark::RegisterBenchmark(
                        ("HammingComputer4/code_size:4" + sfx).c_str(),
                        bench_hamming_computer_width<HammingComputer4>,
                        4,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("HammingComputer8/code_size:8" + sfx).c_str(),
                        bench_hamming_computer_width<HammingComputer8>,
                        8,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("HammingComputer16/code_size:16" + sfx).c_str(),
                        bench_hamming_computer_width<
                                HammingComputer16_tpl<SIMDLevel::NONE>>,
                        16,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("HammingComputer20/code_size:20" + sfx).c_str(),
                        bench_hamming_computer_width<
                                HammingComputer20_tpl<SIMDLevel::NONE>>,
                        20,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("HammingComputer32/code_size:32" + sfx).c_str(),
                        bench_hamming_computer_width<
                                HammingComputer32_tpl<SIMDLevel::NONE>>,
                        32,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("HammingComputer64/code_size:64" + sfx).c_str(),
                        bench_hamming_computer_width<
                                HammingComputer64_tpl<SIMDLevel::NONE>>,
                        64,
                        n1,
                        n2));

        // GenHammingComputer{8,16,32}: each at its natural code_size in bytes.
        set_iters(
                benchmark::RegisterBenchmark(
                        ("GenHammingComputer8/code_size:8" + sfx).c_str(),
                        bench_hamming_computer_width<
                                GenHammingComputer8_tpl<SIMDLevel::NONE>>,
                        8,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("GenHammingComputer16/code_size:16" + sfx).c_str(),
                        bench_hamming_computer_width<
                                GenHammingComputer16_tpl<SIMDLevel::NONE>>,
                        16,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("GenHammingComputer32/code_size:32" + sfx).c_str(),
                        bench_hamming_computer_width<
                                GenHammingComputer32_tpl<SIMDLevel::NONE>>,
                        32,
                        n1,
                        n2));

        // hamming<N>() free-function templates (N in bits).
        set_iters(
                benchmark::RegisterBenchmark(
                        ("hamming_func/bits:64" + sfx).c_str(),
                        bench_hamming_func<64>,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("hamming_func/bits:128" + sfx).c_str(),
                        bench_hamming_func<128>,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("hamming_func/bits:256" + sfx).c_str(),
                        bench_hamming_func<256>,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("hamming_func/bits:384" + sfx).c_str(),
                        bench_hamming_func<384>,
                        n1,
                        n2));
        set_iters(
                benchmark::RegisterBenchmark(
                        ("hamming_func/bits:512" + sfx).c_str(),
                        bench_hamming_func<512>,
                        n1,
                        n2));
    }

    // KNN heap vs counting: nb=30000, nq=10000, k in {1,4,16,64,256}.
    int nb = 30000;
    int nq = 10000;
    std::vector<int> ks = benchmarks::int_list(FLAGS_k, {1, 4, 16, 64, 256});
    std::vector<int> code_sizes_knn =
            benchmarks::int_list(FLAGS_code_size, {16, 32, 64});
    for (int cs : code_sizes_knn) {
        for (int k : ks) {
            std::string hc_name =
                    "hammings_knn_hc/code_size:" + std::to_string(cs) +
                    "/nb:" + std::to_string(nb) + "/nq:" + std::to_string(nq) +
                    "/k:" + std::to_string(k);
            auto* b = benchmark::RegisterBenchmark(
                    hc_name.c_str(), bench_hammings_knn_hc, cs, nb, nq, k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            std::string mc_name =
                    "hammings_knn_mc/code_size:" + std::to_string(cs) +
                    "/nb:" + std::to_string(nb) + "/nq:" + std::to_string(nq) +
                    "/k:" + std::to_string(k);
            b = benchmark::RegisterBenchmark(
                    mc_name.c_str(), bench_hammings_knn_mc, cs, nb, nq, k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
