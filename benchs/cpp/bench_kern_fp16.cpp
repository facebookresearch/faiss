/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// FP16 Conversion Benchmarks

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(
        n,
        "",
        "comma-separated element counts (default: 1000,100000,1000000)");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

static void bench_fp32_to_fp16(benchmark::State& state, int n) {
    std::vector<float> x(n);
    float_rand(x.data(), n, 12345);

    std::vector<uint16_t> y(n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            y[i] = encode_fp16(x[i]);
        }
        benchmark::DoNotOptimize(y[0]);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float));
    state.counters["n"] = n;
}

static void bench_fp16_to_fp32(benchmark::State& state, int n) {
    std::vector<float> tmp(n);
    float_rand(tmp.data(), n, 12345);

    std::vector<uint16_t> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = encode_fp16(tmp[i]);
    }

    std::vector<float> y(n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            y[i] = decode_fp16(x[i]);
        }
        benchmark::DoNotOptimize(y[0]);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint16_t));
    state.counters["n"] = n;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "fp32<->fp16 conversion kernels (encode_fp16, decode_fp16)",
            "--n=100000 --benchmark_filter='fp32_to_fp16/.*'");

    std::vector<int> ns =
            benchmarks::int_list(FLAGS_n, {1000, 100000, 1000000});

    for (int n : ns) {
        std::string suffix = "/n:" + std::to_string(n);

        auto* b = benchmark::RegisterBenchmark(
                ("fp32_to_fp16" + suffix).c_str(), bench_fp32_to_fp16, n);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("fp16_to_fp32" + suffix).c_str(), bench_fp16_to_fp32, n);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
