/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <omp.h>
#include <cstdio>
#include <map>

#include <benchmark/benchmark.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/perf_tests/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

using namespace faiss;
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(n, 2000, "dimension");
DEFINE_uint32(iterations, 20, "iterations");

static void bench_encode(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        int d,
        int n) {
    std::vector<float> x(d * n);

    float_rand(x.data(), d * n, 12345);
    ScalarQuantizer sq(d, type);

    omp_set_num_threads(1);
    size_t code_size = sq.code_size;

    sq.train(n, x.data());
    state.counters["code_size"] = sq.code_size;
    std::vector<uint8_t> codes(code_size * n);

    for (auto _ : state) {
        // encode
        sq.compute_codes(x.data(), codes.data(), n);
    }
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    int iterations = FLAGS_iterations;
    int d = FLAGS_d;
    int n = FLAGS_n;
    auto benchs = ::perf_tests::sq_types();

    for (auto& [bench_name, quantizer_type] : benchs) {
        benchmark::RegisterBenchmark(
                bench_name.c_str(), bench_encode, quantizer_type, d, n)
                ->Iterations(iterations);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
