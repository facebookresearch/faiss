/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <omp.h>
#include <cstdio>
#include <map>

#include <benchmark/benchmark.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/perf_tests/utils.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(n, 2000, "dimension");
DEFINE_uint32(iterations, 20, "iterations");

static void bench_distance(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        int n,
        int d) {
    std::vector<float> x(d * n);

    float_rand(x.data(), d * n, 12345);

    // make sure it's idempotent
    ScalarQuantizer sq(d, type);

    omp_set_num_threads(1);

    sq.train(n, x.data());

    size_t code_size = sq.code_size;
    state.counters["code_size"] = sq.code_size;

    // encode
    std::vector<uint8_t> codes(code_size * n);
    sq.compute_codes(x.data(), codes.data(), n);

    std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer());
    dc->codes = codes.data();
    dc->code_size = sq.code_size;

    for (auto _ : state) {
        float sum_dis = 0;
        for (int i = 0; i < n; i++) {
            dc->set_query(&x[i * d]);
            for (int j = 0; j < n; j++) {
                benchmark::DoNotOptimize(sum_dis += (*dc)(j));
            }
        }
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
                fmt::format("{}_{}d_{}n", bench_name, d, n).c_str(),
                bench_distance,
                quantizer_type,
                d,
                n)
                ->Iterations(iterations);
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
