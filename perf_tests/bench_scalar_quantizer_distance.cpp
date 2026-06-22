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
#include <faiss/MetricType.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/random.h>
#include "utils.h"

using namespace faiss;

DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(n, 2000, "number of vectors");
DEFINE_uint32(iterations, 20, "iterations");

static void bench_distance(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        MetricType metric,
        int d,
        int n) {
    std::vector<float> x(d * n);

    float_rand(x.data(), d * n, 12345);

    ScalarQuantizer sq(d, type);

    omp_set_num_threads(1);

    sq.train(n, x.data());

    size_t code_size = sq.code_size;
    state.counters["code_size"] = sq.code_size;

    // encode
    std::vector<uint8_t> codes(code_size * n);
    sq.compute_codes(x.data(), codes.data(), n);

    std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(metric));
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
        benchmark::ClobberMemory();
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
                (bench_name + "/L2").c_str(),
                bench_distance,
                quantizer_type,
                METRIC_L2,
                d,
                n)
                ->Iterations(iterations);

        benchmark::RegisterBenchmark(
                (bench_name + "/IP").c_str(),
                bench_distance,
                quantizer_type,
                METRIC_INNER_PRODUCT,
                d,
                n)
                ->Iterations(iterations);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
