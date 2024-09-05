/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/perf_tests/utils.h>
#include <fmt/format.h>
#include <gflags/gflags.h>
#include <cstdio>
#include <map>

#include <benchmark/benchmark.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(n, 2000, "dimension");
DEFINE_uint32(iterations, 20, "iterations");

static void bench_reconstruction_error(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        int d,
        int n) {
    std::vector<float> x(d * n);

    float_rand(x.data(), d * n, 12345);

    // make sure it's idempotent
    ScalarQuantizer sq(d, type);

    sq.train(n, x.data());

    size_t code_size = sq.code_size;
    state.counters["code_size"] = sq.code_size;

    // encode
    std::vector<uint8_t> codes(code_size * n);
    sq.compute_codes(x.data(), codes.data(), n);

    // decode
    std::vector<float> x2(d * n);
    sq.decode(codes.data(), x2.data(), n);

    state.counters["sql2_recons_error"] =
            fvec_L2sqr(x.data(), x2.data(), n * d) / n;

    // encode again
    std::vector<uint8_t> codes2(code_size * n);
    sq.compute_codes(x2.data(), codes2.data(), n);

    size_t ndiff = 0;
    for (size_t i = 0; i < codes.size(); i++) {
        if (codes[i] != codes2[i])
            ndiff++;
    }

    state.counters["ndiff_for_idempotence"] = ndiff;

    state.counters["code_size_two"] = codes.size();
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
                bench_reconstruction_error,
                quantizer_type,
                d,
                n)
                ->Iterations(iterations);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
