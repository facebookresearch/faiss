/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <cstdio>
#include <map>

#include <benchmark/benchmark.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;

static void bench(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type) {
    int d = 128;
    int n = 2000;

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

    std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer());
    dc->codes = codes.data();
    dc->code_size = sq.code_size;
    state.counters["code_size_three"] = dc->code_size;

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
    int iterations = 20;
    std::map<std::string, ScalarQuantizer::QuantizerType> benchs = {
            {"QT_8bit", ScalarQuantizer::QT_8bit},
            {"QT_4bit", ScalarQuantizer::QT_4bit},
            {"QT_8bit_uniform", ScalarQuantizer::QT_8bit_uniform},
            {"QT_4bit_uniform", ScalarQuantizer::QT_4bit_uniform},
            {"QT_fp16", ScalarQuantizer::QT_fp16},
            {"QT_8bit_direct", ScalarQuantizer::QT_8bit_direct},
            {"QT_6bit", ScalarQuantizer::QT_6bit},
            {"QT_bf16", ScalarQuantizer::QT_bf16},
            {"QT_8bit_direct_signed", ScalarQuantizer::QT_8bit_direct_signed}};
    for (auto& [bench_name, quantizer_type] : benchs) {
        benchmark::RegisterBenchmark(bench_name.c_str(), bench, quantizer_type)
                ->Iterations(iterations);
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
