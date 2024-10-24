/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <cstdio>

#include <benchmark/benchmark.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;

static void bench(benchmark::State& state) {
    int d = 128;
    int n = 2000;
    state.SetLabel(faiss::get_compile_options());

    std::vector<float> x(d * n);

    float_rand(x.data(), d * n, 12345);

    // make sure it's idempotent
    ScalarQuantizer sq(d, ScalarQuantizer::QT_6bit);

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
// I think maybe n and d should be input arguments
// for things to really make sense, idk.
BENCHMARK(bench)->Iterations(20);
BENCHMARK_MAIN();
