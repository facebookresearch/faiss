/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/utils/utils.h>
#include <omp.h>

namespace faiss {
namespace {

constexpr int kDimension = 8;
constexpr idx_t kDatabaseSize = 1;

struct BenchmarkIndex {
    BenchmarkIndex() : index(kDimension, 2) {
        std::vector<uint8_t> database(kDatabaseSize * kDimension / 8);
        for (size_t i = 0; i < database.size(); ++i) {
            database[i] = static_cast<uint8_t>(i * 17 + 3);
        }

        index.hnsw.efConstruction = 2;
        index.add(kDatabaseSize, database.data());
        index.hnsw.efSearch = 1;
    }

    IndexBinaryHNSW index;
};

const IndexBinaryHNSW& get_index() {
    static const BenchmarkIndex benchmark_index;
    return benchmark_index.index;
}

void bench_binary_hnsw_single_query(benchmark::State& state) {
    omp_set_num_threads(1);
    set_search_stats_enabled(state.range(0) != 0);
    constexpr std::array<uint8_t, kDimension / 8> query{};
    int32_t distance;
    idx_t label;

    for (auto _ : state) {
        get_index().search(1, query.data(), 1, &distance, &label);
        benchmark::DoNotOptimize(distance);
        benchmark::DoNotOptimize(label);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(bench_binary_hnsw_single_query)
        ->ArgName("stats")
        ->Arg(0)
        ->Arg(1)
        ->ThreadRange(1, 64)
        ->UseRealTime();

} // namespace
} // namespace faiss

int main(int argc, char** argv) {
    omp_set_num_threads(1);
    faiss::get_index();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
