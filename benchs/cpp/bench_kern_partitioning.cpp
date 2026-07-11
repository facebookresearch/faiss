/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Partitioning Benchmarks
// Benchmarks CMax_uint16 partition_fuzzy over array sizes, exact/fuzzy
// splits, and id types (int64/int32).

#include <gflags/gflags.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>

#include <benchmark/benchmark.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/partitioning.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(n, "", "comma-separated array sizes (default: 200,2000,20000)");
DEFINE_uint32(maxval, 65536, "max random value for uint16 keys");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

// Benchmark CMax<uint16_t, TI> partition_fuzzy. `fuzzy` selects the
// (q, q+n/4) fuzzy range; otherwise an exact (q, q)
// split is used. TI is the id type (int64_t or int32_t).
template <class TI>
static void bench_partition_uint16(benchmark::State& state, int n, bool fuzzy) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, (int)FLAGS_maxval - 1);
    std::vector<uint16_t> vals(n);
    for (int i = 0; i < n; i++) {
        vals[i] = (uint16_t)dist(rng);
    }
    // ids are a random permutation of [12345, 12345 + n).
    std::vector<TI> ids(n);
    std::iota(ids.begin(), ids.end(), (TI)12345);
    std::shuffle(ids.begin(), ids.end(), rng);

    size_t q_min = n / 2;
    size_t q_max = fuzzy ? (n / 2 + n / 4) : (size_t)(n / 2);

    omp_set_num_threads(1);
    for (auto _ : state) {
        // partition is destructive, so restore the data each iteration.
        std::vector<uint16_t> v(vals);
        std::vector<TI> id(ids);
        size_t q_out;
        partition_fuzzy<CMax<uint16_t, TI>>(
                v.data(), id.data(), n, q_min, q_max, &q_out);
        benchmark::DoNotOptimize(v[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["n"] = n;
    state.counters["q_min"] = q_min;
    state.counters["q_max"] = q_max;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "CMax_uint16 partition_fuzzy over array sizes, exact/fuzzy "
            "splits and id types (int64/int32)",
            "--n=20000 --benchmark_filter='partition/.*'");

    std::vector<int> ns = benchmarks::int_list(FLAGS_n, {200, 2000, 20000});

    struct IdType {
        const char* name;
        bool is_int32;
    };
    const IdType id_types[] = {{"int64", false}, {"int32", true}};

    for (const auto& idt : id_types) {
        for (int n : ns) {
            for (bool fuzzy : {false, true}) {
                std::string name = std::string("partition/id:") + idt.name +
                        (fuzzy ? "/fuzzy/n:" : "/exact/n:") + std::to_string(n);
                auto* b = idt.is_int32
                        ? benchmark::RegisterBenchmark(
                                  name.c_str(),
                                  bench_partition_uint16<int32_t>,
                                  n,
                                  fuzzy)
                        : benchmark::RegisterBenchmark(
                                  name.c_str(),
                                  bench_partition_uint16<int64_t>,
                                  n,
                                  fuzzy);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
