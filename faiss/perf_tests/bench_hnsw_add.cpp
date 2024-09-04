/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/perf_tests/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include "faiss/IndexHNSW.h"

using namespace faiss;
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(n, 2000, "dimension");
DEFINE_uint32(iterations, 20, "iterations");
DEFINE_uint32(M, 32, "connections per vertex");
DEFINE_uint32(num_omp_threads, 1, "number of omp threads");

DEFINE_uint32(
        efConstruction,
        40,
        "size of the dynamic list containing the nearest neighbors, which is used during index time");

static void bench_add(
        benchmark::State& state,
        int num_threads,
        int d,
        int n,
        int M,
        int efConstruction) {
    std::vector<float> x(d * n);

    float_rand(x.data(), d * n, 12345);
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;

    omp_set_num_threads(num_threads);

    for (auto _ : state) {
        index.add(n, x.data());
    }
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    int iterations = FLAGS_iterations;
    int d = FLAGS_d;
    int n = FLAGS_n;
    int M = FLAGS_M;
    int efConstruction = FLAGS_efConstruction;
    int num_threads = FLAGS_num_omp_threads;
    benchmark::RegisterBenchmark(
            fmt::format("hnsw_add_{}d_{}n_{}M", d, n, M).c_str(),
            bench_add,
            num_threads,
            d,
            n,
            M,
            efConstruction)
            ->Iterations(iterations);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
