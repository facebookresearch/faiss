/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Sorting & Bucketing Benchmarks

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(
        n,
        "",
        "comma-separated array sizes (default: 1000,10000,100000)");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

static void bench_fvec_argsort(benchmark::State& state, int n) {
    std::vector<float> vals(n);
    float_rand(vals.data(), n, 12345);

    std::vector<size_t> perm(n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        fvec_argsort(n, vals.data(), perm.data());
        benchmark::DoNotOptimize(perm[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["n"] = n;
}

static void bench_fvec_argsort_parallel(benchmark::State& state, int n) {
    std::vector<float> vals(n);
    float_rand(vals.data(), n, 12345);

    std::vector<size_t> perm(n);

    for (auto _ : state) {
        fvec_argsort_parallel(n, vals.data(), perm.data());
        benchmark::DoNotOptimize(perm[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["n"] = n;
}

static void bench_bucket_sort(benchmark::State& state, int n) {
    uint64_t nbucket = 256;
    std::vector<float> tmp(n);
    float_rand(tmp.data(), n, 12345);

    std::vector<uint64_t> vals(n);
    for (int i = 0; i < n; i++) {
        vals[i] = (uint64_t)((int)(tmp[i] * nbucket) % nbucket);
    }

    std::vector<int64_t> lims(nbucket + 1);
    std::vector<int64_t> perm(n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        bucket_sort(n, vals.data(), nbucket, lims.data(), perm.data());
        benchmark::DoNotOptimize(perm[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["n"] = n;
    state.counters["nbucket"] = nbucket;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "sorting and bucketing kernels (fvec_argsort, "
            "fvec_argsort_parallel, bucket_sort)",
            "--n=100000 --benchmark_filter='fvec_argsort/.*'");

    std::vector<int> ns = benchmarks::int_list(FLAGS_n, {1000, 10000, 100000});

    for (int n : ns) {
        std::string suffix = "/n:" + std::to_string(n);

        auto* b = benchmark::RegisterBenchmark(
                ("fvec_argsort" + suffix).c_str(), bench_fvec_argsort, n);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("fvec_argsort_parallel" + suffix).c_str(),
                bench_fvec_argsort_parallel,
                n);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("bucket_sort" + suffix).c_str(), bench_bucket_sort, n);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
