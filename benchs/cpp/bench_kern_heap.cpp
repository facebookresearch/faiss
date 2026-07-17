/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Heap Operations Benchmarks
//
// Single-threaded replace-top and bulk-addn paths via Google Benchmark.
// heap_addn_cmin keeps the k largest elements (minheap direction);
// heap_addn_cmax keeps the k smallest (maxheap direction).

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(
        k,
        "",
        "comma-separated heap sizes "
        "(default: 1,10,20,50,100,200,500,1000,2000,5000)");
DEFINE_string(
        n_push,
        "",
        "comma-separated numbers of pushed elements "
        "(default: 10000,100000,10000000)");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

static void bench_heap_push_cmax(benchmark::State& state, int k, int n_push) {
    std::vector<float> vals(n_push);
    float_randn(vals.data(), n_push, 12345);

    omp_set_num_threads(1);
    for (auto _ : state) {
        std::vector<float> heap_vals(k, 1e20f);
        std::vector<int64_t> heap_ids(k, -1);
        for (int i = 0; i < n_push; i++) {
            if (vals[i] < heap_vals[0]) {
                heap_replace_top<CMax<float, int64_t>>(
                        k,
                        heap_vals.data(),
                        heap_ids.data(),
                        vals[i],
                        (int64_t)i);
            }
        }
        benchmark::DoNotOptimize(heap_vals[0]);
    }
    state.SetItemsProcessed(state.iterations() * n_push);
    state.counters["k"] = k;
    state.counters["n_push"] = n_push;
}

static void bench_heap_push_cmin(benchmark::State& state, int k, int n_push) {
    std::vector<float> vals(n_push);
    float_randn(vals.data(), n_push, 12345);

    omp_set_num_threads(1);
    for (auto _ : state) {
        std::vector<float> heap_vals(k, -1e20f);
        std::vector<int64_t> heap_ids(k, -1);
        for (int i = 0; i < n_push; i++) {
            if (vals[i] > heap_vals[0]) {
                heap_replace_top<CMin<float, int64_t>>(
                        k,
                        heap_vals.data(),
                        heap_ids.data(),
                        vals[i],
                        (int64_t)i);
            }
        }
        benchmark::DoNotOptimize(heap_vals[0]);
    }
    state.SetItemsProcessed(state.iterations() * n_push);
    state.counters["k"] = k;
    state.counters["n_push"] = n_push;
}

// Minheap direction: keep the k largest elements of the stream.
static void bench_heap_addn_cmin(benchmark::State& state, int k, int n_push) {
    std::vector<float> vals(n_push);
    float_randn(vals.data(), n_push, 12345);

    omp_set_num_threads(1);
    for (auto _ : state) {
        std::vector<float> heap_vals(k, -1e20f);
        std::vector<int64_t> heap_ids(k, -1);
        minheap_addn(
                k,
                heap_vals.data(),
                heap_ids.data(),
                vals.data(),
                nullptr,
                n_push);
        benchmark::DoNotOptimize(heap_vals[0]);
    }
    state.SetItemsProcessed(state.iterations() * n_push);
    state.counters["k"] = k;
    state.counters["n_push"] = n_push;
}

// New-only: maxheap (keep the k smallest elements of the stream).
static void bench_heap_addn_cmax(benchmark::State& state, int k, int n_push) {
    std::vector<float> vals(n_push);
    float_randn(vals.data(), n_push, 12345);

    omp_set_num_threads(1);
    for (auto _ : state) {
        std::vector<float> heap_vals(k, 1e20f);
        std::vector<int64_t> heap_ids(k, -1);
        maxheap_addn(
                k,
                heap_vals.data(),
                heap_ids.data(),
                vals.data(),
                nullptr,
                n_push);
        benchmark::DoNotOptimize(heap_vals[0]);
    }
    state.SetItemsProcessed(state.iterations() * n_push);
    state.counters["k"] = k;
    state.counters["n_push"] = n_push;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "heap operations (heap_replace_top CMax/CMin push loops, "
            "minheap_addn/maxheap_addn bulk add)",
            "--k=100 --n_push=100000 --benchmark_filter='heap_push_cmax/.*'");

    std::vector<int> ks = benchmarks::int_list(
            FLAGS_k, {1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000});
    std::vector<int> n_pushes =
            benchmarks::int_list(FLAGS_n_push, {10000, 100000, 10000000});

    for (int k : ks) {
        for (int n_push : n_pushes) {
            std::string suffix = "/k:" + std::to_string(k) +
                    "/n_push:" + std::to_string(n_push);

            auto* b = benchmark::RegisterBenchmark(
                    ("heap_push_cmax" + suffix).c_str(),
                    bench_heap_push_cmax,
                    k,
                    n_push);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("heap_push_cmin" + suffix).c_str(),
                    bench_heap_push_cmin,
                    k,
                    n_push);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("heap_addn_cmin" + suffix).c_str(),
                    bench_heap_addn_cmin,
                    k,
                    n_push);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("heap_addn_cmax" + suffix).c_str(),
                    bench_heap_addn_cmax,
                    k,
                    n_push);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
