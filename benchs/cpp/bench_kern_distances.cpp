/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Core Distance Computation Benchmarks
//
// Design notes:
// - These benches reuse one query per iteration, so cross-query cache
//   effects are not exercised.

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(
        d,
        "",
        "comma-separated vector dimensions "
        "(default: 16,32,64,128,256,512,768,1536; pairwise group: 128,256,768; "
        "ny group: 3,4,12,36,64)");
DEFINE_string(
        n,
        "",
        "comma-separated batch sizes "
        "(default: 1,100,10000,100000; batch-4 group: 1,100,10000; "
        "pairwise group: 100,1000,10000; ny group: 100000,1000000)");
DEFINE_uint32(nx, 100, "number of query vectors for the pairwise group");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

static void bench_fvec_L2sqr(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d * n);
    AlignedTable<float> y(d);
    float_rand(x.data(), d * n, 12345);
    float_rand(y.data(), d, 54321);

    omp_set_num_threads(1);
    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            result += fvec_L2sqr(x.data() + i * d, y.data(), d);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * d * 2 * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_fvec_inner_product(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d * n);
    AlignedTable<float> y(d);
    float_rand(x.data(), d * n, 12345);
    float_rand(y.data(), d, 54321);

    omp_set_num_threads(1);
    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            result += fvec_inner_product(x.data() + i * d, y.data(), d);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * d * 2 * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_fvec_norm_L2sqr(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d * n);
    float_rand(x.data(), d * n, 12345);

    omp_set_num_threads(1);
    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            result += fvec_norm_L2sqr(x.data() + i * d, d);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_fvec_L1(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d * n);
    AlignedTable<float> y(d);
    float_rand(x.data(), d * n, 12345);
    float_rand(y.data(), d, 54321);

    omp_set_num_threads(1);
    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            result += fvec_L1(x.data() + i * d, y.data(), d);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * d * 2 * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_fvec_Linf(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d * n);
    AlignedTable<float> y(d);
    float_rand(x.data(), d * n, 12345);
    float_rand(y.data(), d, 54321);

    omp_set_num_threads(1);
    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            result += fvec_Linf(x.data() + i * d, y.data(), d);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * d * 2 * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_fvec_L2sqr_batch_4(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d);
    AlignedTable<float> y(d * 4);
    float_rand(x.data(), d, 12345);
    float_rand(y.data(), d * 4, 54321);

    omp_set_num_threads(1);
    float d0, d1, d2, d3;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            fvec_L2sqr_batch_4(
                    x.data(),
                    y.data(),
                    y.data() + d,
                    y.data() + 2 * d,
                    y.data() + 3 * d,
                    d,
                    d0,
                    d1,
                    d2,
                    d3);
            benchmark::DoNotOptimize(d0);
            benchmark::DoNotOptimize(d1);
            benchmark::DoNotOptimize(d2);
            benchmark::DoNotOptimize(d3);
        }
    }
    state.SetItemsProcessed(state.iterations() * n * 4);
    state.SetBytesProcessed(state.iterations() * n * d * 5 * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_fvec_inner_product_batch_4(
        benchmark::State& state,
        int d,
        int n) {
    AlignedTable<float> x(d);
    AlignedTable<float> y(d * 4);
    float_rand(x.data(), d, 12345);
    float_rand(y.data(), d * 4, 54321);

    omp_set_num_threads(1);
    float ip0, ip1, ip2, ip3;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            fvec_inner_product_batch_4(
                    x.data(),
                    y.data(),
                    y.data() + d,
                    y.data() + 2 * d,
                    y.data() + 3 * d,
                    d,
                    ip0,
                    ip1,
                    ip2,
                    ip3);
            benchmark::DoNotOptimize(ip0);
            benchmark::DoNotOptimize(ip1);
            benchmark::DoNotOptimize(ip2);
            benchmark::DoNotOptimize(ip3);
        }
    }
    state.SetItemsProcessed(state.iterations() * n * 4);
    state.SetBytesProcessed(state.iterations() * n * d * 5 * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_pairwise_L2sqr(
        benchmark::State& state,
        int d,
        int nx,
        int n) {
    AlignedTable<float> x(d * nx);
    AlignedTable<float> y(d * n);
    float_rand(x.data(), d * nx, 12345);
    float_rand(y.data(), d * n, 54321);
    std::vector<float> dis((size_t)nx * n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        pairwise_L2sqr(d, nx, x.data(), n, y.data(), dis.data());
        benchmark::DoNotOptimize(dis[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)nx * n);
    state.SetBytesProcessed(state.iterations() * (nx + n) * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["nx"] = nx;
    state.counters["n"] = n;
}

static void bench_fvec_L2sqr_ny(benchmark::State& state, int d, int ny) {
    AlignedTable<float> x(d);
    AlignedTable<float> y((size_t)d * ny);
    float_rand(x.data(), d, 54321);
    float_rand(y.data(), (size_t)d * ny, 12345);
    std::vector<float> dis(ny); // preallocated, outside the timing

    omp_set_num_threads(1);
    for (auto _ : state) {
        fvec_L2sqr_ny(dis.data(), x.data(), y.data(), d, ny);
        benchmark::DoNotOptimize(dis[0]);
    }
    state.SetItemsProcessed(state.iterations() * ny);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)ny * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = ny;
}

static void bench_fvec_inner_products_ny(
        benchmark::State& state,
        int d,
        int ny) {
    AlignedTable<float> x(d);
    AlignedTable<float> y((size_t)d * ny);
    float_rand(x.data(), d, 54321);
    float_rand(y.data(), (size_t)d * ny, 12345);
    std::vector<float> ip(ny); // preallocated, outside the timing

    omp_set_num_threads(1);
    for (auto _ : state) {
        fvec_inner_products_ny(ip.data(), x.data(), y.data(), d, ny);
        benchmark::DoNotOptimize(ip[0]);
    }
    state.SetItemsProcessed(state.iterations() * ny);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)ny * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = ny;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "core distance-computation kernels (fvec_L2sqr, inner product, "
            "L1/Linf, batch-4 variants, pairwise L2sqr)",
            "--d=128,256 --n=10000 --benchmark_filter='fvec_L2sqr/.*'");

    std::vector<int> dims = benchmarks::int_list(
            FLAGS_d, {16, 32, 64, 128, 256, 512, 768, 1536});
    std::vector<int> ns =
            benchmarks::int_list(FLAGS_n, {1, 100, 10000, 100000});

    for (int d : dims) {
        for (int n : ns) {
            std::string suffix =
                    "/d:" + std::to_string(d) + "/n:" + std::to_string(n);

            auto* b = benchmark::RegisterBenchmark(
                    ("fvec_L2sqr" + suffix).c_str(), bench_fvec_L2sqr, d, n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("fvec_inner_product" + suffix).c_str(),
                    bench_fvec_inner_product,
                    d,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("fvec_norm_L2sqr" + suffix).c_str(),
                    bench_fvec_norm_L2sqr,
                    d,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("fvec_L1" + suffix).c_str(), bench_fvec_L1, d, n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("fvec_Linf" + suffix).c_str(), bench_fvec_Linf, d, n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // Batch-4 ILP variants
    std::vector<int> ns_batch4 = benchmarks::int_list(FLAGS_n, {1, 100, 10000});
    for (int d : dims) {
        for (int n : ns_batch4) {
            std::string suffix =
                    "/d:" + std::to_string(d) + "/n:" + std::to_string(n);
            auto* b = benchmark::RegisterBenchmark(
                    ("fvec_L2sqr_batch_4" + suffix).c_str(),
                    bench_fvec_L2sqr_batch_4,
                    d,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("fvec_inner_product_batch_4" + suffix).c_str(),
                    bench_fvec_inner_product_batch_4,
                    d,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // Pairwise L2sqr (BLAS path)
    std::vector<int> dims_pairwise =
            benchmarks::int_list(FLAGS_d, {128, 256, 768});
    std::vector<int> ns_pairwise =
            benchmarks::int_list(FLAGS_n, {100, 1000, 10000});
    for (int d : dims_pairwise) {
        for (int n : ns_pairwise) {
            std::string suffix =
                    "/d:" + std::to_string(d) + "/n:" + std::to_string(n);
            auto* b = benchmark::RegisterBenchmark(
                    ("pairwise_L2sqr" + suffix).c_str(),
                    bench_pairwise_L2sqr,
                    d,
                    (int)FLAGS_nx,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // Standard working point: d=64, nq=4096, nb=16384, METRIC_L2.
    {
        auto* b = benchmark::RegisterBenchmark(
                "pairwise_L2sqr/d:64/nx:4096/n:16384",
                bench_pairwise_L2sqr,
                64,
                4096,
                16384);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // One-query-vs-many _ny kernels (mirrors python bench_kern_distances.py
    // fvec_*_ny: dims including d=3)
    std::vector<int> dims_ny =
            benchmarks::int_list(FLAGS_d, {3, 4, 12, 36, 64});
    std::vector<int> ns_ny = benchmarks::int_list(FLAGS_n, {100000, 1000000});
    for (int d : dims_ny) {
        for (int n : ns_ny) {
            std::string suffix =
                    "/d:" + std::to_string(d) + "/n:" + std::to_string(n);
            auto* b = benchmark::RegisterBenchmark(
                    ("fvec_L2sqr_ny" + suffix).c_str(),
                    bench_fvec_L2sqr_ny,
                    d,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("fvec_inner_products_ny" + suffix).c_str(),
                    bench_fvec_inner_products_ny,
                    d,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
