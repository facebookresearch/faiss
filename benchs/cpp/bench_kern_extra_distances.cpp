/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Extra Distance Types Benchmarks
//
// Design notes:
// - This bench reuses one query (default --nx=1) per iteration, so
//   cross-query cache effects are not exercised.
// - Uses uniform [0,1) data (abs+offset) so JensenShannon/BrayCurtis
//   stay valid.
// - Runs single-threaded (--threads not applicable here).

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/MetricType.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(
        metric,
        "",
        "comma-separated metric names "
        "(default: L1,Linf,Canberra,JensenShannon,Lp_0.5,Lp_1,Lp_2,Lp_3,"
        "BrayCurtis,Jaccard)");
DEFINE_string(d, "", "comma-separated vector dimensions (default: 64,128,768)");
DEFINE_string(
        n,
        "",
        "comma-separated batch sizes (default: 1000,16384,100000)");
DEFINE_uint32(
        nx,
        1,
        "number of query vectors (e.g. --nx=4096 --d=64 --n=16384)");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

static void bench_extra_distance(
        benchmark::State& state,
        MetricType metric,
        int d,
        int nx,
        int n,
        float metric_arg) {
    AlignedTable<float> x((size_t)d * nx);
    AlignedTable<float> y((size_t)d * n);
    float_rand(x.data(), (size_t)d * nx, 12345);
    float_rand(y.data(), (size_t)d * n, 54321);

    // Ensure no zeros for metrics that need positive values
    for (size_t i = 0; i < (size_t)d * n; i++) {
        y[i] = std::abs(y[i]) + 0.001f;
    }
    for (size_t i = 0; i < (size_t)d * nx; i++) {
        x[i] = std::abs(x[i]) + 0.001f;
    }

    std::vector<float> distances((size_t)nx * n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        pairwise_extra_distances(
                d,
                nx,
                x.data(),
                n,
                y.data(),
                metric,
                metric_arg,
                distances.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)nx * n);
    state.SetBytesProcessed(state.iterations() * (nx + n) * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["nx"] = nx;
    state.counters["n"] = n;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "extra distance metrics (L1, Linf, Canberra, Jensen-Shannon, Lp, "
            "Bray-Curtis, Jaccard) via pairwise_extra_distances",
            "--metric=Canberra,Jaccard --d=128 --benchmark_filter='Canberra/.*'");

    struct MetricDef {
        const char* name;
        MetricType metric;
        float metric_arg;
    };

    std::vector<MetricDef> all_metrics = {
            {"L1", METRIC_L1, 0},
            {"Linf", METRIC_Linf, 0},
            {"Canberra", METRIC_Canberra, 0},
            {"JensenShannon", METRIC_JensenShannon, 0},
            {"Lp_0.5", METRIC_Lp, 0.5f},
            {"Lp_1", METRIC_Lp, 1.0f},
            {"Lp_2", METRIC_Lp, 2.0f},
            {"Lp_3", METRIC_Lp, 3.0f},
            {"BrayCurtis", METRIC_BrayCurtis, 0},
            {"Jaccard", METRIC_Jaccard, 0},
    };

    std::vector<std::string> default_metric_names;
    for (const auto& md : all_metrics) {
        default_metric_names.push_back(md.name);
    }
    std::vector<std::string> metric_names =
            benchmarks::str_list(FLAGS_metric, default_metric_names);

    std::vector<MetricDef> metrics;
    for (const auto& name : metric_names) {
        bool found = false;
        for (const auto& md : all_metrics) {
            if (name == md.name) {
                metrics.push_back(md);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr,
                    "bench_kern_extra_distances: unknown metric '%s'\n",
                    name.c_str());
            return 1;
        }
    }

    std::vector<int> dims = benchmarks::int_list(FLAGS_d, {64, 128, 768});
    std::vector<int> ns = benchmarks::int_list(FLAGS_n, {1000, 16384, 100000});

    for (auto& md : metrics) {
        for (int d : dims) {
            for (int n : ns) {
                std::string name = std::string(md.name) +
                        "/d:" + std::to_string(d) + "/n:" + std::to_string(n);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_extra_distance,
                        md.metric,
                        d,
                        (int)FLAGS_nx,
                        n,
                        md.metric_arg);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
