/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Exact Search (IndexFlat) Benchmarks

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        d,
        "",
        "comma-separated vector dimensions (default: 16,32,64,128,256,768)");
DEFINE_string(nb, "", "comma-separated database sizes (default: 10000,100000)");
DEFINE_string(
        nq,
        "",
        "comma-separated query batch sizes "
        "(default: 1,10,100,1000,10000; "
        "sift1m group: 1,10,100 and full query set)");
DEFINE_string(k, "", "comma-separated k values (default: 1,10,100)");
DEFINE_string(
        metric,
        "",
        "comma-separated metrics, L2 and/or IP (default: L2,IP)");

static void bench_flat_add(
        benchmark::State& state,
        int d,
        int nb,
        MetricType metric) {
    AlignedTable<float> xb((size_t)d * nb);
    float_rand(xb.data(), (size_t)d * nb, 12345);

    omp_set_num_threads(FLAGS_threads);
    for (auto _ : state) {
        IndexFlat index(d, metric);
        index.add(nb, xb.data());
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)nb * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_flat_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k,
        MetricType metric) {
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xb.data(), (size_t)d * nb, 12345);
    float_rand(xq.data(), d * nq, 54321);

    IndexFlat index(d, metric);
    omp_set_num_threads(FLAGS_threads);
    index.add(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)nq * nb * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_flat_search_sift1m(
        benchmark::State& state,
        const float* xb,
        const float* xq,
        const int32_t* gt,
        size_t gt_k,
        int d,
        int nb,
        int nq,
        int k,
        MetricType metric) {
    IndexFlat index(d, metric);
    omp_set_num_threads(FLAGS_threads);
    index.add(nb, xb);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)nq * nb * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["dataset"] = 1; // flag: real data
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "exact search with IndexFlat: add and k-NN search for L2 and "
            "inner-product metrics, on synthetic data and SIFT1M",
            "--d=128 --nq=100 --k=10 --benchmark_filter='search/L2/.*'");

    // Force the reservoir result handler for small k.
    faiss::distance_compute_min_k_reservoir = 5;

    std::vector<int> dims =
            benchmarks::int_list(FLAGS_d, {16, 32, 64, 128, 256, 768});
    std::vector<int> nbs = benchmarks::int_list(FLAGS_nb, {10000, 100000});
    std::vector<int> nqs =
            benchmarks::int_list(FLAGS_nq, {1, 10, 100, 1000, 10000});
    std::vector<int> ks = benchmarks::int_list(FLAGS_k, {1, 10, 100});
    std::vector<std::string> metric_names =
            benchmarks::str_list(FLAGS_metric, {"L2", "IP"});

    for (const std::string& metric_name : metric_names) {
        MetricType metric;
        if (metric_name == "L2") {
            metric = METRIC_L2;
        } else if (metric_name == "IP") {
            metric = METRIC_INNER_PRODUCT;
        } else {
            fprintf(stderr,
                    "unknown metric '%s' (expected L2 or IP)\n",
                    metric_name.c_str());
            return 1;
        }
        for (int d : dims) {
            for (int nb : nbs) {
                std::string add_name = std::string("add/") + metric_name +
                        "/d:" + std::to_string(d) + "/nb:" + std::to_string(nb);
                auto* b = benchmark::RegisterBenchmark(
                        add_name.c_str(), bench_flat_add, d, nb, metric);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);

                for (int nq : nqs) {
                    for (int k : ks) {
                        std::string name = std::string("search/") +
                                metric_name + "/d:" + std::to_string(d) +
                                "/nb:" + std::to_string(nb) +
                                "/nq:" + std::to_string(nq) +
                                "/k:" + std::to_string(k);
                        b = benchmark::RegisterBenchmark(
                                name.c_str(),
                                bench_flat_search,
                                d,
                                nb,
                                nq,
                                k,
                                metric);
                        if (FLAGS_iterations > 0)
                            b->Iterations(FLAGS_iterations);
                    }
                }
            }
        }
    }

    // SIFT1M-based benchmarks (if dataset available)
    static benchmarks::DatasetSIFT1M sift;
    if (benchmarks::dataset_available(FLAGS_data_dir) &&
        sift.load(
                FLAGS_data_dir,
                FLAGS_train_file,
                FLAGS_base_file,
                FLAGS_query_file,
                FLAGS_gt_file)) {
        const std::string ds = benchmarks::dataset_label(FLAGS_base_file);
        std::vector<int> sift_nqs =
                benchmarks::int_list(FLAGS_nq, {1, 10, 100, (int)sift.nq});
        for (int nq : sift_nqs) {
            if ((size_t)nq > sift.nq)
                continue;
            for (int k : ks) {
                std::string name = ds + "/search/L2/nq:" + std::to_string(nq) +
                        "/k:" + std::to_string(k);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_flat_search_sift1m,
                        sift.xb.data(),
                        sift.xq.data(),
                        sift.gt.data(),
                        sift.gt_k,
                        (int)sift.d,
                        (int)sift.nb,
                        nq,
                        k,
                        METRIC_L2);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
