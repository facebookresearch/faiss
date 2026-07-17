/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Polysemous PQ Training Benchmarks
// Benchmarks polysemous PQ training and search.
//
// IndexPQ(d, M=16, nbits=8) with do_polysemous_training=True, timing training,
// an ST_PQ baseline search, and a polysemous_ht sweep
// {64,62,58,54,50,46,42,38,34,30} at k=1 reporting per-query time and recall.
//
// Design notes: defaults to --threads=1 for both training and search; uses
// Google Benchmark auto-iteration (the Python port,
// bench_index_flat_quantized.py, keeps rounds=5).

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        M,
        "",
        "comma-separated PQ segment counts M "
        "(default: 8,16; sift1m group: 16)");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");
DEFINE_string(
        polysemous_ht,
        "",
        "comma-separated polysemous Hamming thresholds for the ht sweep "
        "(default: 64,62,58,54,50,46,42,38,34,30)");

static void bench_pq_polysemous_train(
        benchmark::State& state,
        int d,
        int M,
        int nt) {
    AlignedTable<float> xt(d * nt);
    float_rand(xt.data(), d * nt, 12345);

    omp_set_num_threads(FLAGS_threads);

    for (auto _ : state) {
        IndexPQ index(d, M, 8);
        index.do_polysemous_training = true;
        index.verbose = false;
        index.train(nt, xt.data());
        benchmark::DoNotOptimize(index.pq.centroids.data());
    }
    state.SetItemsProcessed(state.iterations() * nt);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nt"] = nt;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_pq_polysemous_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int M,
        int k,
        IndexPQ::Search_type_t search_type,
        int ht) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexPQ index(d, M, 8);
    index.do_polysemous_training = true;
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());

    // Select search type (ST_PQ baseline vs. ST_polysemous with a threshold)
    index.search_type = search_type;
    if (search_type == IndexPQ::ST_polysemous && ht > 0) {
        index.polysemous_ht = ht;
    }

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["M"] = M;
    state.counters["k"] = k;
    if (search_type == IndexPQ::ST_polysemous) {
        state.counters["polysemous_ht"] = index.polysemous_ht;
    }
    state.counters["threads"] = FLAGS_threads;

    // Report recall against exact (flat) ground truth on the synthetic data.
    IndexFlatL2 flat(d);
    flat.add(nb, xb.data());
    std::vector<float> gt_dist(nq * k);
    std::vector<int64_t> gt_lab(nq * k);
    flat.search(nq, xq.data(), k, gt_dist.data(), gt_lab.data());
    std::vector<int32_t> gt32(nq * k);
    for (int i = 0; i < nq * k; i++) {
        gt32[i] = (int32_t)gt_lab[i];
    }
    state.counters["recall"] =
            benchmarks::compute_recall_at(labels.data(), gt32.data(), nq, k, k);
}

// SIFT1M variant: train on real data
static void bench_pq_polysemous_train_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        int d,
        int M) {
    omp_set_num_threads(FLAGS_threads);

    for (auto _ : state) {
        IndexPQ index(d, M, 8);
        index.do_polysemous_training = true;
        index.verbose = false;
        index.train(nt, xt);
        benchmark::DoNotOptimize(index.pq.centroids.data());
    }
    state.SetItemsProcessed(state.iterations() * nt);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nt"] = nt;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant: search on real data
static void bench_pq_polysemous_search_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d,
        int M,
        int k,
        IndexPQ::Search_type_t search_type,
        int ht) {
    IndexPQ index(d, M, 8);
    index.do_polysemous_training = true;
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);

    index.search_type = search_type;
    if (search_type == IndexPQ::ST_polysemous && ht > 0) {
        index.polysemous_ht = ht;
    }

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["M"] = M;
    state.counters["k"] = k;
    if (search_type == IndexPQ::ST_polysemous) {
        state.counters["polysemous_ht"] = index.polysemous_ht;
    }
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;

    // R@1: fraction of queries
    // whose true 1-NN is returned at rank 0. For the k=1 groups "recall"
    // already is R@1, so only report it separately when k > 1.
    if (k > 1) {
        size_t n_top1 = 0;
        for (int i = 0; i < nq; i++) {
            if (labels[i * k] == gt[i * gt_k]) {
                n_top1++;
            }
        }
        state.counters["recall_1"] = (double)n_top1 / nq;
    }
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "polysemous PQ training and search (synthetic data and "
            "SIFT1M)",
            "--M=16 --nq=10 --benchmark_filter='pq_polysemous_search/.*'");

    int d = 128;
    int nt = 50000;

    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {8, 16});
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});
    std::vector<int> hts = benchmarks::int_list(
            FLAGS_polysemous_ht, {64, 62, 58, 54, 50, 46, 42, 38, 34, 30});

    // Polysemous training
    for (int M : Ms) {
        std::string name = "pq_polysemous_train/M:" + std::to_string(M);
        auto* b = benchmark::RegisterBenchmark(
                name.c_str(), bench_pq_polysemous_train, d, M, nt);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // Polysemous search (ST_polysemous with the default threshold, k=32)
    int nb = 100000;
    for (int M : Ms) {
        for (int nq : nqs) {
            std::string name = "pq_polysemous_search/M:" + std::to_string(M) +
                    "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_pq_polysemous_search,
                    d,
                    nb,
                    nq,
                    M,
                    32,
                    IndexPQ::ST_polysemous,
                    0); // ht=0 keeps the default polysemous_ht
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // ST_PQ baseline search: plain PQ search with no polysemous
    // filtering (M=16, k=1)
    for (int M : Ms) {
        for (int nq : nqs) {
            std::string name = "pq_baseline_search/M:" + std::to_string(M) +
                    "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_pq_polysemous_search,
                    d,
                    nb,
                    nq,
                    M,
                    1, // k=1
                    IndexPQ::ST_PQ,
                    0);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // Polysemous_ht sweep (M=16, k=1,
    // ht={64,62,58,54,50,46,42,38,34,30})
    for (int M : Ms) {
        for (int ht : hts) {
            for (int nq : nqs) {
                std::string name =
                        "pq_polysemous_ht_search/M:" + std::to_string(M) +
                        "/ht:" + std::to_string(ht) +
                        "/nq:" + std::to_string(nq);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_pq_polysemous_search,
                        d,
                        nb,
                        nq,
                        M,
                        1, // k=1
                        IndexPQ::ST_polysemous,
                        ht);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // SIFT1M-based polysemous benchmarks (the Python polysemous benches
    // live in bench_index_flat_quantized.py)
    static benchmarks::DatasetSIFT1M sift;
    if (benchmarks::dataset_available(FLAGS_data_dir) &&
        sift.load(
                FLAGS_data_dir,
                FLAGS_train_file,
                FLAGS_base_file,
                FLAGS_query_file,
                FLAGS_gt_file)) {
        const std::string ds = benchmarks::dataset_label(FLAGS_base_file);
        int sd = (int)sift.d;
        int snb = (int)sift.nb;
        int snq = (int)sift.nq;
        int snt = (int)sift.nt;

        std::vector<int> Ms_sift = benchmarks::int_list(FLAGS_M, {16});

        // Training on SIFT1M train set
        for (int M : Ms_sift) {
            std::string name =
                    ds + "/pq_polysemous_train/M:" + std::to_string(M);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_pq_polysemous_train_dataset,
                    sift.xt.data(),
                    snt,
                    sd,
                    M);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }

        // Search on SIFT1M (ST_polysemous, default threshold, k=32)
        for (int M : Ms_sift) {
            std::string name =
                    ds + "/pq_polysemous_search/M:" + std::to_string(M);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_pq_polysemous_search_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    snb,
                    sift.xq.data(),
                    snq,
                    sift.gt.data(),
                    sift.gt_k,
                    sd,
                    M,
                    32,
                    IndexPQ::ST_polysemous,
                    0);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }

        // ST_PQ baseline on SIFT1M: plain PQ search, no polysemous
        // filtering (k=1)
        for (int M : Ms_sift) {
            std::string name =
                    ds + "/pq_baseline_search/M:" + std::to_string(M);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_pq_polysemous_search_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    snb,
                    sift.xq.data(),
                    snq,
                    sift.gt.data(),
                    sift.gt_k,
                    sd,
                    M,
                    1, // k=1
                    IndexPQ::ST_PQ,
                    0);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }

        // Polysemous_ht sweep on SIFT1M (k=1),
        // ht={64,62,58,54,50,46,42,38,34,30}
        for (int M : Ms_sift) {
            for (int ht : hts) {
                std::string name = ds +
                        "/pq_polysemous_ht_search/M:" + std::to_string(M) +
                        "/ht:" + std::to_string(ht);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_pq_polysemous_search_dataset,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        snb,
                        sift.xq.data(),
                        snq,
                        sift.gt.data(),
                        sift.gt_k,
                        sd,
                        M,
                        1, // k=1
                        IndexPQ::ST_polysemous,
                        ht);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
