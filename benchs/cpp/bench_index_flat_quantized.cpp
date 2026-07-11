/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Flat Quantized Index Benchmarks (PQ, PQFastScan, SQ, RaBitQ)
//
// The "pq_nbits/search" group benchmarks IndexPQ(d, M=8, nbits) with the
// nbits sweep {4,6,8,10,12} at k=32, reporting recall.

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(nb, 100000, "database size");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");
DEFINE_string(
        M,
        "",
        "comma-separated PQ subquantizer counts "
        "(default: pq group: 8,16,32; pqfs group: 16,32,64; "
        "sift1m pq group: 8,16; sift1m pqfs group: 8,16,32,64)");
DEFINE_string(
        nbits,
        "",
        "comma-separated PQ bits per code for the nbits sweep "
        "(default: 4,6,8,10,12)");
DEFINE_string(
        implem,
        "",
        "comma-separated FastScan implementation ids (default: 0,12,14)");
DEFINE_string(
        sq_type,
        "",
        "comma-separated scalar quantizer types "
        "(default: QT_8bit,QT_fp16,QT_bf16,QT_4bit)");

static void bench_index_pq_search(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        int nbits,
        int nq,
        int k) {
    // PQ trains M sub-quantizers each with 2^nbits centroids; k-means
    // needs >= 39 * 2^nbits training points to avoid a warning.
    int min_train = 39 * (1 << nbits);
    int nt = std::max(std::min(nb, 50000), min_train);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexPQ index(d, M, nbits);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
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
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
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

static void bench_index_pqfs_search(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        int nq,
        int k,
        int implem) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexPQFastScan index(d, M, 4);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.implem = implem;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    FastScan_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["implem"] = implem;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_index_sq_search(
        benchmark::State& state,
        int d,
        int nb,
        ScalarQuantizer::QuantizerType sq_type,
        int nq,
        int k) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexScalarQuantizer index(d, sq_type);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
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
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant for IndexPQ
static void bench_index_pq_search_dataset(
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
        int nbits,
        int k) {
    IndexPQ index(d, M, nbits);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
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
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;

    // R@1: fraction of queries whose true 1-NN is returned at rank 0.
    size_t n_top1 = 0;
    for (int i = 0; i < nq; i++) {
        if (labels[i * k] == gt[i * gt_k]) {
            n_top1++;
        }
    }
    state.counters["recall_1"] = (double)n_top1 / nq;
}

// SIFT1M variant for IndexPQFastScan
static void bench_index_pqfs_search_dataset(
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
        int k) {
    IndexPQFastScan index(d, M, 4);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
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
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;

    // R@1: fraction of queries whose true 1-NN is returned at rank 0.
    size_t n_top1 = 0;
    for (int i = 0; i < nq; i++) {
        if (labels[i * k] == gt[i * gt_k]) {
            n_top1++;
        }
    }
    state.counters["recall_1"] = (double)n_top1 / nq;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "flat quantized index search: IndexPQ, IndexPQFastScan and "
            "IndexScalarQuantizer on synthetic data and SIFT1M",
            "--M=16 --nq=100 --benchmark_filter='pqfs/search/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    int k = 10;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});

    // IndexPQ — M sweep (nbits fixed at 8)
    std::vector<int> pq_Ms = benchmarks::int_list(FLAGS_M, {8, 16, 32});
    for (int M : pq_Ms) {
        for (int nq : nqs) {
            std::string name = "pq/search/M:" + std::to_string(M) +
                    "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_index_pq_search, d, nb, M, 8, nq, k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // IndexPQ — nbits sweep (M fixed at 8, k=32)
    std::vector<int> pq_nbits =
            benchmarks::int_list(FLAGS_nbits, {4, 6, 8, 10, 12});
    for (int nbits : pq_nbits) {
        for (int nq : nqs) {
            std::string name =
                    "pq_nbits/search/nbits:" + std::to_string(nbits) +
                    "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_index_pq_search,
                    d,
                    nb,
                    8,
                    nbits,
                    nq,
                    32);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // IndexPQFastScan
    std::vector<int> pqfs_Ms = benchmarks::int_list(FLAGS_M, {16, 32, 64});
    std::vector<int> implems = benchmarks::int_list(FLAGS_implem, {0, 12, 14});
    for (int M : pqfs_Ms) {
        if (M > d)
            continue;
        for (int nq : nqs) {
            for (int implem : implems) {
                std::string name = "pqfs/search/M:" + std::to_string(M) +
                        "/nq:" + std::to_string(nq) +
                        "/implem:" + std::to_string(implem);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_index_pqfs_search,
                        d,
                        nb,
                        M,
                        nq,
                        k,
                        implem);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // IndexScalarQuantizer
    struct SQDef {
        const char* name;
        ScalarQuantizer::QuantizerType type;
    };
    std::vector<SQDef> sq_defs = {
            {"QT_8bit", ScalarQuantizer::QT_8bit},
            {"QT_fp16", ScalarQuantizer::QT_fp16},
            {"QT_bf16", ScalarQuantizer::QT_bf16},
            {"QT_4bit", ScalarQuantizer::QT_4bit},
    };
    std::vector<std::string> sq_names = benchmarks::str_list(
            FLAGS_sq_type, {"QT_8bit", "QT_fp16", "QT_bf16", "QT_4bit"});
    std::vector<SQDef> sq_types;
    for (const std::string& sq_name : sq_names) {
        bool found = false;
        for (auto& def : sq_defs) {
            if (sq_name == def.name) {
                sq_types.push_back(def);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr,
                    "unknown scalar quantizer type '%s' "
                    "(expected QT_8bit, QT_fp16, QT_bf16 or QT_4bit)\n",
                    sq_name.c_str());
            return 1;
        }
    }

    for (auto& sq : sq_types) {
        for (int nq : nqs) {
            std::string name = std::string("sq/search/") + sq.name +
                    "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_index_sq_search, d, nb, sq.type, nq, k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
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
        int sd = (int)sift.d;
        int snb = (int)sift.nb;
        int snq = (int)sift.nq;
        int snt = (int)sift.nt;

        // IndexPQ on SIFT1M (matches the PQ benches in
        // bench_index_flat_quantized.py: M sweep, nbits=8)
        std::vector<int> sift_pq_Ms = benchmarks::int_list(FLAGS_M, {8, 16});
        for (int M : sift_pq_Ms) {
            std::string name = ds + "/pq/search/M:" + std::to_string(M);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_index_pq_search_dataset,
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
                    8,
                    32); // nbits=8, k=32
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }

        // IndexPQ nbits sweep on SIFT1M (M=8, nbits={4,6,8,10,12}, k=32)
        std::vector<int> sift_pq_nbits =
                benchmarks::int_list(FLAGS_nbits, {4, 6, 8, 10, 12});
        for (int nbits : sift_pq_nbits) {
            std::string name =
                    ds + "/pq_nbits/search/nbits:" + std::to_string(nbits);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_index_pq_search_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    snb,
                    sift.xq.data(),
                    snq,
                    sift.gt.data(),
                    sift.gt_k,
                    sd,
                    8,
                    nbits,
                    32); // M=8, k=32
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }

        // IndexPQFastScan on SIFT1M (matches the PQ fast-scan benches in
        // bench_index_flat_quantized.py)
        std::vector<int> sift_pqfs_Ms =
                benchmarks::int_list(FLAGS_M, {8, 16, 32, 64});
        for (int M : sift_pqfs_Ms) {
            if (M > sd)
                continue;
            std::string name = ds + "/pqfs/search/M:" + std::to_string(M);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_index_pqfs_search_dataset,
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
                    32); // k=32
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
