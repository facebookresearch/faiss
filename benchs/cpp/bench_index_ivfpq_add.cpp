/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IVFPQ Add Performance Benchmarks
// Benchmarks the add (encode + append) pipeline in IndexIVFPQ at various
// thread counts, measuring the parallelized inverted-list append loop.

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(d, 64, "dimension");
DEFINE_uint32(nb, 100000, "number of vectors to add");
DEFINE_uint32(nlist, 1024, "number of IVF lists");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        M,
        "",
        "comma-separated PQ segment counts M "
        "(default: 8,16,32; sift1m group: 16)");
DEFINE_string(threads, "", "comma-separated thread counts (default: 1,2,4,8)");

static void bench_ivfpq_add(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        int M,
        int threads) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);

    // Pre-train quantizer and PQ
    IndexFlatL2 quantizer(d);
    IndexIVFPQ template_index(&quantizer, d, nlist, M, 8);
    template_index.verbose = false;
    omp_set_num_threads(threads);
    template_index.train(nt, xt.data());

    for (auto _ : state) {
        // Reset index for each iteration (measure pure add cost)
        IndexIVFPQ index(&quantizer, d, nlist, M, 8);
        index.verbose = false;
        index.is_trained = true;
        index.pq = template_index.pq;
        omp_set_num_threads(threads);
        index.add(nb, xb.data());
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["threads"] = threads;
}

static void bench_ivfflat_add(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        int threads) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);

    IndexFlatL2 quantizer(d);
    IndexIVFFlat template_index(&quantizer, d, nlist);
    template_index.verbose = false;
    omp_set_num_threads(threads);
    template_index.train(nt, xt.data());

    for (auto _ : state) {
        IndexIVFFlat index(&quantizer, d, nlist);
        index.verbose = false;
        index.is_trained = true;
        omp_set_num_threads(threads);
        index.add(nb, xb.data());
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["threads"] = threads;
}

// SIFT1M variant: trains on the real train set outside the timed loop,
// times adding the real base set (measure pure add cost)
static void bench_ivfpq_add_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        int d,
        int nlist,
        int M,
        int threads) {
    // Pre-train quantizer and PQ
    IndexFlatL2 quantizer(d);
    IndexIVFPQ template_index(&quantizer, d, nlist, M, 8);
    template_index.verbose = false;
    omp_set_num_threads(threads);
    template_index.train(nt, xt);

    for (auto _ : state) {
        // Reset index for each iteration (measure pure add cost)
        IndexIVFPQ index(&quantizer, d, nlist, M, 8);
        index.verbose = false;
        index.is_trained = true;
        index.pq = template_index.pq;
        omp_set_num_threads(threads);
        index.add(nb, xb);
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["threads"] = threads;
    state.counters["dataset"] = 1;
}

// SIFT1M variant for IVFFlat add
static void bench_ivfflat_add_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        int d,
        int nlist,
        int threads) {
    IndexFlatL2 quantizer(d);
    IndexIVFFlat template_index(&quantizer, d, nlist);
    template_index.verbose = false;
    omp_set_num_threads(threads);
    template_index.train(nt, xt);

    for (auto _ : state) {
        IndexIVFFlat index(&quantizer, d, nlist);
        index.verbose = false;
        index.is_trained = true;
        omp_set_num_threads(threads);
        index.add(nb, xb);
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["threads"] = threads;
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "IVFPQ and IVFFlat add (encode + append) throughput at "
            "various thread counts",
            "--M=16 --threads=1,8 --benchmark_filter='ivfpq_add/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    int nlist = FLAGS_nlist;

    std::vector<int> thread_counts =
            benchmarks::int_list(FLAGS_threads, {1, 2, 4, 8});
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {8, 16, 32});

    // IVFPQ add
    for (int M : Ms) {
        if (M > d)
            continue;
        for (int threads : thread_counts) {
            std::string name = "ivfpq_add/M:" + std::to_string(M) +
                    "/threads:" + std::to_string(threads);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_ivfpq_add, d, nb, nlist, M, threads);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // IVFFlat add
    for (int threads : thread_counts) {
        std::string name = "ivfflat_add/threads:" + std::to_string(threads);
        auto* b = benchmark::RegisterBenchmark(
                name.c_str(), bench_ivfflat_add, d, nb, nlist, threads);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // SIFT1M-based benchmarks (if dataset available). Times adding the real
    // base set to an index pre-trained on the real train set.
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
        int snt = (int)sift.nt;

        // IVFPQ add on the dataset
        std::vector<int> sift_Ms = benchmarks::int_list(FLAGS_M, {16});
        for (int M : sift_Ms) {
            if (M > sd)
                continue;
            for (int threads : thread_counts) {
                std::string name = ds + "/ivfpq_add/M:" + std::to_string(M) +
                        "/threads:" + std::to_string(threads);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivfpq_add_dataset,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        snb,
                        sd,
                        nlist,
                        M,
                        threads);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }

        // IVFFlat add on the dataset
        for (int threads : thread_counts) {
            std::string name =
                    ds + "/ivfflat_add/threads:" + std::to_string(threads);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_ivfflat_add_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    snb,
                    sd,
                    nlist,
                    threads);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
