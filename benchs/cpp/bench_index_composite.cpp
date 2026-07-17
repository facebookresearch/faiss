/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Composite Index Benchmarks

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/VectorTransform.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(nb, 100000, "database size");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");

static void bench_opq_ivfpqfs_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k) {
    int nt = std::min(nb, 50000);
    int M = 16;
    int nlist = 1024;
    int nprobe = 8;

    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    // OPQ transform + IVFPQFastScan
    OPQMatrix* opq = new OPQMatrix(d, M);
    IndexFlatL2* quantizer = new IndexFlatL2(d);
    IndexIVFPQFastScan* ivfpqfs =
            new IndexIVFPQFastScan(quantizer, d, nlist, M, 4);
    ivfpqfs->own_fields = true;
    IndexPreTransform index(opq, ivfpqfs);
    index.own_fields = true;
    index.verbose = false;

    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    ivfpqfs->nprobe = nprobe;

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

static void bench_refine_ivfpq_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k) {
    int nt = std::min(nb, 50000);
    int M = 16;
    int nlist = 256;
    int nprobe = 8;
    int k_factor = 4;

    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFPQ base_index(&quantizer, d, nlist, M, 8);
    base_index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    base_index.train(nt, xt.data());
    base_index.add(nb, xb.data());
    base_index.nprobe = nprobe;

    IndexRefineFlat index(&base_index, xb.data());
    index.k_factor = k_factor;

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
    state.counters["k_factor"] = k_factor;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant: accepts external data pointers, builds index once per call
static void bench_opq_ivfpqfs_search_dataset(
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
        int k) {
    int M = 16;
    int nlist = 1024;
    int nprobe = 8;

    // OPQ transform + IVFPQFastScan
    OPQMatrix* opq = new OPQMatrix(d, M);
    IndexFlatL2* quantizer = new IndexFlatL2(d);
    IndexIVFPQFastScan* ivfpqfs =
            new IndexIVFPQFastScan(quantizer, d, nlist, M, 4);
    ivfpqfs->own_fields = true;
    IndexPreTransform index(opq, ivfpqfs);
    index.own_fields = true;
    index.verbose = false;

    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);
    ivfpqfs->nprobe = nprobe;

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
    state.counters["k"] = k;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

// SIFT1M variant for IndexRefineFlat over IVFPQ
static void bench_refine_ivfpq_search_dataset(
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
        int k) {
    int M = 16;
    int nlist = 256;
    int nprobe = 8;
    int k_factor = 4;

    IndexFlatL2 quantizer(d);
    IndexIVFPQ base_index(&quantizer, d, nlist, M, 8);
    base_index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    base_index.train(nt, xt);
    base_index.add(nb, xb);
    base_index.nprobe = nprobe;

    IndexRefineFlat index(&base_index, xb);
    index.k_factor = k_factor;

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
    state.counters["k"] = k;
    state.counters["k_factor"] = k_factor;
    state.counters["dataset"] = 1;
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
            "composite index search benchmarks (OPQ + IVFPQFastScan, "
            "IndexRefineFlat over IVFPQ)",
            "--nq=100 --benchmark_filter='refine_ivfpq/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    int k = 10;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});

    // OPQ + IVFPQFastScan
    for (int nq : nqs) {
        std::string name = "opq_ivfpqfs/search/nq:" + std::to_string(nq);
        auto* b = benchmark::RegisterBenchmark(
                name.c_str(), bench_opq_ivfpqfs_search, d, nb, nq, k);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // IndexRefineFlat + IVFPQ
    for (int nq : nqs) {
        std::string name = "refine_ivfpq/search/nq:" + std::to_string(nq);
        auto* b = benchmark::RegisterBenchmark(
                name.c_str(), bench_refine_ivfpq_search, d, nb, nq, k);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
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
        int snt = (int)sift.nt;

        // Default to the full query set; the (expensive) train+add happens
        // once per registration, outside the timed loop.
        std::vector<int> sift_nqs =
                benchmarks::int_list(FLAGS_nq, {(int)sift.nq});
        for (int nq : sift_nqs) {
            if ((size_t)nq > sift.nq)
                continue;
            std::string name =
                    ds + "/opq_ivfpqfs/search/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_opq_ivfpqfs_search_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    snb,
                    sift.xq.data(),
                    nq,
                    sift.gt.data(),
                    sift.gt_k,
                    sd,
                    k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            name = ds + "/refine_ivfpq/search/nq:" + std::to_string(nq);
            b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_refine_ivfpq_search_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    snb,
                    sift.xq.data(),
                    nq,
                    sift.gt.data(),
                    sift.gt_k,
                    sd,
                    k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
