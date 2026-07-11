/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// PQ Code Distance Kernels

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        M,
        "",
        "comma-separated numbers of PQ sub-quantizers (default: 8,16,32,64)");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");

// Benchmark PQ distance computation via IndexPQ search
static void bench_pq_code_distance(
        benchmark::State& state,
        int d,
        int M,
        int nb,
        int nq) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexPQ index(d, M, 8);
    index.verbose = false;
    omp_set_num_threads(1);
    index.train(nt, xt.data());
    index.add(nb, xb.data());

    int k = 10;
    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)nq * nb * M * sizeof(uint8_t));
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
}

// SIFT1M variant: PQ code distances over codes of real base vectors
static void bench_pq_code_distance_dataset(
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
        int M) {
    IndexPQ index(d, M, 8);
    index.verbose = false;
    omp_set_num_threads(1);
    index.train(nt, xt);
    index.add(nb, xb);

    int k = 10;
    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)nq * nb * M * sizeof(uint8_t));
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["dataset"] = 1;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "PQ code distance computation kernels via IndexPQ search, "
            "on synthetic data and SIFT1M",
            "--M=16 --nq=100 --benchmark_filter='pq_code_distance/M:16/.*'");

    int d = 128;
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {8, 16, 32, 64});
    int nb = 100000;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});

    for (int M : Ms) {
        for (int nq : nqs) {
            std::string name = "pq_code_distance/M:" + std::to_string(M) +
                    "/nb:" + std::to_string(nb) + "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_pq_code_distance, d, M, nb, nq);
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

        // PQ code distances over the full real base, with recall
        std::vector<int> sift_nqs = benchmarks::int_list(FLAGS_nq, {100});
        for (int M : Ms) {
            if (sd % M != 0)
                continue;
            for (int nq : sift_nqs) {
                if ((size_t)nq > sift.nq)
                    continue;
                std::string name = ds +
                        "/pq_code_distance/M:" + std::to_string(M) +
                        "/nq:" + std::to_string(nq);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_pq_code_distance_dataset,
                        sift.xt.data(),
                        (int)sift.nt,
                        sift.xb.data(),
                        (int)sift.nb,
                        sift.xq.data(),
                        nq,
                        sift.gt.data(),
                        sift.gt_k,
                        sd,
                        M);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
