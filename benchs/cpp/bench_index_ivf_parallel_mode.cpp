/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IVF Parallel Mode Benchmarks
// Benchmarks IVF search with different parallel_mode settings.
// parallel_mode=0: parallelize over queries
// parallel_mode=2: parallelize over probes (useful for single-query,
// high-nprobe).

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 4, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        nq,
        "",
        "comma-separated query batch sizes for the multi-query group "
        "(default: 10,100; sift1m multi-query group: 100; "
        "single-query groups are fixed at nq=1)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values "
        "(default: 64,256; multi-query group: 16,64)");
DEFINE_string(
        pmode,
        "",
        "comma-separated IVF parallel_mode values (default: 0,2)");

static void bench_ivf_parallel_mode(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int nlist,
        int nprobe,
        int k,
        int parallel_mode,
        int threads) {
    int nt = std::max(std::min(nb, 50000), 39 * nlist);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist);
    index.verbose = false;
    omp_set_num_threads(threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;
    index.parallel_mode = parallel_mode;

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
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["parallel_mode"] = parallel_mode;
    state.counters["threads"] = threads;
}

// SIFT1M variant: accepts external data pointers, builds index once per call
static void bench_ivf_parallel_mode_dataset(
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
        int nlist,
        int nprobe,
        int k,
        int parallel_mode,
        int threads) {
    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist);
    index.verbose = false;
    omp_set_num_threads(threads);
    index.train(nt, xt);
    index.add(nb, xb);
    index.nprobe = nprobe;
    index.parallel_mode = parallel_mode;

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
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["parallel_mode"] = parallel_mode;
    state.counters["threads"] = threads;
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
            "IVFFlat search with different parallel_mode settings "
            "(0: parallelize over queries, 2: over probes)",
            "--nprobe=256 --pmode=2 "
            "--benchmark_filter='ivf_parallel_mode/nq:1/.*'");

    int d = 128;
    int nb = 500000;
    int nlist = 4096;
    int k = 10;
    int threads = FLAGS_threads;

    std::vector<int> pmodes = benchmarks::int_list(FLAGS_pmode, {0, 2});

    // Single-query, high-nprobe scenario (where pmode=2 shines)
    std::vector<int> nprobes_single =
            benchmarks::int_list(FLAGS_nprobe, {64, 256});
    for (int nprobe : nprobes_single) {
        for (int parallel_mode : pmodes) {
            std::string name =
                    "ivf_parallel_mode/nq:1/nprobe:" + std::to_string(nprobe) +
                    "/pmode:" + std::to_string(parallel_mode);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_ivf_parallel_mode,
                    d,
                    nb,
                    1,
                    nlist,
                    nprobe,
                    k,
                    parallel_mode,
                    threads);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // Multi-query scenario (where pmode=0 should be better)
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {10, 100});
    std::vector<int> nprobes_multi =
            benchmarks::int_list(FLAGS_nprobe, {16, 64});
    for (int nq : nqs) {
        for (int nprobe : nprobes_multi) {
            for (int parallel_mode : pmodes) {
                std::string name =
                        "ivf_parallel_mode/nq:" + std::to_string(nq) +
                        "/nprobe:" + std::to_string(nprobe) +
                        "/pmode:" + std::to_string(parallel_mode);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivf_parallel_mode,
                        d,
                        nb,
                        nq,
                        nlist,
                        nprobe,
                        k,
                        parallel_mode,
                        threads);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
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
        int sd = (int)sift.d;
        int snb = (int)sift.nb;
        int snt = (int)sift.nt;

        // Single-query, high-nprobe scenario on the dataset
        for (int nprobe : nprobes_single) {
            for (int parallel_mode : pmodes) {
                std::string name = ds + "/ivf_parallel_mode/nq:1/nprobe:" +
                        std::to_string(nprobe) +
                        "/pmode:" + std::to_string(parallel_mode);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivf_parallel_mode_dataset,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        snb,
                        sift.xq.data(),
                        1,
                        sift.gt.data(),
                        sift.gt_k,
                        sd,
                        nlist,
                        nprobe,
                        k,
                        parallel_mode,
                        threads);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }

        // Multi-query scenario on the dataset
        std::vector<int> sift_nqs = benchmarks::int_list(FLAGS_nq, {100});
        for (int nq : sift_nqs) {
            if ((size_t)nq > sift.nq)
                continue;
            for (int nprobe : nprobes_multi) {
                for (int parallel_mode : pmodes) {
                    std::string name = ds +
                            "/ivf_parallel_mode/nq:" + std::to_string(nq) +
                            "/nprobe:" + std::to_string(nprobe) +
                            "/pmode:" + std::to_string(parallel_mode);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_ivf_parallel_mode_dataset,
                            sift.xt.data(),
                            snt,
                            sift.xb.data(),
                            snb,
                            sift.xq.data(),
                            nq,
                            sift.gt.data(),
                            sift.gt_k,
                            sd,
                            nlist,
                            nprobe,
                            k,
                            parallel_mode,
                            threads);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
