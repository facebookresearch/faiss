/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SIFT1M single-GPU search benchmarks.
//
// Two experiments on one GPU:
//   * exact search  — GpuIndexFlatL2 over a k = 1,2,4,...,1024 sweep (recall
//                     is 1 at every k);
//   * approximate   — IVF4096,PQ64 cloned to the GPU with useFloat16 lookup
//                     tables, over an nprobe = 1,2,...,512 sweep at k=100 with
//                     recall@{1,10,100}.
//
// Runs on SIFT1M when --data_dir points at it, otherwise on synthetic data
// with exact CPU-computed ground truth (so recall is still reported).

#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_factory.h>

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/StandardGpuResources.h>

#include "bench_gpu_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_uint32(d, 128, "synthetic dimension (ignored when a dataset loads)");
DEFINE_uint32(nb, 1000000, "synthetic database size");
DEFINE_uint32(nq, 10000, "synthetic query count");
DEFINE_uint32(nt, 100000, "synthetic train count");
DEFINE_int32(device, 0, "GPU device ordinal");
DEFINE_string(
        k,
        "",
        "comma-separated k values for exact search (default: 1,2,...,1024)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values for approximate search "
        "(default: 1,2,4,...,512)");
DEFINE_string(
        factory,
        "IVF4096,PQ64",
        "index_factory key for approximate search");

// Shared resources / working set (built once, reused across benches).
static std::unique_ptr<gpu::StandardGpuResources> g_res;
static benchmarks::GpuBenchData g_data;

// Exact search on a GpuIndexFlatL2 — recall is expected to be 1 at every k.
static void bench_gpu_flat_search(benchmark::State& state, int k) {
    const auto& data = g_data;
    gpu::GpuIndexFlatConfig config;
    config.device = FLAGS_device;
    gpu::GpuIndexFlatL2 index(g_res.get(), (int)data.d, config);
    index.add(data.nb, data.xb.data());

    std::vector<float> distances((size_t)data.nq * k);
    std::vector<int64_t> labels((size_t)data.nq * k);

    // Warmup
    index.search(data.nq, data.xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(
                data.nq, data.xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.nq);
    state.counters["d"] = data.d;
    state.counters["nb"] = data.nb;
    state.counters["nq"] = data.nq;
    state.counters["k"] = k;
    state.counters["recall"] = benchmarks::compute_recall_at(
            labels.data(), data.gt.data(), data.nq, k, data.gt_k);
}

// Approximate search on an IVFPQ index cloned to the GPU (useFloat16 lookup
// tables), swept over nprobe, recall@{1,10,100} at k=100.
static void bench_gpu_ivfpq_search(
        benchmark::State& state,
        const Index* cpu_index,
        int nprobe) {
    const auto& data = g_data;
    const int k = 100;

    gpu::GpuClonerOptions co;
    co.useFloat16 = true; // 64-byte PQ needs 16-bit lookup tables on GPU
    std::unique_ptr<Index> gpu_index(
            gpu::index_cpu_to_gpu(g_res.get(), FLAGS_device, cpu_index, &co));
    auto* ivf = dynamic_cast<gpu::GpuIndexIVF*>(gpu_index.get());
    if (ivf == nullptr) {
        state.SkipWithError("--factory did not clone to a GPU IVF index");
        return;
    }
    ivf->nprobe = nprobe;

    std::vector<float> distances((size_t)data.nq * k);
    std::vector<int64_t> labels((size_t)data.nq * k);

    // Warmup
    gpu_index->search(
            data.nq, data.xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        gpu_index->search(
                data.nq, data.xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.nq);
    state.counters["d"] = data.d;
    state.counters["nb"] = data.nb;
    state.counters["nq"] = data.nq;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    for (int rank : {1, 10, 100}) {
        state.counters["recall@" + std::to_string(rank)] =
                benchmarks::compute_recall_at(
                        labels.data(),
                        data.gt.data(),
                        data.nq,
                        rank,
                        data.gt_k);
    }
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "Single-GPU SIFT1M search: exact GpuIndexFlatL2 (k sweep) and "
            "approximate IVFPQ cloned to GPU (nprobe sweep)",
            "--factory=IVF4096,PQ64 --nprobe=1,8,64 "
            "--benchmark_filter='gpu_flat/.*'");

    g_res = std::make_unique<gpu::StandardGpuResources>();
    g_data = benchmarks::gpu_load_or_synth(
            FLAGS_data_dir,
            FLAGS_train_file,
            FLAGS_base_file,
            FLAGS_query_file,
            FLAGS_gt_file,
            FLAGS_d,
            FLAGS_nb,
            FLAGS_nq,
            FLAGS_nt);
    const std::string tag = g_data.tag;

    // Exact search sweep (default: 1,2,4,...,1024).
    std::vector<int> ks = benchmarks::int_list(
            FLAGS_k, {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
    for (int k : ks) {
        if ((size_t)k > g_data.nb)
            continue;
        std::string name = tag + "/gpu_flat/search/k:" + std::to_string(k);
        auto* b = benchmark::RegisterBenchmark(
                name.c_str(), bench_gpu_flat_search, k);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // Approximate search: train/add the CPU index once, clone per nprobe.
    static std::unique_ptr<Index> cpu_index(
            index_factory((int)g_data.d, FLAGS_factory.c_str(), METRIC_L2));
    cpu_index->train(g_data.nt, g_data.xt.data());
    cpu_index->add(g_data.nb, g_data.xb.data());

    std::vector<int> nprobes = benchmarks::int_list(
            FLAGS_nprobe, {1, 2, 4, 8, 16, 32, 64, 128, 256, 512});
    for (int nprobe : nprobes) {
        std::string name =
                tag + "/gpu_ivfpq/search/nprobe:" + std::to_string(nprobe);
        auto* b = benchmark::RegisterBenchmark(
                name.c_str(), bench_gpu_ivfpq_search, cpu_index.get(), nprobe);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    g_res.reset();
}
