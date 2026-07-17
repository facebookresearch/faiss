/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// hybrid CPU/GPU IVF search benchmarks.
//
// Compares three search topologies that split the two phases of an IVF
// search — coarse quantization (assign queries to nprobe lists) and
// inverted-list scanning — between CPU and GPU. Covers three representative
// operating points on a single GPU:
//   * cpu          — everything on the CPU (baseline);
//   * gpu          — whole index cloned to the GPU;
//   * hybrid       — coarse quantization on a GpuIndexFlatL2, list scanning on
//                    the CPU via IndexIVF::search_preassigned.
// Each is swept over nprobe with recall@{1,10,100}.
//
// Runs on the dataset in --data_dir when present, otherwise synthetic data
// with exact CPU-computed ground truth.

#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
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
DEFINE_uint32(threads, 0, "OpenMP threads for CPU phases (0 = OMP default)");
DEFINE_string(factory, "IVF1024,Flat", "index_factory key");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values (default: 1,4,16,64,256)");
DEFINE_uint32(k, 100, "number of nearest neighbors");

static std::unique_ptr<gpu::StandardGpuResources> g_res;
static benchmarks::GpuBenchData g_data;

static void apply_threads() {
    if (FLAGS_threads > 0) {
        omp_set_num_threads(FLAGS_threads);
    }
}

static void fill_counters(benchmark::State& state, int nprobe, int k) {
    const auto& data = g_data;
    state.SetItemsProcessed(state.iterations() * data.nq);
    state.counters["d"] = data.d;
    state.counters["nb"] = data.nb;
    state.counters["nq"] = data.nq;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
}

static void fill_recall(benchmark::State& state, const int64_t* labels, int k) {
    const auto& data = g_data;
    for (int rank : {1, 10, 100}) {
        if (rank > k)
            continue;
        state.counters["recall@" + std::to_string(rank)] =
                benchmarks::compute_recall_at(
                        labels, data.gt.data(), data.nq, rank, data.gt_k);
    }
}

// Pure-CPU baseline.
static void bench_cpu(
        benchmark::State& state,
        const IndexIVF* cpu_index,
        int nprobe) {
    const auto& data = g_data;
    const int k = FLAGS_k;
    apply_threads();

    // The CPU index is shared across cases; nprobe is set directly on it
    // (via const_cast) before each case's searches. Cases run sequentially,
    // so the mutation cannot race, and every case sets its own nprobe.
    IndexIVF* idx = const_cast<IndexIVF*>(cpu_index);
    idx->nprobe = nprobe;

    std::vector<float> distances((size_t)data.nq * k);
    std::vector<int64_t> labels((size_t)data.nq * k);

    idx->search(data.nq, data.xq.data(), k, distances.data(), labels.data());
    for (auto _ : state) {
        idx->search(
                data.nq, data.xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    fill_counters(state, nprobe, k);
    fill_recall(state, labels.data(), k);
}

// Whole index on the GPU.
static void bench_gpu(
        benchmark::State& state,
        const IndexIVF* cpu_index,
        int nprobe) {
    const auto& data = g_data;
    const int k = FLAGS_k;

    gpu::GpuClonerOptions co;
    std::unique_ptr<Index> gpu_index(
            gpu::index_cpu_to_gpu(g_res.get(), FLAGS_device, cpu_index, &co));
    auto* gpu_ivf = dynamic_cast<gpu::GpuIndexIVF*>(gpu_index.get());
    if (gpu_ivf == nullptr) {
        state.SkipWithError("--factory did not clone to a GPU IVF index");
        return;
    }
    gpu_ivf->nprobe = nprobe;

    std::vector<float> distances((size_t)data.nq * k);
    std::vector<int64_t> labels((size_t)data.nq * k);

    gpu_index->search(
            data.nq, data.xq.data(), k, distances.data(), labels.data());
    for (auto _ : state) {
        gpu_index->search(
                data.nq, data.xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    fill_counters(state, nprobe, k);
    fill_recall(state, labels.data(), k);
}

// Hybrid: GPU coarse quantizer (flat), CPU inverted-list scan. The queries are
// assigned to nprobe lists on the GPU, then IndexIVF::search_preassigned scans
// those lists on the CPU.
static void bench_hybrid(
        benchmark::State& state,
        const IndexIVF* cpu_index,
        int nprobe) {
    const auto& data = g_data;
    const int k = FLAGS_k;
    apply_threads();

    // Move a copy of the (flat) coarse quantizer to the GPU.
    gpu::GpuClonerOptions qco;
    std::unique_ptr<Index> gpu_quantizer(
            gpu::index_cpu_to_gpu(
                    g_res.get(), FLAGS_device, cpu_index->quantizer, &qco));

    IndexIVF* idx = const_cast<IndexIVF*>(cpu_index);
    idx->nprobe = nprobe;

    std::vector<float> coarse_dis((size_t)data.nq * nprobe);
    std::vector<int64_t> assign((size_t)data.nq * nprobe);
    std::vector<float> distances((size_t)data.nq * k);
    std::vector<int64_t> labels((size_t)data.nq * k);

    auto run = [&]() {
        // Phase 1: coarse quantization on the GPU.
        gpu_quantizer->search(
                data.nq,
                data.xq.data(),
                nprobe,
                coarse_dis.data(),
                assign.data());
        // Phase 2: inverted-list scan on the CPU.
        idx->search_preassigned(
                data.nq,
                data.xq.data(),
                k,
                assign.data(),
                coarse_dis.data(),
                distances.data(),
                labels.data(),
                /*store_pairs=*/false);
    };

    run(); // warmup
    for (auto _ : state) {
        run();
        benchmark::DoNotOptimize(distances[0]);
    }
    fill_counters(state, nprobe, k);
    fill_recall(state, labels.data(), k);
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "Hybrid CPU/GPU IVF search: cpu, gpu, and "
            "gpu-coarse-quantizer + cpu-scan, over an nprobe sweep",
            "--factory=IVF1024,Flat --nprobe=1,16,64 "
            "--benchmark_filter='.*/hybrid/.*'");

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

    // Build the shared CPU IVF index once. The factory must produce an
    // IndexIVF with a flat coarse quantizer (default IVF1024,Flat).
    static std::unique_ptr<Index> base_index(
            index_factory((int)g_data.d, FLAGS_factory.c_str(), METRIC_L2));
    base_index->train(g_data.nt, g_data.xt.data());
    base_index->add(g_data.nb, g_data.xb.data());
    auto* cpu_ivf = dynamic_cast<IndexIVF*>(base_index.get());
    if (cpu_ivf == nullptr) {
        fprintf(stderr,
                "bench_gpu_index_ivf_hybrid: --factory=%s did not produce an "
                "IndexIVF\n",
                FLAGS_factory.c_str());
        return 1;
    }

    auto reg = [&](const std::string& name, auto fn, auto... args) {
        auto* b = benchmark::RegisterBenchmark(name.c_str(), fn, args...);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    };

    std::vector<int> nprobes =
            benchmarks::int_list(FLAGS_nprobe, {1, 4, 16, 64, 256});
    for (int nprobe : nprobes) {
        std::string sp = "/nprobe:" + std::to_string(nprobe);
        reg(tag + "/cpu/search" + sp, bench_cpu, cpu_ivf, nprobe);
        reg(tag + "/gpu/search" + sp, bench_gpu, cpu_ivf, nprobe);
        reg(tag + "/hybrid/search" + sp, bench_hybrid, cpu_ivf, nprobe);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    g_res.reset();
}
