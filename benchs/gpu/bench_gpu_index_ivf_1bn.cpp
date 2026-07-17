/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// large-scale IVF add + multi-GPU search benchmarks.
//
// Measures the core of a billion-scale GPU workflow on a tractable working set:
//   * ADD    — build a trained IVF index on the CPU coarse quantizer, then
//              clone-and-add to one or all GPUs (replicated or sharded);
//   * SEARCH — nprobe sweep over the GPU index with recall@{1,10,100}.
//
// The index type is an index_factory key (default IVF4096,PQ64); useFloat16
// and precomputed-tables cloner options are exposed. Multi-GPU cases register
// only when more than one GPU is present.
//
// Runs on the dataset in --data_dir when present (BIGANN/Deep1B/SIFT-style),
// otherwise on synthetic data with exact CPU-computed ground truth.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_factory.h>

#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

#include "bench_gpu_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_uint32(d, 128, "synthetic dimension (ignored when a dataset loads)");
DEFINE_uint32(nb, 1000000, "synthetic database size");
DEFINE_uint32(nq, 10000, "synthetic query count");
DEFINE_uint32(nt, 100000, "synthetic train count");
DEFINE_string(factory, "IVF4096,PQ64", "index_factory key");
DEFINE_int32(ngpu, -1, "number of GPUs to use for multi-GPU cases (-1 = all)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values (default: 1,4,16,64,256)");
DEFINE_uint32(nnn, 100, "number of neighbors to search (k)");
DEFINE_bool(float16, false, "use 16-bit float lookup tables / storage on GPU");
DEFINE_bool(noptables, false, "disable IVFPQ precomputed tables");

static std::unique_ptr<gpu::StandardGpuResources> g_res;
static std::vector<std::unique_ptr<gpu::StandardGpuResources>> g_multi_res;
static benchmarks::GpuBenchData g_data;

static gpu::GpuMultipleClonerOptions make_cloner_opts(bool shard) {
    gpu::GpuMultipleClonerOptions co;
    co.useFloat16 = FLAGS_float16;
    co.usePrecomputed = !FLAGS_noptables;
    co.shard = shard;
    return co;
}

// ADD: time cloning the trained CPU index to GPU(s) and adding all base
// vectors. "single" clones to one GPU; "sharded"/"replicated" span all GPUs.
static void bench_gpu_add(
        benchmark::State& state,
        const Index* trained_cpu_index,
        int ngpu,
        bool shard) {
    const auto& data = g_data;
    gpu::GpuMultipleClonerOptions co = make_cloner_opts(shard);

    std::vector<gpu::GpuResourcesProvider*> provs;
    std::vector<int> devs;
    for (int i = 0; i < ngpu; i++) {
        provs.push_back(g_multi_res[i].get());
        devs.push_back(i);
    }

    for (auto _ : state) {
        state.PauseTiming();
        // Fresh empty clone of the trained index each iteration.
        std::unique_ptr<Index> gpu_index(
                ngpu == 1 ? gpu::index_cpu_to_gpu(
                                    g_res.get(), 0, trained_cpu_index, &co)
                          : gpu::index_cpu_to_gpu_multiple(
                                    provs, devs, trained_cpu_index, &co));
        state.ResumeTiming();

        gpu_index->add(data.nb, data.xb.data());
        benchmark::DoNotOptimize(gpu_index->ntotal);
    }
    state.SetItemsProcessed(state.iterations() * data.nb);
    state.counters["d"] = data.d;
    state.counters["nb"] = data.nb;
    state.counters["ngpu"] = ngpu;
    state.counters["sharded"] = shard;
    state.counters["float16"] = FLAGS_float16;
}

// SEARCH: clone the trained+populated CPU index to GPU(s) once, sweep nprobe.
static void bench_gpu_search(
        benchmark::State& state,
        const Index* populated_cpu_index,
        int ngpu,
        bool shard,
        int nprobe) {
    const auto& data = g_data;
    const int k = FLAGS_nnn;
    gpu::GpuMultipleClonerOptions co = make_cloner_opts(shard);

    std::vector<gpu::GpuResourcesProvider*> provs;
    std::vector<int> devs;
    for (int i = 0; i < ngpu; i++) {
        provs.push_back(g_multi_res[i].get());
        devs.push_back(i);
    }
    std::unique_ptr<Index> gpu_index(
            ngpu == 1 ? gpu::index_cpu_to_gpu(
                                g_res.get(), 0, populated_cpu_index, &co)
                      : gpu::index_cpu_to_gpu_multiple(
                                provs, devs, populated_cpu_index, &co));

    // nprobe is a shared IVF parameter; GpuParameterSpace handles shards/
    // replicas uniformly.
    gpu::GpuParameterSpace ps;
    ps.set_index_parameter(gpu_index.get(), "nprobe", nprobe);

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
    state.counters["ngpu"] = ngpu;
    state.counters["sharded"] = shard;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    for (int rank : {1, 10, 100}) {
        if (rank > k)
            continue;
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
            "Large-scale IVF add + multi-GPU search: clone-and-add and "
            "nprobe search sweep, single/sharded/replicated GPUs",
            "--factory=IVF4096,PQ64 --float16 --nprobe=1,16,64 "
            "--benchmark_filter='.*/search/.*'");

    int total_gpus = gpu::getNumDevices();
    if (total_gpus < 1) {
        fprintf(stderr, "bench_gpu_index_ivf_1bn: no GPU available\n");
        return 1;
    }
    int ngpu = FLAGS_ngpu > 0 ? std::min(FLAGS_ngpu, total_gpus) : total_gpus;

    g_res = std::make_unique<gpu::StandardGpuResources>();
    for (int i = 0; i < ngpu; i++) {
        g_multi_res.push_back(std::make_unique<gpu::StandardGpuResources>());
    }

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

    // Trained (empty) CPU index for ADD; populated CPU index for SEARCH.
    static std::unique_ptr<Index> trained_index(
            index_factory((int)g_data.d, FLAGS_factory.c_str(), METRIC_L2));
    trained_index->train(g_data.nt, g_data.xt.data());

    static std::unique_ptr<Index> populated_index(
            index_factory((int)g_data.d, FLAGS_factory.c_str(), METRIC_L2));
    populated_index->train(g_data.nt, g_data.xt.data());
    populated_index->add(g_data.nb, g_data.xb.data());

    auto reg = [&](const std::string& name, auto fn, auto... args) {
        auto* b = benchmark::RegisterBenchmark(name.c_str(), fn, args...);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    };

    std::vector<int> nprobes =
            benchmarks::int_list(FLAGS_nprobe, {1, 4, 16, 64, 256});

    // Topologies: single GPU always; sharded and replicated when ngpu > 1.
    struct Topo {
        std::string label;
        int ngpu;
        bool shard;
    };
    std::vector<Topo> topos = {{"gpu1", 1, false}};
    if (ngpu > 1) {
        topos.push_back(
                {"gpu" + std::to_string(ngpu) + "_sharded", ngpu, true});
        topos.push_back(
                {"gpu" + std::to_string(ngpu) + "_replicated", ngpu, false});
    }

    for (const auto& t : topos) {
        reg(tag + "/" + t.label + "/add",
            bench_gpu_add,
            trained_index.get(),
            t.ngpu,
            t.shard);
        for (int nprobe : nprobes) {
            reg(tag + "/" + t.label +
                        "/search/nprobe:" + std::to_string(nprobe),
                bench_gpu_search,
                populated_index.get(),
                t.ngpu,
                t.shard,
                nprobe);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    g_multi_res.clear();
    g_res.reset();
}
