/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// GpuIndexIVFFlat classical vs cuVS comparison.
//
// Benchmarks the three IVFFlat operations — TRAIN, ADD and SEARCH — each run
// twice, once with the classical GPU path (use_cuvs=false) and once with cuVS
// (use_cuvs=true), so the two implementations sit side by side in the output.
//
// When faiss is built without cuVS (FAISS_ENABLE_CUVS=OFF), the use_cuvs=true
// cases still register but are skipped at runtime with a SkipWithError so the
// bench is usable on any GPU build; the classical cases always run.
//
// Runs on SIFT1M when --data_dir points at it, otherwise on synthetic data.

#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
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
DEFINE_uint32(nlist, 1024, "number of IVF centroids");
DEFINE_string(nprobe, "", "comma-separated nprobe values (default: 10)");
DEFINE_uint32(k, 10, "number of nearest neighbors");

static std::unique_ptr<gpu::StandardGpuResources> g_res;
static benchmarks::GpuBenchData g_data;

// Guard cuVS-requiring cases when faiss was built without cuVS support.
static bool cuvs_unavailable(benchmark::State& state, bool use_cuvs) {
#if !defined USE_NVIDIA_CUVS
    if (use_cuvs) {
        state.SkipWithError("faiss built without cuVS (FAISS_ENABLE_CUVS=OFF)");
        return true;
    }
#endif
    (void)use_cuvs;
    return false;
}

static const char* impl_label(bool use_cuvs) {
    return use_cuvs ? "cuvs" : "classical";
}

// TRAIN: time GpuIndexIVFFlat::train on the train set.
static void bench_ivfflat_train(benchmark::State& state, bool use_cuvs) {
    if (cuvs_unavailable(state, use_cuvs))
        return;
    const auto& data = g_data;
    for (auto _ : state) {
        gpu::GpuIndexIVFFlatConfig config;
        config.device = FLAGS_device;
        config.use_cuvs = use_cuvs;
        gpu::GpuIndexIVFFlat index(
                g_res.get(), (int)data.d, FLAGS_nlist, METRIC_L2, config);
        index.train(data.nt, data.xt.data());
        benchmark::DoNotOptimize(index.is_trained);
    }
    state.SetItemsProcessed(state.iterations() * data.nt);
    state.counters["d"] = data.d;
    state.counters["nlist"] = FLAGS_nlist;
    state.counters["nt"] = data.nt;
    state.counters["cuvs"] = use_cuvs;
}

// ADD: build a trained index, then time add() (index reset each iteration).
static void bench_ivfflat_add(benchmark::State& state, bool use_cuvs) {
    if (cuvs_unavailable(state, use_cuvs))
        return;
    const auto& data = g_data;
    for (auto _ : state) {
        state.PauseTiming();
        gpu::GpuIndexIVFFlatConfig config;
        config.device = FLAGS_device;
        config.use_cuvs = use_cuvs;
        gpu::GpuIndexIVFFlat index(
                g_res.get(), (int)data.d, FLAGS_nlist, METRIC_L2, config);
        index.train(data.nt, data.xt.data());
        state.ResumeTiming();

        index.add(data.nb, data.xb.data());
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * data.nb);
    state.counters["d"] = data.d;
    state.counters["nlist"] = FLAGS_nlist;
    state.counters["nb"] = data.nb;
    state.counters["cuvs"] = use_cuvs;
}

// SEARCH: clone a trained+populated CPU IVFFlat to the GPU, time search, and
// report recall. Both impls share identical IVF centroids and lists (the CPU
// index is built once) so differences are purely in the GPU codepath.
static void bench_ivfflat_search(
        benchmark::State& state,
        const IndexIVFFlat* cpu_index,
        int nprobe,
        bool use_cuvs) {
    if (cuvs_unavailable(state, use_cuvs))
        return;
    const auto& data = g_data;
    const int k = FLAGS_k;

    gpu::GpuClonerOptions co;
    co.use_cuvs = use_cuvs;
    std::unique_ptr<Index> gpu_index(
            gpu::index_cpu_to_gpu(g_res.get(), FLAGS_device, cpu_index, &co));
    dynamic_cast<gpu::GpuIndexIVF*>(gpu_index.get())->nprobe = nprobe;

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
    state.counters["nlist"] = FLAGS_nlist;
    state.counters["nb"] = data.nb;
    state.counters["nq"] = data.nq;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["cuvs"] = use_cuvs;
    state.counters["recall"] = benchmarks::compute_recall_at(
            labels.data(), data.gt.data(), data.nq, k, data.gt_k);
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "GpuIndexIVFFlat train/add/search: classical GPU vs cuVS",
            "--nlist=1024 --nprobe=10 --benchmark_filter='.*/search/.*'");

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

    auto reg = [&](const std::string& name, auto fn, auto... args) {
        auto* b = benchmark::RegisterBenchmark(name.c_str(), fn, args...);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    };

    // TRAIN and ADD: classical and cuVS.
    for (bool use_cuvs : {false, true}) {
        reg(tag + "/ivfflat/train/" + impl_label(use_cuvs),
            bench_ivfflat_train,
            use_cuvs);
        reg(tag + "/ivfflat/add/" + impl_label(use_cuvs),
            bench_ivfflat_add,
            use_cuvs);
    }

    // SEARCH: build the CPU reference index once, clone per (nprobe, impl).
    static IndexFlatL2 quantizer((int)g_data.d);
    static IndexIVFFlat cpu_index(&quantizer, (int)g_data.d, FLAGS_nlist);
    cpu_index.train(g_data.nt, g_data.xt.data());
    cpu_index.add(g_data.nb, g_data.xb.data());

    std::vector<int> nprobes = benchmarks::int_list(FLAGS_nprobe, {10});
    for (int nprobe : nprobes) {
        for (bool use_cuvs : {false, true}) {
            reg(tag + "/ivfflat/search/nprobe:" + std::to_string(nprobe) + "/" +
                        impl_label(use_cuvs),
                bench_ivfflat_search,
                &cpu_index,
                nprobe,
                use_cuvs);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    g_res.reset();
}
