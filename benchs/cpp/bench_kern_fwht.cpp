/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Fast Walsh-Hadamard Transform (FWHT) Benchmarks
// Benchmarks HadamardRotation vs RandomRotationMatrix:
//   * apply() throughput (hadamard_rotation / random_rotation). The dim
//     sweep includes non-power-of-two dims: {384,768,1536,3072,6144}
//     (HadamardRotation pads to the next power of two internally).
//   * recall@1 with an IVF index built behind an HR/RR/none pre-transform
//     prefix (fwht_ivf_recall): dims {64,128,256,768,1024,2048,4096},
//     nlist=64, nprobe=8, k=1, n=10000, nq=200.
//
// The apply-throughput sweep uses uniform seed-12345 data. The recall
// benchmarks use Gaussian data (float_randn) with seeds 42/4242.

#include <algorithm>
#include <memory>
#include <vector>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/VectorTransform.h>
#include <faiss/index_factory.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        d,
        "",
        "comma-separated vector dimensions for the apply-speed sweep "
        "(default: 64,128,256,384,512,768,1024,1536,2048,3072,4096,6144,"
        "8192)");
DEFINE_string(
        recall_d,
        "",
        "comma-separated dimensions for the fwht_ivf_recall sweep "
        "(default: 64,128,256,768,1024,2048,4096)");
DEFINE_string(
        nlist,
        "",
        "comma-separated IVF nlist values for fwht_ivf_recall (default: 64)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated IVF nprobe values for fwht_ivf_recall (default: 8)");

static void bench_hadamard_rotation(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d * n);
    float_rand(x.data(), d * n, 12345);

    HadamardRotation hr(d, 42);
    omp_set_num_threads(1);

    // Warmup
    std::vector<float> out(d * n);
    hr.apply_noalloc(n, x.data(), out.data());

    for (auto _ : state) {
        hr.apply_noalloc(n, x.data(), out.data());
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)n * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

static void bench_random_rotation(benchmark::State& state, int d, int n) {
    AlignedTable<float> x(d * n);
    float_rand(x.data(), d * n, 12345);

    RandomRotationMatrix rr(d, d);
    rr.init(42);
    omp_set_num_threads(1);

    // Warmup
    std::vector<float> out(d * n);
    rr.apply_noalloc(n, x.data(), out.data());

    for (auto _ : state) {
        rr.apply_noalloc(n, x.data(), out.data());
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(
            state.iterations() * (int64_t)n * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["n"] = n;
}

// Recall@1: build an IVF index behind an
// HR (HadamardRotation) / RR (RandomRotation) / none pre-transform prefix and
// measure recall@1 vs brute-force ground truth. Uses Gaussian data
// (float_randn), n=10000, nq=200, nlist=64, nprobe=8, k=1. The prefix is
// applied via the index_factory string, mirroring the Python version.
static void bench_fwht_ivf_recall(
        benchmark::State& state,
        int d,
        int n,
        int nq,
        int nlist,
        int nprobe,
        const std::string& prefix) {
    AlignedTable<float> xb(d * n);
    AlignedTable<float> xq(d * nq);
    float_randn(xb.data(), d * n, 42);
    float_randn(xq.data(), d * nq, 4242);

    // Brute-force ground truth (no transform).
    std::vector<float> gt_dist(nq);
    std::vector<int64_t> gt_labels(nq);
    {
        IndexFlatL2 gt_index(d);
        gt_index.add(n, xb.data());
        gt_index.search(nq, xq.data(), 1, gt_dist.data(), gt_labels.data());
    }

    std::string factory_str = prefix.empty()
            ? ("IVF" + std::to_string(nlist) + ",Flat")
            : (prefix + ",IVF" + std::to_string(nlist) + ",Flat");
    std::unique_ptr<Index> index(index_factory(d, factory_str.c_str()));
    omp_set_num_threads(1);
    index->train(n, xb.data());
    index->add(n, xb.data());
    ivflib::extract_index_ivf(index.get())->nprobe = nprobe;

    std::vector<float> distances(nq);
    std::vector<int64_t> labels(nq);

    // Warmup
    index->search(nq, xq.data(), 1, distances.data(), labels.data());

    for (auto _ : state) {
        index->search(nq, xq.data(), 1, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);

    // Recall@1: fraction of queries whose top-1 matches the ground truth.
    size_t hits = 0;
    for (int i = 0; i < nq; i++) {
        if (labels[i] == gt_labels[i]) {
            hits++;
        }
    }
    state.counters["recall"] = (double)hits / nq;
    state.counters["d"] = d;
    state.counters["n"] = n;
    state.counters["nq"] = nq;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
}

// Dataset-backed variant of fwht_ivf_recall: the same HR/RR/none pre-transform
// prefixes, but on real train/base/query vectors. The base set may be
// subsampled (the full SIFT1M base is ~100x the synthetic n=10000 scale); when
// it is, exact ground truth is recomputed on the subsample with an IndexFlat,
// mirroring the synthetic benchmark's brute-force reference. When the full
// base set is searched, recall is measured against the dataset ground truth.
static void bench_fwht_ivf_recall_dataset(
        benchmark::State& state,
        const benchmarks::DatasetSIFT1M* sift,
        int n,
        int nq,
        int nlist,
        int nprobe,
        const std::string& prefix) {
    const int d = (int)sift->d;
    const float* xb = sift->xb.data();
    const float* xq = sift->xq.data();
    const bool full_base = (size_t)n == sift->nb;

    // Brute-force ground truth on the subsampled base set (unused when the
    // full base set is searched; the dataset ground truth applies then).
    std::vector<int64_t> gt_labels;
    if (!full_base) {
        gt_labels.resize(nq);
        std::vector<float> gt_dist(nq);
        IndexFlatL2 gt_index(d);
        gt_index.add(n, xb);
        gt_index.search(nq, xq, 1, gt_dist.data(), gt_labels.data());
    }

    std::string factory_str = prefix.empty()
            ? ("IVF" + std::to_string(nlist) + ",Flat")
            : (prefix + ",IVF" + std::to_string(nlist) + ",Flat");
    std::unique_ptr<Index> index(index_factory(d, factory_str.c_str()));
    omp_set_num_threads(1);
    index->train((idx_t)sift->nt, sift->xt.data());
    index->add(n, xb);
    ivflib::extract_index_ivf(index.get())->nprobe = nprobe;

    std::vector<float> distances(nq);
    std::vector<int64_t> labels(nq);

    // Warmup
    index->search(nq, xq, 1, distances.data(), labels.data());

    for (auto _ : state) {
        index->search(nq, xq, 1, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);

    // Recall@1 vs the applicable ground truth.
    if (full_base) {
        state.counters["recall"] = benchmarks::compute_recall_at(
                labels.data(), sift->gt.data(), nq, 1, sift->gt_k);
    } else {
        size_t hits = 0;
        for (int i = 0; i < nq; i++) {
            if (labels[i] == gt_labels[i]) {
                hits++;
            }
        }
        state.counters["recall"] = (double)hits / nq;
    }
    state.counters["dataset"] = 1;
    state.counters["d"] = d;
    state.counters["n"] = n;
    state.counters["nq"] = nq;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "fast Walsh-Hadamard transform (HadamardRotation) vs BLAS-based "
            "RandomRotationMatrix, plus HR/RR/none IVF recall@1",
            "--d=1024,2048 --benchmark_filter='hadamard_rotation/.*'");

    // Apply-speed dims: powers of two plus the non-power-of-two dims that
    // HadamardRotation pads internally.
    std::vector<int> dims = benchmarks::int_list(
            FLAGS_d,
            {64,
             128,
             256,
             384,
             512,
             768,
             1024,
             1536,
             2048,
             3072,
             4096,
             6144,
             8192});
    int n = 10000;

    for (int d : dims) {
        std::string suffix =
                "/d:" + std::to_string(d) + "/n:" + std::to_string(n);

        auto* b = benchmark::RegisterBenchmark(
                ("hadamard_rotation" + suffix).c_str(),
                bench_hadamard_rotation,
                d,
                n);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("random_rotation" + suffix).c_str(),
                bench_random_rotation,
                d,
                n);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // Recall@1 with HR/RR/none pre-transform + IVF.
    std::vector<int> recall_dims = benchmarks::int_list(
            FLAGS_recall_d, {64, 128, 256, 768, 1024, 2048, 4096});
    std::vector<int> nlists = benchmarks::int_list(FLAGS_nlist, {64});
    std::vector<int> nprobes = benchmarks::int_list(FLAGS_nprobe, {8});
    int recall_n = 10000;
    int recall_nq = 200;
    for (int d : recall_dims) {
        for (int nlist : nlists) {
            for (int nprobe : nprobes) {
                for (const auto& prefix :
                     {std::string("HR"),
                      std::string("RR"),
                      std::string("none")}) {
                    std::string name = "fwht_ivf_recall/prefix:" + prefix +
                            "/d:" + std::to_string(d) +
                            "/nlist:" + std::to_string(nlist) +
                            "/nprobe:" + std::to_string(nprobe);
                    auto* rb = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_fwht_ivf_recall,
                            d,
                            recall_n,
                            recall_nq,
                            nlist,
                            nprobe,
                            prefix == "none" ? std::string() : prefix);
                    if (FLAGS_iterations > 0)
                        rb->Iterations(FLAGS_iterations);
                }
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
        // The full SIFT1M base (1M vectors) is ~100x the synthetic recall
        // benchmark's scale, so subsample the base set to at most 100k
        // vectors to keep the per-variant index build cheap; ground truth
        // handling adapts accordingly (see bench_fwht_ivf_recall_dataset).
        // Query count mirrors the synthetic benchmark's nq=200.
        int ds_n = (int)std::min(sift.nb, (size_t)100000);
        int ds_nq = (int)std::min(sift.nq, (size_t)200);
        for (int nlist : nlists) {
            for (int nprobe : nprobes) {
                for (const auto& prefix :
                     {std::string("HR"),
                      std::string("RR"),
                      std::string("none")}) {
                    std::string name = ds +
                            "/fwht_ivf_recall/prefix:" + prefix +
                            "/nlist:" + std::to_string(nlist) +
                            "/nprobe:" + std::to_string(nprobe);
                    auto* rb = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_fwht_ivf_recall_dataset,
                            &sift,
                            ds_n,
                            ds_nq,
                            nlist,
                            nprobe,
                            prefix == "none" ? std::string() : prefix);
                    if (FLAGS_iterations > 0)
                        rb->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
