/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// RaBitQ Index Benchmarks (IndexRaBitQ, IndexIVFRaBitQ)
//
// Index-level counterpart of bench_kern_rabitq.cpp (kernel level) and of
// benchs/python/bench_index_rabitq.py: search over d/nq/nprobe/qb sweeps
// on synthetic data.
//
// Design notes:
//   * Non-RaBitQ baselines (SQ4/6/8, IVF1000,SQ4/6, PQ{32,64,128}x4fs
//     and their IVF variants, HNSW{8,16,32}) are intentionally omitted
//     (covered by the SQ/PQ/HNSW benches); memory-footprint reporting
//     is not included.

#include <map>
#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
DEFINE_uint32(nb, 200000, "database size");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        d,
        "",
        "comma-separated vector dimensions (default: 256,512,768,1024, "
        "chosen to hit distinct SIMD-width tiers)");
DEFINE_string(
        nq,
        "",
        "comma-separated query batch sizes (default: 1,10,100,1000)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated IVF nprobe values (default: 4,16,32)");
DEFINE_string(
        qb,
        "",
        "comma-separated query quantization bits for the IVF variant "
        "(default: 8)");

// k=100 for all measurements
constexpr int kK = 100;
// qb=8 on the flat RaBitQ index before training
constexpr uint8_t kQbFlat = 8;

/// Trained + populated RaBitQ index, built once per (factory, d) and
/// reused across nprobe/nq/qb combinations. Trains on min(nb, 100000)
/// vectors. `qb_flat` (query quantization bits) is set on the index
/// before training when >= 0 for the non-IVF variant.
static Index* get_rabitq_index(
        const std::string& factory,
        int d,
        int nb,
        int qb_flat = -1) {
    static std::map<std::string, std::unique_ptr<Index>> cache;

    std::string key = factory + "/d:" + std::to_string(d) +
            "/nb:" + std::to_string(nb) + "/qb:" + std::to_string(qb_flat);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second.get();
    }

    std::unique_ptr<Index> index(index_factory(d, factory.c_str()));
    if (qb_flat >= 0) {
        auto* rabitq = dynamic_cast<IndexRaBitQ*>(index.get());
        FAISS_THROW_IF_NOT(rabitq != nullptr);
        rabitq->qb = (uint8_t)qb_flat;
    }

    int nt = std::min(nb, 100000);
    std::vector<float> xt((size_t)nt * d);
    float_rand(xt.data(), xt.size(), 12345);
    index->train(nt, xt.data());

    std::vector<float> xb((size_t)nb * d);
    float_rand(xb.data(), xb.size(), 54321);
    index->add(nb, xb.data());

    Index* ptr = index.get();
    cache[key] = std::move(index);
    return ptr;
}

static void bench_rabitq_flat_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq) {
    omp_set_num_threads(FLAGS_threads);
    Index* index = get_rabitq_index("RaBitQ", d, nb, kQbFlat);

    std::vector<float> xq((size_t)nq * d);
    float_rand(xq.data(), xq.size(), 67890);

    std::vector<float> distances((size_t)nq * kK);
    std::vector<int64_t> labels((size_t)nq * kK);

    // Warmup
    index->search(nq, xq.data(), kK, distances.data(), labels.data());

    for (auto _ : state) {
        index->search(nq, xq.data(), kK, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = kK;
    state.counters["qb"] = kQbFlat;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_ivf_rabitq_search(
        benchmark::State& state,
        int d,
        int nb,
        int nprobe,
        int nq,
        int qb) {
    omp_set_num_threads(FLAGS_threads);
    // nlist=1000 fixed.
    Index* index = get_rabitq_index("IVF1000,RaBitQ", d, nb);

    std::vector<float> xq((size_t)nq * d);
    float_rand(xq.data(), xq.size(), 67890);

    std::vector<float> distances((size_t)nq * kK);
    std::vector<int64_t> labels((size_t)nq * kK);

    IVFRaBitQSearchParameters params;
    params.nprobe = nprobe;
    params.qb = (uint8_t)qb;

    // Warmup
    index->search(nq, xq.data(), kK, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        index->search(
                nq, xq.data(), kK, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = kK;
    state.counters["qb"] = qb;
    state.counters["threads"] = FLAGS_threads;
}

/// IVF1000,RaBitQ wrapped in an IndexPreTransform with a random rotation
/// (RandomRotationMatrix(d, d) seeded with 123). Built once per d and
/// reused across nprobe/nq combinations.
static Index* get_rabitq_rrot_index(int d, int nb) {
    static std::map<std::string, std::unique_ptr<Index>> cache;

    std::string key =
            "rrot/d:" + std::to_string(d) + "/nb:" + std::to_string(nb);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second.get();
    }

    auto* rrot = new RandomRotationMatrix(d, d);
    rrot->init(123);
    auto index = std::make_unique<IndexPreTransform>(
            rrot, index_factory(d, "IVF1000,RaBitQ"));
    index->own_fields = true;

    int nt = std::min(nb, 100000);
    std::vector<float> xt((size_t)nt * d);
    float_rand(xt.data(), xt.size(), 12345);
    index->train(nt, xt.data());

    std::vector<float> xb((size_t)nb * d);
    float_rand(xb.data(), xb.size(), 54321);
    index->add(nb, xb.data());

    Index* ptr = index.get();
    cache[key] = std::move(index);
    return ptr;
}

static void bench_ivf_rabitq_search_rrot(
        benchmark::State& state,
        int d,
        int nb,
        int nprobe,
        int nq,
        int qb) {
    omp_set_num_threads(FLAGS_threads);
    // nlist=1000 fixed, behind a random rotation.
    Index* index = get_rabitq_rrot_index(d, nb);

    std::vector<float> xq((size_t)nq * d);
    float_rand(xq.data(), xq.size(), 67890);

    std::vector<float> distances((size_t)nq * kK);
    std::vector<int64_t> labels((size_t)nq * kK);

    // IndexPreTransform forwards the search parameters to the IVF sub-index.
    IVFRaBitQSearchParameters params;
    params.nprobe = nprobe;
    params.qb = (uint8_t)qb;

    // Warmup
    index->search(nq, xq.data(), kK, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        index->search(
                nq, xq.data(), kK, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = kK;
    state.counters["qb"] = qb;
    state.counters["threads"] = FLAGS_threads;
}

/// Trained + populated RaBitQ index on the loaded dataset, built once per
/// (factory, qb) and reused across nprobe/nq/qb combinations.
static Index* get_rabitq_index_dataset(
        const std::string& factory,
        int d,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        int qb_flat = -1) {
    static std::map<std::string, std::unique_ptr<Index>> cache;

    std::string key = factory + "/qb:" + std::to_string(qb_flat);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second.get();
    }

    std::unique_ptr<Index> index(index_factory(d, factory.c_str()));
    if (qb_flat >= 0) {
        auto* rabitq = dynamic_cast<IndexRaBitQ*>(index.get());
        FAISS_THROW_IF_NOT(rabitq != nullptr);
        rabitq->qb = (uint8_t)qb_flat;
    }

    index->train(nt, xt);
    index->add(nb, xb);

    Index* ptr = index.get();
    cache[key] = std::move(index);
    return ptr;
}

// SIFT1M variant for the flat RaBitQ index
static void bench_rabitq_flat_search_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d) {
    omp_set_num_threads(FLAGS_threads);
    Index* index =
            get_rabitq_index_dataset("RaBitQ", d, xt, nt, xb, nb, kQbFlat);

    std::vector<float> distances((size_t)nq * kK);
    std::vector<int64_t> labels((size_t)nq * kK);

    // Warmup
    index->search(nq, xq, kK, distances.data(), labels.data());

    for (auto _ : state) {
        index->search(nq, xq, kK, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = kK;
    state.counters["qb"] = kQbFlat;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, kK, gt_k);
    state.counters["recall"] = recall;
}

// SIFT1M variant for the IVF RaBitQ index
static void bench_ivf_rabitq_search_dataset(
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
        int nprobe,
        int qb) {
    omp_set_num_threads(FLAGS_threads);
    // nlist=1000 fixed.
    Index* index =
            get_rabitq_index_dataset("IVF1000,RaBitQ", d, xt, nt, xb, nb);

    std::vector<float> distances((size_t)nq * kK);
    std::vector<int64_t> labels((size_t)nq * kK);

    IVFRaBitQSearchParameters params;
    params.nprobe = nprobe;
    params.qb = (uint8_t)qb;

    // Warmup
    index->search(nq, xq, kK, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        index->search(nq, xq, kK, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = kK;
    state.counters["qb"] = qb;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, kK, gt_k);
    state.counters["recall"] = recall;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "RaBitQ index search: flat, IVF and random-rotation IVF variants",
            "--d=256 --nprobe=16 --benchmark_filter='ivf_rabitq/search/.*'");

    int nb = FLAGS_nb;
    // Dimensions chosen to hit distinct SIMD-width tiers of the rabitq
    // kernels.
    std::vector<int> dims =
            benchmarks::int_list(FLAGS_d, {256, 512, 768, 1024});
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100, 1000});
    std::vector<int> nprobes = benchmarks::int_list(FLAGS_nprobe, {4, 16, 32});
    std::vector<int> qbs = benchmarks::int_list(FLAGS_qb, {8});

    // Flat RaBitQ search
    for (int d : dims) {
        for (int nq : nqs) {
            std::string name = "rabitq_flat/search/d:" + std::to_string(d) +
                    "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_rabitq_flat_search, d, nb, nq);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // IVF RaBitQ search
    for (int d : dims) {
        for (int nprobe : nprobes) {
            for (int nq : nqs) {
                for (int qb : qbs) {
                    std::string name =
                            "ivf_rabitq/search/d:" + std::to_string(d) +
                            "/nprobe:" + std::to_string(nprobe) +
                            "/nq:" + std::to_string(nq) +
                            "/qb:" + std::to_string(qb);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_ivf_rabitq_search,
                            d,
                            nb,
                            nprobe,
                            nq,
                            qb);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // IVF RaBitQ behind a random rotation ("RROT" variant); qb is
    // fixed at the default 8 to keep the sweep modest.
    for (int d : dims) {
        for (int nprobe : nprobes) {
            for (int nq : nqs) {
                std::string name =
                        "ivf_rabitq_rrot/search/d:" + std::to_string(d) +
                        "/nprobe:" + std::to_string(nprobe) +
                        "/nq:" + std::to_string(nq);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivf_rabitq_search_rrot,
                        d,
                        nb,
                        nprobe,
                        nq,
                        8);
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
        // Default to the full query set so recall against the dataset
        // ground truth is meaningful; --nq overrides (capped to the
        // dataset's query count).
        std::vector<int> sift_nqs =
                benchmarks::int_list(FLAGS_nq, {(int)sift.nq});
        for (int nq : sift_nqs) {
            if ((size_t)nq > sift.nq)
                continue;
            std::string name =
                    ds + "/rabitq_flat/search/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_rabitq_flat_search_dataset,
                    sift.xt.data(),
                    (int)sift.nt,
                    sift.xb.data(),
                    (int)sift.nb,
                    sift.xq.data(),
                    nq,
                    sift.gt.data(),
                    sift.gt_k,
                    (int)sift.d);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            for (int nprobe : nprobes) {
                for (int qb : qbs) {
                    name = ds + "/ivf_rabitq/search/nprobe:" +
                            std::to_string(nprobe) +
                            "/nq:" + std::to_string(nq) +
                            "/qb:" + std::to_string(qb);
                    b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_ivf_rabitq_search_dataset,
                            sift.xt.data(),
                            (int)sift.nt,
                            sift.xb.data(),
                            (int)sift.nb,
                            sift.xq.data(),
                            nq,
                            sift.gt.data(),
                            sift.gt_k,
                            (int)sift.d,
                            nprobe,
                            qb);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
