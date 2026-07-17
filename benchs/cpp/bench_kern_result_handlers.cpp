/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Result-handler / top-k selection Benchmarks
//
// Benchmarks how top-k collection cost grows with k and nprobe, and how it
// varies with the index/quantizer type (--factory). This benchmark exercises
// the built-in result-handler machinery reached through IndexIVF::search.
//
// Design notes:
// - d is fixed per run; additional (d, factory) combinations can be reached
//   via --factory (e.g. --factory='IVF{nlist},SQ4;IVF{nlist},SQfp16').
// - The AVX512 custom reservoir result-handler is intentionally excluded:
//   that handler is not part of libfaiss.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_uint32(d, 64, "dimension");
DEFINE_uint32(nb, 1000000, "database size");
DEFINE_uint32(nq, 100, "number of queries");
DEFINE_uint32(nlist, 1024, "number of IVF lists");
DEFINE_string(
        k,
        "",
        "comma-separated top-k values "
        "(default: 1,4,10,16,20,50,100,200,500,1000)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values (default: 1,2,4,8,16,64)");
DEFINE_string(
        factory,
        "",
        "semicolon-separated IVF index_factory keys (keys contain commas; the "
        "{nlist} placeholder is substituted; default: IVF{nlist},Flat; "
        "e.g. --factory='IVF{nlist},SQ4;IVF{nlist},SQfp16')");

// Substitute the {nlist} placeholder in a factory template.
static std::string subst_nlist(const std::string& tmpl, int nlist) {
    std::string out = tmpl;
    std::string::size_type pos = out.find("{nlist}");
    if (pos != std::string::npos) {
        out.replace(pos, 7, std::to_string(nlist));
    }
    return out;
}

// One trained+populated index per factory key, shared across the whole
// k/nprobe sweep (the result-handler cost is what we measure, not build cost).
static IndexIVF* shared_index(const std::string& factory, int d, int nb) {
    static std::map<std::string, std::unique_ptr<Index>> cache;
    auto it = cache.find(factory);
    if (it == cache.end()) {
        int nt = std::min(nb, 50000);
        std::vector<float> xt(d * nt), xb((size_t)d * nb);
        // rand_smooth_vectors provides a representative data distribution.
        rand_smooth_vectors(nt, d, xt.data(), 1234);
        rand_smooth_vectors(nb, d, xb.data(), 4567);

        std::unique_ptr<Index> index(index_factory(d, factory.c_str()));
        index->verbose = false;
        index->train(nt, xt.data());
        index->add(nb, xb.data());
        it = cache.emplace(factory, std::move(index)).first;
    }
    return dynamic_cast<IndexIVF*>(it->second.get());
}

static void bench_ivf_topk(
        benchmark::State& state,
        const std::string factory,
        int d,
        int nb,
        int nprobe,
        int nq,
        int k) {
    IndexIVF* index = shared_index(factory, d, nb);
    omp_set_num_threads(FLAGS_threads);
    index->nprobe = nprobe;

    std::vector<float> xq(d * nq);
    rand_smooth_vectors(nq, d, xq.data(), 7890);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index->search(nq, xq.data(), k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index->search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = index->nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["nlist_visited"] = indexIVF_stats.nlist;
    state.counters["threads"] = FLAGS_threads;
}

// Dataset-backed counterpart of shared_index: one trained+populated index
// per factory key, built from the real train and base vectors.
static IndexIVF* shared_index_dataset(
        const std::string& factory,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        int d) {
    static std::map<std::string, std::unique_ptr<Index>> cache;
    auto it = cache.find(factory);
    if (it == cache.end()) {
        std::unique_ptr<Index> index(index_factory(d, factory.c_str()));
        index->verbose = false;
        index->train(nt, xt);
        index->add(nb, xb);
        it = cache.emplace(factory, std::move(index)).first;
    }
    return dynamic_cast<IndexIVF*>(it->second.get());
}

// SIFT1M variant: top-k collection through IndexIVF::search over the real
// base set, searched with real queries.
static void bench_ivf_topk_dataset(
        benchmark::State& state,
        const std::string factory,
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
        int k) {
    IndexIVF* index = shared_index_dataset(factory, xt, nt, xb, nb, d);
    omp_set_num_threads(FLAGS_threads);
    index->nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index->search(nq, xq, k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index->search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = index->nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["dataset"] = 1;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["nlist_visited"] = indexIVF_stats.nlist;
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
            "top-k result-handler cost in IVF search over a k, nprobe and "
            "index-type sweep",
            "--k=1,100,1000 --nprobe=1,16 --factory='IVF{nlist},SQ4' "
            "--benchmark_filter='ivf_topk/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    int nq = FLAGS_nq;
    int nlist = FLAGS_nlist;
    std::vector<int> ks = benchmarks::int_list(
            FLAGS_k, {1, 4, 10, 16, 20, 50, 100, 200, 500, 1000});
    std::vector<int> nprobes =
            benchmarks::int_list(FLAGS_nprobe, {1, 2, 4, 8, 16, 64});
    // Default is Flat only; also supports SQ4/SQfp16
    // (and RaBitQ). Factory keys contain commas, so the list is split on ';'
    // (e.g. --factory='IVF{nlist},SQ4;IVF{nlist},SQfp16').
    std::vector<std::string> factories;
    if (FLAGS_factory.empty()) {
        factories = {"IVF{nlist},Flat"};
    } else {
        std::stringstream fss(FLAGS_factory);
        std::string tok;
        while (std::getline(fss, tok, ';')) {
            if (!tok.empty())
                factories.push_back(tok);
        }
    }

    for (const std::string& ftmpl : factories) {
        std::string factory = subst_nlist(ftmpl, nlist);
        for (int nprobe : nprobes) {
            if (nprobe > nlist)
                continue;
            for (int k : ks) {
                std::string name = "ivf_topk/factory:" + factory +
                        "/nprobe:" + std::to_string(nprobe) +
                        "/k:" + std::to_string(k);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivf_topk,
                        factory,
                        d,
                        nb,
                        nprobe,
                        nq,
                        k);
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
        int snq = std::min(nq, (int)sift.nq);
        // Keep the dataset cross-product small: a few representative k and
        // nprobe points per factory key.
        std::vector<int> sift_ks =
                benchmarks::int_list(FLAGS_k, {10, 100, 1000});
        std::vector<int> sift_nprobes =
                benchmarks::int_list(FLAGS_nprobe, {1, 16});
        for (const std::string& ftmpl : factories) {
            std::string factory = subst_nlist(ftmpl, nlist);
            for (int nprobe : sift_nprobes) {
                if (nprobe > nlist)
                    continue;
                for (int k : sift_ks) {
                    std::string name = ds + "/ivf_topk/factory:" + factory +
                            "/nprobe:" + std::to_string(nprobe) +
                            "/k:" + std::to_string(k);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_ivf_topk_dataset,
                            factory,
                            sift.xt.data(),
                            (int)sift.nt,
                            sift.xb.data(),
                            (int)sift.nb,
                            sift.xq.data(),
                            snq,
                            sift.gt.data(),
                            sift.gt_k,
                            (int)sift.d,
                            nprobe,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
