/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IVF IDSelector Overhead Benchmarks
// Measures the runtime overhead of using an IDSelector through
// faiss::ivflib::search_with_parameters versus a plain unconditional
// Index::search, across parallel_mode {0,3} and thread counts. The
// benchmark sweeps three cases: (tt=0) parallel_mode=0, (tt=1) parallel_mode=3,
// (tt=2) parallel_mode=0 with a single thread. We expose parallel_mode and
// thread count as flags and cover the following search variants for each
// combination:
//   * plain           - Index::search (no selector)
//   * null_selector   - search_with_parameters with a default (nullptr) sel
//   * all_selector    - search_with_parameters with an IDSelectorAll
//   * manual_parallel - search_with_parameters sliced across omp threads

#include <gflags/gflags.h>
#include <omp.h>

#include <memory>
#include <vector>

#include <benchmark/benchmark.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;
using idx_t = faiss::idx_t;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_int32(d, 64, "vector dimension");
DEFINE_int32(nb, 1 << 20, "number of database vectors");
DEFINE_int32(nq, 512 * 16, "number of query vectors");
DEFINE_int32(k, 10, "number of neighbors");
DEFINE_string(factory, "IVF1024,SQ8", "index_factory string");
DEFINE_string(
        pmode,
        "",
        "comma-separated IVF parallel_mode values (default: 0,3)");
DEFINE_string(
        threads,
        "",
        "comma-separated thread counts (default: max_threads,1)");

// One of the four search variants exercised by the benchmark.
enum SelectorVariant {
    PLAIN = 0,       // Index::search, no selector
    NULL_SELECTOR,   // search_with_parameters, default (nullptr) selector
    ALL_SELECTOR,    // search_with_parameters, IDSelectorAll
    MANUAL_PARALLEL, // search_with_parameters sliced across threads
};

struct SharedIndex {
    std::unique_ptr<Index> index;
    IndexIVF* index_ivf = nullptr;
    std::vector<float> data;
    const float* xb = nullptr;
    const float* xq = nullptr;
    int d = 0;
    int nb = 0;
    int nq = 0;
};

static SharedIndex& get_shared_index() {
    static SharedIndex si;
    if (si.index) {
        return si;
    }
    si.d = FLAGS_d;
    si.nb = FLAGS_nb;
    si.nq = FLAGS_nq;
    si.data.resize((size_t)(si.nb + si.nq) * si.d);
    si.xb = si.data.data();
    si.xq = si.data.data() + (size_t)si.nb * si.d;
    rand_smooth_vectors(si.nb + si.nq, si.d, si.data.data(), 1234);

    si.index.reset(index_factory(si.d, FLAGS_factory.c_str()));
    si.index->train(si.nb, si.xb);
    si.index->add(si.nb, si.xb);
    si.index_ivf = dynamic_cast<IndexIVF*>(si.index.get());
    FAISS_THROW_IF_NOT_MSG(
            si.index_ivf != nullptr,
            "--factory did not produce an IndexIVF");
    return si;
}

static void bench_ivf_selector(
        benchmark::State& state,
        SelectorVariant variant,
        int parallel_mode,
        int threads) {
    SharedIndex& si = get_shared_index();
    Index* index = si.index.get();
    IndexIVF* index_ivf = si.index_ivf;
    int d = si.d;
    int nq = si.nq;
    int k = FLAGS_k;

    index_ivf->parallel_mode = parallel_mode;
    omp_set_num_threads(threads);

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);

    IDSelectorAll sel;

    auto run = [&]() {
        switch (variant) {
            case PLAIN:
                index->search(nq, si.xq, k, D.data(), I.data());
                break;
            case NULL_SELECTOR: {
                IVFSearchParameters params;
                ivflib::search_with_parameters(
                        index, nq, si.xq, k, D.data(), I.data(), &params);
                break;
            }
            case ALL_SELECTOR: {
                IVFSearchParameters params;
                params.sel = &sel;
                ivflib::search_with_parameters(
                        index, nq, si.xq, k, D.data(), I.data(), &params);
                break;
            }
            case MANUAL_PARALLEL: {
                int nt = omp_get_max_threads();
                IVFSearchParameters params;
#pragma omp parallel for if (nt > 1)
                for (idx_t slice = 0; slice < nt; slice++) {
                    idx_t i0 = (idx_t)nq * slice / nt;
                    idx_t i1 = (idx_t)nq * (slice + 1) / nt;
                    if (i1 > i0) {
                        ivflib::search_with_parameters(
                                index,
                                i1 - i0,
                                si.xq + i0 * d,
                                k,
                                D.data() + i0 * k,
                                I.data() + i0 * k,
                                &params);
                    }
                }
                break;
            }
        }
    };

    // Warmup
    run();

    for (auto _ : state) {
        run();
        benchmark::DoNotOptimize(D[0]);
        benchmark::DoNotOptimize(I[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = si.nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["parallel_mode"] = parallel_mode;
    state.counters["threads"] = threads;
}

// Dataset-backed shared index: trained on the real train vectors and
// populated with the full base set. Built once and reused across all
// dataset registrations.
static SharedIndex& get_dataset_index(
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        int d) {
    static SharedIndex si;
    if (si.index) {
        return si;
    }
    si.d = d;
    si.nb = nb;
    si.nq = nq;
    si.xb = xb;
    si.xq = xq;
    si.index.reset(index_factory(d, FLAGS_factory.c_str()));
    si.index->train(nt, xt);
    si.index->add(nb, xb);
    si.index_ivf = dynamic_cast<IndexIVF*>(si.index.get());
    FAISS_THROW_IF_NOT_MSG(
            si.index_ivf != nullptr,
            "--factory did not produce an IndexIVF");
    return si;
}

// SIFT1M variant: the same search variants against an index trained on the
// real train vectors and populated with the full base set, searched with
// real queries.
static void bench_ivf_selector_dataset(
        benchmark::State& state,
        SelectorVariant variant,
        int parallel_mode,
        int threads,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d) {
    SharedIndex& si = get_dataset_index(xt, nt, xb, nb, xq, nq, d);
    Index* index = si.index.get();
    IndexIVF* index_ivf = si.index_ivf;
    int k = FLAGS_k;

    index_ivf->parallel_mode = parallel_mode;
    omp_set_num_threads(threads);

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);

    IDSelectorAll sel;

    auto run = [&]() {
        switch (variant) {
            case PLAIN:
                index->search(nq, si.xq, k, D.data(), I.data());
                break;
            case NULL_SELECTOR: {
                IVFSearchParameters params;
                ivflib::search_with_parameters(
                        index, nq, si.xq, k, D.data(), I.data(), &params);
                break;
            }
            case ALL_SELECTOR: {
                IVFSearchParameters params;
                params.sel = &sel;
                ivflib::search_with_parameters(
                        index, nq, si.xq, k, D.data(), I.data(), &params);
                break;
            }
            case MANUAL_PARALLEL: {
                int nslice = omp_get_max_threads();
                IVFSearchParameters params;
#pragma omp parallel for if (nslice > 1)
                for (idx_t slice = 0; slice < nslice; slice++) {
                    idx_t i0 = (idx_t)nq * slice / nslice;
                    idx_t i1 = (idx_t)nq * (slice + 1) / nslice;
                    if (i1 > i0) {
                        ivflib::search_with_parameters(
                                index,
                                i1 - i0,
                                si.xq + i0 * d,
                                k,
                                D.data() + i0 * k,
                                I.data() + i0 * k,
                                &params);
                    }
                }
                break;
            }
        }
    };

    // Warmup
    run();

    for (auto _ : state) {
        run();
        benchmark::DoNotOptimize(D[0]);
        benchmark::DoNotOptimize(I[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = si.nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["parallel_mode"] = parallel_mode;
    state.counters["threads"] = threads;
    state.counters["dataset"] = 1;

    // Compute recall from last iteration's results
    double recall = benchmarks::compute_recall_at(I.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "IVF IDSelector overhead: plain search vs search_with_parameters "
            "(nullptr / IDSelectorAll) across parallel_mode and thread counts",
            "--pmode=3 --factory=IVF1024,Flat "
            "--benchmark_filter='ivf_selector/.*/variant:all_selector'");

    std::vector<int> pmodes = benchmarks::int_list(FLAGS_pmode, {0, 3});
    std::vector<int> threads_list =
            benchmarks::int_list(FLAGS_threads, {omp_get_max_threads(), 1});

    struct VariantInfo {
        SelectorVariant variant;
        const char* name;
    };
    const std::vector<VariantInfo> variants = {
            {PLAIN, "plain"},
            {NULL_SELECTOR, "null_selector"},
            {ALL_SELECTOR, "all_selector"},
            {MANUAL_PARALLEL, "manual_parallel"},
    };

    for (int threads : threads_list) {
        for (int parallel_mode : pmodes) {
            for (const auto& vi : variants) {
                std::string name =
                        "ivf_selector/threads:" + std::to_string(threads) +
                        "/pmode:" + std::to_string(parallel_mode) +
                        "/variant:" + vi.name;
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivf_selector,
                        vi.variant,
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
        int snq = std::min(FLAGS_nq, (int)sift.nq);
        // Keep the dataset cross-product small: sweep parallel_mode and
        // variant at the first configured thread count only.
        int sift_threads = threads_list[0];
        for (int parallel_mode : pmodes) {
            for (const auto& vi : variants) {
                std::string name = ds + "/ivf_selector/threads:" +
                        std::to_string(sift_threads) +
                        "/pmode:" + std::to_string(parallel_mode) +
                        "/variant:" + vi.name;
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivf_selector_dataset,
                        vi.variant,
                        parallel_mode,
                        sift_threads,
                        sift.xt.data(),
                        (int)sift.nt,
                        sift.xb.data(),
                        (int)sift.nb,
                        sift.xq.data(),
                        snq,
                        sift.gt.data(),
                        sift.gt_k,
                        (int)sift.d);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
