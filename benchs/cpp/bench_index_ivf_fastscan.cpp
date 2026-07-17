/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IVFPQFastScan Benchmarks
//
// Mirror of benchs/python/bench_index_ivf_fastscan.py, covering
// seven index_factory variants (M/nlist=32/1024, k=1 with the 12-point
// nprobe curve on SIFT1M):
//   IVFxxx,PQ32x4fs              — non-residual PQ fastscan
//   IVFxxx,PQ32x4fsr             — residual PQ fastscan (by_residual)
//   IVFxxx,LSQ30x4fsr_Nlsq2x4    — residual LSQ AQ fastscan
//   IVFxxx,RQ30x4fsr_Nrq2x4      — residual RQ AQ fastscan
//   IVFxxx,LSQ30x4fs_Nlsq2x4     — non-residual LSQ AQ fastscan
//   IVFxxx,RQ30x4fs_Nrq2x4       — non-residual RQ AQ fastscan
// The AQ groups sweep rescale_norm in {0, 1}.
//
// Design notes:
//   * Synthetic PQ groups train on 50000 vectors and the AQ (LSQ/RQ) groups
//     on 40000 (enough for nlist=1024) -- LSQ training is slow and roughly
//     linear beyond.

#include <map>
#include <memory>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFastScan.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/index_factory.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(nb, 100000, "database size");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");
DEFINE_string(nlist, "", "comma-separated IVF list counts (default: 256,1024)");
DEFINE_string(
        M,
        "",
        "comma-separated PQ subquantizer counts "
        "(default: 16,32,64; sift1m group: 16,32)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values "
        "(default: 1,2,4,6,8,12,16,24,32,48,64,128)");
DEFINE_string(
        implem,
        "",
        "comma-separated FastScan implementation ids (default: 0,12,14)");
DEFINE_string(
        rescale,
        "",
        "comma-separated 0/1 rescale_norm settings for the additive "
        "quantizer fastscan groups (default: 0,1)");
DEFINE_string(k, "", "comma-separated knn k values (default: 1,10)");

static void bench_ivfpq_fastscan_search(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        int M,
        int nprobe,
        int nq,
        int k,
        int implem,
        bool by_residual) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFPQFastScan index(&quantizer, d, nlist, M, 4);
    index.by_residual = by_residual;
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;
    index.implem = implem;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    indexIVF_stats.reset();
    FastScan_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["implem"] = implem;
    state.counters["residual"] = by_residual;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["t0"] = FastScan_stats.t0;
    state.counters["t1"] = FastScan_stats.t1;
    state.counters["t2"] = FastScan_stats.t2;
    state.counters["t3"] = FastScan_stats.t3;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant for IVFPQFastScan
static void bench_ivfpq_fastscan_search_dataset(
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
        int M,
        int nprobe,
        int k) {
    IndexFlatL2 quantizer(d);
    IndexIVFPQFastScan index(&quantizer, d, nlist, M, 4);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["dataset"] = 1;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

// Mirror of benchs/python/bench_index_ivf_fastscan.py
// (bench_ivf_lsq_fastscan_search / bench_ivf_rq_fastscan_search):
// additive-quantizer fastscan indexes built with index_factory, e.g.
// "IVF1024,LSQ30x4fsr_Nlsq2x4" — M=32 means 30 vector codebooks plus a
// 2x4-bit norm codec, matching the script's naming logic.
// LSQ training is slow, so each factory string is trained ONCE (lazily,
// static cache)
// and the index is reused across all nprobe/nq/rescale combinations.
static IndexIVFAdditiveQuantizerFastScan* aq_fastscan_index(
        const std::string& factory_string,
        int d,
        int nb) {
    static std::map<std::string, std::unique_ptr<Index>> cache;
    auto it = cache.find(factory_string);
    if (it == cache.end()) {
        // LSQ training is slow; keep nt at 40000 (enough for nlist=1024:
        // 1024*39=39936) rather than the usual 50000.
        int nt = std::min(nb, 40000);
        AlignedTable<float> xt(d * nt);
        AlignedTable<float> xb((size_t)d * nb);
        float_rand(xt.data(), d * nt, 12345);
        float_rand(xb.data(), (size_t)d * nb, 54321);

        std::unique_ptr<Index> index(index_factory(d, factory_string.c_str()));
        index->verbose = false;
        omp_set_num_threads(FLAGS_threads);
        index->train(nt, xt.data());
        index->add(nb, xb.data());
        it = cache.emplace(factory_string, std::move(index)).first;
    }
    auto* index =
            dynamic_cast<IndexIVFAdditiveQuantizerFastScan*>(it->second.get());
    if (index == nullptr) {
        fprintf(stderr,
                "factory '%s' did not produce an "
                "IndexIVFAdditiveQuantizerFastScan\n",
                factory_string.c_str());
        exit(1);
    }
    return index;
}

static void bench_aq_fastscan_search(
        benchmark::State& state,
        std::string factory_string,
        int d,
        int nb,
        int nlist,
        int M,
        int nprobe,
        int nq,
        int k,
        bool rescale_norm,
        bool by_residual) {
    IndexIVFAdditiveQuantizerFastScan* index =
            aq_fastscan_index(factory_string, d, nb);
    omp_set_num_threads(FLAGS_threads);
    index->nprobe = nprobe;
    index->rescale_norm = rescale_norm;

    AlignedTable<float> xq(d * nq);
    float_rand(xq.data(), d * nq, 67890);

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
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["rescale"] = rescale_norm;
    state.counters["residual"] = by_residual;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "IVFPQFastScan index search (SIMD-accelerated IVFPQ) on synthetic "
            "data and SIFT1M",
            "--nlist=1024 --nprobe=8 --M=32 "
            "--benchmark_filter='ivfpq_fastscan/search/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    std::vector<int> ks = benchmarks::int_list(FLAGS_k, {1, 10});
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});
    std::vector<int> nlists = benchmarks::int_list(FLAGS_nlist, {256, 1024});
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {16, 32, 64});
    // 12-point nprobe curve shared by all variant groups.
    std::vector<int> nprobes = benchmarks::int_list(
            FLAGS_nprobe, {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128});
    std::vector<int> implems = benchmarks::int_list(FLAGS_implem, {0, 12, 14});

    for (int k : ks) {
        for (int nlist : nlists) {
            for (int M : Ms) {
                if (M > d)
                    continue;
                for (int nprobe : nprobes) {
                    if (nprobe > nlist)
                        continue;
                    for (int nq : nqs) {
                        for (int implem : implems) {
                            std::string name = "ivfpq_fastscan/search/nlist:" +
                                    std::to_string(nlist) +
                                    "/M:" + std::to_string(M) +
                                    "/nprobe:" + std::to_string(nprobe) +
                                    "/nq:" + std::to_string(nq) +
                                    "/implem:" + std::to_string(implem) +
                                    "/k:" + std::to_string(k);
                            auto* b = benchmark::RegisterBenchmark(
                                    name.c_str(),
                                    bench_ivfpq_fastscan_search,
                                    d,
                                    nb,
                                    nlist,
                                    M,
                                    nprobe,
                                    nq,
                                    k,
                                    implem,
                                    false);
                            if (FLAGS_iterations > 0)
                                b->Iterations(FLAGS_iterations);
                        }
                    }
                }
            }
        }
    }

    // "IVF1024,PQ32x4fsr" — residual PQ fastscan (by_residual=true).
    // Fixed M=32, nlist=1024, 12-point nprobe curve.
    {
        int res_M = 32;
        std::vector<int> res_nlists = benchmarks::int_list(FLAGS_nlist, {1024});
        for (int k : ks) {
            for (int nlist : res_nlists) {
                if (res_M > d)
                    continue;
                for (int nprobe : nprobes) {
                    if (nprobe > nlist)
                        continue;
                    for (int nq : nqs) {
                        for (int implem : implems) {
                            std::string name =
                                    "ivfpq_fastscan_residual/search/nlist:" +
                                    std::to_string(nlist) +
                                    "/M:" + std::to_string(res_M) +
                                    "/nprobe:" + std::to_string(nprobe) +
                                    "/nq:" + std::to_string(nq) +
                                    "/implem:" + std::to_string(implem) +
                                    "/k:" + std::to_string(k);
                            auto* b = benchmark::RegisterBenchmark(
                                    name.c_str(),
                                    bench_ivfpq_fastscan_search,
                                    d,
                                    nb,
                                    nlist,
                                    res_M,
                                    nprobe,
                                    nq,
                                    k,
                                    implem,
                                    true);
                            if (FLAGS_iterations > 0)
                                b->Iterations(FLAGS_iterations);
                        }
                    }
                }
            }
        }
    }

    // Additive-quantizer fastscan groups (LSQ / RQ), mirroring
    // bench_index_ivf_fastscan.py. Fixed M=32 (30 vector codebooks +
    // 2x4-bit norm) and nlist=1024, sweeping nprobe/nq/rescale_norm/k.
    // Both the residual "x4fsr" encoding and the non-residual "x4fs"
    // encoding are covered (the last two variants).
    struct AQDef {
        const char* group;
        const char* factory_tpl; // printf template: nlist, M-2
        bool by_residual;
    };
    std::vector<AQDef> aq_defs = {
            {"ivf_lsq_fastscan", "IVF%d,LSQ%dx4fsr_Nlsq2x4", true},
            {"ivf_rq_fastscan", "IVF%d,RQ%dx4fsr_Nrq2x4", true},
            {"ivf_lsq_fastscan_nonresidual", "IVF%d,LSQ%dx4fs_Nlsq2x4", false},
            {"ivf_rq_fastscan_nonresidual", "IVF%d,RQ%dx4fs_Nrq2x4", false},
    };
    int aq_M = 32;
    std::vector<int> aq_nlists = benchmarks::int_list(FLAGS_nlist, {1024});
    std::vector<int> rescales = benchmarks::int_list(FLAGS_rescale, {0, 1});
    for (auto& aq : aq_defs) {
        for (int nlist : aq_nlists) {
            char factory_string[64];
            snprintf(
                    factory_string,
                    sizeof(factory_string),
                    aq.factory_tpl,
                    nlist,
                    aq_M - 2);
            for (int k : ks) {
                for (int nprobe : nprobes) {
                    if (nprobe > nlist)
                        continue;
                    for (int nq : nqs) {
                        for (int rescale_int : rescales) {
                            bool rescale = rescale_int != 0;
                            std::string name = std::string(aq.group) +
                                    "/search/nlist:" + std::to_string(nlist) +
                                    "/M:" + std::to_string(aq_M) +
                                    "/nprobe:" + std::to_string(nprobe) +
                                    "/nq:" + std::to_string(nq) +
                                    "/rescale:" + std::to_string(rescale) +
                                    "/k:" + std::to_string(k);
                            auto* b = benchmark::RegisterBenchmark(
                                    name.c_str(),
                                    bench_aq_fastscan_search,
                                    std::string(factory_string),
                                    d,
                                    nb,
                                    nlist,
                                    aq_M,
                                    nprobe,
                                    nq,
                                    k,
                                    rescale,
                                    aq.by_residual);
                            if (FLAGS_iterations > 0)
                                b->Iterations(FLAGS_iterations);
                        }
                    }
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
        int sd = (int)sift.d;
        int snb = (int)sift.nb;
        int snq = (int)sift.nq;
        int snt = (int)sift.nt;

        // IVFPQFastScan on SIFT1M (matches bench_index_ivf_fastscan.py).
        // Uses k=1 with the 12-point nprobe curve; sweep ks here too.
        std::vector<int> sift_Ms = benchmarks::int_list(FLAGS_M, {16, 32});
        std::vector<int> sift_nprobes = benchmarks::int_list(
                FLAGS_nprobe, {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128});
        for (int k : ks) {
            for (int nlist : nlists) {
                for (int M : sift_Ms) {
                    for (int nprobe : sift_nprobes) {
                        std::string name = ds +
                                "/ivfpq_fastscan/search/nlist:" +
                                std::to_string(nlist) +
                                "/M:" + std::to_string(M) +
                                "/nprobe:" + std::to_string(nprobe) +
                                "/k:" + std::to_string(k);
                        auto* b = benchmark::RegisterBenchmark(
                                name.c_str(),
                                bench_ivfpq_fastscan_search_dataset,
                                sift.xt.data(),
                                snt,
                                sift.xb.data(),
                                snb,
                                sift.xq.data(),
                                snq,
                                sift.gt.data(),
                                sift.gt_k,
                                sd,
                                nlist,
                                M,
                                nprobe,
                                k);
                        if (FLAGS_iterations > 0)
                            b->Iterations(FLAGS_iterations);
                    }
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
