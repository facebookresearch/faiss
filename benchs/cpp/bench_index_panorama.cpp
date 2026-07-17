/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Panorama Index Benchmarks
// Benchmarks progressive-dimension scanning indexes: FlatL2Panorama,
// IndexIVFFlatPanorama, HNSWFlatPanorama, and IndexRefinePanorama.
//
// The --pca axis adds a PCA pre-transform: pca=1 wraps the index in a
// data-trained PCA rotation, which concentrates energy in the early
// dimensions and is what makes Panorama level pruning effective.
//
// Design notes:
// - Plain (non-Panorama) baselines live in bench_index_flat /
//   bench_index_ivf / bench_index_graph.
// - Default sweep uses d=128 synthetic data plus an optional SIFT1M path.
// - Synthetic sweeps cap nq at 100 (the SIFT1M path defaults to the full
//   query set).

#include <gflags/gflags.h>
#include <omp.h>

#include <memory>

#include <benchmark/benchmark.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFFlatPanorama.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/index_factory.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(nb, 100000, "database size");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");
DEFINE_string(
        n_levels,
        "",
        "comma-separated Panorama level counts "
        "(default: flat 4,8,16; ivf 8,16; hnsw and sift1m groups: 8)");
DEFINE_string(
        batch,
        "",
        "comma-separated FlatL2Panorama batch sizes "
        "(default: 128,512; sift1m group: 512)");
DEFINE_string(
        nlist,
        "",
        "comma-separated IVF list counts "
        "(default: 128,256,1024; sift1m group: 128)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values "
        "(default: 1,2,4,8,16,32,64; refine group: 4,16,64,256; "
        "sift1m group: 4,16,64)");
DEFINE_string(
        M,
        "",
        "comma-separated HNSW connectivity M values (default: 16,32)");
DEFINE_string(
        efSearch,
        "",
        "comma-separated HNSW efSearch values "
        "(default: 16,32,64,128,256,512)");
DEFINE_string(
        k_factor,
        "",
        "comma-separated refine k_factor values "
        "(default: 1,8,64,256,1024)");
DEFINE_string(
        pca,
        "",
        "comma-separated PCA pre-transform toggles (0/1; default: 0,1). "
        "pca=1 adds a PCA pre-transform "
        "(refine group: applied to the refine stage only)");

// Batch size for the IVF Panorama variants (the constructor default is
// 128; 1024 is used here for better throughput).
constexpr size_t kIVFBatchSize = 1024;

// --- FlatL2Panorama ---

static void bench_flat_l2_panorama_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k,
        int n_levels,
        int batch_size,
        int pca) {
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xb.data(), (size_t)d * nb, 12345);
    float_rand(xq.data(), d * nq, 54321);

    IndexFlatL2Panorama flat_index(d, n_levels, batch_size);
    // pca=1: wrap in a PCA pre-transform (trained on a database subsample).
    PCAMatrix pca_mat(d, d);
    IndexPreTransform pre_index(&pca_mat, &flat_index);
    Index& index = pca ? static_cast<Index&>(pre_index)
                       : static_cast<Index&>(flat_index);
    omp_set_num_threads(FLAGS_threads);
    if (pca) {
        index.train(std::min(nb, 50000), xb.data());
    }
    index.add(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        indexPanorama_stats.reset();
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["n_levels"] = n_levels;
    state.counters["batch_size"] = batch_size;
    state.counters["pca"] = pca;
    state.counters["dims_scanned_ratio"] =
            indexPanorama_stats.ratio_dims_scanned;
    state.counters["threads"] = FLAGS_threads;
}

// --- IVFFlatPanorama ---

static void bench_ivf_flat_panorama_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k,
        int nlist,
        int nprobe,
        int n_levels,
        int pca) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    // pca=1: wrap in a PCA pre-transform.
    IndexIVFFlatPanorama ivf_index(
            &quantizer, d, nlist, n_levels, METRIC_L2, true, kIVFBatchSize);
    ivf_index.verbose = false;
    PCAMatrix pca_mat(d, d);
    IndexPreTransform pre_index(&pca_mat, &ivf_index);
    Index& index = pca ? static_cast<Index&>(pre_index)
                       : static_cast<Index&>(ivf_index);
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    ivf_index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        indexPanorama_stats.reset();
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["n_levels"] = n_levels;
    state.counters["batch_size"] = kIVFBatchSize;
    state.counters["pca"] = pca;
    state.counters["dims_scanned_ratio"] =
            indexPanorama_stats.ratio_dims_scanned;
    state.counters["threads"] = FLAGS_threads;
}

// --- HNSW Flat Panorama (HNSW with FlatL2Panorama storage) ---

static void bench_hnsw_flat_panorama_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k,
        int M,
        int efSearch,
        int n_levels,
        int pca) {
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xb.data(), (size_t)d * nb, 12345);
    float_rand(xq.data(), d * nq, 54321);

    // HNSW with Panorama (progressive-dimension) storage.
    // pca=1: wrap in a PCA pre-transform (trained on a database subsample).
    IndexHNSWFlatPanorama hnsw_index(d, M, n_levels);
    hnsw_index.hnsw.efConstruction = 40;
    hnsw_index.verbose = false;
    PCAMatrix pca_mat(d, d);
    IndexPreTransform pre_index(&pca_mat, &hnsw_index);
    Index& index = pca ? static_cast<Index&>(pre_index)
                       : static_cast<Index&>(hnsw_index);
    omp_set_num_threads(FLAGS_threads);
    if (pca) {
        index.train(std::min(nb, 50000), xb.data());
    }
    index.add(nb, xb.data());

    SearchParametersHNSW params;
    params.efSearch = efSearch;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        indexPanorama_stats.reset();
        index.search(
                nq, xq.data(), k, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["M"] = M;
    state.counters["efSearch"] = efSearch;
    state.counters["n_levels"] = n_levels;
    state.counters["pca"] = pca;
    state.counters["dims_scanned_ratio"] =
            indexPanorama_stats.ratio_dims_scanned;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant: accepts external data pointers, builds index once per call
static void bench_flat_l2_panorama_search_dataset(
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
        int k,
        int n_levels,
        int batch_size,
        int pca) {
    IndexFlatL2Panorama flat_index(d, n_levels, batch_size);
    // pca=1: wrap in a PCA pre-transform (trained on the dataset train set).
    PCAMatrix pca_mat(d, d);
    IndexPreTransform pre_index(&pca_mat, &flat_index);
    Index& index = pca ? static_cast<Index&>(pre_index)
                       : static_cast<Index&>(flat_index);
    omp_set_num_threads(FLAGS_threads);
    if (pca) {
        index.train(nt, xt);
    }
    index.add(nb, xb);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    for (auto _ : state) {
        indexPanorama_stats.reset();
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["n_levels"] = n_levels;
    state.counters["batch_size"] = batch_size;
    state.counters["pca"] = pca;
    state.counters["dims_scanned_ratio"] =
            indexPanorama_stats.ratio_dims_scanned;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

// SIFT1M variant for IVFFlatPanorama
static void bench_ivf_flat_panorama_search_dataset(
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
        int k,
        int nlist,
        int nprobe,
        int n_levels,
        int pca) {
    IndexFlatL2 quantizer(d);
    // pca=1: wrap in a PCA pre-transform.
    IndexIVFFlatPanorama ivf_index(
            &quantizer, d, nlist, n_levels, METRIC_L2, true, kIVFBatchSize);
    ivf_index.verbose = false;
    PCAMatrix pca_mat(d, d);
    IndexPreTransform pre_index(&pca_mat, &ivf_index);
    Index& index = pca ? static_cast<Index&>(pre_index)
                       : static_cast<Index&>(ivf_index);
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);
    ivf_index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    for (auto _ : state) {
        indexPanorama_stats.reset();
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["n_levels"] = n_levels;
    state.counters["batch_size"] = kIVFBatchSize;
    state.counters["pca"] = pca;
    state.counters["dims_scanned_ratio"] =
            indexPanorama_stats.ratio_dims_scanned;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

// --- IndexRefinePanorama ---

static void bench_refine_panorama_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k,
        int nlist,
        int nprobe,
        int n_levels,
        float k_factor,
        int pca) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    // Build the fastscan base index. The PQ subquantizer count is derived
    // from d ("IVF256,PQ16x4fs" at d=128). The subquantizer count (d / 8)
    // must divide d and be even (4-bit fastscan packs two codes per byte),
    // so d must be a multiple of 16.
    FAISS_THROW_IF_NOT_MSG(
            d % 16 == 0, "refine base PQ(d/8)x4fs requires d % 16 == 0");
    std::string base_factory = "IVF" + std::to_string(nlist) + ",PQ" +
            std::to_string(d / 8) + "x4fs";
    std::unique_ptr<Index> base_index(index_factory(d, base_factory.c_str()));
    omp_set_num_threads(FLAGS_threads);
    base_index->train(nt, xt.data());
    base_index->add(nb, xb.data());
    ivflib::extract_index_ivf(base_index.get())->nprobe = nprobe;

    // Build panorama refine index. Batch size 1 for the refine stage.
    // pca=1: only the refine stage gets the PCA pre-transform; the
    // fastscan base index stays un-prefixed.
    IndexFlatL2Panorama refine_flat(d, n_levels, 1);
    PCAMatrix pca_mat(d, d);
    IndexPreTransform pre_index(&pca_mat, &refine_flat);
    Index& refine_index = pca ? static_cast<Index&>(pre_index)
                              : static_cast<Index&>(refine_flat);
    if (pca) {
        refine_index.train(nt, xt.data());
    }
    refine_index.add(nb, xb.data());

    IndexRefinePanorama index(base_index.get(), &refine_index);

    IndexRefineSearchParameters params;
    params.k_factor = k_factor;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        indexPanorama_stats.reset();
        index.search(
                nq, xq.data(), k, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["n_levels"] = n_levels;
    state.counters["k_factor"] = k_factor;
    state.counters["pca"] = pca;
    state.counters["dims_scanned_ratio"] =
            indexPanorama_stats.ratio_dims_scanned;
    state.counters["threads"] = FLAGS_threads;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "Panorama progressive-dimension scanning indexes "
            "(FlatL2Panorama, IVFFlatPanorama, HNSW flat, "
            "IndexRefinePanorama)",
            "--nq=10 --n_levels=8 "
            "--benchmark_filter='ivf_flat_panorama/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    int k = 10;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});
    std::vector<int> pcas = benchmarks::int_list(FLAGS_pca, {0, 1});

    // FlatL2Panorama
    std::vector<int> n_levels_flat =
            benchmarks::int_list(FLAGS_n_levels, {4, 8, 16});
    std::vector<int> batch_sizes =
            benchmarks::int_list(FLAGS_batch, {128, 512});
    for (int n_levels : n_levels_flat) {
        for (int batch_size : batch_sizes) {
            for (int nq : nqs) {
                for (int pca : pcas) {
                    std::string name = "flat_l2_panorama/search/n_levels:" +
                            std::to_string(n_levels) +
                            "/batch:" + std::to_string(batch_size) +
                            "/nq:" + std::to_string(nq) +
                            "/pca:" + std::to_string(pca);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_flat_l2_panorama_search,
                            d,
                            nb,
                            nq,
                            k,
                            n_levels,
                            batch_size,
                            pca);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // IVFFlatPanorama
    std::vector<int> nlists =
            benchmarks::int_list(FLAGS_nlist, {128, 256, 1024});
    std::vector<int> nprobes =
            benchmarks::int_list(FLAGS_nprobe, {1, 2, 4, 8, 16, 32, 64});
    std::vector<int> n_levels_ivf =
            benchmarks::int_list(FLAGS_n_levels, {8, 16});
    for (int nlist : nlists) {
        for (int nprobe : nprobes) {
            if (nprobe > nlist)
                continue;
            for (int n_levels : n_levels_ivf) {
                for (int nq : nqs) {
                    for (int pca : pcas) {
                        std::string name = "ivf_flat_panorama/search/nlist:" +
                                std::to_string(nlist) +
                                "/nprobe:" + std::to_string(nprobe) +
                                "/n_levels:" + std::to_string(n_levels) +
                                "/nq:" + std::to_string(nq) +
                                "/pca:" + std::to_string(pca);
                        auto* b = benchmark::RegisterBenchmark(
                                name.c_str(),
                                bench_ivf_flat_panorama_search,
                                d,
                                nb,
                                nq,
                                k,
                                nlist,
                                nprobe,
                                n_levels,
                                pca);
                        if (FLAGS_iterations > 0)
                            b->Iterations(FLAGS_iterations);
                    }
                }
            }
        }
    }

    // HNSW Flat Panorama
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {16, 32});
    std::vector<int> efSearches =
            benchmarks::int_list(FLAGS_efSearch, {16, 32, 64, 128, 256, 512});
    std::vector<int> n_levels_hnsw = benchmarks::int_list(FLAGS_n_levels, {8});
    for (int M : Ms) {
        for (int efSearch : efSearches) {
            for (int n_levels : n_levels_hnsw) {
                for (int nq : nqs) {
                    for (int pca : pcas) {
                        std::string name = "hnsw_flat_panorama/search/M:" +
                                std::to_string(M) +
                                "/efSearch:" + std::to_string(efSearch) +
                                "/n_levels:" + std::to_string(n_levels) +
                                "/nq:" + std::to_string(nq) +
                                "/pca:" + std::to_string(pca);
                        auto* b = benchmark::RegisterBenchmark(
                                name.c_str(),
                                bench_hnsw_flat_panorama_search,
                                d,
                                nb,
                                nq,
                                k,
                                M,
                                efSearch,
                                n_levels,
                                pca);
                        if (FLAGS_iterations > 0)
                            b->Iterations(FLAGS_iterations);
                    }
                }
            }
        }
    }

    // IndexRefinePanorama
    std::vector<int> nprobes_refine =
            benchmarks::int_list(FLAGS_nprobe, {4, 16, 64, 256});
    std::vector<int> k_factors =
            benchmarks::int_list(FLAGS_k_factor, {1, 8, 64, 256, 1024});
    for (int nprobe : nprobes_refine) {
        for (int k_factor_int : k_factors) {
            float k_factor = (float)k_factor_int;
            for (int nq : nqs) {
                for (int pca : pcas) {
                    std::string name = "refine_panorama/search/nprobe:" +
                            std::to_string(nprobe) +
                            "/k_factor:" + std::to_string((int)k_factor) +
                            "/nq:" + std::to_string(nq) +
                            "/pca:" + std::to_string(pca);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_refine_panorama_search,
                            d,
                            nb,
                            nq,
                            k,
                            256,
                            nprobe,
                            8,
                            k_factor,
                            pca);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
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
        int snt = (int)sift.nt;

        // Default to the full query set; the (expensive) build happens once
        // per registration, outside the timed loop.
        std::vector<int> sift_nqs =
                benchmarks::int_list(FLAGS_nq, {(int)sift.nq});
        std::vector<int> sift_n_levels =
                benchmarks::int_list(FLAGS_n_levels, {8});

        // FlatL2Panorama on real data (exact search)
        std::vector<int> sift_batches =
                benchmarks::int_list(FLAGS_batch, {512});
        for (int n_levels : sift_n_levels) {
            for (int batch_size : sift_batches) {
                for (int nq : sift_nqs) {
                    if ((size_t)nq > sift.nq)
                        continue;
                    for (int pca : pcas) {
                        std::string name = ds +
                                "/flat_l2_panorama/search/n_levels:" +
                                std::to_string(n_levels) +
                                "/batch:" + std::to_string(batch_size) +
                                "/nq:" + std::to_string(nq) +
                                "/pca:" + std::to_string(pca);
                        auto* b = benchmark::RegisterBenchmark(
                                name.c_str(),
                                bench_flat_l2_panorama_search_dataset,
                                sift.xt.data(),
                                snt,
                                sift.xb.data(),
                                snb,
                                sift.xq.data(),
                                nq,
                                sift.gt.data(),
                                sift.gt_k,
                                sd,
                                k,
                                n_levels,
                                batch_size,
                                pca);
                        if (FLAGS_iterations > 0)
                            b->Iterations(FLAGS_iterations);
                    }
                }
            }
        }

        // IVFFlatPanorama on real data.
        // nlist=128 is a reasonable default for 1M-scale datasets.
        std::vector<int> sift_nlists = benchmarks::int_list(FLAGS_nlist, {128});
        std::vector<int> sift_nprobes =
                benchmarks::int_list(FLAGS_nprobe, {4, 16, 64});
        for (int nlist : sift_nlists) {
            for (int nprobe : sift_nprobes) {
                if (nprobe > nlist)
                    continue;
                for (int n_levels : sift_n_levels) {
                    for (int nq : sift_nqs) {
                        if ((size_t)nq > sift.nq)
                            continue;
                        for (int pca : pcas) {
                            std::string name = ds +
                                    "/ivf_flat_panorama/search/nlist:" +
                                    std::to_string(nlist) +
                                    "/nprobe:" + std::to_string(nprobe) +
                                    "/n_levels:" + std::to_string(n_levels) +
                                    "/nq:" + std::to_string(nq) +
                                    "/pca:" + std::to_string(pca);
                            auto* b = benchmark::RegisterBenchmark(
                                    name.c_str(),
                                    bench_ivf_flat_panorama_search_dataset,
                                    sift.xt.data(),
                                    snt,
                                    sift.xb.data(),
                                    snb,
                                    sift.xq.data(),
                                    nq,
                                    sift.gt.data(),
                                    sift.gt_k,
                                    sd,
                                    k,
                                    nlist,
                                    nprobe,
                                    n_levels,
                                    pca);
                            if (FLAGS_iterations > 0)
                                b->Iterations(FLAGS_iterations);
                        }
                    }
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
