/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Graph-Based Index Benchmarks (HNSW, NSG)
//
// Covers:
//   * ivf_hnsw_quantizer — IndexIVFFlat with an IndexHNSWFlat coarse
//     quantizer (nlist=16384, quantizer efSearch=64,
//     quantizer_trains_alone=2), build + nprobe search sweep.
//   * hnsw_locks — batched add() throughput with hnsw.retain_locks toggled
//     (use --threads=N for multiple threads).
//   * HNSWSQ QT_8bit/M=16 — the IndexHNSWSQ(d, QT_8bit, 16) config
//     (distinct from the QT_4bit/M=32 prune_headroom case). The HNSW
//     efSearch default sweep and NSG search_L default sweep are widened to
//     {16,32,64,128,256} and {-1,16,32,64,128,256}.
//
// Design notes:
//   * --threads defaults to 1.
//   * IVFFlat search lives in bench_index_ivf; k-means training/assignment
//     lives in bench_codec_kmeans (C++) and bench_codec_kmeans.py (Python).

#include <cstdio>
#include <map>
#include <memory>
#include <tuple>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexNSG.h>
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
DEFINE_string(
        M,
        "",
        "comma-separated HNSW M values "
        "(default: 16,32,64; sift1m group: 32)");
DEFINE_string(
        efConstruction,
        "",
        "comma-separated HNSW efConstruction values (default: 40,128)");
DEFINE_string(
        efSearch,
        "",
        "comma-separated HNSW efSearch values "
        "(default: 16,32,64,128,256)");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");
DEFINE_string(
        bounded,
        "",
        "comma-separated bounded-queue settings, 0/1 (default: 1,0)");
DEFINE_string(R, "", "comma-separated NSG R values (default: 32,64)");
DEFINE_string(
        search_L,
        "",
        "comma-separated NSG search_L values "
        "(default: -1,16,32,64,128,256; -1 = default/full search)");
DEFINE_string(
        nlist,
        "",
        "comma-separated IVF nlist values for the ivf_hnsw_quantizer group "
        "(default: 16384)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated IVF nprobe values for the ivf_hnsw_quantizer group "
        "(default: 1,4,16,64,256)");
DEFINE_string(
        headroom,
        "",
        "comma-separated hnsw.prune_headroom values for the hnsw_sq group "
        "(default: 0.0,0.04,0.08,0.12,0.16,0.20)");

static void bench_hnsw_build(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        int efConstruction) {
    AlignedTable<float> xb((size_t)d * nb);
    float_rand(xb.data(), (size_t)d * nb, 12345);

    omp_set_num_threads(FLAGS_threads);
    for (auto _ : state) {
        IndexHNSWFlat index(d, M);
        index.hnsw.efConstruction = efConstruction;
        index.verbose = false;
        index.add(nb, xb.data());
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["efConstruction"] = efConstruction;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_hnsw_search(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        int efConstruction,
        int efSearch,
        int nq,
        int k,
        bool bounded_queue) {
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xb.data(), (size_t)d * nb, 12345);
    float_rand(xq.data(), d * nq, 54321);

    IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.add(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    SearchParametersHNSW params;
    params.efSearch = efSearch;
    params.bounded_queue = bounded_queue;

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        index.search(
                nq, xq.data(), k, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["efSearch"] = efSearch;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["bounded_queue"] = bounded_queue;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant: accepts external data pointers
static void bench_hnsw_search_dataset(
        benchmark::State& state,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d,
        int M,
        int efConstruction,
        int efSearch,
        int k,
        bool bounded_queue) {
    IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.add(nb, xb);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    SearchParametersHNSW params;
    params.efSearch = efSearch;
    params.bounded_queue = bounded_queue;

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["efSearch"] = efSearch;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["bounded_queue"] = bounded_queue;
    state.counters["dataset"] = 1;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

static void bench_nsg_search(
        benchmark::State& state,
        int d,
        int nb,
        int R,
        int search_L,
        int nq,
        int k) {
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xb.data(), (size_t)d * nb, 12345);
    float_rand(xq.data(), d * nq, 54321);

    IndexNSGFlat index(d, R);
    index.verbose = false;
    index.build_type = 1; // NNDescent build
    omp_set_num_threads(FLAGS_threads);
    index.add(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    index.nsg.search_L = search_L;

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["R"] = R;
    state.counters["search_L"] = search_L;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

// Mirror of benchs/python/bench_index_graph.py (bench_hnsw_sq_build /
// bench_hnsw_sq_search) — IndexHNSWSQ (QT_4bit) build and search with an
// hnsw.prune_headroom sweep.

// headroom values are floats; bench_cli_utils has no float_list, so parse
// the string tokens with std::stof.
static std::vector<float> headroom_list(
        const std::string& flag_value,
        std::vector<std::string> defaults) {
    std::vector<std::string> tokens =
            benchmarks::str_list(flag_value, std::move(defaults));
    std::vector<float> values;
    for (const std::string& token : tokens) {
        try {
            size_t pos = 0;
            float v = std::stof(token, &pos);
            if (pos != token.size()) {
                throw std::invalid_argument(token);
            }
            values.push_back(v);
        } catch (const std::exception&) {
            fprintf(stderr,
                    "cannot parse '%s' as a float list\n",
                    flag_value.c_str());
            exit(1);
        }
    }
    return values;
}

// Format a headroom value for benchmark names without trailing zeros,
// e.g. 0.08f -> "0.08".
static std::string headroom_name(float headroom) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", headroom);
    return buf;
}

static void bench_hnsw_sq_build(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        int efConstruction,
        float headroom) {
    AlignedTable<float> xb((size_t)d * nb);
    float_rand(xb.data(), (size_t)d * nb, 12345);

    omp_set_num_threads(FLAGS_threads);
    for (auto _ : state) {
        // Fresh index per iteration; SQ training is untimed setup, the
        // timed region is only add().
        state.PauseTiming();
        auto index =
                std::make_unique<IndexHNSWSQ>(d, ScalarQuantizer::QT_4bit, M);
        index->hnsw.efConstruction = efConstruction;
        index->hnsw.prune_headroom = headroom;
        index->verbose = false;
        index->train(nb, xb.data());
        state.ResumeTiming();

        index->add(nb, xb.data());
        benchmark::DoNotOptimize(index->ntotal);

        state.PauseTiming();
        index.reset();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["efConstruction"] = efConstruction;
    state.counters["headroom"] = headroom;
    state.counters["threads"] = FLAGS_threads;
}

// Build once per headroom value (static cache) and reuse across the
// efSearch/nq sweep; efConstruction fixed at 40.
static IndexHNSWSQ* hnsw_sq_index(int d, int nb, int M, float headroom) {
    struct Key {
        int M;
        float headroom;
        bool operator<(const Key& o) const {
            return std::tie(M, headroom) < std::tie(o.M, o.headroom);
        }
    };
    static std::map<Key, std::unique_ptr<IndexHNSWSQ>> cache;
    Key key{M, headroom};
    auto it = cache.find(key);
    if (it == cache.end()) {
        AlignedTable<float> xb((size_t)d * nb);
        float_rand(xb.data(), (size_t)d * nb, 12345);

        auto index =
                std::make_unique<IndexHNSWSQ>(d, ScalarQuantizer::QT_4bit, M);
        index->hnsw.efConstruction = 40;
        index->hnsw.prune_headroom = headroom;
        index->verbose = false;
        omp_set_num_threads(FLAGS_threads);
        index->train(nb, xb.data());
        index->add(nb, xb.data());
        it = cache.emplace(key, std::move(index)).first;
    }
    return it->second.get();
}

static void bench_hnsw_sq_search(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        float headroom,
        int efSearch,
        int nq,
        int k) {
    IndexHNSWSQ* index = hnsw_sq_index(d, nb, M, headroom);
    omp_set_num_threads(FLAGS_threads);

    AlignedTable<float> xq(d * nq);
    float_rand(xq.data(), d * nq, 54321);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    SearchParametersHNSW params;
    params.efSearch = efSearch;

    // Warmup
    index->search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        index->search(
                nq, xq.data(), k, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["headroom"] = headroom;
    state.counters["efSearch"] = efSearch;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

// HNSWSQ QT_8bit/M=16: IndexHNSWSQ(d, QT_8bit, 16). Distinct
// from the QT_4bit/M=32 prune_headroom case above. efConstruction fixed at
// 40. Build once per M (static cache) and reuse across the efSearch/nq sweep.
static IndexHNSWSQ* hnsw_sq8_index(int d, int nb, int M) {
    static std::map<int, std::unique_ptr<IndexHNSWSQ>> cache;
    auto it = cache.find(M);
    if (it == cache.end()) {
        AlignedTable<float> xb((size_t)d * nb);
        float_rand(xb.data(), (size_t)d * nb, 12345);
        auto index =
                std::make_unique<IndexHNSWSQ>(d, ScalarQuantizer::QT_8bit, M);
        index->hnsw.efConstruction = 40;
        index->verbose = false;
        omp_set_num_threads(FLAGS_threads);
        index->train(nb, xb.data());
        index->add(nb, xb.data());
        it = cache.emplace(M, std::move(index)).first;
    }
    return it->second.get();
}

static void bench_hnsw_sq8_search(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        int efSearch,
        int nq,
        int k) {
    IndexHNSWSQ* index = hnsw_sq8_index(d, nb, M);
    omp_set_num_threads(FLAGS_threads);

    AlignedTable<float> xq(d * nq);
    float_rand(xq.data(), d * nq, 54321);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    SearchParametersHNSW params;
    params.efSearch = efSearch;

    // Warmup
    index->search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        index->search(
                nq, xq.data(), k, distances.data(), labels.data(), &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["efSearch"] = efSearch;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

// IVF+HNSW-quantizer: IndexIVFFlat whose coarse
// quantizer is an IndexHNSWFlat. nlist=16384, quantizer M=32 with
// efSearch=64, quantizer_trains_alone=2. Build once per (nlist) key and reuse
// across the nprobe sweep.
struct IvfHnswBundle {
    std::unique_ptr<IndexHNSWFlat> quantizer;
    std::unique_ptr<IndexIVFFlat> index;
};

static IndexIVFFlat* ivf_hnsw_quantizer_index(
        int d,
        int nb,
        int nlist,
        int quant_M,
        int quant_efSearch) {
    struct Key {
        int nlist;
        int quant_M;
        int quant_efSearch;
        bool operator<(const Key& o) const {
            return std::tie(nlist, quant_M, quant_efSearch) <
                    std::tie(o.nlist, o.quant_M, o.quant_efSearch);
        }
    };
    static std::map<Key, IvfHnswBundle> cache;
    Key key{nlist, quant_M, quant_efSearch};
    auto it = cache.find(key);
    if (it == cache.end()) {
        AlignedTable<float> xb((size_t)d * nb);
        float_rand(xb.data(), (size_t)d * nb, 12345);

        IvfHnswBundle bundle;
        bundle.quantizer = std::make_unique<IndexHNSWFlat>(d, quant_M);
        bundle.quantizer->hnsw.efSearch = quant_efSearch;
        bundle.index = std::make_unique<IndexIVFFlat>(
                bundle.quantizer.get(), d, nlist);
        bundle.index->cp.min_points_per_centroid = 5; // quiet warning
        bundle.index->quantizer_trains_alone = 2;
        bundle.index->verbose = false;
        omp_set_num_threads(FLAGS_threads);
        bundle.index->train(nb, xb.data());
        bundle.index->add(nb, xb.data());
        it = cache.emplace(key, std::move(bundle)).first;
    }
    return it->second.index.get();
}

static void bench_ivf_hnsw_quantizer_search(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        int quant_M,
        int quant_efSearch,
        int nprobe,
        int nq,
        int k) {
    IndexIVFFlat* index =
            ivf_hnsw_quantizer_index(d, nb, nlist, quant_M, quant_efSearch);
    index->nprobe = nprobe;
    omp_set_num_threads(FLAGS_threads);

    AlignedTable<float> xq(d * nq);
    float_rand(xq.data(), d * nq, 54321);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index->search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index->search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

// HNSW batched-add throughput with retain_locks toggled. retain_locks keeps
// per-node locks across add() calls so concurrent batched adds avoid
// re-locking. Timed region is the batched-add loop over the whole database
// (fresh index per iteration).
static void bench_hnsw_locks_add(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        bool retain_locks) {
    AlignedTable<float> xb((size_t)d * nb);
    float_rand(xb.data(), (size_t)d * nb, 12345);
    int batch_size = std::max(1, nb / 100);

    omp_set_num_threads(FLAGS_threads);
    for (auto _ : state) {
        IndexHNSWFlat index(d, M);
        index.retain_locks = retain_locks;
        index.verbose = false;
        for (int i = 0; i < nb; i += batch_size) {
            int n = std::min(batch_size, nb - i);
            index.add(n, xb.data() + (size_t)i * d);
        }
        benchmark::DoNotOptimize(index.ntotal);
    }
    state.SetItemsProcessed(state.iterations() * nb);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["M"] = M;
    state.counters["retain_locks"] = retain_locks;
    state.counters["threads"] = FLAGS_threads;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "graph-based index benchmarks: HNSW build/search, NSG search, "
            "and SIFT1M HNSW search",
            "--M=32 --efSearch=64 --benchmark_filter='hnsw/search/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    int k = 10;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {16, 32, 64});
    std::vector<int> efCs =
            benchmarks::int_list(FLAGS_efConstruction, {40, 128});
    std::vector<int> efSs =
            benchmarks::int_list(FLAGS_efSearch, {16, 32, 64, 128, 256});
    std::vector<int> boundeds = benchmarks::int_list(FLAGS_bounded, {1, 0});
    std::vector<int> Rs = benchmarks::int_list(FLAGS_R, {32, 64});
    std::vector<int> search_Ls =
            benchmarks::int_list(FLAGS_search_L, {-1, 16, 32, 64, 128, 256});

    // HNSW build
    for (int M : Ms) {
        for (int efC : efCs) {
            std::string name = "hnsw/build/M:" + std::to_string(M) +
                    "/efConstruction:" + std::to_string(efC);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_hnsw_build, d, nb, M, efC);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // HNSW search
    for (int M : Ms) {
        for (int efS : efSs) {
            for (int nq : nqs) {
                for (int bq_int : boundeds) {
                    bool bq = bq_int != 0;
                    std::string name = "hnsw/search/M:" + std::to_string(M) +
                            "/efSearch:" + std::to_string(efS) +
                            "/nq:" + std::to_string(nq) +
                            "/bounded:" + std::to_string(bq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_hnsw_search,
                            d,
                            nb,
                            M,
                            40,
                            efS,
                            nq,
                            k,
                            bq);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // NSG search
    for (int R : Rs) {
        for (int search_L : search_Ls) {
            for (int nq : nqs) {
                std::string name = "nsg/search/R:" + std::to_string(R) +
                        "/search_L:" + std::to_string(search_L) +
                        "/nq:" + std::to_string(nq);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_nsg_search,
                        d,
                        nb,
                        R,
                        search_L,
                        nq,
                        k);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // HNSW+SQ (QT_4bit) prune_headroom sweep, mirroring
    // bench_index_graph.py::bench_hnsw_sq_build / bench_hnsw_sq_search.
    // Defaults: d=128, nb=50000, M=32, efConstruction=40.
    int sq_d = 128;
    int sq_nb = 50000;
    std::vector<int> sq_Ms = benchmarks::int_list(FLAGS_M, {32});
    std::vector<int> sq_efCs = benchmarks::int_list(FLAGS_efConstruction, {40});
    // Default headroom sweep for both build and search.
    std::vector<float> headrooms = headroom_list(
            FLAGS_headroom, {"0.0", "0.04", "0.08", "0.12", "0.16", "0.20"});

    // HNSW+SQ build: train untimed, time only add (fresh index per
    // iteration)
    for (int M : sq_Ms) {
        for (int efC : sq_efCs) {
            for (float headroom : headrooms) {
                std::string name = "hnsw_sq/build/M:" + std::to_string(M) +
                        "/efConstruction:" + std::to_string(efC) +
                        "/headroom:" + headroom_name(headroom);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_hnsw_sq_build,
                        sq_d,
                        sq_nb,
                        M,
                        efC,
                        headroom);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // HNSW+SQ search: one cached index per headroom value
    for (int M : sq_Ms) {
        for (float headroom : headrooms) {
            for (int efS : efSs) {
                for (int nq : nqs) {
                    std::string name = "hnsw_sq/search/M:" + std::to_string(M) +
                            "/headroom:" + headroom_name(headroom) +
                            "/efSearch:" + std::to_string(efS) +
                            "/nq:" + std::to_string(nq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_hnsw_sq_search,
                            sq_d,
                            sq_nb,
                            M,
                            headroom,
                            efS,
                            nq,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // HNSWSQ QT_8bit/M=16 search.
    // Defaults: d=128, nb=50000, M=16, efConstruction=40.
    {
        int d8 = 128;
        int nb8 = 50000;
        std::vector<int> Ms8 = benchmarks::int_list(FLAGS_M, {16});
        for (int M : Ms8) {
            for (int efS : efSs) {
                for (int nq : nqs) {
                    std::string name =
                            "hnsw_sq8/search/M:" + std::to_string(M) +
                            "/efSearch:" + std::to_string(efS) +
                            "/nq:" + std::to_string(nq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_hnsw_sq8_search,
                            d8,
                            nb8,
                            M,
                            efS,
                            nq,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // IVF+HNSW-quantizer search.
    // nlist=16384, quantizer M=32 efSearch=64,
    // quantizer_trains_alone=2, nprobe sweep {1,4,16,64,256}.
    {
        std::vector<int> nlists = benchmarks::int_list(FLAGS_nlist, {16384});
        std::vector<int> nprobes =
                benchmarks::int_list(FLAGS_nprobe, {1, 4, 16, 64, 256});
        int quant_M = 32;
        int quant_efSearch = 64;
        for (int nlist : nlists) {
            for (int nprobe : nprobes) {
                for (int nq : nqs) {
                    std::string name = "ivf_hnsw_quantizer/search/nlist:" +
                            std::to_string(nlist) +
                            "/nprobe:" + std::to_string(nprobe) +
                            "/nq:" + std::to_string(nq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_ivf_hnsw_quantizer_search,
                            d,
                            nb,
                            nlist,
                            quant_M,
                            quant_efSearch,
                            nprobe,
                            nq,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // HNSW batched-add throughput with retain_locks toggled.
    // M=32; run under --threads=N.
    {
        std::vector<int> Ms_locks = benchmarks::int_list(FLAGS_M, {32});
        for (int M : Ms_locks) {
            for (int rl : {0, 1}) {
                std::string name = "hnsw_locks/add/M:" + std::to_string(M) +
                        "/retain_locks:" + std::to_string(rl);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(), bench_hnsw_locks_add, d, nb, M, rl != 0);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // SIFT1M-based HNSW benchmarks (if dataset available)
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

        std::vector<int> Ms_sift = benchmarks::int_list(FLAGS_M, {32});
        std::vector<int> efSs_sift =
                benchmarks::int_list(FLAGS_efSearch, {16, 32, 64, 128, 256});
        for (int M : Ms_sift) {
            for (int efS : efSs_sift) {
                for (int bq_int : boundeds) {
                    bool bq = bq_int != 0;
                    std::string name = ds +
                            "/hnsw/search/M:" + std::to_string(M) +
                            "/efSearch:" + std::to_string(efS) +
                            "/bounded:" + std::to_string(bq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_hnsw_search_dataset,
                            sift.xb.data(),
                            snb,
                            sift.xq.data(),
                            snq,
                            sift.gt.data(),
                            sift.gt_k,
                            sd,
                            M,
                            40,
                            efS,
                            k,
                            bq);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
