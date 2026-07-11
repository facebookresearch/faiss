/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// cppcontrib SaDecodeKernels Benchmarks
//
// Compares the templated faiss::cppcontrib SaDecodeKernels (store / accum)
// against the generic faiss::Index::sa_decode() over a matrix of codecs.
// Each (codec, engine, experiment) triple is a separate Google Benchmark:
//   engine     = faiss   -> Index::sa_decode()
//              = kernel  -> SaDecodeKernels store()/accum()
//   experiment = store_seq  -> decode a contiguous block, one vector per call
//              = store_rnd  -> decode in a random order, one vector per call
//              = accum      -> weighted linear combination, 1 code per call
//              = accum2     -> weighted linear combination, 2 codes per call
//              = accum3     -> weighted linear combination, 3 codes per call
// (accum2/accum3 exist only for the kernels, so faiss registers only
// store_seq/store_rnd/accum.)
//
// Codec families covered:
//   * PQ[64,32,16,8,4]np              -> IndexPQDecoder<128, FINE_SIZE>
//   * IVF256,PQ[64,32,16,8,4]np       -> Index2LevelDecoder<128, 128, FINE>
//   * Residual4x8,PQ[64,32,16,8,4]    -> Index2LevelDecoder<128, 32, FINE>
//
// Condensed cases (to keep the harness manageable; the operations and
// kernels above are representative of them):
//   * PQx10 IndexPQDecoder<...,10> variants.
//   * MinMaxFP16 wrapper variants (IndexMinMaxFP16Decoder<SubT>): the wrapper
//     adds an fp16 min/scale rescale around the same Sub-decoder kernels and
//     an extra minvAccum accumulator argument to accum().
//   * Residual1x10,PQ*x10 unusual-bits Index2LevelDecoder<...,16,10> variants.
//   * accum2u/accum3u "unique centroids" overloads (they pass the same shared
//     centroid tables, so accum2/accum3 shared-centroid cover the same code).
//
// Design notes:
//   * Auto-iterates by default (use --iterations to pin a fixed count).
//   * No kernel-vs-faiss decode error column; this bench focuses on
//     throughput only.

#include <gflags/gflags.h>
#include <omp.h>

#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <benchmark/benchmark.h>
#include <faiss/Index.h>
#include <faiss/Index2Layer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_factory.h>

#include <faiss/cppcontrib/SaDecodeKernels.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_int32(n, 65536 * 16, "number of database vectors (INDEX_SIZE)");
DEFINE_int32(d, 128, "vector dimension");
BENCH_DEFINE_DATASET_FILE_FLAGS();

enum Experiment {
    STORE_SEQ = 0,
    STORE_RND = 1,
    ACCUM = 2,
    ACCUM2 = 3,
    ACCUM3 = 4,
};

// -------- dataset helpers --------

static std::vector<float> generate(size_t n, size_t d) {
    std::vector<float> data(n * d);
    std::minstd_rand rng(345);
    std::uniform_real_distribution<float> ux(0, 1);
    for (size_t k = 0; k < n; k++) {
        for (size_t j = 0; j < d; j++) {
            data[k * d + j] = ux(rng);
        }
    }
    return data;
}

static bool testIfIVFPQ(
        const Index* index,
        const float** pqCoarseCentroidsQ,
        const float** pqFineCentroidsQ) {
    const IndexIVFPQ* const indexQ = dynamic_cast<const IndexIVFPQ*>(index);
    if (indexQ == nullptr) {
        return false;
    }
    const auto coarseIndexQ =
            dynamic_cast<const IndexFlatCodes*>(indexQ->quantizer);
    if (coarseIndexQ == nullptr) {
        return false;
    }
    *pqFineCentroidsQ = indexQ->pq.centroids.data();
    *pqCoarseCentroidsQ =
            reinterpret_cast<const float*>(coarseIndexQ->codes.data());
    return true;
}

static bool testIfResidualPQ(
        const Index* index,
        const float** pqCoarseCentroidsQ,
        const float** pqFineCentroidsQ) {
    const Index2Layer* const indexQ = dynamic_cast<const Index2Layer*>(index);
    if (indexQ == nullptr) {
        return false;
    }
    const auto coarseIndexQ =
            dynamic_cast<const MultiIndexQuantizer*>(indexQ->q1.quantizer);
    if (coarseIndexQ == nullptr) {
        return false;
    }
    *pqFineCentroidsQ = indexQ->pq.centroids.data();
    *pqCoarseCentroidsQ = coarseIndexQ->pq.centroids.data();
    return true;
}

// Set in main() when an on-disk dataset is available; dataset-backed codecs
// train on its train vectors and encode its base vectors.
static const benchmarks::DatasetSIFT1M* g_dataset = nullptr;

// A trained codec plus its encoded dataset and extracted centroid tables,
// cached so the faiss and kernel benchmarks for a codec share one build.
struct Codec {
    std::shared_ptr<Index> index;
    std::vector<uint8_t> encoded;
    size_t codeSize = 0;
    const float* coarse = nullptr; // null for plain PQ
    const float* fine = nullptr;
    std::vector<uint64_t> rndIdx; // random point order
    std::vector<float> weights;   // random accumulation weights
    int n = 0;
    int d = 0;
};

static Codec& get_codec(const std::string& description, bool use_dataset) {
    static std::map<std::string, Codec> cache;
    std::string key = (use_dataset ? "dataset/" : "") + description;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    int n = FLAGS_n;
    int d = FLAGS_d;
    if (use_dataset) {
        // Decode throughput does not need the full base; encoding a subset
        // of the real base vectors keeps per-codec setup cheap.
        n = std::min({(int)FLAGS_n, (int)g_dataset->nb, 65536});
        d = (int)g_dataset->d;
    }

    Codec c;
    c.n = n;
    c.d = d;

    omp_set_num_threads(omp_get_max_threads());
    c.index = std::shared_ptr<Index>(index_factory(d, description.c_str()));
    std::vector<float> data;
    const float* encode_src = nullptr;
    if (use_dataset) {
        // Train on the real train set and encode real base vectors.
        c.index->train(g_dataset->nt, g_dataset->xt.data());
        encode_src = g_dataset->xb.data();
    } else {
        data = generate(n, d);
        c.index->train(n, data.data());
        encode_src = data.data();
    }
    c.codeSize = c.index->sa_code_size();
    c.encoded.resize((size_t)n * c.codeSize);
    c.index->sa_encode(n, encode_src, c.encoded.data());

    // extract centroid tables
    if (!testIfIVFPQ(c.index.get(), &c.coarse, &c.fine)) {
        if (!testIfResidualPQ(c.index.get(), &c.coarse, &c.fine)) {
            const IndexPQ* const indexQ =
                    dynamic_cast<const IndexPQ*>(c.index.get());
            if (indexQ != nullptr) {
                c.fine = indexQ->pq.centroids.data();
            }
        }
    }

    // precompute a random point order and random weights
    std::default_random_engine rng(123);
    std::uniform_int_distribution<uint64_t> un(0, n - 1);
    std::uniform_real_distribution<float> u(0, 1);
    c.rndIdx.resize(n);
    c.weights.resize(n);
    for (int i = 0; i < n; i++) {
        c.rndIdx[i] = un(rng);
        c.weights[i] = u(rng);
    }

    auto res = cache.emplace(key, std::move(c));
    return res.first->second;
}

static void set_common_counters(
        benchmark::State& state,
        const Codec& c,
        int experiment) {
    state.SetItemsProcessed(state.iterations() * c.n);
    state.counters["n"] = c.n;
    state.counters["d"] = c.d;
    state.counters["experiment"] = experiment;
}

// -------- faiss engine (Index::sa_decode) --------

static void bench_faiss(
        benchmark::State& state,
        std::string description,
        int experiment,
        bool use_dataset) {
    Codec& c = get_codec(description, use_dataset);
    const int n = c.n;
    const int d = c.d;
    const size_t cs = c.codeSize;
    omp_set_num_threads(1);

    std::vector<float> out((size_t)n * d, 0);
    std::vector<float> temp(d, 0);

    for (auto _ : state) {
        switch (experiment) {
            case STORE_SEQ:
                c.index->sa_decode(n, c.encoded.data(), out.data());
                break;
            case STORE_RND:
                for (int i = 0; i < n; i++) {
                    c.index->sa_decode(
                            1,
                            c.encoded.data() + c.rndIdx[i] * cs,
                            out.data() + (size_t)i * d);
                }
                break;
            case ACCUM:
                for (int i = 0; i < n; i++) {
                    c.index->sa_decode(
                            1,
                            c.encoded.data() + c.rndIdx[i] * cs,
                            temp.data());
                    const float w = c.weights[i];
                    for (int j = 0; j < d; j++) {
                        out[(size_t)i * d + j] += w * temp[j];
                    }
                }
                break;
        }
        benchmark::DoNotOptimize(out[0]);
    }
    set_common_counters(state, c, experiment);
    if (use_dataset) {
        state.counters["dataset"] = 1;
    }
}

// -------- kernel engine: two-level codecs (IVFPQ, Residual+PQ) --------

template <typename T>
static void bench_kernel_2level(
        benchmark::State& state,
        std::string description,
        int experiment,
        bool use_dataset) {
    Codec& c = get_codec(description, use_dataset);
    const int n = c.n;
    const int d = c.d;
    const size_t cs = c.codeSize;
    const float* coarse = c.coarse;
    const float* fine = c.fine;
    const uint8_t* enc = c.encoded.data();
    const uint64_t* idx = c.rndIdx.data();
    const float* w = c.weights.data();
    omp_set_num_threads(1);

    std::vector<float> out((size_t)n * d, 0);

    for (auto _ : state) {
        switch (experiment) {
            case STORE_SEQ:
                for (int i = 0; i < n; i++) {
                    T::store(
                            coarse,
                            fine,
                            enc + (size_t)i * cs,
                            out.data() + (size_t)i * d);
                }
                break;
            case STORE_RND:
                for (int i = 0; i < n; i++) {
                    T::store(
                            coarse,
                            fine,
                            enc + idx[i] * cs,
                            out.data() + (size_t)i * d);
                }
                break;
            case ACCUM:
                for (int i = 0; i < n; i++) {
                    T::accum(
                            coarse,
                            fine,
                            enc + idx[i] * cs,
                            w[i],
                            out.data() + (size_t)i * d);
                }
                break;
            case ACCUM2:
                for (int i = 0; i + 2 <= n; i += 2) {
                    T::accum(
                            coarse,
                            fine,
                            enc + idx[i] * cs,
                            w[i],
                            enc + idx[i + 1] * cs,
                            w[i + 1],
                            out.data() + (size_t)i * d);
                }
                break;
            case ACCUM3:
                for (int i = 0; i + 3 <= n; i += 3) {
                    T::accum(
                            coarse,
                            fine,
                            enc + idx[i] * cs,
                            w[i],
                            enc + idx[i + 1] * cs,
                            w[i + 1],
                            enc + idx[i + 2] * cs,
                            w[i + 2],
                            out.data() + (size_t)i * d);
                }
                break;
        }
        benchmark::DoNotOptimize(out[0]);
    }
    set_common_counters(state, c, experiment);
    if (use_dataset) {
        state.counters["dataset"] = 1;
    }
}

// -------- kernel engine: plain PQ codecs --------

template <typename T>
static void bench_kernel_pq(
        benchmark::State& state,
        std::string description,
        int experiment,
        bool use_dataset) {
    Codec& c = get_codec(description, use_dataset);
    const int n = c.n;
    const int d = c.d;
    const size_t cs = c.codeSize;
    const float* fine = c.fine;
    const uint8_t* enc = c.encoded.data();
    const uint64_t* idx = c.rndIdx.data();
    const float* w = c.weights.data();
    omp_set_num_threads(1);

    std::vector<float> out((size_t)n * d, 0);

    for (auto _ : state) {
        switch (experiment) {
            case STORE_SEQ:
                for (int i = 0; i < n; i++) {
                    T::store(
                            fine,
                            enc + (size_t)i * cs,
                            out.data() + (size_t)i * d);
                }
                break;
            case STORE_RND:
                for (int i = 0; i < n; i++) {
                    T::store(
                            fine,
                            enc + idx[i] * cs,
                            out.data() + (size_t)i * d);
                }
                break;
            case ACCUM:
                for (int i = 0; i < n; i++) {
                    T::accum(
                            fine,
                            enc + idx[i] * cs,
                            w[i],
                            out.data() + (size_t)i * d);
                }
                break;
            case ACCUM2:
                for (int i = 0; i + 2 <= n; i += 2) {
                    T::accum(
                            fine,
                            enc + idx[i] * cs,
                            w[i],
                            enc + idx[i + 1] * cs,
                            w[i + 1],
                            out.data() + (size_t)i * d);
                }
                break;
            case ACCUM3:
                for (int i = 0; i + 3 <= n; i += 3) {
                    T::accum(
                            fine,
                            enc + idx[i] * cs,
                            w[i],
                            enc + idx[i + 1] * cs,
                            w[i + 1],
                            enc + idx[i + 2] * cs,
                            w[i + 2],
                            out.data() + (size_t)i * d);
                }
                break;
        }
        benchmark::DoNotOptimize(out[0]);
    }
    set_common_counters(state, c, experiment);
    if (use_dataset) {
        state.counters["dataset"] = 1;
    }
}

// -------- registration helpers --------

struct ExpInfo {
    Experiment exp;
    const char* name;
    bool faiss_ok; // whether the faiss engine supports this experiment
};

static const std::vector<ExpInfo> kExperiments = {
        {STORE_SEQ, "store_seq", true},
        {STORE_RND, "store_rnd", true},
        {ACCUM, "accum", true},
        {ACCUM2, "accum2", false},
        {ACCUM3, "accum3", false},
};

static void maybe_iters(benchmark::Benchmark* b) {
    if (FLAGS_iterations > 0)
        b->Iterations(FLAGS_iterations);
}

template <typename T>
static void register_pq_codec(
        const std::string& desc,
        bool use_dataset = false,
        const std::string& name_prefix = "") {
    for (const auto& e : kExperiments) {
        std::string base = name_prefix + "sa_decode/codec:" + desc +
                "/experiment:" + e.name;
        maybe_iters(
                benchmark::RegisterBenchmark(
                        (base + "/engine:kernel").c_str(),
                        bench_kernel_pq<T>,
                        desc,
                        (int)e.exp,
                        use_dataset));
        if (e.faiss_ok) {
            maybe_iters(
                    benchmark::RegisterBenchmark(
                            (base + "/engine:faiss").c_str(),
                            bench_faiss,
                            desc,
                            (int)e.exp,
                            use_dataset));
        }
    }
}

template <typename T>
static void register_2level_codec(
        const std::string& desc,
        bool use_dataset = false,
        const std::string& name_prefix = "") {
    for (const auto& e : kExperiments) {
        std::string base = name_prefix + "sa_decode/codec:" + desc +
                "/experiment:" + e.name;
        maybe_iters(
                benchmark::RegisterBenchmark(
                        (base + "/engine:kernel").c_str(),
                        bench_kernel_2level<T>,
                        desc,
                        (int)e.exp,
                        use_dataset));
        if (e.faiss_ok) {
            maybe_iters(
                    benchmark::RegisterBenchmark(
                            (base + "/engine:faiss").c_str(),
                            bench_faiss,
                            desc,
                            (int)e.exp,
                            use_dataset));
        }
    }
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "cppcontrib SaDecodeKernels store/accum vs Index::sa_decode over "
            "PQ, IVFPQ and Residual+PQ codecs, on synthetic data and SIFT1M",
            "--n=65536 "
            "--benchmark_filter='sa_decode/codec:PQ16np/.*/engine:kernel'");

    // Plain PQx8: IndexPQDecoder<128, FINE_SIZE>
    register_pq_codec<cppcontrib::IndexPQDecoder<128, 2>>("PQ64np");
    register_pq_codec<cppcontrib::IndexPQDecoder<128, 4>>("PQ32np");
    register_pq_codec<cppcontrib::IndexPQDecoder<128, 8>>("PQ16np");
    register_pq_codec<cppcontrib::IndexPQDecoder<128, 16>>("PQ8np");
    register_pq_codec<cppcontrib::IndexPQDecoder<128, 32>>("PQ4np");

    // IVFPQ: Index2LevelDecoder<128, 128, FINE_SIZE>
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 128, 2>>(
            "IVF256,PQ64np");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 128, 4>>(
            "IVF256,PQ32np");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 128, 8>>(
            "IVF256,PQ16np");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 128, 16>>(
            "IVF256,PQ8np");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 128, 32>>(
            "IVF256,PQ4np");

    // Residual+PQ: Index2LevelDecoder<128, 32, FINE_SIZE>
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 32, 2>>(
            "Residual4x8,PQ64");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 32, 4>>(
            "Residual4x8,PQ32");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 32, 8>>(
            "Residual4x8,PQ16");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 32, 16>>(
            "Residual4x8,PQ8");
    register_2level_codec<cppcontrib::Index2LevelDecoder<128, 32, 32>>(
            "Residual4x8,PQ4");

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
        g_dataset = &sift;
        // The kernels above are compiled for d=128, so only a 128-dim
        // dataset (e.g. SIFT1M) can be decoded by them. One codec per
        // family, trained on the real train set and decoding real codes.
        if (sift.d == 128) {
            register_pq_codec<cppcontrib::IndexPQDecoder<128, 8>>(
                    "PQ16np", true, ds + "/");
            register_2level_codec<cppcontrib::Index2LevelDecoder<128, 128, 8>>(
                    "IVF256,PQ16np", true, ds + "/");
            register_2level_codec<cppcontrib::Index2LevelDecoder<128, 32, 8>>(
                    "Residual4x8,PQ16", true, ds + "/");
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
