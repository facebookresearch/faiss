/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Binary Index Benchmarks

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
DEFINE_string(
        d,
        "",
        "comma-separated binary vector dimensions in bits (default: 128,256)");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");
DEFINE_string(
        nlist,
        "",
        "comma-separated BinaryIVF nlist values (default: 256,1024)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated BinaryIVF nprobe values (default: 1,8,32)");
DEFINE_string(M, "", "comma-separated BinaryHNSW M values (default: 16,32)");
DEFINE_string(
        efSearch,
        "",
        "comma-separated BinaryHNSW efSearch values (default: 16,64)");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

static void bench_binary_flat_search(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int k) {
    size_t code_size = d / 8;
    std::vector<uint8_t> xb(code_size * nb);
    std::vector<uint8_t> xq(code_size * nq);

    // Generate random binary data
    std::vector<float> tmp(code_size * (nb + nq));
    float_rand(tmp.data(), code_size * (nb + nq), 12345);
    for (size_t i = 0; i < code_size * nb; i++) {
        xb[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    for (size_t i = 0; i < code_size * nq; i++) {
        xq[i] = (uint8_t)(((int)(tmp[code_size * nb + i] * 256)) & 0xFF);
    }

    IndexBinaryFlat index(d);
    omp_set_num_threads(FLAGS_threads);
    index.add(nb, xb.data());

    std::vector<int32_t> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_binary_ivf_search(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        int nprobe,
        int nq,
        int k) {
    size_t code_size = d / 8;
    std::vector<uint8_t> xb(code_size * nb);
    std::vector<uint8_t> xq(code_size * nq);

    std::vector<float> tmp(code_size * (nb + nq));
    float_rand(tmp.data(), code_size * (nb + nq), 12345);
    for (size_t i = 0; i < code_size * nb; i++) {
        xb[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    for (size_t i = 0; i < code_size * nq; i++) {
        xq[i] = (uint8_t)(((int)(tmp[code_size * nb + i] * 256)) & 0xFF);
    }

    IndexBinaryFlat quantizer(d);
    IndexBinaryIVF index(&quantizer, d, nlist);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nb, xb.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;

    std::vector<int32_t> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
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

static void bench_binary_hnsw_search(
        benchmark::State& state,
        int d,
        int nb,
        int M,
        int efSearch,
        int nq,
        int k) {
    size_t code_size = d / 8;
    std::vector<uint8_t> xb(code_size * nb);
    std::vector<uint8_t> xq(code_size * nq);

    std::vector<float> tmp(code_size * (nb + nq));
    float_rand(tmp.data(), code_size * (nb + nq), 12345);
    for (size_t i = 0; i < code_size * nb; i++) {
        xb[i] = (uint8_t)(((int)(tmp[i] * 256)) & 0xFF);
    }
    for (size_t i = 0; i < code_size * nq; i++) {
        xq[i] = (uint8_t)(((int)(tmp[code_size * nb + i] * 256)) & 0xFF);
    }

    IndexBinaryHNSW index(d, M);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.add(nb, xb.data());

    std::vector<int32_t> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    SearchParametersHNSW params;
    params.efSearch = efSearch;

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
    state.counters["threads"] = FLAGS_threads;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "binary index search benchmarks (IndexBinaryFlat, IndexBinaryIVF, "
            "IndexBinaryHNSW)",
            "--d=256 --nprobe=8 --benchmark_filter='binary_ivf/.*'");

    int nb = 100000;
    int k = 10;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});
    std::vector<int> ds = benchmarks::int_list(FLAGS_d, {128, 256});
    std::vector<int> nlists = benchmarks::int_list(FLAGS_nlist, {256, 1024});
    std::vector<int> nprobes = benchmarks::int_list(FLAGS_nprobe, {1, 8, 32});
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {16, 32});
    std::vector<int> efSearches =
            benchmarks::int_list(FLAGS_efSearch, {16, 64});

    // BinaryFlat
    for (int d : ds) {
        for (int nq : nqs) {
            std::string name = "binary_flat/search/d:" + std::to_string(d) +
                    "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_binary_flat_search, d, nb, nq, k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // BinaryIVF
    for (int d : ds) {
        for (int nlist : nlists) {
            for (int nprobe : nprobes) {
                if (nprobe > nlist)
                    continue;
                for (int nq : nqs) {
                    std::string name =
                            "binary_ivf/search/d:" + std::to_string(d) +
                            "/nlist:" + std::to_string(nlist) +
                            "/nprobe:" + std::to_string(nprobe) +
                            "/nq:" + std::to_string(nq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_binary_ivf_search,
                            d,
                            nb,
                            nlist,
                            nprobe,
                            nq,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // BinaryHNSW
    for (int d : ds) {
        for (int M : Ms) {
            for (int efSearch : efSearches) {
                for (int nq : nqs) {
                    std::string name =
                            "binary_hnsw/search/d:" + std::to_string(d) +
                            "/M:" + std::to_string(M) +
                            "/efSearch:" + std::to_string(efSearch) +
                            "/nq:" + std::to_string(nq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_binary_hnsw_search,
                            d,
                            nb,
                            M,
                            efSearch,
                            nq,
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
