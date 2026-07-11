/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Product Quantizer Benchmarks
//
// Design notes:
//   * Uses uniform random data; table-construction timing is data-independent.
//   * Pins 1 thread for deterministic single-threaded measurement.

#include <algorithm>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        M,
        "",
        "comma-separated numbers of PQ sub-quantizers (default: 8,16,32,64; "
        "tables group: 8,20)");
DEFINE_string(
        nbits,
        "",
        "comma-separated bits per PQ sub-quantizer for the "
        "pq_compute_*_tables benchmarks (default: 4,8)");
DEFINE_string(
        nx,
        "",
        "comma-separated numbers of query vectors for the "
        "pq_compute_*_tables benchmarks (default: 1,10,100)");
DEFINE_string(
        dsub,
        "",
        "comma-separated PQ sub-vector dimensions (d = M*dsub) for the "
        "pq_compute_*_tables benchmarks (default: 2,4,8)");

static void bench_pq_train(benchmark::State& state, int d, int M) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    float_rand(xt.data(), d * nt, 12345);

    omp_set_num_threads(1);
    for (auto _ : state) {
        ProductQuantizer pq(d, M, 8);
        pq.verbose = false;
        pq.train(nt, xt.data());
        benchmark::ClobberMemory();
    }
    state.counters["d"] = d;
    state.counters["M"] = M;
}

static void bench_pq_compute_codes(benchmark::State& state, int d, int M) {
    int nt = 10000;
    int n = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> x(d * n);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(x.data(), d * n, 54321);

    ProductQuantizer pq(d, M, 8);
    pq.verbose = false;
    omp_set_num_threads(1);
    pq.train(nt, xt.data());

    std::vector<uint8_t> codes(pq.code_size * n);

    for (auto _ : state) {
        pq.compute_codes(x.data(), codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
}

static void bench_pq_distance_table(benchmark::State& state, int d, int M) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> query(d);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(query.data(), d, 54321);

    ProductQuantizer pq(d, M, 8);
    pq.verbose = false;
    omp_set_num_threads(1);
    pq.train(nt, xt.data());

    std::vector<float> dis_table(pq.ksub * M);

    for (auto _ : state) {
        pq.compute_distance_table(query.data(), dis_table.data());
        benchmark::DoNotOptimize(dis_table[0]);
    }
    state.counters["d"] = d;
    state.counters["M"] = M;
}

static void bench_pq_inner_prod_table(benchmark::State& state, int d, int M) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> query(d);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(query.data(), d, 54321);

    ProductQuantizer pq(d, M, 8);
    pq.verbose = false;
    omp_set_num_threads(1);
    pq.train(nt, xt.data());

    std::vector<float> ip_table(pq.ksub * M);

    for (auto _ : state) {
        pq.compute_inner_prod_table(query.data(), ip_table.data());
        benchmark::DoNotOptimize(ip_table[0]);
    }
    state.counters["d"] = d;
    state.counters["M"] = M;
}

static void bench_pq_sdc_table(benchmark::State& state, int d, int M) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    float_rand(xt.data(), d * nt, 12345);

    ProductQuantizer pq(d, M, 8);
    pq.verbose = false;
    omp_set_num_threads(1);
    pq.train(nt, xt.data());

    for (auto _ : state) {
        pq.compute_sdc_table();
        benchmark::DoNotOptimize(pq.sdc_table[0]);
    }
    state.counters["d"] = d;
    state.counters["M"] = M;
}

static void bench_pq_search(benchmark::State& state, int d, int M) {
    int nt = 10000;
    int nb = 10000;
    int nq = 100;
    int k = 10;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    ProductQuantizer pq(d, M, 8);
    pq.verbose = false;
    omp_set_num_threads(1);
    pq.train(nt, xt.data());

    std::vector<uint8_t> codes(pq.code_size * nb);
    pq.compute_codes(xb.data(), codes.data(), nb);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);
    float_maxheap_array_t res = {
            (size_t)nq, (size_t)k, labels.data(), distances.data()};

    for (auto _ : state) {
        pq.search(xq.data(), nq, codes.data(), nb, &res);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["M"] = M;
}

// Multi-query distance-table construction (mirrors python
// bench_codec_quantizers.py): dsub swept over
// {2,4,8} (d = dsub*M), train on max(1000, ksub*50) vectors, time only
// compute_distance_tables / compute_inner_prod_tables for nx queries.
static void bench_pq_compute_distance_tables(
        benchmark::State& state,
        int M,
        int nbits,
        int nx,
        int dsub) {
    int d = dsub * M;
    ProductQuantizer pq(d, M, nbits);
    pq.verbose = false;
    size_t nt = std::max((size_t)1000, pq.ksub * 50);
    AlignedTable<float> xt((size_t)d * nt);
    AlignedTable<float> x((size_t)d * nx);
    float_rand(xt.data(), (size_t)d * nt, 12345);
    float_rand(x.data(), (size_t)d * nx, 54321);

    omp_set_num_threads(1);
    pq.train(nt, xt.data());

    std::vector<float> tab((size_t)nx * M * pq.ksub);

    for (auto _ : state) {
        pq.compute_distance_tables(nx, x.data(), tab.data());
        benchmark::DoNotOptimize(tab[0]);
    }
    state.SetItemsProcessed(state.iterations() * nx);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["nx"] = nx;
    state.counters["dsub"] = dsub;
}

static void bench_pq_compute_inner_prod_tables(
        benchmark::State& state,
        int M,
        int nbits,
        int nx,
        int dsub) {
    int d = dsub * M;
    ProductQuantizer pq(d, M, nbits);
    pq.verbose = false;
    size_t nt = std::max((size_t)1000, pq.ksub * 50);
    AlignedTable<float> xt((size_t)d * nt);
    AlignedTable<float> x((size_t)d * nx);
    float_rand(xt.data(), (size_t)d * nt, 12345);
    float_rand(x.data(), (size_t)d * nx, 54321);

    omp_set_num_threads(1);
    pq.train(nt, xt.data());

    std::vector<float> tab((size_t)nx * M * pq.ksub);

    for (auto _ : state) {
        pq.compute_inner_prod_tables(nx, x.data(), tab.data());
        benchmark::DoNotOptimize(tab[0]);
    }
    state.SetItemsProcessed(state.iterations() * nx);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["nx"] = nx;
    state.counters["dsub"] = dsub;
}

// SIFT1M variants: train on the dataset's train vectors, encode and search
// subsets of the base vectors.
static void bench_pq_train_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        int d,
        int M) {
    omp_set_num_threads(1);
    for (auto _ : state) {
        ProductQuantizer pq(d, M, 8);
        pq.verbose = false;
        pq.train(nt, xt);
        benchmark::ClobberMemory();
    }
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nt"] = nt;
    state.counters["dataset"] = 1;
}

static void bench_pq_compute_codes_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        int d,
        int M) {
    ProductQuantizer pq(d, M, 8);
    pq.verbose = false;
    omp_set_num_threads(1);
    pq.train(nt, xt);

    std::vector<uint8_t> codes(pq.code_size * n);

    for (auto _ : state) {
        pq.compute_codes(xb, codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["dataset"] = 1;
}

static void bench_pq_search_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        int d,
        int M) {
    int k = 10;
    ProductQuantizer pq(d, M, 8);
    pq.verbose = false;
    omp_set_num_threads(1);
    pq.train(nt, xt);

    std::vector<uint8_t> codes(pq.code_size * nb);
    pq.compute_codes(xb, codes.data(), nb);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);
    float_maxheap_array_t res = {
            (size_t)nq, (size_t)k, labels.data(), distances.data()};

    for (auto _ : state) {
        pq.search(xq, nq, codes.data(), nb, &res);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "product quantizer train, encode, distance-table and search "
            "operations",
            "--M=16 --benchmark_filter='search/.*'");

    int d = 128;
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {8, 16, 32, 64});

    for (int M : Ms) {
        std::string suffix = "/M:" + std::to_string(M);

        auto* b = benchmark::RegisterBenchmark(
                ("train" + suffix).c_str(), bench_pq_train, d, M);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("compute_codes" + suffix).c_str(),
                bench_pq_compute_codes,
                d,
                M);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("compute_distance_table" + suffix).c_str(),
                bench_pq_distance_table,
                d,
                M);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("compute_inner_prod_table" + suffix).c_str(),
                bench_pq_inner_prod_table,
                d,
                M);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("compute_sdc_table" + suffix).c_str(),
                bench_pq_sdc_table,
                d,
                M);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);

        b = benchmark::RegisterBenchmark(
                ("search" + suffix).c_str(), bench_pq_search, d, M);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // Multi-query table construction (dsub swept over {2,4,8}, d = dsub*M)
    std::vector<int> Ms_tables = benchmarks::int_list(FLAGS_M, {8, 20});
    std::vector<int> nbits_list = benchmarks::int_list(FLAGS_nbits, {4, 8});
    std::vector<int> nxs = benchmarks::int_list(FLAGS_nx, {1, 10, 100});
    std::vector<int> dsubs = benchmarks::int_list(FLAGS_dsub, {2, 4, 8});
    for (int M : Ms_tables) {
        for (int nbits : nbits_list) {
            for (int dsub : dsubs) {
                for (int nx : nxs) {
                    std::string suffix = "/M:" + std::to_string(M) +
                            "/nbits:" + std::to_string(nbits) +
                            "/dsub:" + std::to_string(dsub) +
                            "/nx:" + std::to_string(nx);

                    auto* b = benchmark::RegisterBenchmark(
                            ("pq_compute_distance_tables" + suffix).c_str(),
                            bench_pq_compute_distance_tables,
                            M,
                            nbits,
                            nx,
                            dsub);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);

                    b = benchmark::RegisterBenchmark(
                            ("pq_compute_inner_prod_tables" + suffix).c_str(),
                            bench_pq_compute_inner_prod_tables,
                            M,
                            nbits,
                            nx,
                            dsub);
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
        int snt = (int)sift.nt;
        // Dataset subsets: the timed train benchmark and encode use the
        // synthetic sizes (10k), search scans 100k base codes.
        int train_nt = std::min(snt, 10000);
        int encode_n = std::min((int)sift.nb, 10000);
        int search_nb = std::min((int)sift.nb, 100000);
        int search_nq = std::min((int)sift.nq, 100);

        std::vector<int> sift_Ms = benchmarks::int_list(FLAGS_M, {8, 16});
        for (int M : sift_Ms) {
            std::string suffix = "/M:" + std::to_string(M);

            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/train" + suffix).c_str(),
                    bench_pq_train_dataset,
                    sift.xt.data(),
                    train_nt,
                    sd,
                    M);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    (ds + "/compute_codes" + suffix).c_str(),
                    bench_pq_compute_codes_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    encode_n,
                    sd,
                    M);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    (ds + "/search" + suffix).c_str(),
                    bench_pq_search_dataset,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    search_nb,
                    sift.xq.data(),
                    search_nq,
                    sd,
                    M);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
