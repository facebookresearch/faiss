/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// PQ FastScan Packing & Scanning Benchmarks

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFastScan.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        nsq,
        "",
        "comma-separated numbers of sub-quantizers for pq4_pack_codes "
        "(default: 8,16,32,64)");
DEFINE_string(
        ntotal,
        "",
        "comma-separated numbers of codes to pack for pq4_pack_codes "
        "(default: 32,256,1024,8192)");
DEFINE_string(
        M,
        "",
        "comma-separated numbers of PQ sub-quantizers for pqfs_search "
        "(default: 16,32,64)");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");

static void bench_pq4_pack_codes(benchmark::State& state, int nsq, int ntotal) {
    int bbs = 32;
    size_t nb = ((ntotal + bbs - 1) / bbs) * bbs;
    size_t code_size = (nsq + 1) / 2;

    std::vector<uint8_t> codes(ntotal * code_size);
    for (size_t i = 0; i < codes.size(); i++) {
        codes[i] = (uint8_t)(i % 256);
    }

    size_t block_size = nb * nsq / 2;
    std::vector<uint8_t> blocks(block_size, 0);

    omp_set_num_threads(1);
    for (auto _ : state) {
        pq4_pack_codes(codes.data(), ntotal, nsq, nb, bbs, nsq, blocks.data());
        benchmark::DoNotOptimize(blocks[0]);
    }
    state.SetItemsProcessed(state.iterations() * ntotal);
    state.SetBytesProcessed(state.iterations() * ntotal * code_size);
    state.counters["nsq"] = nsq;
    state.counters["ntotal"] = ntotal;
}

static void bench_pqfs_search(
        benchmark::State& state,
        int d,
        int M,
        int nb,
        int nq) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexPQFastScan index(d, M, 4);
    index.verbose = false;
    omp_set_num_threads(1);
    index.train(nt, xt.data());
    index.add(nb, xb.data());

    int k = 10;
    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    FastScan_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["t0"] = FastScan_stats.t0;
    state.counters["t1"] = FastScan_stats.t1;
    state.counters["t2"] = FastScan_stats.t2;
    state.counters["t3"] = FastScan_stats.t3;
}

// SIFT1M variant for IndexPQFastScan search
static void bench_pqfs_search_dataset(
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
        int M) {
    IndexPQFastScan index(d, M, 4);
    index.verbose = false;
    omp_set_num_threads(1);
    index.train(nt, xt);
    index.add(nb, xb);

    int k = 10;
    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    FastScan_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["dataset"] = 1;
    state.counters["t0"] = FastScan_stats.t0;
    state.counters["t1"] = FastScan_stats.t1;
    state.counters["t2"] = FastScan_stats.t2;
    state.counters["t3"] = FastScan_stats.t3;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "PQ4 FastScan code packing (pq4_pack_codes) and IndexPQFastScan "
            "search, on synthetic data and SIFT1M",
            "--M=32 --nq=10 --benchmark_filter='pqfs_search/.*'");

    // Packing benchmarks
    std::vector<int> nsqs = benchmarks::int_list(FLAGS_nsq, {8, 16, 32, 64});
    std::vector<int> ntotals =
            benchmarks::int_list(FLAGS_ntotal, {32, 256, 1024, 8192});

    for (int nsq : nsqs) {
        for (int ntotal : ntotals) {
            std::string name = "pq4_pack_codes/nsq:" + std::to_string(nsq) +
                    "/ntotal:" + std::to_string(ntotal);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_pq4_pack_codes, nsq, ntotal);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // FastScan search benchmarks
    int d = 128;
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {16, 32, 64});
    int nb = 100000;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});

    for (int M : Ms) {
        if (M > d)
            continue;
        for (int nq : nqs) {
            std::string name = "pqfs_search/M:" + std::to_string(M) +
                    "/nb:" + std::to_string(nb) + "/nq:" + std::to_string(nq);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_pqfs_search, d, M, nb, nq);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
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

        // IndexPQFastScan search on the full real base, with recall
        std::vector<int> sift_Ms = benchmarks::int_list(FLAGS_M, {16, 32, 64});
        std::vector<int> sift_nqs = benchmarks::int_list(FLAGS_nq, {100});
        for (int M : sift_Ms) {
            if (M > sd || sd % M != 0)
                continue;
            for (int nq : sift_nqs) {
                if ((size_t)nq > sift.nq)
                    continue;
                std::string name = ds + "/pqfs_search/M:" + std::to_string(M) +
                        "/nq:" + std::to_string(nq);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_pqfs_search_dataset,
                        sift.xt.data(),
                        (int)sift.nt,
                        sift.xb.data(),
                        (int)sift.nb,
                        sift.xq.data(),
                        nq,
                        sift.gt.data(),
                        sift.gt_k,
                        sd,
                        M);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
