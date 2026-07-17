/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// VisitedTable Benchmarks
// Benchmarks faiss::VisitedTable with various combinations of index size and
// query batch size, exercising both the vector-based and hash-set-based
// strategies via the set()/get()/advance() workload.

#include <gflags/gflags.h>
#include <omp.h>

#include <algorithm>
#include <memory>
#include <random>

#include <benchmark/benchmark.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/VisitedTable.h>

#include "bench_cli_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(
        use_hashset,
        "",
        "comma-separated VisitedTable strategies: 0=vector, 1=hashset "
        "(default: 0,1)");
DEFINE_string(
        ntotal,
        "",
        "comma-separated index sizes (default: 1<<10 .. 1<<26 stepping *4)");
DEFINE_string(
        batch_size,
        "",
        "comma-separated query batch sizes (default: 1,64)");
DEFINE_uint32(ndis, 1000, "number of distances (set/get calls) per query");
DEFINE_string(
        data_dir,
        "sift1M",
        "dataset directory (accepted for CLI uniformity; this benchmark "
        "always runs on synthetic data)");

// Reproduces the set()/get()/advance() workload.
static void bench_visited_table(
        benchmark::State& state,
        bool use_hashset,
        size_t ntotal,
        size_t batch_size,
        size_t ndis) {
    size_t nq = omp_get_max_threads() * batch_size * 2;

    size_t total_queries = 0;
    for (auto _ : state) {
        total_queries += nq;
#pragma omp parallel
        {
            // Distributions may carry mutable state; keep one per thread.
            std::uniform_int_distribution<size_t> randId(0, ntotal - 1);
            std::unique_ptr<VisitedTable> vt =
                    VisitedTable::create(ntotal, use_hashset);

#pragma omp for schedule(static)
            for (size_t q0 = 0; q0 < nq; q0 += batch_size) {
                size_t q1 = std::min(q0 + batch_size, nq);
                for (size_t q = q0; q < q1; ++q) {
                    std::default_random_engine rng1(q);
                    std::default_random_engine rng2(q);
                    size_t added = 0;
                    for (size_t i = 0; i < ndis; ++i) {
                        auto id = randId(rng1);
                        added += vt->set(id);
                    }
                    size_t other_visited = 0;
                    for (size_t i = 0; i < ndis; ++i) {
                        auto id = randId(rng2);
                        other_visited += vt->get(randId(rng1));
                        auto r = vt->get(id);
                        FAISS_ASSERT(r);
                        other_visited += vt->get(randId(rng1));
                    }
                    benchmark::DoNotOptimize(other_visited + added);
                    vt->advance();
                }
            }
        }
    }
    state.SetItemsProcessed(total_queries);
    state.counters["use_hashset"] = use_hashset;
    state.counters["ntotal"] = ntotal;
    state.counters["batch_size"] = batch_size;
    state.counters["ndis"] = ndis;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "VisitedTable set/get/advance workload for the vector and hashset "
            "strategies across index sizes and query batch sizes",
            "--use_hashset=1 --ntotal=1048576 "
            "--benchmark_filter='visited_table/use_hashset:1/.*'");

    std::vector<int> use_hashsets =
            benchmarks::int_list(FLAGS_use_hashset, {0, 1});
    std::vector<int> batch_sizes =
            benchmarks::int_list(FLAGS_batch_size, {1, 64});

    // Default ntotal sweep: 1<<10 .. 1<<26 stepping by *4
    // (CreateRange(1 << 10, 1 << 26, 4)).
    std::vector<int> ntotals_default;
    for (int64_t v = (1 << 10); v <= (1 << 26); v *= 4) {
        ntotals_default.push_back(static_cast<int>(v));
    }
    std::vector<int> ntotals =
            benchmarks::int_list(FLAGS_ntotal, ntotals_default);

    size_t ndis = FLAGS_ndis;

    for (int use_hashset : use_hashsets) {
        for (int ntotal : ntotals) {
            for (int batch_size : batch_sizes) {
                std::string name = "visited_table/use_hashset:" +
                        std::to_string(use_hashset) +
                        "/ntotal:" + std::to_string(ntotal) +
                        "/batch_size:" + std::to_string(batch_size);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_visited_table,
                        use_hashset != 0,
                        static_cast<size_t>(ntotal),
                        static_cast<size_t>(batch_size),
                        ndis);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
