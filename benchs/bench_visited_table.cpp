/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/random.h>
#include <omp.h>

namespace faiss {

// Benchmark VisitedTable with various combinations of index size and query
// batch size, exercising both vector and hash set strategies.
void bench_visited_table(benchmark::State& state) {
    int arg = 0;
    bool use_hashset = state.range(arg++);
    size_t ntotal = state.range(arg++);
    size_t batch_size = state.range(arg++);

    size_t nq = omp_get_max_threads() * batch_size * 2;
    std::uniform_int_distribution<> randId(0, ntotal - 1);
    size_t ndis = 1000;

    size_t total_queries = 0;
    for (auto _ : state) {
        total_queries += nq;
#pragma omp parallel
        {
            VisitedTable vt(ntotal, use_hashset);

#pragma omp for schedule(static)
            for (size_t q0 = 0; q0 < nq; q0 += batch_size) {
                size_t q1 = std::min(q0 + batch_size, nq);
                for (size_t q = q0; q < q1; ++q) {
                    std::default_random_engine rng1(q);
                    std::default_random_engine rng2(q);
                    for (int i = 0; i < ndis; ++i) {
                        auto id = randId(rng1);
                        vt.set(id);
                    }
                    size_t other_visited = 0;
                    for (int i = 0; i < ndis; ++i) {
                        auto id = randId(rng2);
                        other_visited += vt.get(randId(rng1));
                        auto r = vt.get(id);
                        FAISS_ASSERT(r);
                        other_visited += vt.get(randId(rng1));
                    }
                    FAISS_ASSERT(other_visited < 1000);
                    vt.advance();
                }
            }
        }
    }
    state.SetItemsProcessed(total_queries);
}

BENCHMARK(bench_visited_table)
        ->ArgsProduct({
                // use_hashset
                {0, 1},
                // ntotal
                benchmark::CreateRange(1 << 18, 1 << 26, 4),
                // batch_size
                {1, 64},
        });
BENCHMARK_MAIN();

} // namespace faiss
