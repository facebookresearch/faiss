/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <vector>

#include <benchmark/benchmark.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>

namespace faiss {

const int d = 256;
const size_t nb = 1000000;
const size_t nq = 100000;

std::vector<float> random_data(size_t n, int seed) {
    std::vector<float> data(n);
    float_rand(data.data(), n, seed);
    return data;
}

const auto xb = random_data(nb * d, 1234);
const auto xq = random_data(nq * d, 5678);

std::unique_ptr<Index> make_index(const std::string& factory) {
    std::unique_ptr<Index> index{faiss::index_factory(d, factory.c_str())};
    index->train(100000, xb.data()); // train on subset for speed.
    index->add(nb, xb.data());
    return index;
}

std::unique_ptr<IndexHNSW> make_hollow_hnsw() {
    auto index = std::make_unique<IndexHNSW>();
    index->d = d;
    index->own_fields = false;
    auto& hnsw = index->hnsw;
    hnsw.efSearch = 32;
    hnsw.fill_with_random_links(nb);
    index->ntotal = nb;
    return index;
}

using benchmark::Counter;

void bench_index(benchmark::State& state, const Index& index) {
    std::vector<float> distances(nq * 10);
    std::vector<faiss::idx_t> labels(nq * 10);
    for (auto _ : state) {
        index.search(nq, xq.data(), 10, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[1]);
    }
    state.counters["nq"] = Counter(nq, Counter::kAvgThreadsRate);
}

void bench_hnsw(benchmark::State& state, Index& storage) {
    static auto index = make_hollow_hnsw();
    index->storage = &storage;
    hnsw_stats.reset();
    bench_index(state, *index);
    state.counters["ndis"] = Counter(hnsw_stats.ndis, Counter::kAvgThreadsRate);
}

void bench_ivf(benchmark::State& state, Index& index) {
    indexIVF_stats.reset();
    bench_index(state, index);
    state.counters["ndis"] =
            Counter(indexIVF_stats.ndis, Counter::kAvgThreadsRate);
}

// Define and register a benchmark for a quantizer. The function name and the
// index_factory string are both derived from it. Each index is built lazily in
// a function-local static, so filtering to a single benchmark only builds the
// index it actually uses.
#define BENCH_IVF(Q)                                  \
    void BM_IVF_##Q(benchmark::State& state) {        \
        static auto index = make_index("IVF100," #Q); \
        bench_ivf(state, *index);                     \
    }                                                 \
    BENCHMARK(BM_IVF_##Q)                             \
            ->MeasureProcessCPUTime()                 \
            ->Unit(benchmark::kMillisecond)           \
            ->Name("IVF1K," #Q)

#define BENCH_HNSW(Q)                           \
    void BM_HNSW_##Q(benchmark::State& state) { \
        static auto storage = make_index(#Q);   \
        bench_hnsw(state, *storage);            \
    }                                           \
    BENCHMARK(BM_HNSW_##Q)                      \
            ->MeasureProcessCPUTime()           \
            ->Unit(benchmark::kMillisecond)     \
            ->Name("HNSW32," #Q)

BENCH_HNSW(Flat);
BENCH_HNSW(SQ8);
BENCH_HNSW(SQ6);
BENCH_HNSW(SQ4);
BENCH_HNSW(PQ64x8np);
BENCH_HNSW(PQ64x4np);
BENCH_HNSW(RaBitQ1);
BENCH_HNSW(RaBitQ2);

BENCH_IVF(SQ4);
BENCH_IVF(SQ8);
BENCH_IVF(PQ64x8np);

#undef BENCH_IVF
#undef BENCH_HNSW

} // namespace faiss

BENCHMARK_MAIN();
