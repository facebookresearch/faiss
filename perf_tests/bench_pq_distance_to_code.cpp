/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IVFPQ 8-bit code distances at each compiled SIMD level, through the two
// scanner entry points: distance_to_code (one code per call, as callers that
// run their own scan loop use it) and scan_codes (Faiss's batched loop).

#include <gflags/gflags.h>
#include <omp.h>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>

using namespace faiss;

DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(n, 1 << 20, "codes scanned per iteration");
DEFINE_uint32(iterations, 20, "iterations");

namespace {

struct Setup {
    std::unique_ptr<IndexFlatL2> quantizer;
    std::unique_ptr<IndexIVFPQ> index;
    std::vector<uint8_t> codes;
    std::vector<float> query;
};

Setup make_setup(int d, int M, size_t n) {
    Setup s;
    s.quantizer = std::make_unique<IndexFlatL2>(d);
    s.index = std::make_unique<IndexIVFPQ>(s.quantizer.get(), d, 1, M, 8);
    const size_t nt = 10000;
    std::vector<float> xt(nt * d);
    float_rand(xt.data(), xt.size(), 123);
    s.index->train(nt, xt.data());
    s.codes.resize(n * M);
    byte_rand(s.codes.data(), s.codes.size(), 456);
    s.query.resize(d);
    float_rand(s.query.data(), d, 789);
    return s;
}

void bench(benchmark::State& state, int M, SIMDLevel level, bool batched) {
    omp_set_num_threads(1);
    SIMDConfig::set_level(level);
    const size_t n = FLAGS_n;
    Setup s = make_setup(FLAGS_d, M, n);
    std::unique_ptr<InvertedListScanner> scanner(
            s.index->get_InvertedListScanner(false, nullptr, nullptr));
    scanner->set_query(s.query.data());
    scanner->set_list(0, 0);
    const size_t code_size = s.index->code_size;

    const size_t k = 10;
    std::vector<float> dis(k);
    std::vector<idx_t> labels(k);
    std::vector<idx_t> ids(n);
    for (size_t i = 0; i < n; i++) {
        ids[i] = i;
    }

    for (auto _ : state) {
        if (batched) {
            std::fill(dis.begin(), dis.end(), 1e30f);
            std::fill(labels.begin(), labels.end(), -1);
            benchmark::DoNotOptimize(scanner->scan_codes(
                    n,
                    s.codes.data(),
                    ids.data(),
                    dis.data(),
                    labels.data(),
                    k));
        } else {
            float sum = 0;
            const uint8_t* c = s.codes.data();
            for (size_t j = 0; j < n; j++, c += code_size) {
                sum += scanner->distance_to_code(c);
            }
            benchmark::DoNotOptimize(sum);
        }
        benchmark::ClobberMemory();
    }
    state.counters["ns_per_code"] = benchmark::Counter(
            double(n) * state.iterations(),
            benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
    state.counters["level"] = double(int(level));
}

} // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const std::pair<SIMDLevel, const char*> levels[] = {
            {SIMDLevel::NONE, "NONE"},
            {SIMDLevel::AVX2, "AVX2"},
            {SIMDLevel::AVX512, "AVX512"},
            {SIMDLevel::ARM_NEON, "ARM_NEON"},
            {SIMDLevel::ARM_SVE, "ARM_SVE"},
    };
    for (int M : {8, 16, 32, 64}) {
        for (auto [level, name] : levels) {
            if (!SIMDConfig::is_simd_level_available(level)) {
                continue;
            }
            for (bool batched : {false, true}) {
                std::string bench_name =
                        std::string(
                                batched ? "scan_codes" : "distance_to_code") +
                        "/M" + std::to_string(M) + "/" + name;
                benchmark::RegisterBenchmark(
                        bench_name.c_str(), bench, M, level, batched)
                        ->Iterations(FLAGS_iterations)
                        ->Unit(benchmark::kMillisecond);
            }
        }
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
