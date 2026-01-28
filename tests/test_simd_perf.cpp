/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Performance validation test for SIMD Dynamic Dispatch.
 *
 * This test verifies that the DD infrastructure is correctly dispatching
 * to optimized SIMD implementations by checking that:
 * 1. AVX2 is faster than NONE (scalar) implementation
 * 2. The difference is significant for compute-bound operations
 *
 * Note: We use small data sizes that fit in L2 cache to ensure the benchmark
 * is compute-bound rather than memory-bound. With large datasets that exceed
 * cache size, memory bandwidth becomes the bottleneck and SIMD improvements
 * are masked by memory latency.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include <faiss/utils/distances.h>
#include <faiss/utils/simd_levels.h>

class SIMDPerfTest : public ::testing::Test {
   protected:
    void SetUp() override {
        original_level = faiss::SIMDConfig::get_level();

        // Generate random test data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        x.resize(d);
        y.resize(n * d);
        c.resize(n * d); // output buffer for fvec_madd

        for (size_t i = 0; i < d; i++) {
            x[i] = dist(rng);
        }
        for (size_t i = 0; i < n * d; i++) {
            y[i] = dist(rng);
        }
    }

    void TearDown() override {
        faiss::SIMDConfig::set_level(original_level);
    }

    // fvec_L2sqr uses auto-vectorization (same source, different compiler
    // flags) Use many iterations on cache-resident data to measure compute
    // throughput
    double benchmark_fvec_L2sqr(faiss::SIMDLevel level, int n_runs = 10000) {
        faiss::SIMDConfig::set_level(level);

        // Warmup to stabilize CPU frequency
        for (int i = 0; i < 100; i++) {
            for (size_t j = 0; j < n; j++) {
                volatile float result = faiss::fvec_L2sqr(
                        x.data(), y.data() + j * this->d, this->d);
                (void)result;
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < n_runs; run++) {
            for (size_t j = 0; j < n; j++) {
                volatile float result = faiss::fvec_L2sqr(
                        x.data(), y.data() + j * this->d, this->d);
                (void)result;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    // fvec_madd has explicit AVX2 intrinsics - better for testing dispatch
    // Use many iterations on cache-resident data to measure compute throughput
    double benchmark_fvec_madd(faiss::SIMDLevel level, int n_runs = 10000) {
        faiss::SIMDConfig::set_level(level);

        // Warmup to stabilize CPU frequency
        for (int i = 0; i < 100; i++) {
            for (size_t j = 0; j < n; j++) {
                faiss::fvec_madd(
                        this->d,
                        x.data(),
                        0.5f,
                        y.data() + j * this->d,
                        c.data() + j * this->d);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < n_runs; run++) {
            for (size_t j = 0; j < n; j++) {
                faiss::fvec_madd(
                        this->d,
                        x.data(),
                        0.5f,
                        y.data() + j * this->d,
                        c.data() + j * this->d);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    faiss::SIMDLevel original_level = faiss::SIMDLevel::NONE;
    // Use small data that fits in L2 cache (~256KB) to ensure compute-bound
    // benchmark 64 vectors * 128 dims * 4 bytes = 32KB for each of y and c =
    // 64KB total
    size_t d = 128; // dimension
    size_t n = 64;  // number of vectors (small to fit in L2 cache)
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> c; // output buffer

    // Number of benchmark repetitions for statistical robustness
    static constexpr int kBenchmarkReps = 5;

    // Run benchmark multiple times with interleaved measurements, return
    // median. Interleaving ensures both levels experience similar system
    // conditions.
    template <typename BenchmarkFunc>
    std::pair<double, double> benchmark_interleaved_median(
            BenchmarkFunc bench_fn,
            faiss::SIMDLevel level_a,
            faiss::SIMDLevel level_b) {
        std::vector<double> times_a, times_b;
        times_a.reserve(kBenchmarkReps);
        times_b.reserve(kBenchmarkReps);

        for (int i = 0; i < kBenchmarkReps; i++) {
            times_a.push_back(bench_fn(level_a));
            times_b.push_back(bench_fn(level_b));
        }

        std::sort(times_a.begin(), times_a.end());
        std::sort(times_b.begin(), times_b.end());

        // Return median (middle element for odd count)
        return {times_a[kBenchmarkReps / 2], times_b[kBenchmarkReps / 2]};
    }
};

TEST_F(SIMDPerfTest, AVX2FasterThanNONE) {
    // Skip if AVX2 is not available
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    // Test fvec_madd which has explicit AVX2 intrinsics
    // (fvec_L2sqr uses auto-vectorization so speedup is less predictable)

    // Run interleaved benchmarks and take median for robustness
    auto bench_fn = [this](faiss::SIMDLevel level) {
        return benchmark_fvec_madd(level);
    };
    auto [none_time, avx2_time] = benchmark_interleaved_median(
            bench_fn, faiss::SIMDLevel::NONE, faiss::SIMDLevel::AVX2);

    printf("fvec_madd NONE: %.2f ms (median of %d runs)\n",
           none_time,
           kBenchmarkReps);
    printf("fvec_madd AVX2: %.2f ms (median of %d runs)\n",
           avx2_time,
           kBenchmarkReps);

    // AVX2 should be faster than NONE
    double speedup = none_time / avx2_time;
    printf("fvec_madd Speedup: %.2fx\n", speedup);

    // With cache-resident data, AVX2 should be noticeably faster than scalar
    // NONE. The actual speedup is limited by function call overhead for small
    // vectors. We use a conservative 1.1x threshold to verify dispatch is
    // working correctly while accounting for measurement variance and
    // CPU-specific factors.
    EXPECT_GT(speedup, 1.1)
            << "AVX2 should be faster than NONE for fvec_madd. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}

TEST_F(SIMDPerfTest, AVX512FasterThanAVX2IfAvailable) {
    // Skip if AVX512 is not available
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        GTEST_SKIP() << "AVX512 not available on this machine";
    }

    // Run interleaved benchmarks and take median for robustness
    auto bench_fn = [this](faiss::SIMDLevel level) {
        return benchmark_fvec_madd(level);
    };
    auto [avx2_time, avx512_time] = benchmark_interleaved_median(
            bench_fn, faiss::SIMDLevel::AVX2, faiss::SIMDLevel::AVX512);

    printf("fvec_madd AVX2: %.2f ms (median of %d runs)\n",
           avx2_time,
           kBenchmarkReps);
    printf("fvec_madd AVX512: %.2f ms (median of %d runs)\n",
           avx512_time,
           kBenchmarkReps);

    double ratio = avx512_time / avx2_time;
    printf("Ratio (AVX512/AVX2): %.2f\n", ratio);

    // AVX512 should not be significantly slower than AVX2 (allow 25% margin
    // for frequency throttling)
    EXPECT_LT(ratio, 1.25)
            << "AVX512 should not be more than 25% slower than AVX2. "
            << "AVX2=" << avx2_time << "ms, AVX512=" << avx512_time << "ms";
}

// Additional test: Verify fvec_L2sqr dispatch is at least not slower.
// fvec_L2sqr uses auto-vectorization, so AVX2 may only be slightly faster.
//
// NOTE: This test has limited value for validating Dynamic Dispatch. Unlike
// fvec_madd (which uses explicit intrinsics and shows 4x speedup), fvec_L2sqr
// compiles the same source code with different compiler flags (SSE4 vs AVX2
// auto-vectorization). The AVX2 version often shows no speedup (~1.0x) due to:
//   1. Function call overhead (separate translation unit)
//   2. AVX frequency throttling on some CPUs
//   3. Compiler auto-vectorization quality varies
// The 0.8x threshold only catches severe regressions. Consider removing this
// test if it proves to be more noise than signal during code review.
TEST_F(SIMDPerfTest, L2sqrAutoVecDispatchWorks) {
    // Skip if AVX2 is not available
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    // Run interleaved benchmarks and take median for robustness
    auto bench_fn = [this](faiss::SIMDLevel level) {
        return benchmark_fvec_L2sqr(level);
    };
    auto [none_time, avx2_time] = benchmark_interleaved_median(
            bench_fn, faiss::SIMDLevel::NONE, faiss::SIMDLevel::AVX2);

    printf("fvec_L2sqr NONE (SSE4 autovec): %.2f ms (median of %d runs)\n",
           none_time,
           kBenchmarkReps);
    printf("fvec_L2sqr AVX2 (AVX2 autovec): %.2f ms (median of %d runs)\n",
           avx2_time,
           kBenchmarkReps);

    double speedup = none_time / avx2_time;
    printf("fvec_L2sqr Speedup: %.2fx\n", speedup);

    // Auto-vectorization may not show gains due to function call overhead and
    // AVX frequency throttling. Both NONE and AVX2 use the same source code
    // with different compiler flags, so we're testing dispatch works, not that
    // AVX2 auto-vec is faster than SSE4 auto-vec. Allow 0.8x threshold.
    EXPECT_GT(speedup, 0.8)
            << "AVX2 auto-vectorized code should not be significantly slower than SSE4. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}
