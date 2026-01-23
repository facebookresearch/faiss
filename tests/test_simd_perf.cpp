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
 * 2. The difference is significant (at least 1.5x for typical dimensions)
 */

#include <gtest/gtest.h>

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
        c.resize(n * d);  // output buffer for fvec_madd

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

    // fvec_L2sqr uses auto-vectorization (same source, different compiler flags)
    double benchmark_fvec_L2sqr(faiss::SIMDLevel level, int n_runs = 100) {
        faiss::SIMDConfig::set_level(level);

        // Warmup
        for (int i = 0; i < 10; i++) {
            for (size_t j = 0; j < n; j++) {
                volatile float result = faiss::fvec_L2sqr(x.data(), y.data() + j * this->d, this->d);
                (void)result;
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < n_runs; run++) {
            for (size_t j = 0; j < n; j++) {
                volatile float result = faiss::fvec_L2sqr(x.data(), y.data() + j * this->d, this->d);
                (void)result;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    // fvec_madd has explicit AVX2 intrinsics - better for testing dispatch
    double benchmark_fvec_madd(faiss::SIMDLevel level, int n_runs = 100) {
        faiss::SIMDConfig::set_level(level);

        // Warmup
        for (int i = 0; i < 10; i++) {
            for (size_t j = 0; j < n; j++) {
                faiss::fvec_madd(this->d, x.data(), 0.5f, y.data() + j * this->d, c.data() + j * this->d);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < n_runs; run++) {
            for (size_t j = 0; j < n; j++) {
                faiss::fvec_madd(this->d, x.data(), 0.5f, y.data() + j * this->d, c.data() + j * this->d);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    faiss::SIMDLevel original_level;
    size_t d = 128;   // dimension
    size_t n = 10000; // number of vectors
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> c;  // output buffer
};

TEST_F(SIMDPerfTest, AVX2FasterThanNONE) {
    // Skip if AVX2 is not available
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    // Test fvec_madd which has explicit AVX2 intrinsics
    // (fvec_L2sqr uses auto-vectorization so speedup is less predictable)

    // Benchmark NONE
    double none_time = benchmark_fvec_madd(faiss::SIMDLevel::NONE);
    printf("fvec_madd NONE: %.2f ms\n", none_time);

    // Benchmark AVX2
    double avx2_time = benchmark_fvec_madd(faiss::SIMDLevel::AVX2);
    printf("fvec_madd AVX2: %.2f ms\n", avx2_time);

    // AVX2 should be faster than NONE
    double speedup = none_time / avx2_time;
    printf("fvec_madd Speedup: %.2fx\n", speedup);

    // We expect at least 1.5x speedup with AVX2 for fvec_madd (explicit intrinsics)
    // The actual speedup can vary based on CPU, but should be significant
    EXPECT_GT(speedup, 1.5)
            << "AVX2 should be significantly faster than NONE for fvec_madd. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}

TEST_F(SIMDPerfTest, AVX512FasterThanAVX2IfAvailable) {
    // Skip if AVX512 is not available
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        GTEST_SKIP() << "AVX512 not available on this machine";
    }

    // Benchmark AVX2
    double avx2_time = benchmark_fvec_madd(faiss::SIMDLevel::AVX2);
    printf("fvec_madd AVX2: %.2f ms\n", avx2_time);

    // Benchmark AVX512
    double avx512_time = benchmark_fvec_madd(faiss::SIMDLevel::AVX512);
    printf("fvec_madd AVX512: %.2f ms\n", avx512_time);

    double ratio = avx512_time / avx2_time;
    printf("Ratio (AVX512/AVX2): %.2f\n", ratio);

    // AVX512 should not be significantly slower than AVX2 (allow 25% margin
    // for frequency throttling)
    EXPECT_LT(ratio, 1.25)
            << "AVX512 should not be more than 25% slower than AVX2. "
            << "AVX2=" << avx2_time << "ms, AVX512=" << avx512_time << "ms";
}

// Additional test: Verify fvec_L2sqr dispatch is at least not slower
// fvec_L2sqr uses auto-vectorization, so AVX2 may only be slightly faster
TEST_F(SIMDPerfTest, L2sqrAutoVecDispatchWorks) {
    // Skip if AVX2 is not available
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    // Benchmark NONE
    double none_time = benchmark_fvec_L2sqr(faiss::SIMDLevel::NONE);
    printf("fvec_L2sqr NONE (SSE4 autovec): %.2f ms\n", none_time);

    // Benchmark AVX2
    double avx2_time = benchmark_fvec_L2sqr(faiss::SIMDLevel::AVX2);
    printf("fvec_L2sqr AVX2 (AVX2 autovec): %.2f ms\n", avx2_time);

    double speedup = none_time / avx2_time;
    printf("fvec_L2sqr Speedup: %.2fx\n", speedup);

    // Auto-vectorization may not show huge gains, but should not be slower
    // Allow some variance (0.9x) for measurement noise
    EXPECT_GT(speedup, 0.9)
            << "AVX2 auto-vectorized code should not be slower than SSE4. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}
