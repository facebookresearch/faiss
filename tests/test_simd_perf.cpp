/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Checks that DD routes AVX2 to real AVX2 code: fvec_inner_products_ny at d=8
 * (8x8 transpose kernel) must beat NONE, and at d=128 must not be slower.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>

#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>

class SIMDPerfTest : public ::testing::Test {
   protected:
    void SetUp() override {
        original_level = faiss::SIMDConfig::get_level();

        x.resize(d);
        y.resize(ny * d);
        dis.resize(ny);

        faiss::float_rand(x.data(), d, 42);
        faiss::float_rand(y.data(), ny * d, 43);
    }

    void TearDown() override {
        faiss::SIMDConfig::set_level(original_level);
    }

    double benchmark_fvec_inner_products_ny(faiss::SIMDLevel level) {
        faiss::SIMDConfig::set_level(level);

        // Warmup to stabilize CPU frequency
        for (int i = 0; i < 200; i++) {
            faiss::fvec_inner_products_ny(
                    dis.data(), x.data(), y.data(), d, ny);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < 5000; run++) {
            faiss::fvec_inner_products_ny(
                    dis.data(), x.data(), y.data(), d, ny);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    faiss::SIMDLevel original_level = faiss::SIMDLevel::NONE;

    // ny=4096: y=128KB fits in L2 cache, dis=16KB fits in L1.
    size_t d = 8;
    size_t ny = 4096;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> dis;

    static constexpr int kBenchmarkReps = 7;
};

TEST_F(SIMDPerfTest, AVX2FasterThanNONE) {
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    // Interleave the levels and keep the minimum (best) time of each. For
    // microbenchmarks, the minimum is the most stable estimator because
    // system interference only makes runs slower.
    double none_time = std::numeric_limits<double>::max();
    double avx2_time = std::numeric_limits<double>::max();
    for (int i = 0; i < kBenchmarkReps; i++) {
        none_time = std::min(
                none_time,
                benchmark_fvec_inner_products_ny(faiss::SIMDLevel::NONE));
        avx2_time = std::min(
                avx2_time,
                benchmark_fvec_inner_products_ny(faiss::SIMDLevel::AVX2));
    }

    printf("fvec_inner_products_ny NONE: %.2f ms (best of %d runs)\n",
           none_time,
           kBenchmarkReps);
    printf("fvec_inner_products_ny AVX2: %.2f ms (best of %d runs)\n",
           avx2_time,
           kBenchmarkReps);

    double speedup = none_time / avx2_time;
    printf("fvec_inner_products_ny Speedup: %.2fx\n", speedup);

    // Expected 4-8x; 1.5x leaves headroom for CI noise.
    EXPECT_GT(speedup, 1.5)
            << "AVX2 should be faster than NONE for fvec_inner_products_ny. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}

TEST_F(SIMDPerfTest, L2sqrAutoVecDispatchWorks) {
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    // At d=128 fvec_inner_products_ny runs the autovectorized loop, compiled
    // separately per level.
    size_t large_d = 128;
    size_t large_ny = 1024;
    std::vector<float> lx(large_d);
    std::vector<float> ly(large_ny * large_d);
    std::vector<float> ldis(large_ny);
    faiss::float_rand(lx.data(), large_d, 44);
    faiss::float_rand(ly.data(), large_ny * large_d, 45);

    auto bench_fn = [&](faiss::SIMDLevel level) {
        faiss::SIMDConfig::set_level(level);
        for (int i = 0; i < 100; i++) {
            faiss::fvec_inner_products_ny(
                    ldis.data(), lx.data(), ly.data(), large_d, large_ny);
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < 2000; run++) {
            faiss::fvec_inner_products_ny(
                    ldis.data(), lx.data(), ly.data(), large_d, large_ny);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    };

    double none_time = std::numeric_limits<double>::max();
    double avx2_time = std::numeric_limits<double>::max();
    for (int i = 0; i < kBenchmarkReps; i++) {
        none_time = std::min(none_time, bench_fn(faiss::SIMDLevel::NONE));
        avx2_time = std::min(avx2_time, bench_fn(faiss::SIMDLevel::AVX2));
    }

    printf("fvec_inner_products_ny (d=128) NONE: %.2f ms (best of %d runs)\n",
           none_time,
           kBenchmarkReps);
    printf("fvec_inner_products_ny (d=128) AVX2: %.2f ms (best of %d runs)\n",
           avx2_time,
           kBenchmarkReps);

    double speedup = none_time / avx2_time;
    printf("fvec_inner_products_ny (d=128) Speedup: %.2fx\n", speedup);

    EXPECT_GT(speedup, 0.8)
            << "AVX2 code should not be significantly slower than SSE4. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}
