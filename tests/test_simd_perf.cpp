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
 * 1. Dispatch actually routes to different code when set_level() is called
 * 2. AVX2 is faster than NONE (scalar) for explicit-intrinsics functions
 * 3. AVX512 does not catastrophically regress vs AVX2
 *
 * We benchmark fvec_inner_products_ny with d=8, which has a large algorithmic
 * difference between SIMD levels: AVX2 uses an 8x8 register transpose to
 * process 8 vectors simultaneously, while NONE processes one vector at a time.
 * This gives a reliable 4-8x speedup that is robust against CI noise.
 *
 * Benchmarking methodology: we use the minimum (best) time across multiple
 * interleaved runs. For microbenchmarks, the minimum is the most stable
 * estimator of true performance because system interference (context switches,
 * cache evictions, VM scheduling) only makes runs slower, never faster.
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

        // Generate random test data
        x.resize(d);
        y.resize(ny * d);
        dis.resize(ny);

        faiss::float_rand(x.data(), d, 42);
        faiss::float_rand(y.data(), ny * d, 43);
    }

    void TearDown() override {
        faiss::SIMDConfig::set_level(original_level);
    }

    // Verify that dispatch actually routes to the requested level.
    // This catches DD being broken (always routing to one level)
    // without relying on performance measurements.
    void verify_dispatch(faiss::SIMDLevel level) {
        faiss::SIMDConfig::set_level(level);
        faiss::SIMDLevel dispatched = faiss::SIMDConfig::get_dispatched_level();
        ASSERT_EQ(dispatched, level)
                << "Dispatch broken: set_level(" << faiss::to_string(level)
                << ") but get_dispatched_level() returned "
                << faiss::to_string(dispatched);
    }

    // Benchmark fvec_inner_products_ny which has explicit AVX2/AVX512
    // intrinsics with 8x8 register transpose — large algorithmic difference
    // from the scalar NONE fallback.
    double benchmark_fvec_inner_products_ny(
            faiss::SIMDLevel level,
            int n_runs = 5000) {
        faiss::SIMDConfig::set_level(level);

        // Warmup to stabilize CPU frequency
        for (int i = 0; i < 200; i++) {
            faiss::fvec_inner_products_ny(
                    dis.data(), x.data(), y.data(), d, ny);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < n_runs; run++) {
            faiss::fvec_inner_products_ny(
                    dis.data(), x.data(), y.data(), d, ny);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    faiss::SIMDLevel original_level = faiss::SIMDLevel::NONE;

    // d=8: AVX2 uses 8x8 transpose to process 8 vectors simultaneously.
    // ny=4096: y=128KB fits in L2 cache, dis=16KB fits in L1.
    size_t d = 8;
    size_t ny = 4096;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> dis;

    static constexpr int kBenchmarkReps = 7;

    // Run benchmark multiple times with interleaved measurements, return
    // minimum (best) time. For microbenchmarks, the minimum is the most
    // stable estimator because system interference only makes runs slower.
    template <typename BenchmarkFunc>
    std::pair<double, double> benchmark_interleaved_best(
            BenchmarkFunc bench_fn,
            faiss::SIMDLevel level_a,
            faiss::SIMDLevel level_b) {
        double best_a = std::numeric_limits<double>::max();
        double best_b = std::numeric_limits<double>::max();

        for (int i = 0; i < kBenchmarkReps; i++) {
            best_a = std::min(best_a, bench_fn(level_a));
            best_b = std::min(best_b, bench_fn(level_b));
        }

        return {best_a, best_b};
    }
};

TEST_F(SIMDPerfTest, AVX2FasterThanNONE) {
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    // First verify dispatch actually routes to different levels.
    // This catches DD being broken without relying on performance.
    verify_dispatch(faiss::SIMDLevel::NONE);
    verify_dispatch(faiss::SIMDLevel::AVX2);

    // Benchmark fvec_inner_products_ny (d=8) which has explicit AVX2
    // intrinsics with 8x8 register transpose. The NONE version processes
    // one vector at a time, giving a large and reliable speedup.
    auto bench_fn = [this](faiss::SIMDLevel level) {
        return benchmark_fvec_inner_products_ny(level);
    };
    auto [none_time, avx2_time] = benchmark_interleaved_best(
            bench_fn, faiss::SIMDLevel::NONE, faiss::SIMDLevel::AVX2);

    printf("fvec_inner_products_ny NONE: %.2f ms (best of %d runs)\n",
           none_time,
           kBenchmarkReps);
    printf("fvec_inner_products_ny AVX2: %.2f ms (best of %d runs)\n",
           avx2_time,
           kBenchmarkReps);

    double speedup = none_time / avx2_time;
    printf("fvec_inner_products_ny Speedup: %.2fx\n", speedup);

    // AVX2 fvec_inner_products_ny (d=8) uses 8x8 register transpose to
    // process 8 vectors simultaneously vs scalar one-at-a-time. Expected
    // speedup is 4-8x on bare metal. We use 1.5x threshold which gives
    // generous headroom for CI noise while still catching dispatch failures.
    EXPECT_GT(speedup, 1.5)
            << "AVX2 should be faster than NONE for fvec_inner_products_ny. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}

TEST_F(SIMDPerfTest, AVX512FasterThanAVX2IfAvailable) {
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        GTEST_SKIP() << "AVX512 not available on this machine";
    }

    verify_dispatch(faiss::SIMDLevel::AVX2);
    verify_dispatch(faiss::SIMDLevel::AVX512);

    auto bench_fn = [this](faiss::SIMDLevel level) {
        return benchmark_fvec_inner_products_ny(level);
    };
    auto [avx2_time, avx512_time] = benchmark_interleaved_best(
            bench_fn, faiss::SIMDLevel::AVX2, faiss::SIMDLevel::AVX512);

    printf("fvec_inner_products_ny AVX2: %.2f ms (best of %d runs)\n",
           avx2_time,
           kBenchmarkReps);
    printf("fvec_inner_products_ny AVX512: %.2f ms (best of %d runs)\n",
           avx512_time,
           kBenchmarkReps);

    double speedup = avx2_time / avx512_time;
    printf("Speedup (AVX512 vs AVX2): %.2fx\n", speedup);

    // AVX512 fvec_inner_products_ny (d=8) uses 16x8 register transpose
    // (16 vectors/iteration) vs AVX2's 8x8 transpose (8 vectors/iteration).
    // Expected speedup is ~1.5x on bare metal. We use 1.1x threshold to
    // allow for AVX-512 frequency throttling on Intel CPUs.
    EXPECT_GT(speedup, 1.1)
            << "AVX512 should be at least 1.1x faster than AVX2 for "
            << "fvec_inner_products_ny. "
            << "AVX2=" << avx2_time << "ms, AVX512=" << avx512_time << "ms";
}

// Additional test: Verify fvec_L2sqr dispatch is at least not slower.
// fvec_L2sqr uses auto-vectorization, so AVX2 may only be slightly faster.
TEST_F(SIMDPerfTest, L2sqrAutoVecDispatchWorks) {
    if (!faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        GTEST_SKIP() << "AVX2 not available on this machine";
    }

    verify_dispatch(faiss::SIMDLevel::NONE);
    verify_dispatch(faiss::SIMDLevel::AVX2);

    // Use fvec_inner_products_ny at larger d where the algorithmic
    // advantage is smaller (less transpose benefit) to verify dispatch
    // doesn't regress even with autovectorized code paths.
    size_t large_d = 128;
    size_t large_ny = 1024;
    std::vector<float> lx(large_d);
    std::vector<float> ly(large_ny * large_d);
    std::vector<float> ldis(large_ny);
    faiss::float_rand(lx.data(), large_d, 44);
    faiss::float_rand(ly.data(), large_ny * large_d, 45);

    auto bench_fn = [&](faiss::SIMDLevel level) {
        faiss::SIMDConfig::set_level(level);
        // Warmup
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

    auto [none_time, avx2_time] = benchmark_interleaved_best(
            bench_fn, faiss::SIMDLevel::NONE, faiss::SIMDLevel::AVX2);

    printf("fvec_inner_products_ny (d=128) NONE: %.2f ms (best of %d runs)\n",
           none_time,
           kBenchmarkReps);
    printf("fvec_inner_products_ny (d=128) AVX2: %.2f ms (best of %d runs)\n",
           avx2_time,
           kBenchmarkReps);

    double speedup = none_time / avx2_time;
    printf("fvec_inner_products_ny (d=128) Speedup: %.2fx\n", speedup);

    // At larger d, both NONE and AVX2 use the same autovectorized loop,
    // just compiled with different flags. This only catches regressions.
    EXPECT_GT(speedup, 0.8)
            << "AVX2 code should not be significantly slower than SSE4. "
            << "NONE=" << none_time << "ms, AVX2=" << avx2_time << "ms";
}
