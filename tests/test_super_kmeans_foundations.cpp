/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/impl/AdSampling.h>
#include <faiss/impl/PdxLayout.h>
#include <faiss/utils/simd_impl/super_kmeans_dispatch.h>
#include <faiss/utils/simd_impl/super_kmeans_kernels.h>
#include <faiss/utils/simd_levels.h>

// =====================================================================
// AdSampling tests
// =====================================================================

// Reference values from scipy.stats.norm.ppf, cited so a reader can
// reproduce. Covers all three branches: lower tail (p < 0.02425), central
// region, and upper tail.
TEST(AdSampling, NormalQuantile_KnownValues) {
    // scipy.stats.norm.ppf(0.001) == -3.0902323061678135
    // scipy.stats.norm.ppf(0.025) == -1.959963984540054
    // scipy.stats.norm.ppf(0.5)   ==  0.0
    // scipy.stats.norm.ppf(0.975) ==  1.959963984540054
    // scipy.stats.norm.ppf(0.999) ==  3.090232306167813
    EXPECT_NEAR(
            faiss::detail::normal_quantile(0.001), -3.0902323061678135, 1e-6);
    EXPECT_NEAR(
            faiss::detail::normal_quantile(0.025), -1.959963984540054, 1e-6);
    EXPECT_NEAR(faiss::detail::normal_quantile(0.5), 0.0, 1e-9);
    EXPECT_NEAR(faiss::detail::normal_quantile(0.975), 1.959963984540054, 1e-6);
    EXPECT_NEAR(faiss::detail::normal_quantile(0.999), 3.090232306167813, 1e-6);
}

// Sweep across a representative (d, p) range; each scipy value cited inline.
TEST(AdSampling, WilsonHilferty_SweepWithin2pct) {
    // {d, p, scipy.stats.chi2.ppf(1 - 1/d, p)}
    constexpr struct {
        int d;
        int p;
        double scipy;
    } refs[] = {
            {128, 16, 32.8175588210},
            {128, 64, 94.5448777626},
            {128, 128, 169.8866700943},
            {256, 16, 35.0547925053},
            {512, 64, 101.5518691071},
            {1024, 64, 104.8262690216},
            {1024, 256, 331.8453440901},
            {1024, 1024, 1169.9132946160},
            {2048, 128, 187.4098077764},
            {4096, 64, 111.0328823202},
            {4096, 1024, 1189.2900132113},
            {4096, 4096, 4419.0780429111},
    };
    for (const auto& r : refs) {
        const double alpha = 1.0 - 1.0 / r.d;
        const double wh = faiss::detail::chi2_quantile_wh(r.p, alpha);
        const double rel_err = std::abs(wh - r.scipy) / r.scipy;
        EXPECT_LT(rel_err, 0.02) << "d=" << r.d << " p=" << r.p
                                 << " alpha=" << alpha << " scipy=" << r.scipy
                                 << " wh=" << wh << " rel_err=" << rel_err;
    }
}

// chi-squared quantile is monotone in degrees of freedom (distribution
// property, not implementation choice). Validated range is p >= 16, so
// loop from 17 to keep p-1 in range.
TEST(AdSampling, ThresholdsMonotone) {
    constexpr int d = 768;
    const std::vector<float> coeff =
            faiss::detail::precompute_ad_thresholds(d, 1.0 / d);

    ASSERT_EQ(coeff.size(), static_cast<size_t>(d + 1));

    for (int p = 17; p < d; p++) {
        EXPECT_GT(coeff[p], coeff[p - 1])
                << "monotonicity violated at p = " << p;
    }
}

// =====================================================================
// PdxLayout tests
// =====================================================================

// pdxify . de_pdxify == identity. d_trail=200, block=64 -> 3 full blocks +
// 1 trailing block of 8 dims. Bit-equality: both transforms are pure memcpy.
TEST(PdxLayout, RoundTrip) {
    constexpr int k = 64;
    constexpr int d_trail = 200;
    constexpr int pdx_block_size = 64;

    std::vector<float> Y(k * d_trail);
    for (int j = 0; j < k; ++j) {
        for (int dim = 0; dim < d_trail; ++dim) {
            Y[j * d_trail + dim] =
                    static_cast<float>(j) * 1000.0f + static_cast<float>(dim);
        }
    }

    std::vector<float> Y_pdx(k * d_trail);
    faiss::detail::pdxify(Y.data(), k, d_trail, pdx_block_size, Y_pdx.data());

    std::vector<float> Y_back(k * d_trail);
    faiss::detail::de_pdxify(
            Y_pdx.data(), k, d_trail, pdx_block_size, Y_back.data());

    ASSERT_EQ(Y, Y_back);
}

// Trailing dims are large/varied so a "summed too far" bug shows up
// loudly (rows 1 and 2 expectations would jump to ~405 / 10000.5).
TEST(PdxLayout, ComputePartialNormsMatchesReference) {
    constexpr int n = 4;
    constexpr int d = 8;
    constexpr int p = 3;
    // clang-format off
    const std::vector<float> X = {
        // row 0: ||row[0:3]||^2 = 1 + 4 + 9 = 14
        1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // row 1: ||row[0:3]||^2 = 0
        0.0f, 0.0f, 0.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f,
        // row 2: ||row[0:3]||^2 = 0.25 + 0 + 0.25 = 0.5
        0.5f, 0.0f, 0.5f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // row 3: ||row[0:3]||^2 = 0.01 + 0.04 + 0.09 = 0.14
        0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    // clang-format on
    std::vector<float> norms(n, -1.0f); // sentinel
    faiss::detail::compute_partial_norms(X.data(), n, d, p, norms.data());
    EXPECT_NEAR(norms[0], 14.0f, 1e-6f);
    EXPECT_NEAR(norms[1], 0.0f, 1e-6f);
    EXPECT_NEAR(norms[2], 0.5f, 1e-6f);
    EXPECT_NEAR(norms[3], 0.14f, 1e-6f);
}

// =====================================================================
// SIMD kernel tests
// =====================================================================

TEST(BlockL2, ScalarMatchesReference) {
    // y[m] = x[m] + 1 -> diff^2 = 1 for each m -> sum over n dims = n.
    constexpr int N = 64;
    std::vector<float> x(N), y(N);
    for (int m = 0; m < N; ++m) {
        x[m] = static_cast<float>(m);
        y[m] = static_cast<float>(m + 1);
    }

    EXPECT_NEAR(
            faiss::detail::block_l2<faiss::SIMDLevel::NONE>(
                    x.data(), y.data(), N),
            static_cast<float>(N),
            1e-4f);

    for (int n = 1; n < N; ++n) {
        const float ref = static_cast<float>(n);
        EXPECT_NEAR(
                faiss::detail::block_l2<faiss::SIMDLevel::NONE>(
                        x.data(), y.data(), n),
                ref,
                1e-4f)
                << "n=" << n;
    }
}

TEST(BlockL2, Avx512MatchesScalar) {
#if defined(__x86_64__) && defined(__AVX512F__)
    constexpr int N = 64;
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(N), y(N);
    for (int m = 0; m < N; ++m) {
        x[m] = dist(rng);
        y[m] = dist(rng);
    }

    const float scalar = faiss::detail::block_l2<faiss::SIMDLevel::NONE>(
            x.data(), y.data(), N);
    const float avx512 = faiss::detail::block_l2<faiss::SIMDLevel::AVX512>(
            x.data(), y.data(), N);
    EXPECT_NEAR(avx512, scalar, 1e-4f);

    for (int n = 1; n < N; ++n) {
        const float s = faiss::detail::block_l2<faiss::SIMDLevel::NONE>(
                x.data(), y.data(), n);
        const float a = faiss::detail::block_l2<faiss::SIMDLevel::AVX512>(
                x.data(), y.data(), n);
        EXPECT_NEAR(a, s, 1e-4f) << "n=" << n;
    }
#else
    GTEST_SKIP() << "AVX-512 test requires __AVX512F__";
#endif
}

TEST(BlockL2, DispatchMatchesScalar) {
    constexpr int N = 64;
    std::mt19937 rng(44);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(N), y(N);
    for (int m = 0; m < N; ++m) {
        x[m] = dist(rng);
        y[m] = dist(rng);
    }

    const float scalar = faiss::detail::block_l2<faiss::SIMDLevel::NONE>(
            x.data(), y.data(), N);
    const float dispatched =
            faiss::detail::block_l2_dispatch(x.data(), y.data(), N);
    EXPECT_NEAR(dispatched, scalar, 1e-4f);

    for (int n = 1; n < N; ++n) {
        const float s = faiss::detail::block_l2<faiss::SIMDLevel::NONE>(
                x.data(), y.data(), n);
        const float d = faiss::detail::block_l2_dispatch(x.data(), y.data(), n);
        EXPECT_NEAR(d, s, 1e-4f) << "n=" << n;
    }
}
