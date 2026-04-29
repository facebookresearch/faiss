/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Per-SIMD-level cross-equivalence tests for the templated distance
// functions in faiss/utils/distances.h:
// fvec_L1, fvec_Linf, fvec_L2sqr_batch_4, fvec_inner_product_batch_4,
// fvec_L2sqr_ny_transposed, fvec_L2sqr_ny_nearest.
//
// IndexFlat exercises these transitively via search, but transitive
// coverage can mask tail-handling bugs. This test runs each function at
// every SIMDConfig-available level and asserts equivalence against the
// SIMDLevel::NONE reference. Sweeps dimensions {1, 3, 7, 8, 9, 15, 16,
// 17, 31, 32, 33, 64} so non-multiple-of-8 / non-multiple-of-16 dims
// exercise the tail paths.
//
// Tolerances:
// * Functions returning an integer (fvec_L2sqr_ny_nearest's chosen
//   index) are compared bit-exactly via EXPECT_EQ.
// * Functions returning a single float (fvec_L1, fvec_Linf) accumulate
//   over d terms; rounding-order differences across SIMD widths can
//   reach a few ULPs. Comparison uses an absolute tolerance scaled by
//   d to bound worst-case FP-reassociation drift.
// * Functions returning multiple floats (the batch_4 variants and
//   fvec_L2sqr_ny_transposed) use the same scaled tolerance per element.
//
// NaN/Inf inputs are out of scope: these paths are not exercised by
// random uniform input in [-1, 1].

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <faiss/utils/distances.h>
#include <faiss/utils/simd_levels.h>

using namespace faiss;

namespace {

class SIMDLevelGuard {
   public:
    SIMDLevelGuard() : prev_(SIMDConfig::get_level()) {}
    ~SIMDLevelGuard() {
        SIMDConfig::set_level(prev_);
    }

   private:
    SIMDLevel prev_;
};

std::vector<SIMDLevel> available_levels() {
    static const std::vector<SIMDLevel> all = {
            SIMDLevel::NONE,
            SIMDLevel::AVX2,
            SIMDLevel::AVX512,
            SIMDLevel::AVX512_SPR,
            SIMDLevel::ARM_NEON,
            SIMDLevel::ARM_SVE,
    };
    std::vector<SIMDLevel> out;
    for (auto lv : all) {
        if (SIMDConfig::is_simd_level_available(lv)) {
            out.push_back(lv);
        }
    }
    return out;
}

std::vector<float> rand_vec(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = u(rng);
    }
    return v;
}

// dimensions covering aligned (multiple of 8 / 16) and tail cases.
const std::vector<size_t> kDims = {1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64};

// Tolerance per accumulator term. With inputs in [-1, 1], each squared term
// is in [0, 4]; FP-reassociation differences across SIMD widths grow at
// most linearly with d. 1e-6 per term gives a generous bound (e.g. 6.4e-5
// at d=64) that is still many orders of magnitude tighter than any real
// dispatch bug would produce.
constexpr float kPerTermEps = 1e-6f;

float tol_for_d(size_t d) {
    // Floor at 1e-6 so d=1 still has a sensible (non-zero) tolerance.
    return std::max(static_cast<float>(d) * kPerTermEps, 1e-6f);
}

#define SKIP_IF_SINGLE_LEVEL(levels)                                         \
    if ((levels).size() <= 1) {                                              \
        GTEST_SKIP() << "only one SIMD level available; nothing to compare"; \
    }

template <typename Fn>
void check_floats_at_levels(
        Fn fn,
        const std::vector<SIMDLevel>& levels,
        float tol) {
    auto ref = (SIMDConfig::set_level(SIMDLevel::NONE), fn());
    for (auto lv : levels) {
        if (lv == SIMDLevel::NONE) {
            continue;
        }
        SIMDConfig::set_level(lv);
        auto got = fn();
        ASSERT_EQ(ref.size(), got.size())
                << "size mismatch at level " << static_cast<int>(lv);
        for (size_t i = 0; i < ref.size(); ++i) {
            EXPECT_NEAR(ref[i], got[i], tol)
                    << "diverged at level " << static_cast<int>(lv) << " idx "
                    << i;
        }
    }
}

// For functions returning an integer index (cast to size_t). Bit-exact
// comparison: a correct dispatch must return the same chosen index.
template <typename Fn>
void check_index_at_levels(Fn fn, const std::vector<SIMDLevel>& levels) {
    auto ref = (SIMDConfig::set_level(SIMDLevel::NONE), fn());
    for (auto lv : levels) {
        if (lv == SIMDLevel::NONE) {
            continue;
        }
        SIMDConfig::set_level(lv);
        auto got = fn();
        EXPECT_EQ(ref, got)
                << "index diverged at level " << static_cast<int>(lv);
    }
}

} // namespace

TEST(DistancesDispatch, FvecL1_AllLevels) {
    SIMDLevelGuard guard;
    auto levels = available_levels();
    SKIP_IF_SINGLE_LEVEL(levels);
    for (size_t d : kDims) {
        auto x = rand_vec(d, 1);
        auto y = rand_vec(d, 2);
        check_floats_at_levels(
                [&]() {
                    return std::vector<float>{fvec_L1(x.data(), y.data(), d)};
                },
                levels,
                tol_for_d(d));
    }
}

TEST(DistancesDispatch, FvecLinf_AllLevels) {
    SIMDLevelGuard guard;
    auto levels = available_levels();
    SKIP_IF_SINGLE_LEVEL(levels);
    for (size_t d : kDims) {
        auto x = rand_vec(d, 3);
        auto y = rand_vec(d, 4);
        // Linf is a max, not a sum — no FP-reassociation drift; bit-exact.
        check_floats_at_levels(
                [&]() {
                    return std::vector<float>{fvec_Linf(x.data(), y.data(), d)};
                },
                levels,
                0.0f);
    }
}

TEST(DistancesDispatch, FvecL2sqrBatch4_AllLevels) {
    SIMDLevelGuard guard;
    auto levels = available_levels();
    SKIP_IF_SINGLE_LEVEL(levels);
    for (size_t d : kDims) {
        auto x = rand_vec(d, 10);
        auto y0 = rand_vec(d, 11);
        auto y1 = rand_vec(d, 12);
        auto y2 = rand_vec(d, 13);
        auto y3 = rand_vec(d, 14);
        check_floats_at_levels(
                [&]() {
                    float d0, d1, d2, d3;
                    fvec_L2sqr_batch_4(
                            x.data(),
                            y0.data(),
                            y1.data(),
                            y2.data(),
                            y3.data(),
                            d,
                            d0,
                            d1,
                            d2,
                            d3);
                    return std::vector<float>{d0, d1, d2, d3};
                },
                levels,
                tol_for_d(d));
    }
}

TEST(DistancesDispatch, FvecInnerProductBatch4_AllLevels) {
    SIMDLevelGuard guard;
    auto levels = available_levels();
    SKIP_IF_SINGLE_LEVEL(levels);
    for (size_t d : kDims) {
        auto x = rand_vec(d, 20);
        auto y0 = rand_vec(d, 21);
        auto y1 = rand_vec(d, 22);
        auto y2 = rand_vec(d, 23);
        auto y3 = rand_vec(d, 24);
        check_floats_at_levels(
                [&]() {
                    float d0, d1, d2, d3;
                    fvec_inner_product_batch_4(
                            x.data(),
                            y0.data(),
                            y1.data(),
                            y2.data(),
                            y3.data(),
                            d,
                            d0,
                            d1,
                            d2,
                            d3);
                    return std::vector<float>{d0, d1, d2, d3};
                },
                levels,
                tol_for_d(d));
    }
}

TEST(DistancesDispatch, FvecL2sqrNyTransposed_AllLevels) {
    SIMDLevelGuard guard;
    auto levels = available_levels();
    SKIP_IF_SINGLE_LEVEL(levels);
    constexpr size_t ny = 13;
    for (size_t d : kDims) {
        // d_offset is the stride between consecutive components in the
        // transposed y layout; pick > ny to exercise non-trivial striding
        // and to cover all (i, k) accesses (i in [0, ny), k in [0, d)).
        const size_t d_offset = ny + 5;
        auto x = rand_vec(d, 30);
        // y[i + k * d_offset] is the k-th component of the i-th vector
        // (per fvec_L2sqr_ny_transposed's contract). Required size is
        // (d - 1) * d_offset + ny.
        auto y = rand_vec((d == 0 ? 0 : (d - 1) * d_offset) + ny, 31);
        std::vector<float> y_sqlen(ny);
        for (size_t i = 0; i < ny; ++i) {
            float s = 0;
            for (size_t k = 0; k < d; ++k) {
                float v = y[i + k * d_offset];
                s += v * v;
            }
            y_sqlen[i] = s;
        }
        check_floats_at_levels(
                [&]() {
                    std::vector<float> dis(ny);
                    fvec_L2sqr_ny_transposed(
                            dis.data(),
                            x.data(),
                            y.data(),
                            y_sqlen.data(),
                            d,
                            d_offset,
                            ny);
                    return dis;
                },
                levels,
                tol_for_d(d));
    }
}

TEST(DistancesDispatch, FvecL2sqrNyNearest_AllLevels) {
    SIMDLevelGuard guard;
    auto levels = available_levels();
    SKIP_IF_SINGLE_LEVEL(levels);
    constexpr size_t ny = 23;
    for (size_t d : kDims) {
        auto x = rand_vec(d, 40);
        auto y = rand_vec(d * ny, 41);
        check_index_at_levels(
                [&]() {
                    // distances_tmp_buffer is an internal scratch; only the
                    // returned index is part of the public contract. Some
                    // SIMD impls don't write into the buffer.
                    std::vector<float> tmp(ny);
                    return fvec_L2sqr_ny_nearest(
                            tmp.data(), x.data(), y.data(), d, ny);
                },
                levels);
    }
}
