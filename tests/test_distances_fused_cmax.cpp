/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Correctness checks for the fused L2 + top-1 kernel
// (faiss/utils/distances_fused/{rvv,avx512,simdlib_based}.cpp).
//
// Reference: brute-force scan over y, distance computed as
// sum((x - y)^2) in double precision. The fused kernels compute
// |x|^2 + |y|^2 - 2<x,y> with clamped negatives, so distances are compared
// with a tolerance and the index is only compared when the two best
// distances are far enough apart to be distinguishable.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/distances_fused/distances_fused.h>
#include <faiss/utils/simd_levels.h>

using namespace faiss;

namespace {

struct BruteForceTop1 {
    std::vector<float> dis;
    std::vector<int64_t> ids;
};

// Scalar reference: strict greedy min over j, ties resolved to the smaller
// index (same contract as the fused kernel's post-processing loop).
BruteForceTop1 brute_force_top1(
        const std::vector<float>& x,
        const std::vector<float>& y,
        size_t d,
        size_t nx,
        size_t ny) {
    BruteForceTop1 out;
    out.dis.assign(nx, 0.0f);
    out.ids.assign(nx, -1);
    for (size_t i = 0; i < nx; i++) {
        double best = std::numeric_limits<double>::infinity();
        int64_t best_j = -1;
        for (size_t j = 0; j < ny; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < d; k++) {
                double diff = double(x[i * d + k]) - double(y[j * d + k]);
                sum += diff * diff;
            }
            if (sum < best ||
                (sum == best && best_j >= 0 && int64_t(j) < best_j)) {
                best = sum;
                best_j = int64_t(j);
            }
        }
        if (best_j >= 0) {
            out.dis[i] = float(best < 0 ? 0 : best);
            out.ids[i] = best_j;
        }
    }
    return out;
}

std::vector<float> make_vectors(size_t n, unsigned seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& f : v) {
        f = dist(rng);
    }
    return v;
}

// Checks one result set against the brute-force reference.
void expect_matches_reference(
        const BruteForceTop1& expected,
        const std::vector<float>& dis,
        const std::vector<int64_t>& ids,
        size_t nx,
        const char* label) {
    size_t id_mismatches = 0;
    size_t checked = 0;
    for (size_t i = 0; i < nx; i++) {
        const float tol =
                1e-3f * std::max(1.0f, std::fabs(expected.dis[i])) + 1e-4f;
        EXPECT_NEAR(dis[i], expected.dis[i], tol)
                << label << ": query " << i << " distance";
        // Only compare the argmin when no near-tie can flip the answer.
        if (expected.ids[i] >= 0 && ids[i] >= 0 &&
            std::fabs(dis[i] - expected.dis[i]) <= 0.5f * tol) {
            checked++;
            if (ids[i] != expected.ids[i]) {
                id_mismatches++;
            }
        }
    }
    // Near-ties are extremely rare with random data; allow a tiny number.
    EXPECT_LE(id_mismatches, checked / 100)
            << label << ": " << id_mismatches << "/" << checked
            << " argmin mismatches";
}

// Cross-checks knn_L2sqr(k=1), which dispatches to the fused kernel when the
// SIMD level provides one, against the brute-force reference. Runs on every
// platform: on x86 the fused kernel is AVX512/AVX2, on RISC-V the RVV one.
void check_knn_top1(
        size_t d,
        size_t nx,
        size_t ny,
        float lo,
        float hi,
        const char* label) {
    std::vector<float> x = make_vectors(nx * d, 11, lo, hi);
    std::vector<float> y = make_vectors(ny * d, 22, lo, hi);
    std::vector<float> y_norms(ny);
    if (ny > 0) {
        fvec_norms_L2sqr(y_norms.data(), y.data(), d, ny);
    }

    std::vector<float> dis(nx, -1.0f);
    std::vector<int64_t> ids(nx, -2);
    if (nx > 0 && ny > 0) {
        knn_L2sqr(
                x.data(),
                y.data(),
                d,
                nx,
                ny,
                size_t(1),
                dis.data(),
                ids.data(),
                y_norms.data(),
                nullptr);
    }

    if (nx == 0 || ny == 0) {
        return;
    }
    auto expected = brute_force_top1(x, y, d, nx, ny);
    expect_matches_reference(expected, dis, ids, nx, label);
}

} // namespace

// Dimensions covered by the reference kernels, and a few outside them so the
// fallback path is exercised too.
TEST(DistancesFusedCmax, KnnTop1MatchesReference) {
    for (size_t d :
         {size_t(1), size_t(4), size_t(12), size_t(16), size_t(17)}) {
        // ny below one vector width, exactly the width and a few full-width
        // blocks plus leftovers.
        for (size_t ny : {size_t(3), size_t(16), size_t(37), size_t(1024)}) {
            check_knn_top1(d, 100, ny, -1.0f, 1.0f, "random [-1,1]");
            check_knn_top1(d, 97, ny, 0.0f, 100.0f, "random [0,100]");
        }
    }
}

TEST(DistancesFusedCmax, DegenerateInputs) {
    // nx == 0 and ny == 0 must not crash and must leave no results.
    check_knn_top1(12, 0, 64, -1.0f, 1.0f, "nx=0");
    check_knn_top1(12, 64, 0, -1.0f, 1.0f, "ny=0");
    check_knn_top1(12, 64, 1, -1.0f, 1.0f, "ny=1");
    check_knn_top1(12, 1, 64, -1.0f, 1.0f, "nx=1");
}

TEST(DistancesFusedCmax, IdenticalVectors) {
    // Every database vector equals the same query vector: distances must be
    // 0 (not negative) and the argmin must be the smallest index.
    const size_t d = 12;
    const size_t nx = 8;
    const size_t ny = 100;
    std::vector<float> x = make_vectors(nx * d, 3, -1.0f, 1.0f);
    std::vector<float> y(ny * d);
    for (size_t j = 0; j < ny; j++) {
        for (size_t k = 0; k < d; k++) {
            y[j * d + k] = x[k]; // all rows copy query 0
        }
    }
    std::vector<float> y_norms(ny);
    fvec_norms_L2sqr(y_norms.data(), y.data(), d, ny);
    std::vector<float> dis(nx, -1.0f);
    std::vector<int64_t> ids(nx, -2);
    knn_L2sqr(
            x.data(),
            y.data(),
            d,
            nx,
            ny,
            size_t(1),
            dis.data(),
            ids.data(),
            y_norms.data(),
            nullptr);

    for (size_t i = 0; i < nx; i++) {
        EXPECT_GE(dis[i], 0.0f) << "query " << i;
    }
    EXPECT_NEAR(dis[0], 0.0f, 1e-3f);
    EXPECT_EQ(ids[0], 0);
}

#if defined(__riscv) && defined(__riscv_vector)

// Explicit check of the RVV specialization itself, so that a silently
// disabled RVV path cannot hide behind the platform-generic test above.
TEST(DistancesFusedCmax, RvvKernelMatchesReference) {
    for (size_t d :
         {size_t(1), size_t(4), size_t(12), size_t(16), size_t(32)}) {
        for (size_t ny : {size_t(5), size_t(16), size_t(100), size_t(1024)}) {
            const size_t nx = 40;
            std::vector<float> x = make_vectors(nx * d, 101, -1.0f, 1.0f);
            std::vector<float> y = make_vectors(ny * d, 202, -1.0f, 1.0f);
            std::vector<float> y_norms(ny);
            fvec_norms_L2sqr(y_norms.data(), y.data(), d, ny);

            std::vector<float> dis(nx, 12345.0f);
            std::vector<int64_t> ids(nx, 7);
            Top1BlockResultHandler<CMax<float, int64_t>> res(
                    nx, dis.data(), ids.data());

            const bool applied =
                    exhaustive_L2sqr_fused_cmax<SIMDLevel::RISCV_RVV>(
                            x.data(), y.data(), d, nx, ny, res, y_norms.data());
            EXPECT_TRUE(applied)
                    << "RVV fused kernel declined d=" << d << " ny=" << ny;

            auto expected = brute_force_top1(x, y, d, nx, ny);
            expect_matches_reference(expected, dis, ids, nx, "rvv direct call");
        }
    }
}

TEST(DistancesFusedCmax, RvvHandlesUnsupportedDimensionality) {
    // d is outside the compiled dispatch table: the kernel must decline rather
    // than produce garbage, so the caller can fall back.
    const size_t d = 33;
    const size_t nx = 8;
    const size_t ny = 64;
    std::vector<float> x = make_vectors(nx * d, 5, -1.0f, 1.0f);
    std::vector<float> y = make_vectors(ny * d, 6, -1.0f, 1.0f);
    std::vector<float> y_norms(ny);
    fvec_norms_L2sqr(y_norms.data(), y.data(), d, ny);
    std::vector<float> dis(nx, 12345.0f);
    std::vector<int64_t> ids(nx, 7);
    Top1BlockResultHandler<CMax<float, int64_t>> res(
            nx, dis.data(), ids.data());
    EXPECT_FALSE(
            exhaustive_L2sqr_fused_cmax<SIMDLevel::RISCV_RVV>(
                    x.data(), y.data(), d, nx, ny, res, y_norms.data()));
}

#endif
