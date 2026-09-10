/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * AVX512_VPOPCNT RaBitQ kernels against the scalar reference.
 *
 * Separate from test_rabitq_simd.cpp because the AVX512_VPOPCNT
 * specializations only exist in a dynamic dispatch build, and that file also
 * builds in static mode. The target is dynamic dispatch only.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_levels.h>

using faiss::SIMDLevel;

namespace {

std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<uint8_t>(rng());
    }
    return v;
}

// Widths on either side of every register boundary the kernels step over.
const std::vector<size_t> kDims = {
        1,
        8,
        16,
        31,
        32,
        33,
        100,
        128,
        255,
        256,
        257,
        384,
        512,
        768,
        1024,
        2048};

} // namespace

TEST(RaBitQVpopcnt, BitwiseKernelsMatchScalarAcrossTails) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                SIMDLevel::AVX512_VPOPCNT)) {
        GTEST_SKIP() << "AVX512_VPOPCNT is not available on this CPU";
    }

    for (size_t d : kDims) {
        for (size_t qb = 1; qb <= 8; qb++) {
            const size_t size = (d + 7) / 8;
            const auto data = random_bytes(size, 19717);
            const auto q = random_bytes(size * qb, 51691);

            const uint64_t expected_and =
                    faiss::rabitq::bitwise_and_dot_product<SIMDLevel::NONE>(
                            q.data(), data.data(), size, qb);
            const uint64_t expected_xor =
                    faiss::rabitq::bitwise_xor_dot_product<SIMDLevel::NONE>(
                            q.data(), data.data(), size, qb);
            const uint64_t expected_pop =
                    faiss::rabitq::popcount<SIMDLevel::NONE>(data.data(), size);

            const uint64_t actual_and = faiss::rabitq::bitwise_and_dot_product<
                    SIMDLevel::AVX512_VPOPCNT>(q.data(), data.data(), size, qb);
            const uint64_t actual_xor = faiss::rabitq::bitwise_xor_dot_product<
                    SIMDLevel::AVX512_VPOPCNT>(q.data(), data.data(), size, qb);
            const uint64_t actual_pop =
                    faiss::rabitq::popcount<SIMDLevel::AVX512_VPOPCNT>(
                            data.data(), size);
            const auto actual_fused =
                    faiss::rabitq::bitwise_and_dot_product_with_popcount<
                            SIMDLevel::AVX512_VPOPCNT>(
                            q.data(), data.data(), size, qb);

            EXPECT_EQ(actual_and, expected_and) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_xor, expected_xor) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_pop, expected_pop) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_fused.dot_product, expected_and)
                    << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_fused.popcount, expected_pop)
                    << "d=" << d << " qb=" << qb;
        }
    }
}
