/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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
    std::vector<uint8_t> values(n);
    for (uint8_t& value : values) {
        value = static_cast<uint8_t>(rng());
    }
    return values;
}

} // namespace

TEST(RaBitQNeon, BitwiseKernelsMatchScalarAcrossTails) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::ARM_NEON)) {
        GTEST_SKIP() << "ARM_NEON is not available on this CPU";
    }

    const std::vector<size_t> dimensions = {
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

    for (size_t d : dimensions) {
        const size_t size = (d + 7) / 8;
        const auto data = random_bytes(size, 19717 + d);
        for (size_t qb = 1; qb <= 8; qb++) {
            const auto query = random_bytes(size * qb, 51691 + d + qb);

            const uint64_t expected_and =
                    faiss::rabitq::bitwise_and_dot_product<SIMDLevel::NONE>(
                            query.data(), data.data(), size, qb);
            const uint64_t expected_xor =
                    faiss::rabitq::bitwise_xor_dot_product<SIMDLevel::NONE>(
                            query.data(), data.data(), size, qb);
            const uint64_t expected_popcount =
                    faiss::rabitq::popcount<SIMDLevel::NONE>(data.data(), size);

            const uint64_t actual_and =
                    faiss::rabitq::bitwise_and_dot_product<SIMDLevel::ARM_NEON>(
                            query.data(), data.data(), size, qb);
            const uint64_t actual_xor =
                    faiss::rabitq::bitwise_xor_dot_product<SIMDLevel::ARM_NEON>(
                            query.data(), data.data(), size, qb);
            const uint64_t actual_popcount =
                    faiss::rabitq::popcount<SIMDLevel::ARM_NEON>(
                            data.data(), size);
            const auto actual_fused =
                    faiss::rabitq::bitwise_and_dot_product_with_popcount<
                            SIMDLevel::ARM_NEON>(
                            query.data(), data.data(), size, qb);

            EXPECT_EQ(actual_and, expected_and) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_xor, expected_xor) << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_popcount, expected_popcount)
                    << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_fused.dot_product, expected_and)
                    << "d=" << d << " qb=" << qb;
            EXPECT_EQ(actual_fused.popcount, expected_popcount)
                    << "d=" << d << " qb=" << qb;
        }
    }
}

TEST(RaBitQNeon, MultiBitInnerProductMatchesScalarAcrossTails) {
    if (!faiss::SIMDConfig::is_simd_level_available(SIMDLevel::ARM_NEON)) {
        GTEST_SKIP() << "ARM_NEON is not available on this CPU";
    }

    std::mt19937 rng(918273);
    std::uniform_real_distribution<float> values(-2.0f, 2.0f);
    const std::vector<size_t> dimensions = {
            1, 7, 8, 9, 15, 16, 31, 32, 100, 255, 256, 768, 1024, 2048};

    for (size_t d : dimensions) {
        const auto sign_bits = random_bytes((d + 7) / 8, 13579 + d);
        std::vector<float> query(d);
        for (float& value : query) {
            value = values(rng);
        }
        for (size_t ex_bits = 1; ex_bits <= 7; ex_bits++) {
            // ip_scalar reads an 8-byte extraction window, matching the padded
            // production code layout.
            const auto extra_bits = random_bytes(
                    (d * ex_bits + 7) / 8 + 7, 24680 + d + ex_bits);
            const float cb = -static_cast<float>((1u << ex_bits) - 0.5f);
            const float expected =
                    faiss::rabitq::multibit::compute_inner_product<
                            SIMDLevel::NONE>(
                            sign_bits.data(),
                            extra_bits.data(),
                            query.data(),
                            d,
                            ex_bits,
                            cb);
            const float actual = faiss::rabitq::multibit::compute_inner_product<
                    SIMDLevel::ARM_NEON>(
                    sign_bits.data(),
                    extra_bits.data(),
                    query.data(),
                    d,
                    ex_bits,
                    cb);

            const float tolerance = 2e-5f * std::max(1.0f, std::abs(expected));
            EXPECT_NEAR(actual, expected, tolerance)
                    << "d=" << d << " ex_bits=" << ex_bits;
        }
    }
}
