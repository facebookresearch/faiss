/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_levels.h>

using faiss::SIMDLevel;

// Random qb-bit-per-dimension query codes (one byte each, value in [0, 2^qb)).
static std::vector<uint8_t> random_codes(size_t d, size_t qb, uint32_t seed) {
    std::mt19937 rng(seed);
    const uint8_t code_mask = static_cast<uint8_t>((1u << qb) - 1);
    std::vector<uint8_t> q(d);
    for (size_t i = 0; i < d; i++) {
        q[i] = static_cast<uint8_t>(rng()) & code_mask;
    }
    return q;
}

// 32-d chunks and chunk boundaries.
static const std::vector<size_t> kDims =
        {1, 8, 16, 31, 32, 33, 255, 256, 257, 512, 1024, 2048};

// Note: scalar kernel's own correctness is covered end-to-end by
// tests/test_rabitq.py. This target is x86-only (see BUCK).
TEST(RaBitQRearrangeBitPlanes, Avx2MatchesScalar) {
    for (size_t d : kDims) {
        for (size_t qb = 1; qb <= 8; qb++) {
            const auto q = random_codes(d, qb, 10996);
            const size_t out_bytes = ((d + 7) / 8) * qb;

            std::vector<uint8_t> scalar(out_bytes);
            faiss::rabitq::rearrange_bit_planes<SIMDLevel::NONE>(
                    q.data(), d, qb, scalar.data());

            std::vector<uint8_t> avx2(out_bytes);
            faiss::rabitq::rearrange_bit_planes<SIMDLevel::AVX2>(
                    q.data(), d, qb, avx2.data());

            EXPECT_EQ(avx2, scalar) << "d=" << d << " qb=" << qb;
        }
    }
}
