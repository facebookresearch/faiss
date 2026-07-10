/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/VectorTransform.h>
#include <faiss/impl/RaBitQUtils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

uint8_t round_clamped_ref(float x, uint8_t max_code) {
    float y = std::round(x);
    y = std::max(0.0f, std::min(y, static_cast<float>(max_code)));
    return static_cast<uint8_t>(y);
}

void check_linear_transform_batch_match(bool have_bias) {
    constexpr int d_in = 7;
    constexpr int d_out = 5;
    faiss::LinearTransform transform(d_in, d_out, have_bias);
    transform.is_trained = true;

    transform.A.resize(d_in * d_out);
    for (size_t i = 0; i < transform.A.size(); i++) {
        transform.A[i] = std::sin(static_cast<float>(i) * 0.37f);
    }

    if (have_bias) {
        transform.b.resize(d_out);
        for (int i = 0; i < d_out; i++) {
            transform.b[i] = 0.25f * i - 0.5f;
        }
    }

    std::vector<float> x(d_in);
    for (int i = 0; i < d_in; i++) {
        x[i] = std::cos(static_cast<float>(i) * 0.19f);
    }

    std::vector<float> one_out(d_out);
    transform.apply_noalloc(1, x.data(), one_out.data());

    std::vector<float> batch_x(4 * d_in);
    for (int i = 0; i < 4; i++) {
        std::copy(x.begin(), x.end(), batch_x.begin() + i * d_in);
    }

    std::vector<float> batch_out(4 * d_out);
    transform.apply_noalloc(4, batch_x.data(), batch_out.data());

    for (int i = 0; i < d_out; i++) {
        EXPECT_NEAR(one_out[i], batch_out[i], 1e-5f);
    }
}

} // namespace

TEST(RaBitQUtils, RoundNonnegativeToUint8) {
    using faiss::rabitq_utils::round_nonnegative_to_uint8;

    EXPECT_EQ(round_nonnegative_to_uint8(0.0f), 0);
    EXPECT_EQ(round_nonnegative_to_uint8(0.49f), 0);
    EXPECT_EQ(round_nonnegative_to_uint8(0.5f), 1);
    EXPECT_EQ(round_nonnegative_to_uint8(1.5f), 2);
    EXPECT_EQ(round_nonnegative_to_uint8(254.5f), 255);
    EXPECT_EQ(round_nonnegative_to_uint8(255.0f), 255);

    const float below_half = std::nextafter(0.5f, 0.0f);
    EXPECT_LE(
            std::abs(
                    static_cast<int>(round_nonnegative_to_uint8(below_half)) -
                    static_cast<int>(std::round(below_half))),
            1);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 255.49f);
    for (int i = 0; i < 10000; i++) {
        const float x = dist(rng);
        EXPECT_LE(
                std::abs(
                        static_cast<int>(round_nonnegative_to_uint8(x)) -
                        static_cast<int>(std::round(x))),
                1);
    }
}

TEST(RaBitQUtils, RoundNonnegativeToUint16) {
    using faiss::rabitq_utils::round_nonnegative_to_uint16;

    EXPECT_EQ(round_nonnegative_to_uint16(0.0f), 0);
    EXPECT_EQ(round_nonnegative_to_uint16(0.5f), 1);
    EXPECT_EQ(round_nonnegative_to_uint16(65534.5f), 65535);
    EXPECT_EQ(round_nonnegative_to_uint16(65535.0f), 65535);
}

TEST(RaBitQUtils, RoundClampedToUint8) {
    using faiss::rabitq_utils::round_clamped_to_uint8;

    EXPECT_EQ(round_clamped_to_uint8(-1.0f, 15), 0);
    EXPECT_EQ(round_clamped_to_uint8(0.49f, 15), 0);
    EXPECT_EQ(round_clamped_to_uint8(0.5f, 15), 1);
    EXPECT_EQ(round_clamped_to_uint8(14.5f, 15), 15);
    EXPECT_EQ(round_clamped_to_uint8(15.4f, 15), 15);
    EXPECT_EQ(round_clamped_to_uint8(16.0f, 15), 15);

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-4.0f, 260.0f);
    for (int i = 0; i < 10000; i++) {
        const float x = dist(rng);
        EXPECT_LE(
                std::abs(
                        static_cast<int>(round_clamped_to_uint8(x, 255)) -
                        static_cast<int>(round_clamped_ref(x, 255))),
                1);
    }
}

TEST(LinearTransform, SingleVectorMatchesBatchPath) {
    check_linear_transform_batch_match(false);
    check_linear_transform_batch_match(true);
}
