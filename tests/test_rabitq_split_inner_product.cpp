/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// rabitq::sign_inner_product and multibit::compute_inner_product_with_sign_ip
// score a multi-bit RaBitQ code whose sign bits are stored in pieces, e.g.
// partly in memory and partly on disk. They are checked against a double
// reference and against the fused multibit::compute_inner_product, at every
// SIMD level available on this machine.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_levels.h>

namespace {

using faiss::SIMDLevel;

struct RestoreSIMD {
    SIMDLevel saved = faiss::SIMDConfig::get_level();
    ~RestoreSIMD() {
        faiss::SIMDConfig::set_level(saved);
    }
};

std::vector<SIMDLevel> available_levels() {
    std::vector<SIMDLevel> levels;
    for (auto level :
         {faiss::SIMDConfig::get_level(),
          SIMDLevel::NONE,
          SIMDLevel::AVX2,
          SIMDLevel::AVX512,
          SIMDLevel::AVX512_VPOPCNT,
          SIMDLevel::ARM_NEON,
          SIMDLevel::RISCV_RVV}) {
        if (faiss::SIMDConfig::is_simd_level_available(level) &&
            std::find(levels.begin(), levels.end(), level) == levels.end()) {
            levels.push_back(level);
        }
    }
    return levels;
}

// Dispatched the way multibit::compute_inner_product is.
template <typename Action>
float dispatch(Action&& action) {
    return faiss::with_selected_simd_levels<
            faiss::AVAILABLE_SIMD_LEVELS_BASE_WITH_VPOPCNT>(
            std::forward<Action>(action));
}

float sign_ip(const uint8_t* sign_bits, const float* rotated_q, size_t d) {
    return dispatch([&]<SIMDLevel SL>() {
        return faiss::rabitq::sign_inner_product<SL>(sign_bits, rotated_q, d);
    });
}

float split_ip(
        const uint8_t* ex_code,
        const float* rotated_q,
        size_t d,
        size_t ex_bits,
        float cb,
        float sign_ip_value,
        float q_sum) {
    return dispatch([&]<SIMDLevel SL>() {
        return faiss::rabitq::multibit::compute_inner_product_with_sign_ip<SL>(
                ex_code, rotated_q, d, ex_bits, cb, sign_ip_value, q_sum);
    });
}

float fused_ip(
        const uint8_t* sign_bits,
        const uint8_t* ex_code,
        const float* rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    return dispatch([&]<SIMDLevel SL>() {
        return faiss::rabitq::multibit::compute_inner_product<SL>(
                sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    });
}

// Dimensions on either side of the 8-, 16- and 32-dim steps of the kernels.
const std::vector<size_t> kDims = {1,   7,   8,   9,    15,   16,  17,
                                   24,  31,  32,  33,   63,   64,  65,
                                   100, 129, 768, 1000, 1280, 1536};

std::vector<float> random_query(size_t d, std::mt19937& rng) {
    std::normal_distribution<float> normal;
    std::vector<float> q(d);
    for (auto& v : q) {
        v = normal(rng);
    }
    return q;
}

// One extra byte in front, so that the codes start at an odd address.
std::vector<uint8_t> random_codes(size_t n_bytes, std::mt19937& rng) {
    std::vector<uint8_t> codes(1 + n_bytes);
    for (auto& v : codes) {
        v = static_cast<uint8_t>(rng());
    }
    return codes;
}

int get_bit(const uint8_t* bits, size_t i) {
    return (bits[i / 8] >> (i % 8)) & 1;
}

unsigned get_ex(const uint8_t* ex_code, size_t i, size_t ex_bits) {
    unsigned value = 0;
    for (size_t b = 0; b < ex_bits; b++) {
        value |= unsigned(get_bit(ex_code, i * ex_bits + b)) << b;
    }
    return value;
}

} // namespace

TEST(RaBitQSplitInnerProduct, SignInnerProductMatchesReference) {
    RestoreSIMD restore;
    std::mt19937 rng(123);
    for (size_t d : kDims) {
        const auto q = random_query(d, rng);
        const auto codes = random_codes((d + 7) / 8, rng);
        const uint8_t* sign_bits = codes.data() + 1;

        double expected = 0, magnitude = 0;
        for (size_t i = 0; i < d; i++) {
            expected += get_bit(sign_bits, i) * double(q[i]);
            magnitude += std::abs(q[i]);
        }
        for (auto level : available_levels()) {
            faiss::SIMDConfig::set_level(level);
            EXPECT_NEAR(
                    sign_ip(sign_bits, q.data(), d),
                    expected,
                    1e-5 * magnitude + 1e-5)
                    << "d=" << d << " level=" << faiss::to_string(level);
        }
    }
}

TEST(RaBitQSplitInnerProduct, SignInnerProductSplitsAtByteBoundaries) {
    RestoreSIMD restore;
    std::mt19937 rng(456);
    const size_t d = 1280;
    const auto q = random_query(d, rng);
    const auto codes = random_codes(d / 8, rng);
    const uint8_t* sign_bits = codes.data() + 1;

    for (auto level : available_levels()) {
        faiss::SIMDConfig::set_level(level);
        const float full = sign_ip(sign_bits, q.data(), d);
        for (size_t m : {0, 8, 24, 64, 640, 1024, 1272, 1280}) {
            const float prefix = sign_ip(sign_bits, q.data(), m);
            const float suffix =
                    sign_ip(sign_bits + m / 8, q.data() + m, d - m);
            EXPECT_NEAR(prefix + suffix, full, 1e-3)
                    << "m=" << m << " level=" << faiss::to_string(level);
        }
    }
}

TEST(RaBitQSplitInnerProduct, WithSignIpMatchesFusedKernel) {
    RestoreSIMD restore;
    std::mt19937 rng(789);
    for (size_t d : kDims) {
        for (size_t ex_bits = 1; ex_bits <= 8; ex_bits++) {
            const float cb = -(float(1u << ex_bits) - 0.5f);
            const auto q = random_query(d, rng);
            const auto signs = random_codes((d + 7) / 8, rng);
            const size_t ex_bytes = (d * ex_bits + 7) / 8;
            // The fused kernel may read 8 bytes past the codes, so it gets a
            // padded copy. The split kernel gets codes of the exact size.
            const auto ex = random_codes(ex_bytes, rng);
            std::vector<uint8_t> ex_padded(ex.begin(), ex.end());
            ex_padded.resize(ex.size() + 8);
            const uint8_t* sign_bits = signs.data() + 1;
            const uint8_t* ex_code = ex.data() + 1;

            double expected = 0, magnitude = 0, q_sum = 0;
            for (size_t i = 0; i < d; i++) {
                const double recon = (get_bit(sign_bits, i) << ex_bits) +
                        double(get_ex(ex_code, i, ex_bits)) + cb;
                expected += q[i] * recon;
                magnitude += std::abs(q[i] * recon);
                q_sum += q[i];
            }
            const double tolerance = 1e-5 * magnitude + 1e-5;

            for (auto level : available_levels()) {
                faiss::SIMDConfig::set_level(level);
                const float s = sign_ip(sign_bits, q.data(), d);
                const float split =
                        split_ip(ex_code, q.data(), d, ex_bits, cb, s, q_sum);
                const float fused = fused_ip(
                        sign_bits,
                        ex_padded.data() + 1,
                        q.data(),
                        d,
                        ex_bits,
                        cb);
                EXPECT_NEAR(split, expected, tolerance)
                        << "d=" << d << " ex_bits=" << ex_bits
                        << " level=" << faiss::to_string(level);
                EXPECT_NEAR(split, fused, tolerance)
                        << "d=" << d << " ex_bits=" << ex_bits
                        << " level=" << faiss::to_string(level);
            }
        }
    }
}

// The use the kernels are for: sign bits split between a resident prefix and
// a stored suffix, the extra bits in the stored part only.
TEST(RaBitQSplitInnerProduct, SignTermAccumulatedInPieces) {
    RestoreSIMD restore;
    std::mt19937 rng(1011);
    const size_t d = 1536;
    for (size_t mem_dim : {0, 64, 384, 768, 1472, 1536}) {
        for (size_t ex_bits : {1, 4, 8}) {
            const float cb = -(float(1u << ex_bits) - 0.5f);
            const auto q = random_query(d, rng);
            const auto signs = random_codes(d / 8, rng);
            auto ex = random_codes((d * ex_bits + 7) / 8, rng);
            ex.resize(ex.size() + 8); // for the fused kernel
            const uint8_t* sign_bits = signs.data() + 1;
            const uint8_t* ex_code = ex.data() + 1;

            // Each piece in its own buffer, as if read from a different place.
            const std::vector<uint8_t> prefix(
                    sign_bits, sign_bits + mem_dim / 8);
            const std::vector<uint8_t> suffix(
                    sign_bits + mem_dim / 8, sign_bits + d / 8);

            double expected = 0, magnitude = 0, q_sum = 0;
            for (size_t i = 0; i < d; i++) {
                const double recon = (get_bit(sign_bits, i) << ex_bits) +
                        double(get_ex(ex_code, i, ex_bits)) + cb;
                expected += q[i] * recon;
                magnitude += std::abs(q[i] * recon);
                q_sum += q[i];
            }
            const double tolerance = 1e-5 * magnitude + 1e-5;

            for (auto level : available_levels()) {
                faiss::SIMDConfig::set_level(level);
                const float sign_term =
                        sign_ip(prefix.data(), q.data(), mem_dim) +
                        sign_ip(suffix.data(), q.data() + mem_dim, d - mem_dim);
                const float split = split_ip(
                        ex_code, q.data(), d, ex_bits, cb, sign_term, q_sum);
                const float fused =
                        fused_ip(sign_bits, ex_code, q.data(), d, ex_bits, cb);
                EXPECT_NEAR(split, expected, tolerance)
                        << "mem_dim=" << mem_dim << " ex_bits=" << ex_bits
                        << " level=" << faiss::to_string(level);
                EXPECT_NEAR(split, fused, tolerance)
                        << "mem_dim=" << mem_dim << " ex_bits=" << ex_bits
                        << " level=" << faiss::to_string(level);
            }
        }
    }
}

TEST(RaBitQSplitInnerProduct, WithSignIpWithoutExtraBits) {
    RestoreSIMD restore;
    std::mt19937 rng(1213);
    const size_t d = 100;
    const auto q = random_query(d, rng);
    const auto signs = random_codes((d + 7) / 8, rng);
    const uint8_t* sign_bits = signs.data() + 1;
    const float cb = -0.5f;

    double expected = 0, q_sum = 0;
    for (size_t i = 0; i < d; i++) {
        expected += q[i] * (get_bit(sign_bits, i) + cb);
        q_sum += q[i];
    }
    for (auto level : available_levels()) {
        faiss::SIMDConfig::set_level(level);
        const float s = sign_ip(sign_bits, q.data(), d);
        EXPECT_NEAR(
                split_ip(nullptr, q.data(), d, 0, cb, s, q_sum), expected, 1e-4)
                << "level=" << faiss::to_string(level);
    }
}
