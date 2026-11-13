/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_simd.h>
#include <lsxintrin.h>
#include <bit>
#include <cstring>

namespace faiss::rabitq {
namespace {

struct LSXBits {
    using Register = __m128i;
    static constexpr size_t width = 16;
    static Register load(const uint8_t* p) {
        return __lsx_vld(p, 0);
    }
    static Register bit_and(Register a, Register b) {
        return __lsx_vand_v(a, b);
    }
    static Register bit_xor(Register a, Register b) {
        return __lsx_vxor_v(a, b);
    }
    static uint64_t popcount(Register v) {
        v = __lsx_vpcnt_d(v);
        v = __lsx_vadd_d(v, __lsx_vbsrl_v(v, 8));
        return static_cast<uint64_t>(__lsx_vpickve2gr_d(v, 0));
    }
};

using Bits = LSXBits;
template <size_t N>
__m128i lsx_load_short_code(const uint8_t* p) {
    static_assert(N == 4 || N == 8);
    uint64_t word = 0;
    std::memcpy(&word, p, N);
    return __lsx_vinsgr2vr_d(__lsx_vldi(0), word, 0);
}

// Reduce each register into uint64_t before weighting; narrow lane
// accumulators can overflow for long codes or high query bit counts.
template <bool use_xor, bool count_data>
BitwiseAndDotProductResult lsx_bitwise_dot(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0, count = 0;
    size_t i = 0;
    auto block = [&]<typename B, size_t N>() {
        auto load = [](const uint8_t* p) {
            if constexpr (N == B::width)
                return B::load(p);
            else
                return lsx_load_short_code<N>(p);
        };
        const auto value = load(data + i);
        if constexpr (count_data)
            count += B::popcount(value);
        for (size_t j = 0; j < qb; ++j) {
            const auto q = load(query + j * size + i);
            const auto combined =
                    use_xor ? B::bit_xor(value, q) : B::bit_and(value, q);
            sum += B::popcount(combined) << j;
        }
        i += N;
    };
    for (; i + Bits::width <= size;)
        block.template operator()<Bits, Bits::width>();
    // Constant-size partial loads avoid reading past short codes.
    using Narrow = LSXBits;
    if (i + 8 <= size)
        block.template operator()<Narrow, 8>();
    if (i + 4 <= size)
        block.template operator()<Narrow, 4>();
    for (; i < size; ++i) {
        if constexpr (count_data)
            count += popcount32(data[i]);
        for (size_t j = 0; j < qb; ++j) {
            unsigned value = use_xor ? data[i] ^ query[j * size + i]
                                     : data[i] & query[j * size + i];
            sum += uint64_t(popcount32(value)) << j;
        }
    }
    return {sum, count};
}
} // namespace
template <>
uint64_t bitwise_and_dot_product<SIMDLevel::LOONGARCH_LSX>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return lsx_bitwise_dot<false, false>(query, data, size, qb).dot_product;
}
template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::LOONGARCH_LSX>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return lsx_bitwise_dot<true, false>(query, data, size, qb).dot_product;
}
template <>
BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount<
        SIMDLevel::LOONGARCH_LSX>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return lsx_bitwise_dot<false, true>(query, data, size, qb);
}
template <>
uint64_t popcount<SIMDLevel::LOONGARCH_LSX>(const uint8_t* data, size_t size) {
    uint64_t result = 0;
    size_t i = 0;
    for (; i + Bits::width <= size; i += Bits::width)
        result += Bits::popcount(Bits::load(data + i));
    using Narrow = LSXBits;
    if (i + 8 <= size) {
        result += Narrow::popcount(lsx_load_short_code<8>(data + i));
        i += 8;
    }
    if (i + 4 <= size) {
        result += Narrow::popcount(lsx_load_short_code<4>(data + i));
        i += 4;
    }
    for (; i < size; ++i)
        result += popcount32(data[i]);
    return result;
}
} // namespace faiss::rabitq

namespace faiss::rabitq::multibit {
template <>
float compute_inner_product<SIMDLevel::LOONGARCH_LSX>(
        const uint8_t* sign_bits,
        const uint8_t* ex_code,
        const float* rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    if (ex_bits == 8) {
        return compute_inner_product<SIMDLevel::NONE>(
                sign_bits, ex_code, rotated_q, d, ex_bits, cb);
    }
    __m128 acc[2] = {};
    const __m128 bias = (__m128)__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(cb));
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        uint64_t packed = 0;
        memcpy(&packed, ex_code + i * ex_bits / 8, ex_bits);
        const uint32_t words[2] = {
                uint32_t(packed), uint32_t(packed >> (4 * ex_bits))};
        for (int j = 0; j < 2; ++j) {
            const uint32_t offsets[4] = {
                    0,
                    uint32_t(ex_bits),
                    uint32_t(2 * ex_bits),
                    uint32_t(3 * ex_bits)};
            auto extra = __lsx_vsrl_w(
                    __lsx_vreplgr2vr_w(words[j]), __lsx_vld(offsets, 0));
            extra = __lsx_vand_v(
                    extra, __lsx_vreplgr2vr_w((1u << ex_bits) - 1));
            const uint32_t sign_offsets[4] = {
                    uint32_t(j * 4),
                    uint32_t(j * 4 + 1),
                    uint32_t(j * 4 + 2),
                    uint32_t(j * 4 + 3)};
            auto sign = __lsx_vsrl_w(
                    __lsx_vreplgr2vr_w(sign_bits[i / 8]),
                    __lsx_vld(sign_offsets, 0));
            sign = __lsx_vand_v(sign, __lsx_vreplgr2vr_w(1));
            sign = __lsx_vsll_w(sign, __lsx_vreplgr2vr_w(ex_bits));
            const __m128 decoded = __lsx_vffint_s_w(__lsx_vadd_w(sign, extra));
            const __m128 query = (__m128)__lsx_vld(rotated_q + i + j * 4, 0);
            acc[j] = __lsx_vfadd_s(
                    acc[j], __lsx_vfmul_s(query, __lsx_vfadd_s(decoded, bias)));
        }
    }
    __m128 sum = __lsx_vfadd_s(acc[0], acc[1]);
    sum = __lsx_vfadd_s(sum, (__m128)__lsx_vshuf4i_w((__m128i)sum, 0x4e));
    sum = __lsx_vfadd_s(sum, (__m128)__lsx_vshuf4i_w((__m128i)sum, 0xb1));
    float result = std::bit_cast<float>(
            static_cast<uint32_t>(__lsx_vpickve2gr_w((__m128i)sum, 0)));
    for (; i < d; ++i) {
        const size_t bit = i * ex_bits;
        uint32_t value = ex_code[bit / 8] >> (bit % 8);
        if (bit % 8 + ex_bits > 8)
            value |= uint32_t(ex_code[bit / 8 + 1]) << (8 - bit % 8);
        value &= (1u << ex_bits) - 1;
        value += ((sign_bits[i / 8] >> (i % 8)) & 1u) << ex_bits;
        result += rotated_q[i] * (float(value) + cb);
    }
    return result;
}
} // namespace faiss::rabitq::multibit
