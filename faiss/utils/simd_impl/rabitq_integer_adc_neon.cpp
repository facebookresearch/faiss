/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>

#include <faiss/impl/FaissAssert.h>

#include <arm_neon.h>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace faiss::rabitq_integer_adc {

bool arm_dotprod_supported() {
#if defined(__linux__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

namespace {

// Reduce at bounded intervals so the int32 lanes cannot overflow even for
// unusually high dimensions. One lane receives at most 1024 products per
// 4096-dimensional chunk, well below INT32_MAX for signed bytes.
constexpr size_t kDotChunk = 4096;

template <typename Function, size_t... Indices>
inline void for_fixed_lanes_impl(
        Function&& function,
        std::index_sequence<Indices...>) {
    (function(std::integral_constant<size_t, Indices>{}), ...);
}

template <int N, typename Function>
inline void for_fixed_lanes(Function&& function) {
    for_fixed_lanes_impl(
            std::forward<Function>(function), std::make_index_sequence<N>{});
}

template <int N>
void dot_product_batch_fixed_arm(
        const int8_t* query,
        const int8_t* const* levels,
        size_t d,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                accumulators[lane] = vdotq_s32(
                        accumulators[lane], q, vld1q_s8(levels[lane] + j));
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += int64_t(query[j]) * int64_t(levels[lane][j]);
        });
    }
}

FAISS_ALWAYS_INLINE int8x16_t
decode_packed_2bit_block(const uint8_t* signs, const uint8_t* extra) {
    static const uint8_t kSignMasksData[16] = {
            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t sign_bytes =
            vcombine_u8(vdup_n_u8(signs[0]), vdup_n_u8(signs[1]));
    const uint8x16_t extra_bytes =
            vcombine_u8(vdup_n_u8(extra[0]), vdup_n_u8(extra[1]));
    const uint8x16_t masks = vld1q_u8(kSignMasksData);
    const uint8x16_t sign_levels =
            vandq_u8(vtstq_u8(sign_bytes, masks), vdupq_n_u8(2));
    const uint8x16_t low_levels =
            vandq_u8(vtstq_u8(extra_bytes, masks), vdupq_n_u8(1));
    return vreinterpretq_s8_u8(
            vsubq_u8(vaddq_u8(sign_levels, low_levels), vdupq_n_u8(2)));
}

FAISS_ALWAYS_INLINE int8_t
decode_packed_2bit_tail(const uint8_t* code, const uint8_t* extra, size_t j) {
    const int low = (extra[j >> 3] >> (j & 7)) & 1;
    const int sign = (code[j >> 3] >> (j & 7)) & 1;
    return static_cast<int8_t>((sign << 1) + low - 2);
}

template <int N>
void dot_product_packed_2bit_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t extra_offset,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            const size_t sign_offset = j >> 3;
            const size_t extra_byte_offset = j >> 3;
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                const int8x16_t levels = decode_packed_2bit_block(
                        code + sign_offset,
                        code + extra_offset + extra_byte_offset);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            const uint8_t* code = codes[lane];
            dots[lane] += int64_t(query[j]) *
                    int64_t(decode_packed_2bit_tail(
                            code, code + extra_offset, j));
        });
    }
}

FAISS_ALWAYS_INLINE int8x16_t
decode_packed_4bit_block(const uint8_t* signs, const uint8_t* extra) {
    static const uint8_t kLowIndicesData[8] = {0, 0, 0, 1, 1, 1, 2, 2};
    static const uint8_t kHighIndicesData[8] = {1, 1, 1, 2, 2, 2, 3, 3};
    static const int16_t kShiftsData[8] = {0, -3, -6, -1, -4, -7, -2, -5};
    const auto decode8 = [&](const uint8_t* input) {
        uint32_t word = 0;
        memcpy(&word, input, 3);
        const uint8x8_t source = vcreate_u8(word);
        const uint16x8_t windows = vorrq_u16(
                vmovl_u8(vtbl1_u8(source, vld1_u8(kLowIndicesData))),
                vshlq_n_u16(
                        vmovl_u8(vtbl1_u8(source, vld1_u8(kHighIndicesData))),
                        8));
        return vmovn_u16(vandq_u16(
                vshlq_u16(windows, vld1q_s16(kShiftsData)), vdupq_n_u16(7)));
    };
    const uint8x16_t lows = vcombine_u8(decode8(extra), decode8(extra + 3));
    static const uint8_t kSignMasksData[16] = {
            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t sign_bytes =
            vcombine_u8(vdup_n_u8(signs[0]), vdup_n_u8(signs[1]));
    const uint8x16_t sign_levels = vandq_u8(
            vtstq_u8(sign_bytes, vld1q_u8(kSignMasksData)), vdupq_n_u8(8));
    return vreinterpretq_s8_u8(
            vsubq_u8(vorrq_u8(sign_levels, lows), vdupq_n_u8(8)));
}

FAISS_ALWAYS_INLINE int8_t
decode_packed_4bit_tail(const uint8_t* code, const uint8_t* extra, size_t j) {
    const size_t bit_position = 3 * j;
    uint32_t low = 0;
    for (size_t bit = 0; bit < 3; ++bit) {
        const size_t position = bit_position + bit;
        low |= ((extra[position >> 3] >> (position & 7)) & 1) << bit;
    }
    const int sign = (code[j >> 3] >> (j & 7)) & 1;
    return static_cast<int8_t>((sign << 3) + low - 8);
}

FAISS_ALWAYS_INLINE uint8x16_t
expand_bit_plane_16(const uint8_t* plane, uint8_t value) {
    static const uint8_t kMasksData[16] = {
            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t bytes =
            vcombine_u8(vdup_n_u8(plane[0]), vdup_n_u8(plane[1]));
    return vandq_u8(vtstq_u8(bytes, vld1q_u8(kMasksData)), vdupq_n_u8(value));
}

FAISS_ALWAYS_INLINE int8x16_t decode_split_4bit_block(
        const uint8_t* signs,
        const uint8_t* tails,
        size_t plane_stride) {
    const uint8x16_t levels = vorrq_u8(
            vorrq_u8(
                    expand_bit_plane_16(signs, 8),
                    expand_bit_plane_16(tails, 1)),
            vorrq_u8(
                    expand_bit_plane_16(tails + plane_stride, 2),
                    expand_bit_plane_16(tails + 2 * plane_stride, 4)));
    return vreinterpretq_s8_u8(vsubq_u8(levels, vdupq_n_u8(8)));
}

FAISS_ALWAYS_INLINE int8_t decode_split_4bit_tail(
        const uint8_t* signs,
        const uint8_t* tails,
        size_t plane_stride,
        size_t j) {
    int low = 0;
    for (size_t bit = 0; bit < 3; ++bit) {
        low |= ((tails[bit * plane_stride + (j >> 3)] >> (j & 7)) & 1) << bit;
    }
    const int sign = (signs[j >> 3] >> (j & 7)) & 1;
    return static_cast<int8_t>((sign << 3) + low - 8);
}

template <int N>
void dot_product_packed_4bit_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t extra_offset,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            const size_t sign_offset = j >> 3;
            const size_t extra_byte_offset = 3 * j / 8;
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                const int8x16_t levels = decode_packed_4bit_block(
                        code + sign_offset,
                        code + extra_offset + extra_byte_offset);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            const uint8_t* code = codes[lane];
            dots[lane] += int64_t(query[j]) *
                    int64_t(decode_packed_4bit_tail(
                            code, code + extra_offset, j));
        });
    }
}

void dot_product_packed_4bit_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        int64_t* dots) {
    switch (count) {
        case 1:
            return dot_product_packed_4bit_batch_fixed_arm<1>(
                    query, codes, d, extra_offset, dots);
        case 2:
            return dot_product_packed_4bit_batch_fixed_arm<2>(
                    query, codes, d, extra_offset, dots);
        case 3:
            return dot_product_packed_4bit_batch_fixed_arm<3>(
                    query, codes, d, extra_offset, dots);
        case 4:
            return dot_product_packed_4bit_batch_fixed_arm<4>(
                    query, codes, d, extra_offset, dots);
        case 5:
            return dot_product_packed_4bit_batch_fixed_arm<5>(
                    query, codes, d, extra_offset, dots);
        case 6:
            return dot_product_packed_4bit_batch_fixed_arm<6>(
                    query, codes, d, extra_offset, dots);
        case 7:
            return dot_product_packed_4bit_batch_fixed_arm<7>(
                    query, codes, d, extra_offset, dots);
        case 8:
            return dot_product_packed_4bit_batch_fixed_arm<8>(
                    query, codes, d, extra_offset, dots);
        case 16:
            return dot_product_packed_4bit_batch_fixed_arm<16>(
                    query, codes, d, extra_offset, dots);
    }
    FAISS_THROW_MSG("packed 4-bit batch size must be 1..8 or 16");
}

template <int N>
void dot_product_split_4bit_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* signs,
        const uint8_t* const* tails,
        size_t d,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    const size_t plane_stride = (d + 7) / 8;
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            const size_t sign_offset = j >> 3;
            const size_t tail_offset = j >> 3;
            for_fixed_lanes<N>([&](auto lane) {
                const int8x16_t levels = decode_split_4bit_block(
                        signs[lane] + sign_offset,
                        tails[lane] + tail_offset,
                        plane_stride);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += int64_t(query[j]) *
                    int64_t(decode_split_4bit_tail(
                            signs[lane], tails[lane], plane_stride, j));
        });
    }
}

void dot_product_split_4bit_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* signs,
        const uint8_t* const* tails,
        int count,
        size_t d,
        int64_t* dots) {
    switch (count) {
        case 1:
            return dot_product_split_4bit_batch_fixed_arm<1>(
                    query, signs, tails, d, dots);
        case 2:
            return dot_product_split_4bit_batch_fixed_arm<2>(
                    query, signs, tails, d, dots);
        case 3:
            return dot_product_split_4bit_batch_fixed_arm<3>(
                    query, signs, tails, d, dots);
        case 4:
            return dot_product_split_4bit_batch_fixed_arm<4>(
                    query, signs, tails, d, dots);
        case 5:
            return dot_product_split_4bit_batch_fixed_arm<5>(
                    query, signs, tails, d, dots);
        case 6:
            return dot_product_split_4bit_batch_fixed_arm<6>(
                    query, signs, tails, d, dots);
        case 7:
            return dot_product_split_4bit_batch_fixed_arm<7>(
                    query, signs, tails, d, dots);
        case 8:
            return dot_product_split_4bit_batch_fixed_arm<8>(
                    query, signs, tails, d, dots);
        case 16:
            return dot_product_split_4bit_batch_fixed_arm<16>(
                    query, signs, tails, d, dots);
    }
    FAISS_THROW_MSG("split 4-bit batch size must be 1..8 or 16");
}

FAISS_ALWAYS_INLINE uint8x16_t
decode_progressive_prefix_raw_16(const uint8_t* prefix) {
    uint32_t word;
    memcpy(&word, prefix, sizeof(word));
    const uint8x16_t source = vreinterpretq_u8_u32(vdupq_n_u32(word));
    static const uint8_t kIndicesData[16] = {
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    static const int8_t kShiftsData[16] = {
            0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    return vandq_u8(
            vshlq_u8(
                    vqtbl1q_u8(source, vld1q_u8(kIndicesData)),
                    vld1q_s8(kShiftsData)),
            vdupq_n_u8(3));
}

FAISS_ALWAYS_INLINE int8x16_t
load_nested_lut4_signed_table(const uint8_t* lut) {
    alignas(16) int8_t table[16];
    for (int symbol = 0; symbol < 16; ++symbol) {
        const int magnitude = lut[symbol & 7];
        table[symbol] =
                static_cast<int8_t>((symbol & 8) ? magnitude : -1 - magnitude);
    }
    return vld1q_s8(table);
}

FAISS_ALWAYS_INLINE int8x16_t
decode_nested_lut4_nibble_16(const uint8_t* packed, int8x16_t table) {
    const uint8x8_t bytes = vld1_u8(packed);
    const uint8x8_t low = vand_u8(bytes, vdup_n_u8(15));
    const uint8x8_t high = vshr_n_u8(bytes, 4);
    const uint8x16_t symbols = vzip1q_u8(
            vcombine_u8(low, vdup_n_u8(0)), vcombine_u8(high, vdup_n_u8(0)));
    return vreinterpretq_s8_u8(vqtbl1q_u8(vreinterpretq_u8_s8(table), symbols));
}

FAISS_ALWAYS_INLINE int8x16_t decode_nested_lut4_staged_16(
        const uint8_t* prefix,
        const uint8_t* local,
        int8x16_t table) {
    const uint8x16_t prefix_raw = decode_progressive_prefix_raw_16(prefix);
    const uint8x16_t local_raw = decode_progressive_prefix_raw_16(local);
    const uint8x16_t positive = vshrq_n_u8(prefix_raw, 1);
    const uint8x16_t coarse = veorq_u8(
            vandq_u8(prefix_raw, vdupq_n_u8(1)),
            veorq_u8(positive, vdupq_n_u8(1)));
    const uint8x16_t symbols = vorrq_u8(
            vshlq_n_u8(positive, 3),
            vorrq_u8(vshlq_n_u8(coarse, 2), local_raw));
    return vreinterpretq_s8_u8(vqtbl1q_u8(vreinterpretq_u8_s8(table), symbols));
}

FAISS_ALWAYS_INLINE int8_t decode_nested_lut4_tail(
        const uint8_t* code,
        size_t local_offset,
        size_t j,
        const uint8_t* lut,
        bool nibble_layout) {
    int symbol;
    if (nibble_layout) {
        symbol = (code[j >> 1] >> (4 * (j & 1))) & 15;
    } else {
        const int prefix = (code[j >> 2] >> (2 * (j & 3))) & 3;
        const int positive = prefix >> 1;
        const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
        const int local = (code[local_offset + (j >> 2)] >> (2 * (j & 3))) & 3;
        symbol = (positive << 3) | (coarse << 2) | local;
    }
    const int magnitude = lut[symbol & 7];
    return static_cast<int8_t>((symbol & 8) ? magnitude : -1 - magnitude);
}

template <int N>
void dot_product_nested_lut4_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    const int8x16_t table = load_nested_lut4_signed_table(lut);
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                const int8x16_t levels = nibble_layout
                        ? decode_nested_lut4_nibble_16(code + (j >> 1), table)
                        : decode_nested_lut4_staged_16(
                                  code + (j >> 2),
                                  code + local_offset + (j >> 2),
                                  table);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += int64_t(query[j]) *
                    int64_t(decode_nested_lut4_tail(
                            codes[lane], local_offset, j, lut, nibble_layout));
        });
    }
}

void dot_product_nested_lut4_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t* dots) {
#define FAISS_NESTED_LUT4_CASE(N)                          \
    case N:                                                \
        return dot_product_nested_lut4_batch_fixed_arm<N>( \
                query, codes, d, local_offset, lut, nibble_layout, dots)
    switch (count) {
        FAISS_NESTED_LUT4_CASE(1);
        FAISS_NESTED_LUT4_CASE(2);
        FAISS_NESTED_LUT4_CASE(3);
        FAISS_NESTED_LUT4_CASE(4);
        FAISS_NESTED_LUT4_CASE(5);
        FAISS_NESTED_LUT4_CASE(6);
        FAISS_NESTED_LUT4_CASE(7);
        FAISS_NESTED_LUT4_CASE(8);
        FAISS_NESTED_LUT4_CASE(16);
    }
#undef FAISS_NESTED_LUT4_CASE
    FAISS_THROW_MSG("nested LUT4 batch size must be 1..8 or 16");
}

FAISS_ALWAYS_INLINE int8x16_t
decode_progressive_prefix_16(const uint8_t* prefix) {
    return vreinterpretq_s8_u8(
            vsubq_u8(decode_progressive_prefix_raw_16(prefix), vdupq_n_u8(2)));
}

FAISS_ALWAYS_INLINE int8x16_t
decode_progressive_full6_16(const uint8_t* prefix, const uint8_t* tail) {
    static const uint8_t kLowIndicesData[8] = {0, 0, 1, 1, 2, 3, 3, 4};
    static const uint8_t kHighIndicesData[8] = {1, 1, 2, 2, 3, 4, 4, 5};
    static const int16_t kShiftsData[8] = {0, -5, -2, -7, -4, -1, -6, -3};
    const auto decode8 = [&](const uint8_t* input) {
        uint64_t word = 0;
        memcpy(&word, input, 5);
        const uint8x8_t source = vcreate_u8(word);
        const uint16x8_t windows = vorrq_u16(
                vmovl_u8(vtbl1_u8(source, vld1_u8(kLowIndicesData))),
                vshlq_n_u16(
                        vmovl_u8(vtbl1_u8(source, vld1_u8(kHighIndicesData))),
                        8));
        return vmovn_u16(vandq_u16(
                vshlq_u16(windows, vld1q_s16(kShiftsData)), vdupq_n_u16(31)));
    };
    const uint8x16_t tails = vcombine_u8(decode8(tail), decode8(tail + 5));
    const uint8x16_t signs =
            vshrq_n_u8(decode_progressive_prefix_raw_16(prefix), 1);
    const uint8x16_t levels = vorrq_u8(vshlq_n_u8(signs, 5), tails);
    return vreinterpretq_s8_u8(vsubq_u8(levels, vdupq_n_u8(32)));
}

FAISS_ALWAYS_INLINE int8x16_t
decode_progressive_full7_16(const uint8_t* prefix, const uint8_t* tail) {
    // Six-bit packing repeats every four coordinates:
    //   [aaaaaa bb][bbbb cccc][cc dddddd].
    // Gather the three source bytes for four groups, reconstruct the four
    // values in parallel, then interleave them in coordinate order.
    alignas(16) uint8_t padded[16] = {};
    memcpy(padded, tail, 12);
    const uint8x16_t source = vld1q_u8(padded);
    static const uint8_t kByte0Indices[16] = {
            0,
            3,
            6,
            9,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255};
    static const uint8_t kByte1Indices[16] = {
            1,
            4,
            7,
            10,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255};
    static const uint8_t kByte2Indices[16] = {
            2,
            5,
            8,
            11,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255};
    const uint8x16_t b0 = vqtbl1q_u8(source, vld1q_u8(kByte0Indices));
    const uint8x16_t b1 = vqtbl1q_u8(source, vld1q_u8(kByte1Indices));
    const uint8x16_t b2 = vqtbl1q_u8(source, vld1q_u8(kByte2Indices));
    const uint8x16_t mask = vdupq_n_u8(63);
    const uint8x16_t value0 = vandq_u8(b0, mask);
    const uint8x16_t value1 =
            vandq_u8(vorrq_u8(vshrq_n_u8(b0, 6), vshlq_n_u8(b1, 2)), mask);
    const uint8x16_t value2 =
            vandq_u8(vorrq_u8(vshrq_n_u8(b1, 4), vshlq_n_u8(b2, 4)), mask);
    const uint8x16_t value3 = vshrq_n_u8(b2, 2);
    const uint8x16_t pairs01 = vzip1q_u8(value0, value1);
    const uint8x16_t pairs23 = vzip1q_u8(value2, value3);
    const uint8x16_t tails = vreinterpretq_u8_u16(vzip1q_u16(
            vreinterpretq_u16_u8(pairs01), vreinterpretq_u16_u8(pairs23)));
    const uint8x16_t signs =
            vshrq_n_u8(decode_progressive_prefix_raw_16(prefix), 1);
    const uint8x16_t levels = vorrq_u8(vshlq_n_u8(signs, 6), tails);
    return vreinterpretq_s8_u8(vsubq_u8(levels, vdupq_n_u8(64)));
}

FAISS_ALWAYS_INLINE int8x16_t decode_progressive_tail4_16(const uint8_t* tail) {
    const uint8x8_t packed_tail = vld1_u8(tail);
    const uint8x8_t low = vand_u8(packed_tail, vdup_n_u8(15));
    const uint8x8_t high = vshr_n_u8(packed_tail, 4);
    const uint8x16_t tails = vzip1q_u8(
            vcombine_u8(low, vdup_n_u8(0)), vcombine_u8(high, vdup_n_u8(0)));
    return vreinterpretq_s8_u8(vsubq_u8(tails, vdupq_n_u8(8)));
}

FAISS_ALWAYS_INLINE int8_t
decode_progressive_prefix_tail(const uint8_t* code, size_t j) {
    return static_cast<int8_t>(((code[j >> 2] >> (2 * (j & 3))) & 3) - 2);
}

FAISS_ALWAYS_INLINE int8_t decode_progressive_full6_tail(
        const uint8_t* code,
        size_t tail_offset,
        size_t j) {
    const int prefix = (code[j >> 2] >> (2 * (j & 3))) & 3;
    const size_t bit_position = 5 * j;
    int tail = 0;
    for (size_t bit = 0; bit < 5; ++bit) {
        const size_t position = bit_position + bit;
        tail |= ((code[tail_offset + position / 8] >> (position % 8)) & 1)
                << bit;
    }
    const int sign = prefix >> 1;
    return static_cast<int8_t>((sign << 5) + tail - 32);
}

FAISS_ALWAYS_INLINE int8_t decode_progressive_full7_tail(
        const uint8_t* code,
        size_t tail_offset,
        size_t j) {
    const int prefix = (code[j >> 2] >> (2 * (j & 3))) & 3;
    const size_t bit_position = 6 * j;
    int tail = 0;
    for (size_t bit = 0; bit < 6; ++bit) {
        const size_t position = bit_position + bit;
        tail |= ((code[tail_offset + position / 8] >> (position % 8)) & 1)
                << bit;
    }
    const int sign = prefix >> 1;
    return static_cast<int8_t>((sign << 6) + tail - 64);
}

template <int N, bool PrefixOnly>
void dot_product_progressive6_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                const int8x16_t levels = PrefixOnly
                        ? decode_progressive_prefix_16(code + j / 4)
                        : decode_progressive_full6_16(
                                  code + j / 4, code + tail_offset + 5 * j / 8);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            const int8_t level = PrefixOnly
                    ? decode_progressive_prefix_tail(codes[lane], j)
                    : decode_progressive_full6_tail(
                              codes[lane], tail_offset, j);
            dots[lane] += int64_t(query[j]) * int64_t(level);
        });
    }
}

template <bool PrefixOnly>
void dot_product_progressive6_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
    switch (count) {
        case 1:
            return dot_product_progressive6_batch_fixed_arm<1, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 2:
            return dot_product_progressive6_batch_fixed_arm<2, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 3:
            return dot_product_progressive6_batch_fixed_arm<3, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 4:
            return dot_product_progressive6_batch_fixed_arm<4, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 5:
            return dot_product_progressive6_batch_fixed_arm<5, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 6:
            return dot_product_progressive6_batch_fixed_arm<6, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 7:
            return dot_product_progressive6_batch_fixed_arm<7, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 8:
            return dot_product_progressive6_batch_fixed_arm<8, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
        case 16:
            return dot_product_progressive6_batch_fixed_arm<16, PrefixOnly>(
                    query, codes, d, tail_offset, dots);
    }
    FAISS_THROW_MSG("progressive batch size must be 1..8 or 16");
}

template <int N>
void dot_product_progressive7_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                const int8x16_t levels = decode_progressive_full7_16(
                        code + j / 4, code + tail_offset + 3 * j / 4);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            const int8_t level =
                    decode_progressive_full7_tail(codes[lane], tail_offset, j);
            dots[lane] += int64_t(query[j]) * int64_t(level);
        });
    }
}

void dot_product_progressive7_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
#define FAISS_PROGRESSIVE7_CASE(N)                          \
    case N:                                                 \
        return dot_product_progressive7_batch_fixed_arm<N>( \
                query, codes, d, tail_offset, dots)
    switch (count) {
        FAISS_PROGRESSIVE7_CASE(1);
        FAISS_PROGRESSIVE7_CASE(2);
        FAISS_PROGRESSIVE7_CASE(3);
        FAISS_PROGRESSIVE7_CASE(4);
        FAISS_PROGRESSIVE7_CASE(5);
        FAISS_PROGRESSIVE7_CASE(6);
        FAISS_PROGRESSIVE7_CASE(7);
        FAISS_PROGRESSIVE7_CASE(8);
        FAISS_PROGRESSIVE7_CASE(16);
    }
#undef FAISS_PROGRESSIVE7_CASE
    FAISS_THROW_MSG("progressive batch size must be 1..8 or 16");
}

struct NestedLut7Tables {
    uint8x16x2_t cell0;
    uint8x16x2_t cell1;
};

FAISS_ALWAYS_INLINE NestedLut7Tables
load_nested_lut7_tables(const uint8_t* lut) {
    NestedLut7Tables tables;
    tables.cell0.val[0] = vld1q_u8(lut);
    tables.cell0.val[1] = vld1q_u8(lut + 16);
    tables.cell1.val[0] = vld1q_u8(lut + 32);
    tables.cell1.val[1] = vld1q_u8(lut + 48);
    return tables;
}

FAISS_ALWAYS_INLINE uint8x16_t decode_nested_high1_16(const uint8_t* high1) {
    uint16_t high_word;
    memcpy(&high_word, high1, sizeof(high_word));
    const uint8x16_t repeated = vreinterpretq_u8_u16(vdupq_n_u16(high_word));
    static const uint8_t kHighIndices[16] = {
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    static const int8_t kHighShifts[16] = {
            0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7};
    return vandq_u8(
            vshlq_u8(
                    vqtbl1q_u8(repeated, vld1q_u8(kHighIndices)),
                    vld1q_s8(kHighShifts)),
            vdupq_n_u8(1));
}

FAISS_ALWAYS_INLINE uint8x16_t
decode_nested_lut7_local_16(const uint8_t* low4, const uint8_t* high1) {
    const uint8x8_t packed_low = vld1_u8(low4);
    const uint8x8_t low_nibbles = vand_u8(packed_low, vdup_n_u8(15));
    const uint8x8_t high_nibbles = vshr_n_u8(packed_low, 4);
    const uint8x16_t local_low = vzip1q_u8(
            vcombine_u8(low_nibbles, vdup_n_u8(0)),
            vcombine_u8(high_nibbles, vdup_n_u8(0)));
    const uint8x16_t local_high = decode_nested_high1_16(high1);
    return vorrq_u8(local_low, vshlq_n_u8(local_high, 4));
}

FAISS_ALWAYS_INLINE uint8x16_t load_nested_lut3_table(const uint8_t* lut) {
    // Each high local bit represents one ordered half of a 32-entry cell.
    // The middle codepoint is a deterministic representative that requires
    // no extra serialized metadata.
    const uint8_t values[16] = {
            lut[7],
            lut[23],
            lut[39],
            lut[55],
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0};
    return vld1q_u8(values);
}

FAISS_ALWAYS_INLINE int8x16_t decode_nested_lut3_16(
        const uint8_t* prefix,
        const uint8_t* high1,
        uint8x16_t table) {
    const uint8x16_t raw_prefix = decode_progressive_prefix_raw_16(prefix);
    const uint8x16_t positive = vshrq_n_u8(raw_prefix, 1);
    const uint8x16_t negative_mask = vceqq_u8(positive, vdupq_n_u8(0));
    const uint8x16_t coarse = veorq_u8(
            vandq_u8(raw_prefix, vdupq_n_u8(1)),
            vandq_u8(negative_mask, vdupq_n_u8(1)));
    const uint8x16_t local_high = decode_nested_high1_16(high1);
    const uint8x16_t selector = vorrq_u8(vshlq_n_u8(coarse, 1), local_high);
    const uint8x16_t magnitude = vqtbl1q_u8(table, selector);
    return vreinterpretq_s8_u8(vbslq_u8(
            vcgtq_u8(positive, vdupq_n_u8(0)), magnitude, vmvnq_u8(magnitude)));
}

FAISS_ALWAYS_INLINE int8_t decode_nested_lut3_tail(
        const uint8_t* code,
        size_t high1_offset,
        const uint8_t* lut,
        size_t j) {
    const int prefix = (code[j >> 2] >> (2 * (j & 3))) & 3;
    const int positive = prefix >> 1;
    const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
    const int local_high = (code[high1_offset + j / 8] >> (j % 8)) & 1;
    const int magnitude = lut[coarse * 32 + local_high * 16 + 7];
    return static_cast<int8_t>(positive ? magnitude : -1 - magnitude);
}

template <int N>
void dot_product_nested_lut3_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
    const uint8x16_t table = load_nested_lut3_table(lut);
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const int8x16_t levels = decode_nested_lut3_16(
                        codes[lane] + j / 4,
                        codes[lane] + high1_offset + j / 8,
                        table);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += int64_t(query[j]) *
                    int64_t(decode_nested_lut3_tail(
                            codes[lane], high1_offset, lut, j));
        });
    }
}

void dot_product_nested_lut3_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
#define FAISS_NESTED_LUT3_CASE(N)                          \
    case N:                                                \
        return dot_product_nested_lut3_batch_fixed_arm<N>( \
                query, codes, d, high1_offset, lut, dots)
    switch (count) {
        FAISS_NESTED_LUT3_CASE(1);
        FAISS_NESTED_LUT3_CASE(2);
        FAISS_NESTED_LUT3_CASE(3);
        FAISS_NESTED_LUT3_CASE(4);
        FAISS_NESTED_LUT3_CASE(5);
        FAISS_NESTED_LUT3_CASE(6);
        FAISS_NESTED_LUT3_CASE(7);
        FAISS_NESTED_LUT3_CASE(8);
        FAISS_NESTED_LUT3_CASE(16);
    }
#undef FAISS_NESTED_LUT3_CASE
    FAISS_THROW_MSG("nested LUT3 batch size must be 1..8 or 16");
}

FAISS_ALWAYS_INLINE int8x16_t decode_nested_lut7_16(
        const uint8_t* prefix,
        const uint8_t* low4,
        const uint8_t* high1,
        const NestedLut7Tables& tables) {
    const uint8x16_t raw_prefix = decode_progressive_prefix_raw_16(prefix);
    const uint8x16_t positive = vshrq_n_u8(raw_prefix, 1);
    const uint8x16_t negative_mask = vceqq_u8(positive, vdupq_n_u8(0));
    const uint8x16_t coarse = veorq_u8(
            vandq_u8(raw_prefix, vdupq_n_u8(1)),
            vandq_u8(negative_mask, vdupq_n_u8(1)));
    const uint8x16_t local = decode_nested_lut7_local_16(low4, high1);
    const uint8x16_t magnitude = vbslq_u8(
            vcgtq_u8(coarse, vdupq_n_u8(0)),
            vqtbl2q_u8(tables.cell1, local),
            vqtbl2q_u8(tables.cell0, local));
    return vreinterpretq_s8_u8(vbslq_u8(
            vcgtq_u8(positive, vdupq_n_u8(0)), magnitude, vmvnq_u8(magnitude)));
}

FAISS_ALWAYS_INLINE int8x16_t decode_nested_exact7_16(
        const uint8_t* prefix,
        const uint8_t* low4,
        const uint8_t* high1) {
    const uint8x16_t raw_prefix = decode_progressive_prefix_raw_16(prefix);
    const uint8x16_t positive = vshrq_n_u8(raw_prefix, 1);
    const uint8x16_t negative_mask = vceqq_u8(positive, vdupq_n_u8(0));
    const uint8x16_t coarse = veorq_u8(
            vandq_u8(raw_prefix, vdupq_n_u8(1)),
            vandq_u8(negative_mask, vdupq_n_u8(1)));
    const uint8x16_t local = decode_nested_lut7_local_16(low4, high1);
    const uint8x16_t magnitude = vorrq_u8(local, vshlq_n_u8(coarse, 5));
    return vreinterpretq_s8_u8(vbslq_u8(
            vcgtq_u8(positive, vdupq_n_u8(0)), magnitude, vmvnq_u8(magnitude)));
}

FAISS_ALWAYS_INLINE int8_t decode_nested_lut7_tail(
        const uint8_t* code,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        size_t j) {
    const int prefix = (code[j >> 2] >> (2 * (j & 3))) & 3;
    const int positive = prefix >> 1;
    const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
    const int local_low =
            (code[low4_offset + (4 * j) / 8] >> ((4 * j) % 8)) & 15;
    const int local_high = (code[high1_offset + j / 8] >> (j % 8)) & 1;
    const int magnitude = lut[coarse * 32 + local_low + 16 * local_high];
    return static_cast<int8_t>(positive ? magnitude : -1 - magnitude);
}

template <int N>
void dot_product_nested_lut7_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
    const NestedLut7Tables tables = load_nested_lut7_tables(lut);
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                const int8x16_t levels = decode_nested_lut7_16(
                        code + j / 4,
                        code + low4_offset + j / 2,
                        code + high1_offset + j / 8,
                        tables);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += int64_t(query[j]) *
                    int64_t(decode_nested_lut7_tail(
                            codes[lane], low4_offset, high1_offset, lut, j));
        });
    }
}

void dot_product_nested_lut7_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
#define FAISS_NESTED_LUT7_CASE(N)                          \
    case N:                                                \
        return dot_product_nested_lut7_batch_fixed_arm<N>( \
                query, codes, d, low4_offset, high1_offset, lut, dots)
    switch (count) {
        FAISS_NESTED_LUT7_CASE(1);
        FAISS_NESTED_LUT7_CASE(2);
        FAISS_NESTED_LUT7_CASE(3);
        FAISS_NESTED_LUT7_CASE(4);
        FAISS_NESTED_LUT7_CASE(5);
        FAISS_NESTED_LUT7_CASE(6);
        FAISS_NESTED_LUT7_CASE(7);
        FAISS_NESTED_LUT7_CASE(8);
        FAISS_NESTED_LUT7_CASE(16);
    }
#undef FAISS_NESTED_LUT7_CASE
    FAISS_THROW_MSG("nested LUT batch size must be 1..8 or 16");
}

template <int N>
void dot_product_nested_exact7_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        int64_t* dots) {
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t accumulators[N];
        for_fixed_lanes<N>(
                [&](auto lane) { accumulators[lane] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                const int8x16_t levels = decode_nested_exact7_16(
                        code + j / 4,
                        code + low4_offset + j / 2,
                        code + high1_offset + j / 8);
                accumulators[lane] = vdotq_s32(accumulators[lane], q, levels);
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] += static_cast<int64_t>(vaddvq_s32(accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            const uint8_t* code = codes[lane];
            const int prefix = (code[j >> 2] >> (2 * (j & 3))) & 3;
            const int positive = prefix >> 1;
            const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
            const int local_low =
                    (code[low4_offset + (4 * j) / 8] >> ((4 * j) % 8)) & 15;
            const int local_high = (code[high1_offset + j / 8] >> (j % 8)) & 1;
            const int magnitude = local_low | (local_high << 4) | (coarse << 5);
            const int level = positive ? magnitude : -1 - magnitude;
            dots[lane] += int64_t(query[j]) * int64_t(level);
        });
    }
}

void dot_product_nested_exact7_batch_dispatch(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        int64_t* dots) {
#define FAISS_NESTED_EXACT7_CASE(N)                          \
    case N:                                                  \
        return dot_product_nested_exact7_batch_fixed_arm<N>( \
                query, codes, d, low4_offset, high1_offset, dots)
    switch (count) {
        FAISS_NESTED_EXACT7_CASE(1);
        FAISS_NESTED_EXACT7_CASE(2);
        FAISS_NESTED_EXACT7_CASE(3);
        FAISS_NESTED_EXACT7_CASE(4);
        FAISS_NESTED_EXACT7_CASE(5);
        FAISS_NESTED_EXACT7_CASE(6);
        FAISS_NESTED_EXACT7_CASE(7);
        FAISS_NESTED_EXACT7_CASE(8);
        FAISS_NESTED_EXACT7_CASE(16);
    }
#undef FAISS_NESTED_EXACT7_CASE
    FAISS_THROW_MSG("nested exact RQ7 batch size must be 1..8 or 16");
}

template <int N>
void dot_product_progressive_refine6_batch_fixed_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t tail_offset,
        int64_t* prefix_dots,
        int64_t* tail_dots) {
    for_fixed_lanes<N>([&](auto lane) {
        prefix_dots[lane] = 0;
        tail_dots[lane] = 0;
    });
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t prefix_accumulators[N];
        int32x4_t tail_accumulators[N];
        for_fixed_lanes<N>([&](auto lane) {
            prefix_accumulators[lane] = vdupq_n_s32(0);
            tail_accumulators[lane] = vdupq_n_s32(0);
        });
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const uint8_t* code = codes[lane];
                prefix_accumulators[lane] = vdotq_s32(
                        prefix_accumulators[lane],
                        q,
                        decode_progressive_prefix_16(code + j / 4));
                tail_accumulators[lane] = vdotq_s32(
                        tail_accumulators[lane],
                        q,
                        decode_progressive_tail4_16(
                                code + tail_offset + j / 2));
            });
        }
        for_fixed_lanes<N>([&](auto lane) {
            prefix_dots[lane] +=
                    static_cast<int64_t>(vaddvq_s32(prefix_accumulators[lane]));
            tail_dots[lane] +=
                    static_cast<int64_t>(vaddvq_s32(tail_accumulators[lane]));
        });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            const uint8_t* code = codes[lane];
            const int8_t prefix = decode_progressive_prefix_tail(code, j);
            const uint8_t packed = code[tail_offset + (j >> 1)];
            const int8_t tail =
                    static_cast<int8_t>(((packed >> (4 * (j & 1))) & 15) - 8);
            prefix_dots[lane] += int64_t(query[j]) * int64_t(prefix);
            tail_dots[lane] += int64_t(query[j]) * int64_t(tail);
        });
    }
}

template <int N>
void dot_product_progressive_refine6_batch_chunk(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        size_t tail_offset,
        int64_t* prefix_dots,
        int64_t* tail_dots) {
    dot_product_progressive_refine6_batch_fixed_arm<N>(
            query, codes, d, tail_offset, prefix_dots, tail_dots);
}

} // namespace

int64_t dot_product_arm(const int8_t* query, const int8_t* levels, size_t d) {
    int64_t result = 0;
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        for (; j + 32 <= end; j += 32) {
            acc0 = vdotq_s32(acc0, vld1q_s8(query + j), vld1q_s8(levels + j));
            acc1 = vdotq_s32(
                    acc1, vld1q_s8(query + j + 16), vld1q_s8(levels + j + 16));
        }
        if (j + 16 <= end) {
            acc0 = vdotq_s32(acc0, vld1q_s8(query + j), vld1q_s8(levels + j));
            j += 16;
        }
        result += static_cast<int64_t>(vaddvq_s32(acc0)) +
                static_cast<int64_t>(vaddvq_s32(acc1));
    }
    return result + dot_product_scalar(query + j, levels + j, d - j);
}

void dot_product_batch_4_arm(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3) {
    dot0 = dot1 = dot2 = dot3 = 0;
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            acc0 = vdotq_s32(acc0, q, vld1q_s8(levels0 + j));
            acc1 = vdotq_s32(acc1, q, vld1q_s8(levels1 + j));
            acc2 = vdotq_s32(acc2, q, vld1q_s8(levels2 + j));
            acc3 = vdotq_s32(acc3, q, vld1q_s8(levels3 + j));
        }
        dot0 += vaddvq_s32(acc0);
        dot1 += vaddvq_s32(acc1);
        dot2 += vaddvq_s32(acc2);
        dot3 += vaddvq_s32(acc3);
    }
    int64_t tail0, tail1, tail2, tail3;
    dot_product_batch_4_scalar(
            query + j,
            levels0 + j,
            levels1 + j,
            levels2 + j,
            levels3 + j,
            d - j,
            tail0,
            tail1,
            tail2,
            tail3);
    dot0 += tail0;
    dot1 += tail1;
    dot2 += tail2;
    dot3 += tail3;
}

void dot_product_batch_8_arm(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t dots[8]) {
    for (size_t k = 0; k < 8; ++k) {
        dots[k] = 0;
    }
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            acc0 = vdotq_s32(acc0, q, vld1q_s8(levels[0] + j));
            acc1 = vdotq_s32(acc1, q, vld1q_s8(levels[1] + j));
            acc2 = vdotq_s32(acc2, q, vld1q_s8(levels[2] + j));
            acc3 = vdotq_s32(acc3, q, vld1q_s8(levels[3] + j));
            acc4 = vdotq_s32(acc4, q, vld1q_s8(levels[4] + j));
            acc5 = vdotq_s32(acc5, q, vld1q_s8(levels[5] + j));
            acc6 = vdotq_s32(acc6, q, vld1q_s8(levels[6] + j));
            acc7 = vdotq_s32(acc7, q, vld1q_s8(levels[7] + j));
        }
        dots[0] += vaddvq_s32(acc0);
        dots[1] += vaddvq_s32(acc1);
        dots[2] += vaddvq_s32(acc2);
        dots[3] += vaddvq_s32(acc3);
        dots[4] += vaddvq_s32(acc4);
        dots[5] += vaddvq_s32(acc5);
        dots[6] += vaddvq_s32(acc6);
        dots[7] += vaddvq_s32(acc7);
    }
    const int8_t* tail_levels[8];
    for (size_t k = 0; k < 8; ++k) {
        tail_levels[k] = levels[k] + j;
    }
    int64_t tails[8];
    dot_product_batch_8_scalar(query + j, tail_levels, d - j, tails);
    for (size_t k = 0; k < 8; ++k) {
        dots[k] += tails[k];
    }
}

void dot_product_batch_16_arm(
        const int8_t* query,
        const int8_t* const levels[16],
        size_t d,
        int64_t dots[16]) {
    for (size_t k = 0; k < 16; ++k) {
        dots[k] = 0;
    }
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        int32x4_t acc8 = vdupq_n_s32(0);
        int32x4_t acc9 = vdupq_n_s32(0);
        int32x4_t acc10 = vdupq_n_s32(0);
        int32x4_t acc11 = vdupq_n_s32(0);
        int32x4_t acc12 = vdupq_n_s32(0);
        int32x4_t acc13 = vdupq_n_s32(0);
        int32x4_t acc14 = vdupq_n_s32(0);
        int32x4_t acc15 = vdupq_n_s32(0);
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            acc0 = vdotq_s32(acc0, q, vld1q_s8(levels[0] + j));
            acc1 = vdotq_s32(acc1, q, vld1q_s8(levels[1] + j));
            acc2 = vdotq_s32(acc2, q, vld1q_s8(levels[2] + j));
            acc3 = vdotq_s32(acc3, q, vld1q_s8(levels[3] + j));
            acc4 = vdotq_s32(acc4, q, vld1q_s8(levels[4] + j));
            acc5 = vdotq_s32(acc5, q, vld1q_s8(levels[5] + j));
            acc6 = vdotq_s32(acc6, q, vld1q_s8(levels[6] + j));
            acc7 = vdotq_s32(acc7, q, vld1q_s8(levels[7] + j));
            acc8 = vdotq_s32(acc8, q, vld1q_s8(levels[8] + j));
            acc9 = vdotq_s32(acc9, q, vld1q_s8(levels[9] + j));
            acc10 = vdotq_s32(acc10, q, vld1q_s8(levels[10] + j));
            acc11 = vdotq_s32(acc11, q, vld1q_s8(levels[11] + j));
            acc12 = vdotq_s32(acc12, q, vld1q_s8(levels[12] + j));
            acc13 = vdotq_s32(acc13, q, vld1q_s8(levels[13] + j));
            acc14 = vdotq_s32(acc14, q, vld1q_s8(levels[14] + j));
            acc15 = vdotq_s32(acc15, q, vld1q_s8(levels[15] + j));
        }
        dots[0] += vaddvq_s32(acc0);
        dots[1] += vaddvq_s32(acc1);
        dots[2] += vaddvq_s32(acc2);
        dots[3] += vaddvq_s32(acc3);
        dots[4] += vaddvq_s32(acc4);
        dots[5] += vaddvq_s32(acc5);
        dots[6] += vaddvq_s32(acc6);
        dots[7] += vaddvq_s32(acc7);
        dots[8] += vaddvq_s32(acc8);
        dots[9] += vaddvq_s32(acc9);
        dots[10] += vaddvq_s32(acc10);
        dots[11] += vaddvq_s32(acc11);
        dots[12] += vaddvq_s32(acc12);
        dots[13] += vaddvq_s32(acc13);
        dots[14] += vaddvq_s32(acc14);
        dots[15] += vaddvq_s32(acc15);
    }
    const int8_t* tail_levels[16];
    for (size_t k = 0; k < 16; ++k) {
        tail_levels[k] = levels[k] + j;
    }
    int64_t tails[16];
    dot_product_batch_16_scalar(query + j, tail_levels, d - j, tails);
    for (size_t k = 0; k < 16; ++k) {
        dots[k] += tails[k];
    }
}

void dot_product_batch_tail_arm(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots) {
    switch (count) {
        case 1:
            return dot_product_batch_fixed_arm<1>(query, levels, d, dots);
        case 2:
            return dot_product_batch_fixed_arm<2>(query, levels, d, dots);
        case 3:
            return dot_product_batch_fixed_arm<3>(query, levels, d, dots);
        case 4:
            return dot_product_batch_fixed_arm<4>(query, levels, d, dots);
        case 5:
            return dot_product_batch_fixed_arm<5>(query, levels, d, dots);
        case 6:
            return dot_product_batch_fixed_arm<6>(query, levels, d, dots);
        case 7:
            return dot_product_batch_fixed_arm<7>(query, levels, d, dots);
    }
    dot_product_batch_tail_scalar(query, levels, count, d, dots);
}

int64_t dot_product_packed_2bit_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t extra_offset) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_packed_2bit_batch_fixed_arm<1>(
            query, codes, d, extra_offset, &dot);
    return dot;
}

void dot_product_packed_2bit_batch_4_arm(
        const int8_t* query,
        const uint8_t* const codes[4],
        size_t d,
        size_t extra_offset,
        int64_t dots[4]) {
    dot_product_packed_2bit_batch_fixed_arm<4>(
            query, codes, d, extra_offset, dots);
}

void dot_product_packed_2bit_batch_8_arm(
        const int8_t* query,
        const uint8_t* const codes[8],
        size_t d,
        size_t extra_offset,
        int64_t dots[8]) {
    dot_product_packed_2bit_batch_fixed_arm<8>(
            query, codes, d, extra_offset, dots);
}

void dot_product_packed_2bit_batch_16_arm(
        const int8_t* query,
        const uint8_t* const codes[16],
        size_t d,
        size_t extra_offset,
        int64_t dots[16]) {
    dot_product_packed_2bit_batch_fixed_arm<16>(
            query, codes, d, extra_offset, dots);
}

void dot_product_packed_2bit_batch_tail_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        int64_t* dots) {
    switch (count) {
        case 1:
            return dot_product_packed_2bit_batch_fixed_arm<1>(
                    query, codes, d, extra_offset, dots);
        case 2:
            return dot_product_packed_2bit_batch_fixed_arm<2>(
                    query, codes, d, extra_offset, dots);
        case 3:
            return dot_product_packed_2bit_batch_fixed_arm<3>(
                    query, codes, d, extra_offset, dots);
        case 4:
            return dot_product_packed_2bit_batch_fixed_arm<4>(
                    query, codes, d, extra_offset, dots);
        case 5:
            return dot_product_packed_2bit_batch_fixed_arm<5>(
                    query, codes, d, extra_offset, dots);
        case 6:
            return dot_product_packed_2bit_batch_fixed_arm<6>(
                    query, codes, d, extra_offset, dots);
        case 7:
            return dot_product_packed_2bit_batch_fixed_arm<7>(
                    query, codes, d, extra_offset, dots);
    }
    FAISS_THROW_MSG("packed 2-bit tail batch size must be between 1 and 7");
}

int64_t dot_product_packed_4bit_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t extra_offset) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_packed_4bit_batch_dispatch(
            query, codes, 1, d, extra_offset, &dot);
    return dot;
}

void dot_product_packed_4bit_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        int64_t* dots) {
    dot_product_packed_4bit_batch_dispatch(
            query, codes, count, d, extra_offset, dots);
}

void dot_product_split_4bit_batch_arm(
        const int8_t* query,
        const uint8_t* const* signs,
        const uint8_t* const* tails,
        int count,
        size_t d,
        int64_t* dots) {
    dot_product_split_4bit_batch_dispatch(query, signs, tails, count, d, dots);
}

int64_t dot_product_progressive_prefix_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_progressive6_batch_fixed_arm<1, true>(query, codes, d, 0, &dot);
    return dot;
}

int64_t dot_product_progressive_full6_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_progressive6_batch_fixed_arm<1, false>(
            query, codes, d, tail_offset, &dot);
    return dot;
}

int64_t dot_product_progressive_full7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_progressive7_batch_fixed_arm<1>(
            query, codes, d, tail_offset, &dot);
    return dot;
}

void dot_product_progressive_prefix_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        int64_t* dots) {
    dot_product_progressive6_batch_dispatch<true>(
            query, codes, count, d, 0, dots);
}

void dot_product_progressive_full6_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
    dot_product_progressive6_batch_dispatch<false>(
            query, codes, count, d, tail_offset, dots);
}

void dot_product_progressive_full7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
    dot_product_progressive7_batch_dispatch(
            query, codes, count, d, tail_offset, dots);
}

int64_t dot_product_nested_lut7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_nested_lut7_batch_fixed_arm<1>(
            query, codes, d, low4_offset, high1_offset, lut, &dot);
    return dot;
}

void dot_product_nested_lut7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
    dot_product_nested_lut7_batch_dispatch(
            query, codes, count, d, low4_offset, high1_offset, lut, dots);
}

int64_t dot_product_nested_lut4_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_nested_lut4_batch_fixed_arm<1>(
            query, codes, d, local_offset, lut, nibble_layout, &dot);
    return dot;
}

void dot_product_nested_lut4_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t* dots) {
    dot_product_nested_lut4_batch_dispatch(
            query, codes, count, d, local_offset, lut, nibble_layout, dots);
}

int64_t dot_product_nested_exact7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t low4_offset,
        size_t high1_offset) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_nested_exact7_batch_fixed_arm<1>(
            query, codes, d, low4_offset, high1_offset, &dot);
    return dot;
}

void dot_product_nested_exact7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        int64_t* dots) {
    dot_product_nested_exact7_batch_dispatch(
            query, codes, count, d, low4_offset, high1_offset, dots);
}

int64_t dot_product_nested_lut3_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t high1_offset,
        const uint8_t* lut) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_nested_lut3_batch_fixed_arm<1>(
            query, codes, d, high1_offset, lut, &dot);
    return dot;
}

void dot_product_nested_lut3_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
    dot_product_nested_lut3_batch_dispatch(
            query, codes, count, d, high1_offset, lut, dots);
}

void dot_product_progressive_refine6_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset,
        int64_t& prefix_dot,
        int64_t& tail_dot) {
    const uint8_t* codes[1] = {code};
    dot_product_progressive_refine6_batch_fixed_arm<1>(
            query, codes, d, tail_offset, &prefix_dot, &tail_dot);
}

void dot_product_progressive_refine6_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* prefix_dots,
        int64_t* tail_dots) {
    switch (count) {
        case 1:
            return dot_product_progressive_refine6_batch_chunk<1>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 2:
            return dot_product_progressive_refine6_batch_chunk<2>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 3:
            return dot_product_progressive_refine6_batch_chunk<3>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 4:
            return dot_product_progressive_refine6_batch_chunk<4>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 5:
            return dot_product_progressive_refine6_batch_chunk<5>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 6:
            return dot_product_progressive_refine6_batch_chunk<6>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 7:
            return dot_product_progressive_refine6_batch_chunk<7>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 8:
            return dot_product_progressive_refine6_batch_chunk<8>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
        case 16:
            dot_product_progressive_refine6_batch_chunk<8>(
                    query, codes, d, tail_offset, prefix_dots, tail_dots);
            return dot_product_progressive_refine6_batch_chunk<8>(
                    query,
                    codes + 8,
                    d,
                    tail_offset,
                    prefix_dots + 8,
                    tail_dots + 8);
    }
    FAISS_THROW_MSG("progressive refinement batch size must be 1..8 or 16");
}

} // namespace faiss::rabitq_integer_adc
