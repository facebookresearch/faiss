/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>

#include <faiss/impl/FaissAssert.h>

#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

namespace faiss::rabitq_integer_adc {

namespace {

// Keep each int32 lane far from overflow before reducing into int64. A lane
// receives one sum of four byte products per 64 input dimensions.
constexpr size_t kDotChunk = 4096;

inline int64_t reduce(__m512i accumulator) {
    return _mm512_reduce_add_epi32(accumulator);
}

inline int64_t biased_tail_product(int8_t query, int8_t level) {
    // VPDPBUSD multiplies unsigned bytes by signed bytes. Flipping the sign
    // bit maps a signed level l to the unsigned value l + 128. The caller's
    // query_correction subtracts 128 * sum(query) once from the full dot.
    const uint8_t biased_level = uint8_t(level) ^ uint8_t(0x80);
    return int64_t(query) * int64_t(biased_level);
}

template <typename Function, size_t... Indices>
FAISS_ALWAYS_INLINE void for_fixed_lanes_impl(
        Function&& function,
        std::index_sequence<Indices...>) {
    (function(std::integral_constant<size_t, Indices>{}), ...);
}

template <int N, typename Function>
FAISS_ALWAYS_INLINE void for_fixed_lanes(Function&& function) {
    for_fixed_lanes_impl(
            std::forward<Function>(function), std::make_index_sequence<N>{});
}

/** Expand 64 consecutive packed two-bit values to 64 bytes. VBMI
 * VPMULTISHIFTQB extracts eight fields from each duplicated 64-bit source
 * lane; masking keeps the low two bits of each extracted byte. */
FAISS_ALWAYS_INLINE __m512i unpack_2bit_64(const uint8_t* packed) {
    const __m512i input = _mm512_castsi128_si512(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed)));
    const __m512i duplicated = _mm512_permutexvar_epi64(
            _mm512_set_epi64(1, 1, 1, 1, 0, 0, 0, 0), input);
    const __m512i shifts = _mm512_set_epi64(
            0x3e3c3a3836343230ULL,
            0x2e2c2a2826242220ULL,
            0x1e1c1a1816141210ULL,
            0x0e0c0a0806040200ULL,
            0x3e3c3a3836343230ULL,
            0x2e2c2a2826242220ULL,
            0x1e1c1a1816141210ULL,
            0x0e0c0a0806040200ULL);
    return _mm512_and_si512(
            _mm512_multishift_epi64_epi8(shifts, duplicated),
            _mm512_set1_epi8(3));
}

/** Expand one packed bit plane to 64 bytes containing zero or one. */
FAISS_ALWAYS_INLINE __m512i unpack_1bit_64(const uint8_t* packed) {
    uint64_t input;
    memcpy(&input, packed, sizeof(input));
    return _mm512_maskz_set1_epi8(static_cast<__mmask64>(input), 1);
}

FAISS_ALWAYS_INLINE __m512i unpack_nibbles_64(const uint8_t* packed) {
    const __m512i input = _mm512_castsi256_si512(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed)));
    const __m512i duplicated = _mm512_permutexvar_epi64(
            _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0), input);
    const __m512i shifts = _mm512_set_epi64(
            0x3c3834302c282420ULL,
            0x1c1814100c080400ULL,
            0x3c3834302c282420ULL,
            0x1c1814100c080400ULL,
            0x3c3834302c282420ULL,
            0x1c1814100c080400ULL,
            0x3c3834302c282420ULL,
            0x1c1814100c080400ULL);
    return _mm512_and_si512(
            _mm512_multishift_epi64_epi8(shifts, duplicated),
            _mm512_set1_epi8(15));
}

FAISS_ALWAYS_INLINE __m512i make_nested_lut4_table(const uint8_t* lut) {
    alignas(64) uint8_t table[64];
    for (size_t lane = 0; lane < 4; ++lane) {
        for (size_t symbol = 0; symbol < 16; ++symbol) {
            const uint8_t magnitude = lut[symbol & 7];
            table[16 * lane + symbol] =
                    symbol & 8 ? magnitude : uint8_t(~magnitude);
        }
    }
    return _mm512_load_si512(table);
}

FAISS_ALWAYS_INLINE __m512i decode_nested_lut4_block(
        const uint8_t* code,
        size_t j,
        size_t local_offset,
        bool nibble_layout,
        __m512i table) {
    __m512i symbols;
    if (nibble_layout) {
        symbols = unpack_nibbles_64(code + j / 2);
    } else {
        const __m512i prefix = unpack_2bit_64(code + j / 4);
        const __m512i local = unpack_2bit_64(code + local_offset + j / 4);
        const __m512i positive = _mm512_and_si512(
                _mm512_srli_epi16(prefix, 1), _mm512_set1_epi8(1));
        const __m512i coarse = _mm512_and_si512(
                _mm512_xor_si512(
                        _mm512_xor_si512(prefix, positive),
                        _mm512_set1_epi8(1)),
                _mm512_set1_epi8(1));
        symbols = _mm512_or_si512(
                local,
                _mm512_or_si512(
                        _mm512_slli_epi16(coarse, 2),
                        _mm512_slli_epi16(positive, 3)));
    }
    return _mm512_shuffle_epi8(table, symbols);
}

FAISS_ALWAYS_INLINE int8_t decode_nested_lut4_tail(
        const uint8_t* code,
        size_t j,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout) {
    if (nibble_layout) {
        const uint8_t symbol = (code[j / 2] >> (4 * (j % 2))) & 15;
        const int magnitude = lut[symbol & 7];
        return static_cast<int8_t>(symbol & 8 ? magnitude : -1 - magnitude);
    }
    const uint8_t prefix = (code[j / 4] >> (2 * (j % 4))) & 3;
    const int positive = prefix >> 1;
    const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
    const int local = (code[local_offset + j / 4] >> (2 * (j % 4))) & 3;
    const int magnitude = lut[coarse * 4 + local];
    return static_cast<int8_t>(positive ? magnitude : -1 - magnitude);
}

struct NestedLut7Tables {
    __m512i negative;
    __m512i positive;
};

FAISS_ALWAYS_INLINE NestedLut7Tables
make_nested_lut7_tables(const uint8_t* lut) {
    const __m512i positive = _mm512_loadu_si512(lut);
    // The positive table is [cell0, cell1]. Prefix values 2 and 3 select
    // those halves directly. Prefix values 0 and 1 need the reversed cells
    // with one's-complement sign encoding: [~cell1, ~cell0].
    const __m512i reversed = _mm512_shuffle_i64x2(positive, positive, 0x4e);
    return {_mm512_xor_si512(reversed, _mm512_set1_epi8(-1)), positive};
}

FAISS_ALWAYS_INLINE __m512i decode_nested_lut7_block(
        const uint8_t* code,
        size_t j,
        size_t low4_offset,
        size_t high1_offset,
        const NestedLut7Tables& tables) {
    const __m512i prefix = unpack_2bit_64(code + j / 4);
    const __m512i local_low = unpack_nibbles_64(code + low4_offset + j / 2);
    const __m512i local_high = unpack_1bit_64(code + high1_offset + j / 8);
    const __m512i indices = _mm512_or_si512(
            _mm512_or_si512(local_low, _mm512_slli_epi16(local_high, 4)),
            _mm512_slli_epi16(prefix, 5));
    return _mm512_permutex2var_epi8(tables.negative, indices, tables.positive);
}

FAISS_ALWAYS_INLINE __m512i
decode_progressive_prefix_block(const uint8_t* code, size_t j) {
    return _mm512_sub_epi8(unpack_2bit_64(code + j / 4), _mm512_set1_epi8(2));
}

FAISS_ALWAYS_INLINE int8_t
decode_progressive_prefix_tail(const uint8_t* code, size_t j) {
    return static_cast<int8_t>(((code[j / 4] >> (2 * (j % 4))) & 3) - 2);
}

FAISS_ALWAYS_INLINE int8_t decode_nested_lut7_tail(
        const uint8_t* code,
        size_t j,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut) {
    const uint8_t prefix = (code[j / 4] >> (2 * (j % 4))) & 3;
    const int positive = prefix >> 1;
    const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
    const int local_low = (code[low4_offset + j / 2] >> (4 * (j % 2))) & 15;
    const int local_high = (code[high1_offset + j / 8] >> (j % 8)) & 1;
    const int magnitude = lut[coarse * 32 + local_low + 16 * local_high];
    return static_cast<int8_t>(positive ? magnitude : -1 - magnitude);
}

template <int N, typename DecodeBlock, typename DecodeTail>
void dot_product_compact_batch_fixed_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        int64_t query_correction,
        int64_t* dots,
        DecodeBlock& decode_block,
        DecodeTail& decode_tail) {
    const __m512i sign_bit = _mm512_set1_epi32(0x80808080);
    for_fixed_lanes<N>([&](auto lane) { dots[lane] = query_correction; });

    size_t j = 0;
    while (j + 64 <= d) {
        const size_t end = std::min(d - (d - j) % 64, j + kDotChunk);
        __m512i accumulators[N];
        for_fixed_lanes<N>([&](auto lane) {
            accumulators[lane] = _mm512_setzero_si512();
        });
        for (; j + 64 <= end; j += 64) {
            const __m512i q = _mm512_loadu_si512(query + j);
            for_fixed_lanes<N>([&](auto lane) {
                const __m512i levels = decode_block(codes[lane], j);
                const __m512i biased_levels =
                        _mm512_xor_si512(levels, sign_bit);
                accumulators[lane] = _mm512_dpbusd_epi32(
                        accumulators[lane], biased_levels, q);
            });
        }
        for_fixed_lanes<N>(
                [&](auto lane) { dots[lane] += reduce(accumulators[lane]); });
    }
    for (; j < d; ++j) {
        for_fixed_lanes<N>([&](auto lane) {
            dots[lane] +=
                    biased_tail_product(query[j], decode_tail(codes[lane], j));
        });
    }
}

template <typename DecodeBlock, typename DecodeTail>
void dot_product_compact_batch_dispatch_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        int64_t query_correction,
        int64_t* dots,
        DecodeBlock&& decode_block,
        DecodeTail&& decode_tail) {
    FAISS_THROW_IF_NOT_MSG(
            count >= 1 && (count <= 8 || count == 16),
            "compact AVX-512 batch size must be 1..8 or 16");
    if (count == 16) {
        return dot_product_compact_batch_fixed_avx512_vnni<16>(
                query,
                codes,
                d,
                query_correction,
                dots,
                decode_block,
                decode_tail);
    }
    if (count == 8) {
        return dot_product_compact_batch_fixed_avx512_vnni<8>(
                query,
                codes,
                d,
                query_correction,
                dots,
                decode_block,
                decode_tail);
    }
    int offset = 0;
    if (count >= 4) {
        dot_product_compact_batch_fixed_avx512_vnni<4>(
                query,
                codes,
                d,
                query_correction,
                dots,
                decode_block,
                decode_tail);
        offset = 4;
    }
    if (count - offset >= 2) {
        dot_product_compact_batch_fixed_avx512_vnni<2>(
                query,
                codes + offset,
                d,
                query_correction,
                dots + offset,
                decode_block,
                decode_tail);
        offset += 2;
    }
    if (count != offset) {
        dot_product_compact_batch_fixed_avx512_vnni<1>(
                query,
                codes + offset,
                d,
                query_correction,
                dots + offset,
                decode_block,
                decode_tail);
    }
}

} // namespace

int64_t dot_product_avx512_vnni(
        const int8_t* query,
        const int8_t* levels,
        size_t d,
        int64_t query_correction) {
    const __m512i sign_bit = _mm512_set1_epi32(0x80808080);
    int64_t result = query_correction;
    size_t j = 0;
    while (j + 64 <= d) {
        const size_t end = std::min(d - (d - j) % 64, j + kDotChunk);
        __m512i accumulator = _mm512_setzero_si512();
        for (; j + 64 <= end; j += 64) {
            const __m512i q = _mm512_loadu_si512(query + j);
            const __m512i biased_levels =
                    _mm512_xor_si512(_mm512_loadu_si512(levels + j), sign_bit);
            accumulator = _mm512_dpbusd_epi32(accumulator, biased_levels, q);
        }
        result += reduce(accumulator);
    }
    for (; j < d; j++) {
        result += biased_tail_product(query[j], levels[j]);
    }
    return result;
}

void dot_product_batch_4_avx512_vnni(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t query_correction,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3) {
    const __m512i sign_bit = _mm512_set1_epi32(0x80808080);
    dot0 = dot1 = dot2 = dot3 = query_correction;
    size_t j = 0;
    while (j + 64 <= d) {
        const size_t end = std::min(d - (d - j) % 64, j + kDotChunk);
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        for (; j + 64 <= end; j += 64) {
            const __m512i q = _mm512_loadu_si512(query + j);
            const __m512i level0 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels0 + j), sign_bit);
            const __m512i level1 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels1 + j), sign_bit);
            const __m512i level2 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels2 + j), sign_bit);
            const __m512i level3 =
                    _mm512_xor_si512(_mm512_loadu_si512(levels3 + j), sign_bit);
            acc0 = _mm512_dpbusd_epi32(acc0, level0, q);
            acc1 = _mm512_dpbusd_epi32(acc1, level1, q);
            acc2 = _mm512_dpbusd_epi32(acc2, level2, q);
            acc3 = _mm512_dpbusd_epi32(acc3, level3, q);
        }
        dot0 += reduce(acc0);
        dot1 += reduce(acc1);
        dot2 += reduce(acc2);
        dot3 += reduce(acc3);
    }
    for (; j < d; j++) {
        dot0 += biased_tail_product(query[j], levels0[j]);
        dot1 += biased_tail_product(query[j], levels1[j]);
        dot2 += biased_tail_product(query[j], levels2[j]);
        dot3 += biased_tail_product(query[j], levels3[j]);
    }
}

void dot_product_batch_8_avx512_vnni(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t query_correction,
        int64_t dots[8]) {
    const __m512i sign_bit = _mm512_set1_epi32(0x80808080);
    for (size_t k = 0; k < 8; ++k) {
        dots[k] = query_correction;
    }
    size_t j = 0;
    while (j + 64 <= d) {
        const size_t end = std::min(d - (d - j) % 64, j + kDotChunk);
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        __m512i acc4 = _mm512_setzero_si512();
        __m512i acc5 = _mm512_setzero_si512();
        __m512i acc6 = _mm512_setzero_si512();
        __m512i acc7 = _mm512_setzero_si512();
        for (; j + 64 <= end; j += 64) {
            const __m512i q = _mm512_loadu_si512(query + j);
            acc0 = _mm512_dpbusd_epi32(
                    acc0,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[0] + j), sign_bit),
                    q);
            acc1 = _mm512_dpbusd_epi32(
                    acc1,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[1] + j), sign_bit),
                    q);
            acc2 = _mm512_dpbusd_epi32(
                    acc2,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[2] + j), sign_bit),
                    q);
            acc3 = _mm512_dpbusd_epi32(
                    acc3,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[3] + j), sign_bit),
                    q);
            acc4 = _mm512_dpbusd_epi32(
                    acc4,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[4] + j), sign_bit),
                    q);
            acc5 = _mm512_dpbusd_epi32(
                    acc5,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[5] + j), sign_bit),
                    q);
            acc6 = _mm512_dpbusd_epi32(
                    acc6,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[6] + j), sign_bit),
                    q);
            acc7 = _mm512_dpbusd_epi32(
                    acc7,
                    _mm512_xor_si512(
                            _mm512_loadu_si512(levels[7] + j), sign_bit),
                    q);
        }
        dots[0] += reduce(acc0);
        dots[1] += reduce(acc1);
        dots[2] += reduce(acc2);
        dots[3] += reduce(acc3);
        dots[4] += reduce(acc4);
        dots[5] += reduce(acc5);
        dots[6] += reduce(acc6);
        dots[7] += reduce(acc7);
    }
    for (; j < d; ++j) {
        for (size_t k = 0; k < 8; ++k) {
            dots[k] += biased_tail_product(query[j], levels[k][j]);
        }
    }
}

int64_t dot_product_nested_lut4_avx512_vnni(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t query_correction) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_nested_lut4_batch_avx512_vnni(
            query,
            codes,
            1,
            d,
            local_offset,
            lut,
            nibble_layout,
            query_correction,
            &dot);
    return dot;
}

void dot_product_nested_lut4_batch_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t query_correction,
        int64_t* dots) {
    const __m512i table = make_nested_lut4_table(lut);
    dot_product_compact_batch_dispatch_avx512_vnni(
            query,
            codes,
            count,
            d,
            query_correction,
            dots,
            [=](const uint8_t* row, size_t j) {
                return decode_nested_lut4_block(
                        row, j, local_offset, nibble_layout, table);
            },
            [=](const uint8_t* row, size_t j) {
                return decode_nested_lut4_tail(
                        row, j, local_offset, lut, nibble_layout);
            });
}

int64_t dot_product_nested_lut7_avx512_vnni(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t query_correction) {
    const uint8_t* codes[1] = {code};
    int64_t dot;
    dot_product_nested_lut7_batch_avx512_vnni(
            query,
            codes,
            1,
            d,
            low4_offset,
            high1_offset,
            lut,
            query_correction,
            &dot);
    return dot;
}

void dot_product_nested_lut7_batch_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t query_correction,
        int64_t* dots) {
    const NestedLut7Tables tables = make_nested_lut7_tables(lut);
    dot_product_compact_batch_dispatch_avx512_vnni(
            query,
            codes,
            count,
            d,
            query_correction,
            dots,
            [=](const uint8_t* row, size_t j) {
                return decode_nested_lut7_block(
                        row, j, low4_offset, high1_offset, tables);
            },
            [=](const uint8_t* row, size_t j) {
                return decode_nested_lut7_tail(
                        row, j, low4_offset, high1_offset, lut);
            });
}

void dot_product_progressive_prefix_batch_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        int64_t query_correction,
        int64_t* dots) {
    dot_product_compact_batch_dispatch_avx512_vnni(
            query,
            codes,
            count,
            d,
            query_correction,
            dots,
            [](const uint8_t* row, size_t j) {
                return decode_progressive_prefix_block(row, j);
            },
            [](const uint8_t* row, size_t j) {
                return decode_progressive_prefix_tail(row, j);
            });
}

} // namespace faiss::rabitq_integer_adc
