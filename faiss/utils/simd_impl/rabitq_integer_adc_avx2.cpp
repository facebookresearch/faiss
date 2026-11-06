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

namespace faiss::rabitq_integer_adc {
namespace {

constexpr size_t kDotChunk = 4096;

// Widen signed bytes before multiplying. MADDUBSW would saturate for some
// signed-byte inputs, including the -128 level used by eight-bit RaBitQ.
FAISS_ALWAYS_INLINE __m256i
multiply_add(__m256i accumulator, __m256i query, __m256i levels) {
    const __m256i low = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(query)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(levels)));
    const __m256i high = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(query, 1)),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(levels, 1)));
    return _mm256_add_epi32(accumulator, _mm256_add_epi32(low, high));
}

FAISS_ALWAYS_INLINE int64_t reduce(__m256i accumulator) {
    __m128i sum = _mm_add_epi32(
            _mm256_castsi256_si128(accumulator),
            _mm256_extracti128_si256(accumulator, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

// Each source byte becomes four bytes. OR, rather than multiplication,
// prevents carries between the overlapping copies before masking.
FAISS_ALWAYS_INLINE __m256i unpack_2bit_32(const uint8_t* input) {
    const __m256i words = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input)));
    return _mm256_and_si256(
            _mm256_or_si256(
                    _mm256_or_si256(words, _mm256_slli_epi32(words, 6)),
                    _mm256_or_si256(
                            _mm256_slli_epi32(words, 12),
                            _mm256_slli_epi32(words, 18))),
            _mm256_set1_epi8(3));
}

FAISS_ALWAYS_INLINE __m256i unpack_nibbles_32(const uint8_t* input) {
    const __m256i words = _mm256_cvtepu8_epi16(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
    return _mm256_and_si256(
            _mm256_or_si256(words, _mm256_slli_epi16(words, 4)),
            _mm256_set1_epi8(15));
}

FAISS_ALWAYS_INLINE __m256i unpack_1bit_32(const uint8_t* input) {
    uint32_t word;
    memcpy(&word, input, sizeof(word));
    const __m256i bytes = _mm256_shuffle_epi8(
            _mm256_set1_epi32(word),
            _mm256_setr_epi8(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3));
    const __m256i masks = _mm256_set1_epi64x(0x8040201008040201ULL);
    return _mm256_and_si256(
            _mm256_cmpeq_epi8(_mm256_and_si256(bytes, masks), masks),
            _mm256_set1_epi8(1));
}

inline uint8_t extract_bits(const uint8_t* input, size_t j, size_t bits) {
    const size_t position = j * bits;
    unsigned word = input[position / 8];
    if (position % 8 + bits > 8) {
        word |= unsigned(input[position / 8 + 1]) << 8;
    }
    return (word >> (position % 8)) & ((1u << bits) - 1);
}

// General packed levels use vector shifts over eight small windows at a
// time. Loads stop at the actual bit field, including on the final block.
FAISS_ALWAYS_INLINE __m256i
unpack_bits8(const uint8_t* input, size_t j, size_t bits) {
    alignas(32) int words[8];
    alignas(32) int shifts[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        const size_t position = (j + lane) * bits;
        words[lane] = input[position / 8];
        shifts[lane] = position % 8;
        if (position % 8 + bits > 8) {
            words[lane] |= unsigned(input[position / 8 + 1]) << 8;
        }
    }
    return _mm256_and_si256(
            _mm256_srlv_epi32(
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(words)),
                    _mm256_load_si256(
                            reinterpret_cast<const __m256i*>(shifts))),
            _mm256_set1_epi32((1u << bits) - 1));
}

FAISS_ALWAYS_INLINE __m256i
unpack_bits32(const uint8_t* input, size_t j, size_t bits) {
    if (bits == 1) {
        return unpack_1bit_32(input + j / 8);
    }
    if (bits == 2) {
        return unpack_2bit_32(input + j / 4);
    }
    if (bits == 4) {
        return unpack_nibbles_32(input + j / 2);
    }
    const __m256i first = _mm256_packs_epi32(
            unpack_bits8(input, j, bits), unpack_bits8(input, j + 8, bits));
    const __m256i second = _mm256_packs_epi32(
            unpack_bits8(input, j + 16, bits),
            unpack_bits8(input, j + 24, bits));
    return _mm256_permutevar8x32_epi32(
            _mm256_packs_epi16(first, second),
            _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

template <int N, typename DecodeBlock, typename DecodeTail>
void dot_batch_fixed(
        const int8_t* query,
        size_t d,
        int64_t* dots,
        DecodeBlock& block,
        DecodeTail& tail) {
    std::fill_n(dots, N, int64_t(0));
    size_t j = 0;
    while (j + 32 <= d) {
        const size_t end = std::min(d - (d - j) % 32, j + kDotChunk);
        __m256i accumulators[N];
        for (int lane = 0; lane < N; ++lane) {
            accumulators[lane] = _mm256_setzero_si256();
        }
        for (; j + 32 <= end; j += 32) {
            const __m256i q = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(query + j));
            for (int lane = 0; lane < N; ++lane) {
                accumulators[lane] =
                        multiply_add(accumulators[lane], q, block(lane, j));
            }
        }
        for (int lane = 0; lane < N; ++lane) {
            dots[lane] += reduce(accumulators[lane]);
        }
    }
    for (; j < d; ++j) {
        for (int lane = 0; lane < N; ++lane) {
            dots[lane] += int64_t(query[j]) * tail(lane, j);
        }
    }
}

template <typename DecodeBlock, typename DecodeTail>
void dot_batch(
        const int8_t* query,
        int count,
        size_t d,
        int64_t* dots,
        DecodeBlock block,
        DecodeTail tail) {
    FAISS_THROW_IF_NOT(count >= 1 && count <= 16);
    // Four accumulators leave enough YMM registers for unpacking and lookup.
    for (int first = 0; first < count; first += 4) {
        auto b = [&](int lane, size_t j) { return block(first + lane, j); };
        auto t = [&](int lane, size_t j) { return tail(first + lane, j); };
        switch (std::min(4, count - first)) {
            case 1:
                dot_batch_fixed<1>(query, d, dots + first, b, t);
                break;
            case 2:
                dot_batch_fixed<2>(query, d, dots + first, b, t);
                break;
            case 3:
                dot_batch_fixed<3>(query, d, dots + first, b, t);
                break;
            case 4:
                dot_batch_fixed<4>(query, d, dots + first, b, t);
        }
    }
}

} // namespace

void dot_product_batch_avx2(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots) {
    dot_batch(
            query,
            count,
            d,
            dots,
            [&](int row, size_t j) {
                return _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(levels[row] + j));
            },
            [&](int row, size_t j) { return levels[row][j]; });
}

void dot_product_packed_batch_avx2(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        size_t ex_bits,
        int64_t* dots) {
    FAISS_THROW_IF_NOT(ex_bits >= 1 && ex_bits <= 7);
    dot_batch(
            query,
            count,
            d,
            dots,
            [&](int row, size_t j) {
                const __m256i low =
                        unpack_bits32(codes[row] + extra_offset, j, ex_bits);
                const __m256i sign = unpack_1bit_32(codes[row] + j / 8);
                // A zero sign contributes -2^ex_bits; a set sign contributes
                // zero.
                return _mm256_sub_epi8(
                        low,
                        _mm256_andnot_si256(
                                _mm256_cmpeq_epi8(sign, _mm256_set1_epi8(1)),
                                _mm256_set1_epi8(uint8_t(1u << ex_bits))));
            },
            [&](int row, size_t j) {
                const int sign = (codes[row][j / 8] >> (j % 8)) & 1;
                return int(extract_bits(
                               codes[row] + extra_offset, j, ex_bits)) -
                        (sign ? 0 : int(1u << ex_bits));
            });
}

void dot_product_progressive_batch_avx2(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        size_t tail_bits,
        bool prefix_only,
        int64_t* dots) {
    FAISS_THROW_IF_NOT(prefix_only || (tail_bits >= 1 && tail_bits <= 7));
    dot_batch(
            query,
            count,
            d,
            dots,
            [&](int row, size_t j) {
                const __m256i prefix = unpack_2bit_32(codes[row] + j / 4);
                if (prefix_only) {
                    return _mm256_sub_epi8(prefix, _mm256_set1_epi8(2));
                }
                const __m256i low =
                        unpack_bits32(codes[row] + tail_offset, j, tail_bits);
                return _mm256_sub_epi8(
                        low,
                        _mm256_andnot_si256(
                                _mm256_cmpgt_epi8(prefix, _mm256_set1_epi8(1)),
                                _mm256_set1_epi8(uint8_t(1u << tail_bits))));
            },
            [&](int row, size_t j) {
                const int prefix = (codes[row][j / 4] >> (2 * (j % 4))) & 3;
                return prefix_only
                        ? prefix - 2
                        : int(extract_bits(
                                  codes[row] + tail_offset, j, tail_bits)) -
                                (prefix >= 2 ? 0 : int(1u << tail_bits));
            });
}

void dot_product_nested_lut4_batch_avx2(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t* dots) {
    alignas(16) uint8_t table[16];
    for (size_t symbol = 0; symbol < 16; ++symbol) {
        table[symbol] =
                symbol & 8 ? lut[symbol & 7] : uint8_t(~lut[symbol & 7]);
    }
    const __m256i lookup = _mm256_broadcastsi128_si256(
            _mm_load_si128(reinterpret_cast<const __m128i*>(table)));
    dot_batch(
            query,
            count,
            d,
            dots,
            [&](int row, size_t j) {
                __m256i symbols;
                if (nibble_layout) {
                    symbols = unpack_nibbles_32(codes[row] + j / 2);
                } else {
                    const __m256i prefix = unpack_2bit_32(codes[row] + j / 4);
                    const __m256i positive = _mm256_and_si256(
                            _mm256_srli_epi16(prefix, 1), _mm256_set1_epi8(1));
                    const __m256i coarse = _mm256_and_si256(
                            _mm256_xor_si256(
                                    _mm256_xor_si256(prefix, positive),
                                    _mm256_set1_epi8(1)),
                            _mm256_set1_epi8(1));
                    symbols = _mm256_or_si256(
                            unpack_2bit_32(codes[row] + local_offset + j / 4),
                            _mm256_or_si256(
                                    _mm256_slli_epi16(coarse, 2),
                                    _mm256_slli_epi16(positive, 3)));
                }
                return _mm256_shuffle_epi8(lookup, symbols);
            },
            [&](int row, size_t j) {
                if (nibble_layout) {
                    const int symbol =
                            (codes[row][j / 2] >> (4 * (j % 2))) & 15;
                    const int magnitude = lut[symbol & 7];
                    return symbol & 8 ? magnitude : -1 - magnitude;
                }
                const int prefix = (codes[row][j / 4] >> (2 * (j % 4))) & 3;
                const bool positive = prefix >= 2;
                const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
                const int local =
                        (codes[row][local_offset + j / 4] >> (2 * (j % 4))) & 3;
                const int magnitude = lut[4 * coarse + local];
                return positive ? magnitude : -1 - magnitude;
            });
}

void dot_product_nested_lut7_batch_avx2(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
    __m256i tables[4];
    for (size_t i = 0; i < 4; ++i) {
        tables[i] = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(lut + 16 * i)));
    }
    dot_batch(
            query,
            count,
            d,
            dots,
            [&](int row, size_t j) {
                const __m256i prefix = unpack_2bit_32(codes[row] + j / 4);
                const __m256i positive =
                        _mm256_cmpgt_epi8(prefix, _mm256_set1_epi8(1));
                const __m256i coarse = _mm256_and_si256(
                        _mm256_xor_si256(
                                prefix,
                                _mm256_andnot_si256(
                                        positive, _mm256_set1_epi8(1))),
                        _mm256_set1_epi8(1));
                const __m256i index = _mm256_or_si256(
                        unpack_nibbles_32(codes[row] + low4_offset + j / 2),
                        _mm256_slli_epi16(
                                unpack_1bit_32(
                                        codes[row] + high1_offset + j / 8),
                                4));
                const __m256i high =
                        _mm256_cmpgt_epi8(index, _mm256_set1_epi8(15));
                const __m256i first = _mm256_blendv_epi8(
                        _mm256_shuffle_epi8(tables[0], index),
                        _mm256_shuffle_epi8(tables[1], index),
                        high);
                const __m256i second = _mm256_blendv_epi8(
                        _mm256_shuffle_epi8(tables[2], index),
                        _mm256_shuffle_epi8(tables[3], index),
                        high);
                const __m256i magnitude = _mm256_blendv_epi8(
                        first,
                        second,
                        _mm256_cmpeq_epi8(coarse, _mm256_set1_epi8(1)));
                return _mm256_xor_si256(
                        magnitude,
                        _mm256_andnot_si256(positive, _mm256_set1_epi8(-1)));
            },
            [&](int row, size_t j) {
                const int prefix = (codes[row][j / 4] >> (2 * (j % 4))) & 3;
                const bool positive = prefix >= 2;
                const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
                const int local =
                        ((codes[row][low4_offset + j / 2] >> (4 * (j % 2))) &
                         15) |
                        (((codes[row][high1_offset + j / 8] >> (j % 8)) & 1)
                         << 4);
                const int magnitude = lut[coarse * 32 + local];
                return positive ? magnitude : -1 - magnitude;
            });
}

} // namespace faiss::rabitq_integer_adc
