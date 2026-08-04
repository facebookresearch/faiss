/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/EDENQuantizer.h>
#include <faiss/impl/simdlib/simdlib_avx2.h>
#include <faiss/impl/simdlib/simdlib_avx512.h>

#include <immintrin.h>

namespace faiss {

namespace eden_distance {

namespace {

constexpr int kCodeDotLUTKindHalfByte = 2;

inline __m512i load_byte_indexes_16(const uint8_t* const code[16], size_t i) {
    return _mm512_set_epi32(
            code[15][i],
            code[14][i],
            code[13][i],
            code[12][i],
            code[11][i],
            code[10][i],
            code[9][i],
            code[8][i],
            code[7][i],
            code[6][i],
            code[5][i],
            code[4][i],
            code[3][i],
            code[2][i],
            code[1][i],
            code[0][i]);
}

inline __m512i load_low_half_byte_indexes_16(
        const uint8_t* const code[16],
        size_t i) {
    return _mm512_set_epi32(
            code[15][i] & 0x0f,
            code[14][i] & 0x0f,
            code[13][i] & 0x0f,
            code[12][i] & 0x0f,
            code[11][i] & 0x0f,
            code[10][i] & 0x0f,
            code[9][i] & 0x0f,
            code[8][i] & 0x0f,
            code[7][i] & 0x0f,
            code[6][i] & 0x0f,
            code[5][i] & 0x0f,
            code[4][i] & 0x0f,
            code[3][i] & 0x0f,
            code[2][i] & 0x0f,
            code[1][i] & 0x0f,
            code[0][i] & 0x0f);
}

inline __m512i load_high_half_byte_indexes_16(
        const uint8_t* const code[16],
        size_t i) {
    return _mm512_set_epi32(
            16 + (code[15][i] >> 4),
            16 + (code[14][i] >> 4),
            16 + (code[13][i] >> 4),
            16 + (code[12][i] >> 4),
            16 + (code[11][i] >> 4),
            16 + (code[10][i] >> 4),
            16 + (code[9][i] >> 4),
            16 + (code[8][i] >> 4),
            16 + (code[7][i] >> 4),
            16 + (code[6][i] >> 4),
            16 + (code[5][i] >> 4),
            16 + (code[4][i] >> 4),
            16 + (code[3][i] >> 4),
            16 + (code[2][i] >> 4),
            16 + (code[1][i] >> 4),
            16 + (code[0][i] >> 4));
}

inline __m512i load_byte_indexes_8(const uint8_t* const code[8], size_t i) {
    const __m256i indexes = _mm256_setr_epi32(
            code[0][i],
            code[1][i],
            code[2][i],
            code[3][i],
            code[4][i],
            code[5][i],
            code[6][i],
            code[7][i]);
    return _mm512_castsi256_si512(indexes);
}

inline __m512 gather_lower_8_lanes(const float* table, __m512i indexes) {
    constexpr __mmask16 active_lanes = 0x00ff;
    return _mm512_mask_i32gather_ps(
            _mm512_setzero_ps(), active_lanes, indexes, table, 4);
}

inline __m512i load_half_byte_indexes_8(
        const uint8_t* const code[8],
        size_t i) {
    return _mm512_set_epi32(
            16 + (code[7][i] >> 4),
            16 + (code[6][i] >> 4),
            16 + (code[5][i] >> 4),
            16 + (code[4][i] >> 4),
            16 + (code[3][i] >> 4),
            16 + (code[2][i] >> 4),
            16 + (code[1][i] >> 4),
            16 + (code[0][i] >> 4),
            code[7][i] & 0x0f,
            code[6][i] & 0x0f,
            code[5][i] & 0x0f,
            code[4][i] & 0x0f,
            code[3][i] & 0x0f,
            code[2][i] & 0x0f,
            code[1][i] & 0x0f,
            code[0][i] & 0x0f);
}

inline __m512i load_byte_pair_indexes_8(
        const uint8_t* const code[8],
        size_t i) {
    return _mm512_set_epi32(
            256 + code[7][i + 1],
            256 + code[6][i + 1],
            256 + code[5][i + 1],
            256 + code[4][i + 1],
            256 + code[3][i + 1],
            256 + code[2][i + 1],
            256 + code[1][i + 1],
            256 + code[0][i + 1],
            code[7][i],
            code[6][i],
            code[5][i],
            code[4][i],
            code[3][i],
            code[2][i],
            code[1][i],
            code[0][i]);
}

} // namespace

void compute_code_dot_lut_batch_8_avx512(
        const uint8_t* const code[8],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[8]) {
    __m512 acc = _mm512_setzero_ps();

    if (lut_kind == kCodeDotLUTKindHalfByte) {
        for (size_t i = 0; i < packed_size; i++) {
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(
                            load_half_byte_indexes_8(code, i), lut, 4));
            lut += 32;
        }
    } else {
        size_t i = 0;
        for (; i + 1 < packed_size; i += 2) {
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(
                            load_byte_pair_indexes_8(code, i), lut, 4));
            lut += 512;
        }
        if (i < packed_size) {
            acc = _mm512_add_ps(
                    acc,
                    gather_lower_8_lanes(lut, load_byte_indexes_8(code, i)));
        }
    }

    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, acc);
    for (size_t i = 0; i < 8; i++) {
        dots[i] = lanes[i] + lanes[i + 8];
    }
}

void compute_code_dot_lut_batch_16_avx512(
        const uint8_t* const code[16],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[16]) {
    __m512 acc = _mm512_setzero_ps();

    if (lut_kind == kCodeDotLUTKindHalfByte) {
        for (size_t i = 0; i < packed_size; i++) {
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(
                            load_low_half_byte_indexes_16(code, i), lut, 4));
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(
                            load_high_half_byte_indexes_16(code, i), lut, 4));
            lut += 32;
        }
    } else {
        for (size_t i = 0; i < packed_size; i++) {
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(load_byte_indexes_16(code, i), lut, 4));
            lut += 256;
        }
    }

    _mm512_storeu_ps(dots, acc);
}

} // namespace eden_distance

} // namespace faiss

#endif
