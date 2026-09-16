/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/fp16_linear_transform.h>

#include <immintrin.h>
#include <vector>

#include <faiss/utils/fp16.h>

namespace faiss::fp16_linear_transform {

template <>
bool supported<SIMDLevel::AVX512_SPR>() {
    return true;
}

template <>
bool apply<SIMDLevel::AVX512_SPR>(
        const uint16_t* matrix,
        size_t rows,
        size_t columns,
        const float* input,
        float* output) {
    std::vector<uint16_t> input_fp16(columns);
    const __m512 max_fp16 = _mm512_set1_ps(65504.0f);
    const __m512i absolute_value_mask = _mm512_set1_epi32(0x7fffffff);
    size_t column = 0;
    for (; column + 16 <= columns; column += 16) {
        const __m512 values = _mm512_loadu_ps(input + column);
        const __m512 absolute_values = _mm512_castsi512_ps(_mm512_and_si512(
                _mm512_castps_si512(values), absolute_value_mask));
        if (_mm512_cmp_ps_mask(absolute_values, max_fp16, _CMP_LE_OQ) !=
            0xffff) {
            return false;
        }
        const __m256i converted =
                _mm512_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(input_fp16.data() + column),
                converted);
    }
    for (; column < columns; ++column) {
        if (!(input[column] >= -65504.0f && input[column] <= 65504.0f)) {
            return false;
        }
        input_fp16[column] = encode_fp16(input[column]);
    }

    for (size_t row = 0; row < rows; ++row) {
        const uint16_t* weights = matrix + row * columns;
        __m512 accumulator = _mm512_setzero_ps();
        column = 0;
        for (; column + 16 <= columns; column += 16) {
            const __m256i weights_fp16 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weights + column));
            const __m256i input_values_fp16 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                            input_fp16.data() + column));
            accumulator = _mm512_fmadd_ps(
                    _mm512_cvtph_ps(weights_fp16),
                    _mm512_cvtph_ps(input_values_fp16),
                    accumulator);
        }
        float sum = _mm512_reduce_add_ps(accumulator);
        for (; column < columns; ++column) {
            sum += decode_fp16(weights[column]) *
                    decode_fp16(input_fp16[column]);
        }
        output[row] = sum;
    }
    return true;
}

} // namespace faiss::fp16_linear_transform
