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

bool supported() {
    return true;
}

void apply(
        const uint16_t* matrix,
        size_t rows,
        size_t columns,
        const float* input,
        float* output) {
    std::vector<uint16_t> input_fp16(columns);
    size_t column = 0;
    for (; column + 16 <= columns; column += 16) {
        const __m512 values = _mm512_loadu_ps(input + column);
        const __m256i converted =
                _mm512_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(input_fp16.data() + column),
                converted);
    }
    for (; column < columns; ++column) {
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
}

} // namespace faiss::fp16_linear_transform
