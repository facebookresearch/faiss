/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Code-to-code inner product estimate for 8-bit RaBitQ codes.
 *
 * Used by RaBitQDistanceComputer::symmetric_dis() when the metric is
 * METRIC_INNER_PRODUCT and nb_bits == 8, which is what HNSW construction
 * calls to compare two stored vectors.
 *
 * An 8-bit code stores, per dimension, a sign bit b and 7 extra bits e. The
 * reconstructed coordinate is proportional to z = (256 b + 2 e - 255) / 2,
 * an odd half-integer in [-127.5, 127.5]. With w = 2 z (an odd integer in
 * [-255, 255]), the inner product of two residuals r_a, r_b (relative to the
 * centroid c) is estimated as
 *
 *     <r_a, r_b> ~= f_a * f_b * sum_k(w_a[k] * w_b[k]) / 4
 *
 * where f is ExtraBitsFactors::f_rescale_ex. With a =
 * ExtraBitsFactors::f_add_ex, which for METRIC_INNER_PRODUCT holds <c, r>,
 * the inner product of the original vectors x = c + r is estimated as
 *
 *     <x_a, x_b> ~= |c|^2 + a_a + a_b + f_a * f_b * sum_k(w_a[k] w_b[k]) / 4
 *
 * The centroid terms are kept exactly and only the residual cross term is
 * approximated. The estimate is symmetric in a and b.
 *
 * The exact form used is restore() below. The integer sum is exact, and the
 * pair (a, b) is ordered by id by the caller, so the result does not depend
 * on the order of the arguments.
 */

#include <cstdint>
#include <cstring>
#include <limits>

#include <faiss/impl/FaissAssert.h>

#if defined(COMPILE_SIMD_AVX512) && defined(__AVX512BW__)
#include <immintrin.h>
#endif

namespace faiss {
namespace rabitq_ip {

/// Byte offset of the extra bits in an 8-bit code: sign bits, then the 12
/// bytes of SignBitFactorsWithError.
inline size_t extra_offset(size_t d) {
    return (d + 7) / 8 + 12;
}

/// Byte offset of the ExtraBitsFactors (f_add_ex, f_rescale_ex).
inline size_t ext_factors_offset(size_t d) {
    return extra_offset(d) + (7 * d + 7) / 8;
}

/// Size of an 8-bit code of dimension d.
inline size_t code_size_8bit(size_t d) {
    return ext_factors_offset(d) + 8;
}

/// w = 256 b + 2 e - 255 of dimension k.
inline int16_t coefficient(const uint8_t* code, size_t d, size_t k) {
    const size_t bit = k * 7;
    const uint8_t* extra = code + extra_offset(d);
    unsigned value = extra[bit / 8];
    // do not read past the end of the extra bits for the last dimension
    if (bit % 8 > 1) {
        value |= unsigned(extra[bit / 8 + 1]) << 8;
    }
    const int e = (value >> (bit % 8)) & 127;
    const int b = (code[k / 8] >> (k % 8)) & 1;
    return int16_t(256 * b + 2 * e - 255);
}

/// sum_k w_a[k] * w_b[k], exact.
inline int64_t dot_scalar(const uint8_t* a, const uint8_t* b, size_t d) {
    FAISS_THROW_IF_NOT(
            d <= size_t(std::numeric_limits<int64_t>::max() / 65025));
    int64_t sum = 0;
    for (size_t k = 0; k < d; ++k) {
        sum += int64_t(coefficient(a, d, k)) * coefficient(b, d, k);
    }
    return sum;
}

#if defined(COMPILE_SIMD_AVX512) && defined(__AVX512BW__)
/// Same result as dot_scalar(). The int32 lanes cannot overflow for
/// d <= INT32_MAX / 65025; larger d falls back to dot_scalar().
inline int64_t dot_avx512(const uint8_t* a, const uint8_t* b, size_t d) {
    if (d > size_t(std::numeric_limits<int32_t>::max() / 65025)) {
        return dot_scalar(a, b, d);
    }
    __m512i sum = _mm512_setzero_si512();
    alignas(64) int16_t wa[32], wb[32];
    size_t k = 0;
    for (; k + 32 <= d; k += 32) {
        for (size_t t = 0; t < 32; ++t) {
            wa[t] = coefficient(a, d, k + t);
            wb[t] = coefficient(b, d, k + t);
        }
        sum = _mm512_add_epi32(
                sum,
                _mm512_madd_epi16(
                        _mm512_load_si512(wa), _mm512_load_si512(wb)));
    }
    int64_t result = _mm512_reduce_add_epi32(sum);
    for (; k < d; ++k) {
        result += int64_t(coefficient(a, d, k)) * coefficient(b, d, k);
    }
    return result;
}
#endif

/** Inner product estimate of the vectors of codes a and b from the exact
 * integer dot = sum_k w_a[k] * w_b[k]. center_norm_sq is |c|^2.
 * Evaluated in double precision with a fixed order of operations. */
inline float restore(
        const uint8_t* a,
        const uint8_t* b,
        size_t d,
        double center_norm_sq,
        int64_t dot) {
    const size_t offset = ext_factors_offset(d);
    float af[2], bf[2];
    std::memcpy(af, a + offset, sizeof(af));
    std::memcpy(bf, b + offset, sizeof(bf));
    const double base = (center_norm_sq + double(af[0])) + double(bf[0]);
    const double scale = double(af[1]) * double(bf[1]);
    return float(base + scale * (double(dot) * 0.25));
}

} // namespace rabitq_ip
} // namespace faiss
