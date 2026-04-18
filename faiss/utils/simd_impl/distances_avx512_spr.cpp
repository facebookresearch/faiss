/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The BF16-specific distance computation (VDPBF16PS) lives in
// sq-avx512-spr.cpp, not here.

#ifdef COMPILE_SIMD_AVX512_SPR

#include <faiss/utils/distances.h>

#include <immintrin.h>

#define THE_SIMD_LEVEL SIMDLevel::AVX512_SPR
#include <faiss/utils/simd_impl/distances_autovec-inl.h>
#include <faiss/utils/simd_impl/distances_sse-inl.h>
#include <faiss/utils/transpose/transpose-avx512-inl.h>

namespace faiss {

// Forward declarations of AVX512 explicit specializations
// (defined in distances_avx512.cpp).
template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::AVX512>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::AVX512>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny);

template <>
void fvec_madd<SIMDLevel::AVX512_SPR>(
        const size_t n,
        const float* __restrict a,
        const float bf,
        const float* __restrict b,
        float* __restrict c) {
    fvec_madd<SIMDLevel::AVX512>(n, a, bf, b, c);
}

template <>
void fvec_inner_products_ny<SIMDLevel::AVX512_SPR>(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny<SIMDLevel::AVX512>(ip, x, y, d, ny);
}

template <>
void fvec_L2sqr_ny<SIMDLevel::AVX512_SPR>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny<SIMDLevel::AVX512>(dis, x, y, d, ny);
}

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::AVX512_SPR>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    fvec_L2sqr_ny_transposed<SIMDLevel::AVX512>(
            dis, x, y, y_sqlen, d, d_offset, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::AVX512_SPR>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest<SIMDLevel::AVX512>(
            distances_tmp_buffer, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::AVX512_SPR>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::AVX512>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

template <>
int fvec_madd_and_argmin<SIMDLevel::AVX512_SPR>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    return fvec_madd_and_argmin<SIMDLevel::AVX512>(n, a, bf, b, c);
}

} // namespace faiss

#endif // COMPILE_SIMD_AVX512_SPR
