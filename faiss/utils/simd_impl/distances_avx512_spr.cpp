/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AVX512_SPR distance function specializations.
// These delegate to the AVX512 implementations since distance functions do not
// currently use any SPR-only instructions.

#include <faiss/utils/distances.h>

namespace faiss {

template <>
float fvec_L2sqr<SIMDLevel::AVX512_SPR>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_L2sqr<SIMDLevel::AVX512>(x, y, d);
}

template <>
float fvec_inner_product<SIMDLevel::AVX512_SPR>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_inner_product<SIMDLevel::AVX512>(x, y, d);
}

template <>
float fvec_L1<SIMDLevel::AVX512_SPR>(const float* x, const float* y, size_t d) {
    return fvec_L1<SIMDLevel::AVX512>(x, y, d);
}

template <>
float fvec_Linf<SIMDLevel::AVX512_SPR>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_Linf<SIMDLevel::AVX512>(x, y, d);
}

template <>
float fvec_norm_L2sqr<SIMDLevel::AVX512_SPR>(const float* x, size_t d) {
    return fvec_norm_L2sqr<SIMDLevel::AVX512>(x, d);
}

template <>
void fvec_inner_product_batch_4<SIMDLevel::AVX512_SPR>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_inner_product_batch_4<SIMDLevel::AVX512>(
            x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

template <>
void fvec_L2sqr_batch_4<SIMDLevel::AVX512_SPR>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_L2sqr_batch_4<SIMDLevel::AVX512>(
            x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

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
