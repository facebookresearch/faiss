/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file distances_dispatch.h
 * @brief Inlineable dispatch wrappers for distance functions.
 *
 * This is a PRIVATE header. Do not include in public APIs or user code.
 *
 * These wrappers call DISPATCH_SIMDLevel to route to the correct SIMD
 * implementation. They are plain inline functions with a _dispatch suffix
 * (e.g. fvec_L2sqr_dispatch). Internal callers that want inlining include
 * this header and call the _dispatch variants directly.
 *
 * The public API functions (fvec_L2sqr, etc.) are defined as regular extern
 * functions in distances.cpp and simply delegate to these _dispatch variants.
 */

#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances.h>

namespace faiss {

inline float fvec_L1_dispatch(const float* x, const float* y, size_t d) {
    DISPATCH_SIMDLevel(fvec_L1, x, y, d);
}

inline float fvec_Linf_dispatch(const float* x, const float* y, size_t d) {
    DISPATCH_SIMDLevel(fvec_Linf, x, y, d);
}

inline float fvec_norm_L2sqr_dispatch(const float* x, size_t d) {
    DISPATCH_SIMDLevel(fvec_norm_L2sqr, x, d);
}

inline float fvec_L2sqr_dispatch(const float* x, const float* y, size_t d) {
    DISPATCH_SIMDLevel(fvec_L2sqr, x, y, d);
}

inline float fvec_inner_product_dispatch(
        const float* x,
        const float* y,
        size_t d) {
    DISPATCH_SIMDLevel(fvec_inner_product, x, y, d);
}

inline void fvec_inner_product_batch_4_dispatch(
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
    DISPATCH_SIMDLevel(
            fvec_inner_product_batch_4,
            x,
            y0,
            y1,
            y2,
            y3,
            d,
            dis0,
            dis1,
            dis2,
            dis3);
}

inline void fvec_L2sqr_batch_4_dispatch(
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
    DISPATCH_SIMDLevel(
            fvec_L2sqr_batch_4, x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

inline void fvec_L2sqr_ny_transposed_dispatch(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    DISPATCH_SIMDLevel(
            fvec_L2sqr_ny_transposed, dis, x, y, y_sqlen, d, d_offset, ny);
}

inline void fvec_inner_products_ny_dispatch(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    DISPATCH_SIMDLevel(fvec_inner_products_ny, ip, x, y, d, ny);
}

inline void fvec_L2sqr_ny_dispatch(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    DISPATCH_SIMDLevel(fvec_L2sqr_ny, dis, x, y, d, ny);
}

inline size_t fvec_L2sqr_ny_nearest_dispatch(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    DISPATCH_SIMDLevel(
            fvec_L2sqr_ny_nearest, distances_tmp_buffer, x, y, d, ny);
}

inline size_t fvec_L2sqr_ny_nearest_y_transposed_dispatch(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    DISPATCH_SIMDLevel(
            fvec_L2sqr_ny_nearest_y_transposed,
            distances_tmp_buffer,
            x,
            y,
            y_sqlen,
            d,
            d_offset,
            ny);
}

inline void fvec_madd_dispatch(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    DISPATCH_SIMDLevel(fvec_madd, n, a, bf, b, c);
}

inline int fvec_madd_and_argmin_dispatch(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    DISPATCH_SIMDLevel(fvec_madd_and_argmin, n, a, bf, b, c);
}

} // namespace faiss
