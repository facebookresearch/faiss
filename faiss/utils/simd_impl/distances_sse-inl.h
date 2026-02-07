/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/utils/distances.h>

#include <immintrin.h>

namespace faiss {

[[maybe_unused]] inline void fvec_madd_sse(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    n >>= 2;
    __m128 bf4 = _mm_set_ps1(bf);
    __m128* a4 = (__m128*)a;
    __m128* b4 = (__m128*)b;
    __m128* c4 = (__m128*)c;

    while (n--) {
        *c4 = _mm_add_ps(*a4, _mm_mul_ps(bf4, *b4));
        b4++;
        a4++;
        c4++;
    }
}

/// helper function
inline float horizontal_sum(const __m128 v) {
    // say, v is [x0, x1, x2, x3]

    // v0 is [x2, x3, ..., ...]
    const __m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
    // v1 is [x0 + x2, x1 + x3, ..., ...]
    const __m128 v1 = _mm_add_ps(v, v0);
    // v2 is [x1 + x3, ..., .... ,...]
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    // v3 is [x0 + x1 + x2 + x3, ..., ..., ...]
    const __m128 v3 = _mm_add_ps(v1, v2);
    // return v3[0]
    return _mm_cvtss_f32(v3);
}

/// Function that does a component-wise operation between x and y
/// to compute inner products
struct ElementOpIP {
    static float op(float x, float y) {
        return x * y;
    }

    static __m128 op(__m128 x, __m128 y) {
        return _mm_mul_ps(x, y);
    }
};

/// Function that does a component-wise operation between x and y
/// to compute L2 distances. ElementOp can then be used in the fvec_op_ny
/// functions below
struct ElementOpL2 {
    static float op(float x, float y) {
        float tmp = x - y;
        return tmp * tmp;
    }

    static __m128 op(__m128 x, __m128 y) {
        __m128 tmp = _mm_sub_ps(x, y);
        return _mm_mul_ps(tmp, tmp);
    }
};

template <class ElementOp>
void fvec_op_ny_D1(float* dis, const float* x, const float* y, size_t ny) {
    float x0s = x[0];
    __m128 x0 = _mm_set_ps(x0s, x0s, x0s, x0s);

    size_t i;
    for (i = 0; i + 3 < ny; i += 4) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        dis[i] = _mm_cvtss_f32(accu);
        __m128 tmp = _mm_shuffle_ps(accu, accu, 1);
        dis[i + 1] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 2);
        dis[i + 2] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 3] = _mm_cvtss_f32(tmp);
    }
    while (i < ny) { // handle non-multiple-of-4 case
        dis[i++] = ElementOp::op(x0s, *y++);
    }
}

template <class ElementOp>
void fvec_op_ny_D2(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_set_ps(x[1], x[0], x[1], x[0]);

    size_t i;
    for (i = 0; i + 1 < ny; i += 2) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
        accu = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 1] = _mm_cvtss_f32(accu);
    }
    if (i < ny) { // handle odd case
        dis[i] = ElementOp::op(x[0], y[0]) + ElementOp::op(x[1], y[1]);
    }
}

template <class ElementOp>
void fvec_op_ny_D4(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        dis[i] = horizontal_sum(accu);
    }
}

template <class ElementOp>
void fvec_op_ny_D8(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x1, _mm_loadu_ps(y)));
        y += 4;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

template <class ElementOp>
void fvec_op_ny_D12(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);
    __m128 x2 = _mm_loadu_ps(x + 8);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x1, _mm_loadu_ps(y)));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x2, _mm_loadu_ps(y)));
        y += 4;
        dis[i] = horizontal_sum(accu);
    }
}

template <class ElementOpIP>
void fvec_inner_products_ny_ref(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
#define DISPATCH(dval)                                  \
    case dval:                                          \
        fvec_op_ny_D##dval<ElementOpIP>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        default:
            fvec_inner_products_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}

template <class ElementOpL2>
void fvec_L2sqr_ny_ref(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    // optimized for a few special cases

#define DISPATCH(dval)                                  \
    case dval:                                          \
        fvec_op_ny_D##dval<ElementOpL2>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        default:
            fvec_L2sqr_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}

template <SIMDLevel>
size_t fvec_L2sqr_ny_nearest_D2(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny);

template <SIMDLevel>
size_t fvec_L2sqr_ny_nearest_D4(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny);

template <SIMDLevel>
size_t fvec_L2sqr_ny_nearest_D8(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny);

template <SIMDLevel SIMD>
size_t fvec_L2sqr_ny_nearest_x86(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny,
        size_t (*fvec_L2sqr_ny_nearest_D2_func)(
                float*,
                const float*,
                const float*,
                size_t) = &fvec_L2sqr_ny_nearest_D2<SIMD>,
        size_t (*fvec_L2sqr_ny_nearest_D4_func)(
                float*,
                const float*,
                const float*,
                size_t) = &fvec_L2sqr_ny_nearest_D4<SIMD>,
        size_t (*fvec_L2sqr_ny_nearest_D8_func)(
                float*,
                const float*,
                const float*,
                size_t) = &fvec_L2sqr_ny_nearest_D8<SIMD>);

template <SIMDLevel SIMD>
size_t fvec_L2sqr_ny_nearest_x86(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny,
        size_t (*fvec_L2sqr_ny_nearest_D2_func)(
                float*,
                const float*,
                const float*,
                size_t),
        size_t (*fvec_L2sqr_ny_nearest_D4_func)(
                float*,
                const float*,
                const float*,
                size_t),
        size_t (*fvec_L2sqr_ny_nearest_D8_func)(
                float*,
                const float*,
                const float*,
                size_t)) {
    switch (d) {
        case 2:
            return fvec_L2sqr_ny_nearest_D2_func(
                    distances_tmp_buffer, x, y, ny);
        case 4:
            return fvec_L2sqr_ny_nearest_D4_func(
                    distances_tmp_buffer, x, y, ny);
        case 8:
            return fvec_L2sqr_ny_nearest_D8_func(
                    distances_tmp_buffer, x, y, ny);
    }

    return fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, d, ny);
}

template <SIMDLevel SIMD>
inline size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

inline int fvec_madd_and_argmin_sse_ref(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    n >>= 2;
    __m128 bf4 = _mm_set_ps1(bf);
    __m128 vmin4 = _mm_set_ps1(1e20);
    __m128i imin4 = _mm_set1_epi32(-1);
    __m128i idx4 = _mm_set_epi32(3, 2, 1, 0);
    __m128i inc4 = _mm_set1_epi32(4);
    __m128* a4 = (__m128*)a;
    __m128* b4 = (__m128*)b;
    __m128* c4 = (__m128*)c;

    while (n--) {
        __m128 vc4 = _mm_add_ps(*a4, _mm_mul_ps(bf4, *b4));
        *c4 = vc4;
        __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(vmin4, vc4));
        // imin4 = _mm_blendv_epi8 (imin4, idx4, mask); // slower!

        imin4 = _mm_or_si128(
                _mm_and_si128(mask, idx4), _mm_andnot_si128(mask, imin4));
        vmin4 = _mm_min_ps(vmin4, vc4);
        b4++;
        a4++;
        c4++;
        idx4 = _mm_add_epi32(idx4, inc4);
    }

    // 4 values -> 2
    {
        idx4 = _mm_shuffle_epi32(imin4, 3 << 2 | 2);
        __m128 vc4 = _mm_shuffle_ps(vmin4, vmin4, 3 << 2 | 2);
        __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(vmin4, vc4));
        imin4 = _mm_or_si128(
                _mm_and_si128(mask, idx4), _mm_andnot_si128(mask, imin4));
        vmin4 = _mm_min_ps(vmin4, vc4);
    }
    // 2 values -> 1
    {
        idx4 = _mm_shuffle_epi32(imin4, 1);
        __m128 vc4 = _mm_shuffle_ps(vmin4, vmin4, 1);
        __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(vmin4, vc4));
        imin4 = _mm_or_si128(
                _mm_and_si128(mask, idx4), _mm_andnot_si128(mask, imin4));
        // vmin4 = _mm_min_ps (vmin4, vc4);
    }
    return _mm_cvtsi128_si32(imin4);
}

inline int fvec_madd_and_argmin_sse(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    if ((n & 3) == 0 && ((((long)a) | ((long)b) | ((long)c)) & 15) == 0) {
        return fvec_madd_and_argmin_sse_ref(n, a, bf, b, c);
    } else {
        return fvec_madd_and_argmin<SIMDLevel::NONE>(n, a, bf, b, c);
    }
}

// reads 0 <= d < 4 floats as __m128
inline __m128 masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
            [[fallthrough]];
        case 2:
            buf[1] = x[1];
            [[fallthrough]];
        case 1:
            buf[0] = x[0];
            break;
        default:
            break;
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

} // namespace faiss
