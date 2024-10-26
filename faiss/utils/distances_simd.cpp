/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/simdlib.h>

#ifdef __SSE3__
#include <immintrin.h>
#endif

#if defined(__AVX512F__)
#include <faiss/utils/transpose/transpose-avx512-inl.h>
#elif defined(__AVX2__)
#include <faiss/utils/transpose/transpose-avx2-inl.h>
#endif

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace faiss {

#ifdef __AVX__
#define USE_AVX
#endif

/*********************************************************
 * Optimized distance computations
 *********************************************************/

/* Functions to compute:
   - L2 distance between 2 vectors
   - inner product between 2 vectors
   - L2 norm of a vector

   The functions should probably not be invoked when a large number of
   vectors are be processed in batch (in which case Matrix multiply
   is faster), but may be useful for comparing vectors isolated in
   memory.

   Works with any vectors of any dimension, even unaligned (in which
   case they are slower).

*/

/*********************************************************
 * Reference implementations
 */

float fvec_L1_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += fabs(tmp);
    }
    return res;
}

float fvec_Linf_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        res = fmax(res, fabs(x[i] - y[i]));
    }
    return res;
}

void fvec_L2sqr_ny_ref(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr(x, y, d);
        y += d;
    }
}

void fvec_L2sqr_ny_y_transposed_ref(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        x_sqlen += x[j] * x[j];
    }

    for (size_t i = 0; i < ny; i++) {
        float dp = 0;
        for (size_t j = 0; j < d; j++) {
            dp += x[j] * y[i + j * d_offset];
        }

        dis[i] = x_sqlen + y_sqlen[i] - 2 * dp;
    }
}

size_t fvec_L2sqr_ny_nearest_ref(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny(distances_tmp_buffer, x, y, d, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

size_t fvec_L2sqr_ny_nearest_y_transposed_ref(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    fvec_L2sqr_ny_y_transposed_ref(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

void fvec_inner_products_ny_ref(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    // BLAS slower for the use cases here
#if 0
    {
        FINTEGER di = d;
        FINTEGER nyi = ny;
        float one = 1.0, zero = 0.0;
        FINTEGER onei = 1;
        sgemv_ ("T", &di, &nyi, &one, y, &di, x, &onei, &zero, ip, &onei);
    }
#endif
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product(x, y, d);
        y += d;
    }
}

/*********************************************************
 * Autovectorized implementations
 */

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float fvec_inner_product(const float* x, const float* y, size_t d) {
    float res = 0.F;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * y[i];
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float fvec_L2sqr(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/// Special version of inner product that computes 4 distances
/// between x and yi
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void fvec_inner_product_batch_4(
        const float* __restrict x,
        const float* __restrict y0,
        const float* __restrict y1,
        const float* __restrict y2,
        const float* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * y0[i];
        d1 += x[i] * y1[i];
        d2 += x[i] * y2[i];
        d3 += x[i] * y3[i];
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void fvec_L2sqr_batch_4(
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
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/*********************************************************
 * SSE and AVX implementations
 */

#ifdef __SSE3__

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float* x) {
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
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

namespace {

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

#ifdef __AVX2__
/// helper function for AVX2
inline float horizontal_sum(const __m256 v) {
    // add high and low parts
    const __m128 v0 =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    // perform horizontal sum on v0
    return horizontal_sum(v0);
}
#endif

#ifdef __AVX512F__
/// helper function for AVX512
inline float horizontal_sum(const __m512 v) {
    // performs better than adding the high and low parts
    return _mm512_reduce_add_ps(v);
}
#endif

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

#ifdef __AVX2__
    static __m256 op(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(x, y);
        return _mm256_mul_ps(tmp, tmp);
    }
#endif

#ifdef __AVX512F__
    static __m512 op(__m512 x, __m512 y) {
        __m512 tmp = _mm512_sub_ps(x, y);
        return _mm512_mul_ps(tmp, tmp);
    }
#endif
};

/// Function that does a component-wise operation between x and y
/// to compute inner products
struct ElementOpIP {
    static float op(float x, float y) {
        return x * y;
    }

    static __m128 op(__m128 x, __m128 y) {
        return _mm_mul_ps(x, y);
    }

#ifdef __AVX2__
    static __m256 op(__m256 x, __m256 y) {
        return _mm256_mul_ps(x, y);
    }
#endif

#ifdef __AVX512F__
    static __m512 op(__m512 x, __m512 y) {
        return _mm512_mul_ps(x, y);
    }
#endif
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

#if defined(__AVX512F__)

template <>
void fvec_op_ny_D2<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D2-vectors per loop.
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);

        for (i = 0; i < ny16 * 16; i += 16) {
            _mm_prefetch((const char*)(y + 64), _MM_HINT_T0);

            // load 16x2 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;

            transpose_16x2(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    v0,
                    v1);

            // compute distances (dot product)
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 32; // move to the next set of 16x2 elements
        }
    }

    if (i < ny) {
        // process leftovers
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float distance = x0 * y[0] + x1 * y[1];
            y += 2;
            dis[i] = distance;
        }
    }
}

template <>
void fvec_op_ny_D2<ElementOpL2>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D2-vectors per loop.
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);

        for (i = 0; i < ny16 * 16; i += 16) {
            _mm_prefetch((const char*)(y + 64), _MM_HINT_T0);

            // load 16x2 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;

            transpose_16x2(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    v0,
                    v1);

            // compute differences
            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);

            // compute squares of differences
            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 32; // move to the next set of 16x2 elements
        }
    }

    if (i < ny) {
        // process leftovers
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float sub0 = x0 - y[0];
            float sub1 = x1 - y[1];
            float distance = sub0 * sub0 + sub1 * sub1;

            y += 2;
            dis[i] = distance;
        }
    }
}

#elif defined(__AVX2__)

template <>
void fvec_op_ny_D2<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny8 = ny / 8;
    size_t i = 0;

    if (ny8 > 0) {
        // process 8 D2-vectors per loop.
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 16), _MM_HINT_T0);

        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);

        for (i = 0; i < ny8 * 8; i += 8) {
            _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

            // load 8x2 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m256 v0;
            __m256 v1;

            transpose_8x2(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    v0,
                    v1);

            // compute distances
            __m256 distances = _mm256_mul_ps(m0, v0);
            distances = _mm256_fmadd_ps(m1, v1, distances);

            // store
            _mm256_storeu_ps(dis + i, distances);

            y += 16;
        }
    }

    if (i < ny) {
        // process leftovers
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float distance = x0 * y[0] + x1 * y[1];
            y += 2;
            dis[i] = distance;
        }
    }
}

template <>
void fvec_op_ny_D2<ElementOpL2>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny8 = ny / 8;
    size_t i = 0;

    if (ny8 > 0) {
        // process 8 D2-vectors per loop.
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 16), _MM_HINT_T0);

        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);

        for (i = 0; i < ny8 * 8; i += 8) {
            _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

            // load 8x2 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m256 v0;
            __m256 v1;

            transpose_8x2(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    v0,
                    v1);

            // compute differences
            const __m256 d0 = _mm256_sub_ps(m0, v0);
            const __m256 d1 = _mm256_sub_ps(m1, v1);

            // compute squares of differences
            __m256 distances = _mm256_mul_ps(d0, d0);
            distances = _mm256_fmadd_ps(d1, d1, distances);

            // store
            _mm256_storeu_ps(dis + i, distances);

            y += 16;
        }
    }

    if (i < ny) {
        // process leftovers
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float sub0 = x0 - y[0];
            float sub1 = x1 - y[1];
            float distance = sub0 * sub0 + sub1 * sub1;

            y += 2;
            dis[i] = distance;
        }
    }
}

#endif

template <class ElementOp>
void fvec_op_ny_D4(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        dis[i] = horizontal_sum(accu);
    }
}

#if defined(__AVX512F__)

template <>
void fvec_op_ny_D4<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D4-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x4 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;

            transpose_16x4(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    _mm512_loadu_ps(y + 2 * 16),
                    _mm512_loadu_ps(y + 3 * 16),
                    v0,
                    v1,
                    v2,
                    v3);

            // compute distances
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);
            distances = _mm512_fmadd_ps(m2, v2, distances);
            distances = _mm512_fmadd_ps(m3, v3, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 64; // move to the next set of 16x4 elements
        }
    }

    if (i < ny) {
        // process leftovers
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpIP::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D4<ElementOpL2>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D4-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x4 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;

            transpose_16x4(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    _mm512_loadu_ps(y + 2 * 16),
                    _mm512_loadu_ps(y + 3 * 16),
                    v0,
                    v1,
                    v2,
                    v3);

            // compute differences
            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);

            // compute squares of differences
            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 64; // move to the next set of 16x4 elements
        }
    }

    if (i < ny) {
        // process leftovers
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

#elif defined(__AVX2__)

template <>
void fvec_op_ny_D4<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny8 = ny / 8;
    size_t i = 0;

    if (ny8 > 0) {
        // process 8 D4-vectors per loop.
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);
        const __m256 m2 = _mm256_set1_ps(x[2]);
        const __m256 m3 = _mm256_set1_ps(x[3]);

        for (i = 0; i < ny8 * 8; i += 8) {
            // load 8x4 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m256 v0;
            __m256 v1;
            __m256 v2;
            __m256 v3;

            transpose_8x4(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    _mm256_loadu_ps(y + 2 * 8),
                    _mm256_loadu_ps(y + 3 * 8),
                    v0,
                    v1,
                    v2,
                    v3);

            // compute distances
            __m256 distances = _mm256_mul_ps(m0, v0);
            distances = _mm256_fmadd_ps(m1, v1, distances);
            distances = _mm256_fmadd_ps(m2, v2, distances);
            distances = _mm256_fmadd_ps(m3, v3, distances);

            // store
            _mm256_storeu_ps(dis + i, distances);

            y += 32;
        }
    }

    if (i < ny) {
        // process leftovers
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpIP::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D4<ElementOpL2>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny8 = ny / 8;
    size_t i = 0;

    if (ny8 > 0) {
        // process 8 D4-vectors per loop.
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);
        const __m256 m2 = _mm256_set1_ps(x[2]);
        const __m256 m3 = _mm256_set1_ps(x[3]);

        for (i = 0; i < ny8 * 8; i += 8) {
            // load 8x4 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m256 v0;
            __m256 v1;
            __m256 v2;
            __m256 v3;

            transpose_8x4(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    _mm256_loadu_ps(y + 2 * 8),
                    _mm256_loadu_ps(y + 3 * 8),
                    v0,
                    v1,
                    v2,
                    v3);

            // compute differences
            const __m256 d0 = _mm256_sub_ps(m0, v0);
            const __m256 d1 = _mm256_sub_ps(m1, v1);
            const __m256 d2 = _mm256_sub_ps(m2, v2);
            const __m256 d3 = _mm256_sub_ps(m3, v3);

            // compute squares of differences
            __m256 distances = _mm256_mul_ps(d0, d0);
            distances = _mm256_fmadd_ps(d1, d1, distances);
            distances = _mm256_fmadd_ps(d2, d2, distances);
            distances = _mm256_fmadd_ps(d3, d3, distances);

            // store
            _mm256_storeu_ps(dis + i, distances);

            y += 32;
        }
    }

    if (i < ny) {
        // process leftovers
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

#endif

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

#if defined(__AVX512F__)

template <>
void fvec_op_ny_D8<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D16-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);
        const __m512 m4 = _mm512_set1_ps(x[4]);
        const __m512 m5 = _mm512_set1_ps(x[5]);
        const __m512 m6 = _mm512_set1_ps(x[6]);
        const __m512 m7 = _mm512_set1_ps(x[7]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x8 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;
            __m512 v4;
            __m512 v5;
            __m512 v6;
            __m512 v7;

            transpose_16x8(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    _mm512_loadu_ps(y + 2 * 16),
                    _mm512_loadu_ps(y + 3 * 16),
                    _mm512_loadu_ps(y + 4 * 16),
                    _mm512_loadu_ps(y + 5 * 16),
                    _mm512_loadu_ps(y + 6 * 16),
                    _mm512_loadu_ps(y + 7 * 16),
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7);

            // compute distances
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);
            distances = _mm512_fmadd_ps(m2, v2, distances);
            distances = _mm512_fmadd_ps(m3, v3, distances);
            distances = _mm512_fmadd_ps(m4, v4, distances);
            distances = _mm512_fmadd_ps(m5, v5, distances);
            distances = _mm512_fmadd_ps(m6, v6, distances);
            distances = _mm512_fmadd_ps(m7, v7, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 128; // 16 floats * 8 rows
        }
    }

    if (i < ny) {
        // process leftovers
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpIP::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D8<ElementOpL2>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D16-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);
        const __m512 m4 = _mm512_set1_ps(x[4]);
        const __m512 m5 = _mm512_set1_ps(x[5]);
        const __m512 m6 = _mm512_set1_ps(x[6]);
        const __m512 m7 = _mm512_set1_ps(x[7]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x8 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;
            __m512 v4;
            __m512 v5;
            __m512 v6;
            __m512 v7;

            transpose_16x8(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    _mm512_loadu_ps(y + 2 * 16),
                    _mm512_loadu_ps(y + 3 * 16),
                    _mm512_loadu_ps(y + 4 * 16),
                    _mm512_loadu_ps(y + 5 * 16),
                    _mm512_loadu_ps(y + 6 * 16),
                    _mm512_loadu_ps(y + 7 * 16),
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7);

            // compute differences
            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);
            const __m512 d4 = _mm512_sub_ps(m4, v4);
            const __m512 d5 = _mm512_sub_ps(m5, v5);
            const __m512 d6 = _mm512_sub_ps(m6, v6);
            const __m512 d7 = _mm512_sub_ps(m7, v7);

            // compute squares of differences
            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);
            distances = _mm512_fmadd_ps(d4, d4, distances);
            distances = _mm512_fmadd_ps(d5, d5, distances);
            distances = _mm512_fmadd_ps(d6, d6, distances);
            distances = _mm512_fmadd_ps(d7, d7, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 128; // 16 floats * 8 rows
        }
    }

    if (i < ny) {
        // process leftovers
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

#elif defined(__AVX2__)

template <>
void fvec_op_ny_D8<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny8 = ny / 8;
    size_t i = 0;

    if (ny8 > 0) {
        // process 8 D8-vectors per loop.
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);
        const __m256 m2 = _mm256_set1_ps(x[2]);
        const __m256 m3 = _mm256_set1_ps(x[3]);
        const __m256 m4 = _mm256_set1_ps(x[4]);
        const __m256 m5 = _mm256_set1_ps(x[5]);
        const __m256 m6 = _mm256_set1_ps(x[6]);
        const __m256 m7 = _mm256_set1_ps(x[7]);

        for (i = 0; i < ny8 * 8; i += 8) {
            // load 8x8 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m256 v0;
            __m256 v1;
            __m256 v2;
            __m256 v3;
            __m256 v4;
            __m256 v5;
            __m256 v6;
            __m256 v7;

            transpose_8x8(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    _mm256_loadu_ps(y + 2 * 8),
                    _mm256_loadu_ps(y + 3 * 8),
                    _mm256_loadu_ps(y + 4 * 8),
                    _mm256_loadu_ps(y + 5 * 8),
                    _mm256_loadu_ps(y + 6 * 8),
                    _mm256_loadu_ps(y + 7 * 8),
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7);

            // compute distances
            __m256 distances = _mm256_mul_ps(m0, v0);
            distances = _mm256_fmadd_ps(m1, v1, distances);
            distances = _mm256_fmadd_ps(m2, v2, distances);
            distances = _mm256_fmadd_ps(m3, v3, distances);
            distances = _mm256_fmadd_ps(m4, v4, distances);
            distances = _mm256_fmadd_ps(m5, v5, distances);
            distances = _mm256_fmadd_ps(m6, v6, distances);
            distances = _mm256_fmadd_ps(m7, v7, distances);

            // store
            _mm256_storeu_ps(dis + i, distances);

            y += 64;
        }
    }

    if (i < ny) {
        // process leftovers
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpIP::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D8<ElementOpL2>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny8 = ny / 8;
    size_t i = 0;

    if (ny8 > 0) {
        // process 8 D8-vectors per loop.
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);
        const __m256 m2 = _mm256_set1_ps(x[2]);
        const __m256 m3 = _mm256_set1_ps(x[3]);
        const __m256 m4 = _mm256_set1_ps(x[4]);
        const __m256 m5 = _mm256_set1_ps(x[5]);
        const __m256 m6 = _mm256_set1_ps(x[6]);
        const __m256 m7 = _mm256_set1_ps(x[7]);

        for (i = 0; i < ny8 * 8; i += 8) {
            // load 8x8 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m256 v0;
            __m256 v1;
            __m256 v2;
            __m256 v3;
            __m256 v4;
            __m256 v5;
            __m256 v6;
            __m256 v7;

            transpose_8x8(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    _mm256_loadu_ps(y + 2 * 8),
                    _mm256_loadu_ps(y + 3 * 8),
                    _mm256_loadu_ps(y + 4 * 8),
                    _mm256_loadu_ps(y + 5 * 8),
                    _mm256_loadu_ps(y + 6 * 8),
                    _mm256_loadu_ps(y + 7 * 8),
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7);

            // compute differences
            const __m256 d0 = _mm256_sub_ps(m0, v0);
            const __m256 d1 = _mm256_sub_ps(m1, v1);
            const __m256 d2 = _mm256_sub_ps(m2, v2);
            const __m256 d3 = _mm256_sub_ps(m3, v3);
            const __m256 d4 = _mm256_sub_ps(m4, v4);
            const __m256 d5 = _mm256_sub_ps(m5, v5);
            const __m256 d6 = _mm256_sub_ps(m6, v6);
            const __m256 d7 = _mm256_sub_ps(m7, v7);

            // compute squares of differences
            __m256 distances = _mm256_mul_ps(d0, d0);
            distances = _mm256_fmadd_ps(d1, d1, distances);
            distances = _mm256_fmadd_ps(d2, d2, distances);
            distances = _mm256_fmadd_ps(d3, d3, distances);
            distances = _mm256_fmadd_ps(d4, d4, distances);
            distances = _mm256_fmadd_ps(d5, d5, distances);
            distances = _mm256_fmadd_ps(d6, d6, distances);
            distances = _mm256_fmadd_ps(d7, d7, distances);

            // store
            _mm256_storeu_ps(dis + i, distances);

            y += 64;
        }
    }

    if (i < ny) {
        // process leftovers
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

#endif

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

} // anonymous namespace

void fvec_L2sqr_ny(
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
            fvec_L2sqr_ny_ref(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}

void fvec_inner_products_ny(
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
            fvec_inner_products_ny_ref(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}

#if defined(__AVX512F__)

template <size_t DIM>
void fvec_L2sqr_ny_y_transposed_D(
        float* distances,
        const float* x,
        const float* y,
        const float* y_sqlen,
        const size_t d_offset,
        size_t ny) {
    // current index being processed
    size_t i = 0;

    // squared length of x
    float x_sqlen = 0;
    for (size_t j = 0; j < DIM; j++) {
        x_sqlen += x[j] * x[j];
    }

    // process 16 vectors per loop
    const size_t ny16 = ny / 16;

    if (ny16 > 0) {
        // m[i] = (2 * x[i], ... 2 * x[i])
        __m512 m[DIM];
        for (size_t j = 0; j < DIM; j++) {
            m[j] = _mm512_set1_ps(x[j]);
            m[j] = _mm512_add_ps(m[j], m[j]); // m[j] = 2 * x[j]
        }

        __m512 x_sqlen_ymm = _mm512_set1_ps(x_sqlen);

        for (; i < ny16 * 16; i += 16) {
            // Load vectors for 16 dimensions
            __m512 v[DIM];
            for (size_t j = 0; j < DIM; j++) {
                v[j] = _mm512_loadu_ps(y + j * d_offset);
            }

            // Compute dot products
            __m512 dp = _mm512_fnmadd_ps(m[0], v[0], x_sqlen_ymm);
            for (size_t j = 1; j < DIM; j++) {
                dp = _mm512_fnmadd_ps(m[j], v[j], dp);
            }

            // Compute y^2 - (2 * x, y) + x^2
            __m512 distances_v = _mm512_add_ps(_mm512_loadu_ps(y_sqlen), dp);

            _mm512_storeu_ps(distances + i, distances_v);

            // Scroll y and y_sqlen forward
            y += 16;
            y_sqlen += 16;
        }
    }

    if (i < ny) {
        // Process leftovers
        for (; i < ny; i++) {
            float dp = 0;
            for (size_t j = 0; j < DIM; j++) {
                dp += x[j] * y[j * d_offset];
            }

            // Compute y^2 - 2 * (x, y), which is sufficient for looking for the
            // lowest distance.
            const float distance = y_sqlen[0] - 2 * dp + x_sqlen;
            distances[i] = distance;

            y += 1;
            y_sqlen += 1;
        }
    }
}

#elif defined(__AVX2__)

template <size_t DIM>
void fvec_L2sqr_ny_y_transposed_D(
        float* distances,
        const float* x,
        const float* y,
        const float* y_sqlen,
        const size_t d_offset,
        size_t ny) {
    // current index being processed
    size_t i = 0;

    // squared length of x
    float x_sqlen = 0;
    for (size_t j = 0; j < DIM; j++) {
        x_sqlen += x[j] * x[j];
    }

    // process 8 vectors per loop.
    const size_t ny8 = ny / 8;

    if (ny8 > 0) {
        // m[i] = (2 * x[i], ... 2 * x[i])
        __m256 m[DIM];
        for (size_t j = 0; j < DIM; j++) {
            m[j] = _mm256_set1_ps(x[j]);
            m[j] = _mm256_add_ps(m[j], m[j]);
        }

        __m256 x_sqlen_ymm = _mm256_set1_ps(x_sqlen);

        for (; i < ny8 * 8; i += 8) {
            // collect dim 0 for 8 D4-vectors.
            const __m256 v0 = _mm256_loadu_ps(y + 0 * d_offset);

            // compute dot products
            // this is x^2 - 2x[0]*y[0]
            __m256 dp = _mm256_fnmadd_ps(m[0], v0, x_sqlen_ymm);

            for (size_t j = 1; j < DIM; j++) {
                // collect dim j for 8 D4-vectors.
                const __m256 vj = _mm256_loadu_ps(y + j * d_offset);
                dp = _mm256_fnmadd_ps(m[j], vj, dp);
            }

            // we've got x^2 - (2x, y) at this point

            // y^2 - (2x, y) + x^2
            __m256 distances_v = _mm256_add_ps(_mm256_loadu_ps(y_sqlen), dp);

            _mm256_storeu_ps(distances + i, distances_v);

            // scroll y and y_sqlen forward.
            y += 8;
            y_sqlen += 8;
        }
    }

    if (i < ny) {
        // process leftovers
        for (; i < ny; i++) {
            float dp = 0;
            for (size_t j = 0; j < DIM; j++) {
                dp += x[j] * y[j * d_offset];
            }

            // compute y^2 - 2 * (x, y), which is sufficient for looking for the
            //   lowest distance.
            const float distance = y_sqlen[0] - 2 * dp + x_sqlen;
            distances[i] = distance;

            y += 1;
            y_sqlen += 1;
        }
    }
}

#endif

void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    // optimized for a few special cases

#ifdef __AVX2__
#define DISPATCH(dval)                             \
    case dval:                                     \
        return fvec_L2sqr_ny_y_transposed_D<dval>( \
                dis, x, y, y_sqlen, d_offset, ny);

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        default:
            return fvec_L2sqr_ny_y_transposed_ref(
                    dis, x, y, y_sqlen, d, d_offset, ny);
    }
#undef DISPATCH
#else
    // non-AVX2 case
    return fvec_L2sqr_ny_y_transposed_ref(dis, x, y, y_sqlen, d, d_offset, ny);
#endif
}

#if defined(__AVX512F__)

size_t fvec_L2sqr_ny_nearest_D2(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    size_t i = 0;
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    const size_t ny16 = ny / 16;
    if (ny16 > 0) {
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

        __m512 min_distances = _mm512_set1_ps(HUGE_VALF);
        __m512i min_indices = _mm512_set1_epi32(0);

        __m512i current_indices = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i indices_increment = _mm512_set1_epi32(16);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);

        for (; i < ny16 * 16; i += 16) {
            _mm_prefetch((const char*)(y + 64), _MM_HINT_T0);

            __m512 v0;
            __m512 v1;

            transpose_16x2(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    v0,
                    v1);

            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);

            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);

            __mmask16 comparison =
                    _mm512_cmp_ps_mask(distances, min_distances, _CMP_LT_OS);

            min_distances = _mm512_min_ps(distances, min_distances);
            min_indices = _mm512_mask_blend_epi32(
                    comparison, min_indices, current_indices);

            current_indices =
                    _mm512_add_epi32(current_indices, indices_increment);

            y += 32;
        }

        alignas(64) float min_distances_scalar[16];
        alignas(64) uint32_t min_indices_scalar[16];
        _mm512_store_ps(min_distances_scalar, min_distances);
        _mm512_store_epi32(min_indices_scalar, min_indices);

        for (size_t j = 0; j < 16; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float sub0 = x0 - y[0];
            float sub1 = x1 - y[1];
            float distance = sub0 * sub0 + sub1 * sub1;

            y += 2;

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

size_t fvec_L2sqr_ny_nearest_D4(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    size_t i = 0;
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    const size_t ny16 = ny / 16;

    if (ny16 > 0) {
        __m512 min_distances = _mm512_set1_ps(HUGE_VALF);
        __m512i min_indices = _mm512_set1_epi32(0);

        __m512i current_indices = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i indices_increment = _mm512_set1_epi32(16);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (; i < ny16 * 16; i += 16) {
            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;

            transpose_16x4(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    _mm512_loadu_ps(y + 2 * 16),
                    _mm512_loadu_ps(y + 3 * 16),
                    v0,
                    v1,
                    v2,
                    v3);

            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);

            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);

            __mmask16 comparison =
                    _mm512_cmp_ps_mask(distances, min_distances, _CMP_LT_OS);

            min_distances = _mm512_min_ps(distances, min_distances);
            min_indices = _mm512_mask_blend_epi32(
                    comparison, min_indices, current_indices);

            current_indices =
                    _mm512_add_epi32(current_indices, indices_increment);

            y += 64;
        }

        alignas(64) float min_distances_scalar[16];
        alignas(64) uint32_t min_indices_scalar[16];
        _mm512_store_ps(min_distances_scalar, min_distances);
        _mm512_store_epi32(min_indices_scalar, min_indices);

        for (size_t j = 0; j < 16; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            const float distance = horizontal_sum(accu);

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

size_t fvec_L2sqr_ny_nearest_D8(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    size_t i = 0;
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    const size_t ny16 = ny / 16;
    if (ny16 > 0) {
        __m512 min_distances = _mm512_set1_ps(HUGE_VALF);
        __m512i min_indices = _mm512_set1_epi32(0);

        __m512i current_indices = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i indices_increment = _mm512_set1_epi32(16);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        const __m512 m4 = _mm512_set1_ps(x[4]);
        const __m512 m5 = _mm512_set1_ps(x[5]);
        const __m512 m6 = _mm512_set1_ps(x[6]);
        const __m512 m7 = _mm512_set1_ps(x[7]);

        for (; i < ny16 * 16; i += 16) {
            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;
            __m512 v4;
            __m512 v5;
            __m512 v6;
            __m512 v7;

            transpose_16x8(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    _mm512_loadu_ps(y + 2 * 16),
                    _mm512_loadu_ps(y + 3 * 16),
                    _mm512_loadu_ps(y + 4 * 16),
                    _mm512_loadu_ps(y + 5 * 16),
                    _mm512_loadu_ps(y + 6 * 16),
                    _mm512_loadu_ps(y + 7 * 16),
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7);

            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);
            const __m512 d4 = _mm512_sub_ps(m4, v4);
            const __m512 d5 = _mm512_sub_ps(m5, v5);
            const __m512 d6 = _mm512_sub_ps(m6, v6);
            const __m512 d7 = _mm512_sub_ps(m7, v7);

            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);
            distances = _mm512_fmadd_ps(d4, d4, distances);
            distances = _mm512_fmadd_ps(d5, d5, distances);
            distances = _mm512_fmadd_ps(d6, d6, distances);
            distances = _mm512_fmadd_ps(d7, d7, distances);

            __mmask16 comparison =
                    _mm512_cmp_ps_mask(distances, min_distances, _CMP_LT_OS);

            min_distances = _mm512_min_ps(distances, min_distances);
            min_indices = _mm512_mask_blend_epi32(
                    comparison, min_indices, current_indices);

            current_indices =
                    _mm512_add_epi32(current_indices, indices_increment);

            y += 128;
        }

        alignas(64) float min_distances_scalar[16];
        alignas(64) uint32_t min_indices_scalar[16];
        _mm512_store_ps(min_distances_scalar, min_distances);
        _mm512_store_epi32(min_indices_scalar, min_indices);

        for (size_t j = 0; j < 16; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            const float distance = horizontal_sum(accu);

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

#elif defined(__AVX2__)

size_t fvec_L2sqr_ny_nearest_D2(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    // current index being processed
    size_t i = 0;

    // min distance and the index of the closest vector so far
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    // process 8 D2-vectors per loop.
    const size_t ny8 = ny / 8;
    if (ny8 > 0) {
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 16), _MM_HINT_T0);

        // track min distance and the closest vector independently
        // for each of 8 AVX2 components.
        __m256 min_distances = _mm256_set1_ps(HUGE_VALF);
        __m256i min_indices = _mm256_set1_epi32(0);

        __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i indices_increment = _mm256_set1_epi32(8);

        // 1 value per register
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);

        for (; i < ny8 * 8; i += 8) {
            _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

            __m256 v0;
            __m256 v1;

            transpose_8x2(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    v0,
                    v1);

            // compute differences
            const __m256 d0 = _mm256_sub_ps(m0, v0);
            const __m256 d1 = _mm256_sub_ps(m1, v1);

            // compute squares of differences
            __m256 distances = _mm256_mul_ps(d0, d0);
            distances = _mm256_fmadd_ps(d1, d1, distances);

            // compare the new distances to the min distances
            // for each of 8 AVX2 components.
            __m256 comparison =
                    _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS);

            // update min distances and indices with closest vectors if needed.
            min_distances = _mm256_min_ps(distances, min_distances);
            min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(current_indices),
                    _mm256_castsi256_ps(min_indices),
                    comparison));

            // update current indices values. Basically, +8 to each of the
            // 8 AVX2 components.
            current_indices =
                    _mm256_add_epi32(current_indices, indices_increment);

            // scroll y forward (8 vectors 2 DIM each).
            y += 16;
        }

        // dump values and find the minimum distance / minimum index
        float min_distances_scalar[8];
        uint32_t min_indices_scalar[8];
        _mm256_storeu_ps(min_distances_scalar, min_distances);
        _mm256_storeu_si256((__m256i*)(min_indices_scalar), min_indices);

        for (size_t j = 0; j < 8; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        // process leftovers.
        // the following code is not optimal, but it is rarely invoked.
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float sub0 = x0 - y[0];
            float sub1 = x1 - y[1];
            float distance = sub0 * sub0 + sub1 * sub1;

            y += 2;

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

size_t fvec_L2sqr_ny_nearest_D4(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    // current index being processed
    size_t i = 0;

    // min distance and the index of the closest vector so far
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    // process 8 D4-vectors per loop.
    const size_t ny8 = ny / 8;

    if (ny8 > 0) {
        // track min distance and the closest vector independently
        // for each of 8 AVX2 components.
        __m256 min_distances = _mm256_set1_ps(HUGE_VALF);
        __m256i min_indices = _mm256_set1_epi32(0);

        __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i indices_increment = _mm256_set1_epi32(8);

        // 1 value per register
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);
        const __m256 m2 = _mm256_set1_ps(x[2]);
        const __m256 m3 = _mm256_set1_ps(x[3]);

        for (; i < ny8 * 8; i += 8) {
            __m256 v0;
            __m256 v1;
            __m256 v2;
            __m256 v3;

            transpose_8x4(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    _mm256_loadu_ps(y + 2 * 8),
                    _mm256_loadu_ps(y + 3 * 8),
                    v0,
                    v1,
                    v2,
                    v3);

            // compute differences
            const __m256 d0 = _mm256_sub_ps(m0, v0);
            const __m256 d1 = _mm256_sub_ps(m1, v1);
            const __m256 d2 = _mm256_sub_ps(m2, v2);
            const __m256 d3 = _mm256_sub_ps(m3, v3);

            // compute squares of differences
            __m256 distances = _mm256_mul_ps(d0, d0);
            distances = _mm256_fmadd_ps(d1, d1, distances);
            distances = _mm256_fmadd_ps(d2, d2, distances);
            distances = _mm256_fmadd_ps(d3, d3, distances);

            // compare the new distances to the min distances
            // for each of 8 AVX2 components.
            __m256 comparison =
                    _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS);

            // update min distances and indices with closest vectors if needed.
            min_distances = _mm256_min_ps(distances, min_distances);
            min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(current_indices),
                    _mm256_castsi256_ps(min_indices),
                    comparison));

            // update current indices values. Basically, +8 to each of the
            // 8 AVX2 components.
            current_indices =
                    _mm256_add_epi32(current_indices, indices_increment);

            // scroll y forward (8 vectors 4 DIM each).
            y += 32;
        }

        // dump values and find the minimum distance / minimum index
        float min_distances_scalar[8];
        uint32_t min_indices_scalar[8];
        _mm256_storeu_ps(min_distances_scalar, min_distances);
        _mm256_storeu_si256((__m256i*)(min_indices_scalar), min_indices);

        for (size_t j = 0; j < 8; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        // process leftovers
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            const float distance = horizontal_sum(accu);

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

size_t fvec_L2sqr_ny_nearest_D8(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    // current index being processed
    size_t i = 0;

    // min distance and the index of the closest vector so far
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    // process 8 D8-vectors per loop.
    const size_t ny8 = ny / 8;
    if (ny8 > 0) {
        // track min distance and the closest vector independently
        // for each of 8 AVX2 components.
        __m256 min_distances = _mm256_set1_ps(HUGE_VALF);
        __m256i min_indices = _mm256_set1_epi32(0);

        __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i indices_increment = _mm256_set1_epi32(8);

        // 1 value per register
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);
        const __m256 m2 = _mm256_set1_ps(x[2]);
        const __m256 m3 = _mm256_set1_ps(x[3]);

        const __m256 m4 = _mm256_set1_ps(x[4]);
        const __m256 m5 = _mm256_set1_ps(x[5]);
        const __m256 m6 = _mm256_set1_ps(x[6]);
        const __m256 m7 = _mm256_set1_ps(x[7]);

        for (; i < ny8 * 8; i += 8) {
            __m256 v0;
            __m256 v1;
            __m256 v2;
            __m256 v3;
            __m256 v4;
            __m256 v5;
            __m256 v6;
            __m256 v7;

            transpose_8x8(
                    _mm256_loadu_ps(y + 0 * 8),
                    _mm256_loadu_ps(y + 1 * 8),
                    _mm256_loadu_ps(y + 2 * 8),
                    _mm256_loadu_ps(y + 3 * 8),
                    _mm256_loadu_ps(y + 4 * 8),
                    _mm256_loadu_ps(y + 5 * 8),
                    _mm256_loadu_ps(y + 6 * 8),
                    _mm256_loadu_ps(y + 7 * 8),
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7);

            // compute differences
            const __m256 d0 = _mm256_sub_ps(m0, v0);
            const __m256 d1 = _mm256_sub_ps(m1, v1);
            const __m256 d2 = _mm256_sub_ps(m2, v2);
            const __m256 d3 = _mm256_sub_ps(m3, v3);
            const __m256 d4 = _mm256_sub_ps(m4, v4);
            const __m256 d5 = _mm256_sub_ps(m5, v5);
            const __m256 d6 = _mm256_sub_ps(m6, v6);
            const __m256 d7 = _mm256_sub_ps(m7, v7);

            // compute squares of differences
            __m256 distances = _mm256_mul_ps(d0, d0);
            distances = _mm256_fmadd_ps(d1, d1, distances);
            distances = _mm256_fmadd_ps(d2, d2, distances);
            distances = _mm256_fmadd_ps(d3, d3, distances);
            distances = _mm256_fmadd_ps(d4, d4, distances);
            distances = _mm256_fmadd_ps(d5, d5, distances);
            distances = _mm256_fmadd_ps(d6, d6, distances);
            distances = _mm256_fmadd_ps(d7, d7, distances);

            // compare the new distances to the min distances
            // for each of 8 AVX2 components.
            __m256 comparison =
                    _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS);

            // update min distances and indices with closest vectors if needed.
            min_distances = _mm256_min_ps(distances, min_distances);
            min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(current_indices),
                    _mm256_castsi256_ps(min_indices),
                    comparison));

            // update current indices values. Basically, +8 to each of the
            // 8 AVX2 components.
            current_indices =
                    _mm256_add_epi32(current_indices, indices_increment);

            // scroll y forward (8 vectors 8 DIM each).
            y += 64;
        }

        // dump values and find the minimum distance / minimum index
        float min_distances_scalar[8];
        uint32_t min_indices_scalar[8];
        _mm256_storeu_ps(min_distances_scalar, min_distances);
        _mm256_storeu_si256((__m256i*)(min_indices_scalar), min_indices);

        for (size_t j = 0; j < 8; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        // process leftovers
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            const float distance = horizontal_sum(accu);

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

#else
size_t fvec_L2sqr_ny_nearest_D2(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, 2, ny);
}

size_t fvec_L2sqr_ny_nearest_D4(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, 4, ny);
}

size_t fvec_L2sqr_ny_nearest_D8(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, 8, ny);
}
#endif

size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    // optimized for a few special cases
#define DISPATCH(dval) \
    case dval:         \
        return fvec_L2sqr_ny_nearest_D##dval(distances_tmp_buffer, x, y, ny);

    switch (d) {
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        default:
            return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, d, ny);
    }
#undef DISPATCH
}

#if defined(__AVX512F__)

template <size_t DIM>
size_t fvec_L2sqr_ny_nearest_y_transposed_D(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        const size_t d_offset,
        size_t ny) {
    // This implementation does not use distances_tmp_buffer.

    // Current index being processed
    size_t i = 0;

    // Min distance and the index of the closest vector so far
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    // Process 16 vectors per loop
    const size_t ny16 = ny / 16;

    if (ny16 > 0) {
        // Track min distance and the closest vector independently
        // for each of 16 AVX-512 components.
        __m512 min_distances = _mm512_set1_ps(HUGE_VALF);
        __m512i min_indices = _mm512_set1_epi32(0);

        __m512i current_indices = _mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i indices_increment = _mm512_set1_epi32(16);

        // m[i] = (2 * x[i], ... 2 * x[i])
        __m512 m[DIM];
        for (size_t j = 0; j < DIM; j++) {
            m[j] = _mm512_set1_ps(x[j]);
            m[j] = _mm512_add_ps(m[j], m[j]);
        }

        for (; i < ny16 * 16; i += 16) {
            // Compute dot products
            const __m512 v0 = _mm512_loadu_ps(y + 0 * d_offset);
            __m512 dp = _mm512_mul_ps(m[0], v0);
            for (size_t j = 1; j < DIM; j++) {
                const __m512 vj = _mm512_loadu_ps(y + j * d_offset);
                dp = _mm512_fmadd_ps(m[j], vj, dp);
            }

            // Compute y^2 - (2 * x, y), which is sufficient for looking for the
            // lowest distance.
            // x^2 is the constant that can be avoided.
            const __m512 distances =
                    _mm512_sub_ps(_mm512_loadu_ps(y_sqlen), dp);

            // Compare the new distances to the min distances
            __mmask16 comparison =
                    _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);

            // Update min distances and indices with closest vectors if needed
            min_distances =
                    _mm512_mask_blend_ps(comparison, distances, min_distances);
            min_indices = _mm512_castps_si512(_mm512_mask_blend_ps(
                    comparison,
                    _mm512_castsi512_ps(current_indices),
                    _mm512_castsi512_ps(min_indices)));

            // Update current indices values. Basically, +16 to each of the 16
            // AVX-512 components.
            current_indices =
                    _mm512_add_epi32(current_indices, indices_increment);

            // Scroll y and y_sqlen forward.
            y += 16;
            y_sqlen += 16;
        }

        // Dump values and find the minimum distance / minimum index
        float min_distances_scalar[16];
        uint32_t min_indices_scalar[16];
        _mm512_storeu_ps(min_distances_scalar, min_distances);
        _mm512_storeu_si512((__m512i*)(min_indices_scalar), min_indices);

        for (size_t j = 0; j < 16; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        // Process leftovers
        for (; i < ny; i++) {
            float dp = 0;
            for (size_t j = 0; j < DIM; j++) {
                dp += x[j] * y[j * d_offset];
            }

            // Compute y^2 - 2 * (x, y), which is sufficient for looking for the
            // lowest distance.
            const float distance = y_sqlen[0] - 2 * dp;

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }

            y += 1;
            y_sqlen += 1;
        }
    }

    return current_min_index;
}

#elif defined(__AVX2__)

template <size_t DIM>
size_t fvec_L2sqr_ny_nearest_y_transposed_D(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        const size_t d_offset,
        size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    // current index being processed
    size_t i = 0;

    // min distance and the index of the closest vector so far
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    // process 8 vectors per loop.
    const size_t ny8 = ny / 8;

    if (ny8 > 0) {
        // track min distance and the closest vector independently
        // for each of 8 AVX2 components.
        __m256 min_distances = _mm256_set1_ps(HUGE_VALF);
        __m256i min_indices = _mm256_set1_epi32(0);

        __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i indices_increment = _mm256_set1_epi32(8);

        // m[i] = (2 * x[i], ... 2 * x[i])
        __m256 m[DIM];
        for (size_t j = 0; j < DIM; j++) {
            m[j] = _mm256_set1_ps(x[j]);
            m[j] = _mm256_add_ps(m[j], m[j]);
        }

        for (; i < ny8 * 8; i += 8) {
            // collect dim 0 for 8 D4-vectors.
            const __m256 v0 = _mm256_loadu_ps(y + 0 * d_offset);
            // compute dot products
            __m256 dp = _mm256_mul_ps(m[0], v0);

            for (size_t j = 1; j < DIM; j++) {
                // collect dim j for 8 D4-vectors.
                const __m256 vj = _mm256_loadu_ps(y + j * d_offset);
                dp = _mm256_fmadd_ps(m[j], vj, dp);
            }

            // compute y^2 - (2 * x, y), which is sufficient for looking for the
            //   lowest distance.
            // x^2 is the constant that can be avoided.
            const __m256 distances =
                    _mm256_sub_ps(_mm256_loadu_ps(y_sqlen), dp);

            // compare the new distances to the min distances
            // for each of 8 AVX2 components.
            const __m256 comparison =
                    _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS);

            // update min distances and indices with closest vectors if needed.
            min_distances =
                    _mm256_blendv_ps(distances, min_distances, comparison);
            min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(current_indices),
                    _mm256_castsi256_ps(min_indices),
                    comparison));

            // update current indices values. Basically, +8 to each of the
            // 8 AVX2 components.
            current_indices =
                    _mm256_add_epi32(current_indices, indices_increment);

            // scroll y and y_sqlen forward.
            y += 8;
            y_sqlen += 8;
        }

        // dump values and find the minimum distance / minimum index
        float min_distances_scalar[8];
        uint32_t min_indices_scalar[8];
        _mm256_storeu_ps(min_distances_scalar, min_distances);
        _mm256_storeu_si256((__m256i*)(min_indices_scalar), min_indices);

        for (size_t j = 0; j < 8; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        // process leftovers
        for (; i < ny; i++) {
            float dp = 0;
            for (size_t j = 0; j < DIM; j++) {
                dp += x[j] * y[j * d_offset];
            }

            // compute y^2 - 2 * (x, y), which is sufficient for looking for the
            //   lowest distance.
            const float distance = y_sqlen[0] - 2 * dp;

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }

            y += 1;
            y_sqlen += 1;
        }
    }

    return current_min_index;
}

#endif

size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    // optimized for a few special cases
#ifdef __AVX2__
#define DISPATCH(dval)                                     \
    case dval:                                             \
        return fvec_L2sqr_ny_nearest_y_transposed_D<dval>( \
                distances_tmp_buffer, x, y, y_sqlen, d_offset, ny);

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        default:
            return fvec_L2sqr_ny_nearest_y_transposed_ref(
                    distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
    }
#undef DISPATCH
#else
    // non-AVX2 case
    return fvec_L2sqr_ny_nearest_y_transposed_ref(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
#endif
}

#endif

#ifdef USE_AVX

float fvec_L1(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();
    __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffUL));

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = _mm256_sub_ps(mx, my);
        msum1 = _mm256_add_ps(msum1, _mm256_and_ps(signmask, a_m_b));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));
    __m128 signmask2 = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_and_ps(signmask2, a_m_b));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();
    __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffUL));

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = _mm256_sub_ps(mx, my);
        msum1 = _mm256_max_ps(msum1, _mm256_and_ps(signmask, a_m_b));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_max_ps(msum2, _mm256_extractf128_ps(msum1, 0));
    __m128 signmask2 = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
    }

    msum2 = _mm_max_ps(_mm_movehl_ps(msum2, msum2), msum2);
    msum2 = _mm_max_ps(msum2, _mm_shuffle_ps(msum2, msum2, 1));
    return _mm_cvtss_f32(msum2);
}

#elif defined(__SSE3__) // But not AVX

float fvec_L1(const float* x, const float* y, size_t d) {
    return fvec_L1_ref(x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    return fvec_Linf_ref(x, y, d);
}

#elif defined(__ARM_FEATURE_SVE)

struct ElementOpIP {
    static svfloat32_t op(svbool_t pg, svfloat32_t x, svfloat32_t y) {
        return svmul_f32_x(pg, x, y);
    }
    static svfloat32_t merge(
            svbool_t pg,
            svfloat32_t z,
            svfloat32_t x,
            svfloat32_t y) {
        return svmla_f32_x(pg, z, x, y);
    }
};

template <typename ElementOp>
void fvec_op_ny_sve_d1(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    size_t i = 0;
    for (; i + lanes4 < ny; i += lanes4) {
        svfloat32_t y0 = svld1_f32(pg, y);
        svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        svfloat32_t y3 = svld1_f32(pg, y + lanes3);
        y0 = ElementOp::op(pg, x0, y0);
        y1 = ElementOp::op(pg, x0, y1);
        y2 = ElementOp::op(pg, x0, y2);
        y3 = ElementOp::op(pg, x0, y3);
        svst1_f32(pg, dis, y0);
        svst1_f32(pg, dis + lanes, y1);
        svst1_f32(pg, dis + lanes2, y2);
        svst1_f32(pg, dis + lanes3, y3);
        y += lanes4;
        dis += lanes4;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svbool_t pg1 = svwhilelt_b32_u64(i + lanes, ny);
    const svbool_t pg2 = svwhilelt_b32_u64(i + lanes2, ny);
    const svbool_t pg3 = svwhilelt_b32_u64(i + lanes3, ny);
    svfloat32_t y0 = svld1_f32(pg0, y);
    svfloat32_t y1 = svld1_f32(pg1, y + lanes);
    svfloat32_t y2 = svld1_f32(pg2, y + lanes2);
    svfloat32_t y3 = svld1_f32(pg3, y + lanes3);
    y0 = ElementOp::op(pg0, x0, y0);
    y1 = ElementOp::op(pg1, x0, y1);
    y2 = ElementOp::op(pg2, x0, y2);
    y3 = ElementOp::op(pg3, x0, y3);
    svst1_f32(pg0, dis, y0);
    svst1_f32(pg1, dis + lanes, y1);
    svst1_f32(pg2, dis + lanes2, y2);
    svst1_f32(pg3, dis + lanes3, y3);
}

template <typename ElementOp>
void fvec_op_ny_sve_d2(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    const svfloat32_t x1 = svdup_n_f32(x[1]);
    size_t i = 0;
    for (; i + lanes2 < ny; i += lanes2) {
        const svfloat32x2_t y0 = svld2_f32(pg, y);
        const svfloat32x2_t y1 = svld2_f32(pg, y + lanes2);
        svfloat32_t y00 = svget2_f32(y0, 0);
        const svfloat32_t y01 = svget2_f32(y0, 1);
        svfloat32_t y10 = svget2_f32(y1, 0);
        const svfloat32_t y11 = svget2_f32(y1, 1);
        y00 = ElementOp::op(pg, x0, y00);
        y10 = ElementOp::op(pg, x0, y10);
        y00 = ElementOp::merge(pg, y00, x1, y01);
        y10 = ElementOp::merge(pg, y10, x1, y11);
        svst1_f32(pg, dis, y00);
        svst1_f32(pg, dis + lanes, y10);
        y += lanes4;
        dis += lanes2;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svbool_t pg1 = svwhilelt_b32_u64(i + lanes, ny);
    const svfloat32x2_t y0 = svld2_f32(pg0, y);
    const svfloat32x2_t y1 = svld2_f32(pg1, y + lanes2);
    svfloat32_t y00 = svget2_f32(y0, 0);
    const svfloat32_t y01 = svget2_f32(y0, 1);
    svfloat32_t y10 = svget2_f32(y1, 0);
    const svfloat32_t y11 = svget2_f32(y1, 1);
    y00 = ElementOp::op(pg0, x0, y00);
    y10 = ElementOp::op(pg1, x0, y10);
    y00 = ElementOp::merge(pg0, y00, x1, y01);
    y10 = ElementOp::merge(pg1, y10, x1, y11);
    svst1_f32(pg0, dis, y00);
    svst1_f32(pg1, dis + lanes, y10);
}

template <typename ElementOp>
void fvec_op_ny_sve_d4(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    const svfloat32_t x1 = svdup_n_f32(x[1]);
    const svfloat32_t x2 = svdup_n_f32(x[2]);
    const svfloat32_t x3 = svdup_n_f32(x[3]);
    size_t i = 0;
    for (; i + lanes < ny; i += lanes) {
        const svfloat32x4_t y0 = svld4_f32(pg, y);
        svfloat32_t y00 = svget4_f32(y0, 0);
        const svfloat32_t y01 = svget4_f32(y0, 1);
        svfloat32_t y02 = svget4_f32(y0, 2);
        const svfloat32_t y03 = svget4_f32(y0, 3);
        y00 = ElementOp::op(pg, x0, y00);
        y02 = ElementOp::op(pg, x2, y02);
        y00 = ElementOp::merge(pg, y00, x1, y01);
        y02 = ElementOp::merge(pg, y02, x3, y03);
        y00 = svadd_f32_x(pg, y00, y02);
        svst1_f32(pg, dis, y00);
        y += lanes4;
        dis += lanes;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svfloat32x4_t y0 = svld4_f32(pg0, y);
    svfloat32_t y00 = svget4_f32(y0, 0);
    const svfloat32_t y01 = svget4_f32(y0, 1);
    svfloat32_t y02 = svget4_f32(y0, 2);
    const svfloat32_t y03 = svget4_f32(y0, 3);
    y00 = ElementOp::op(pg0, x0, y00);
    y02 = ElementOp::op(pg0, x2, y02);
    y00 = ElementOp::merge(pg0, y00, x1, y01);
    y02 = ElementOp::merge(pg0, y02, x3, y03);
    y00 = svadd_f32_x(pg0, y00, y02);
    svst1_f32(pg0, dis, y00);
}

template <typename ElementOp>
void fvec_op_ny_sve_d8(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes4 = lanes * 4;
    const size_t lanes8 = lanes * 8;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    const svfloat32_t x1 = svdup_n_f32(x[1]);
    const svfloat32_t x2 = svdup_n_f32(x[2]);
    const svfloat32_t x3 = svdup_n_f32(x[3]);
    const svfloat32_t x4 = svdup_n_f32(x[4]);
    const svfloat32_t x5 = svdup_n_f32(x[5]);
    const svfloat32_t x6 = svdup_n_f32(x[6]);
    const svfloat32_t x7 = svdup_n_f32(x[7]);
    size_t i = 0;
    for (; i + lanes < ny; i += lanes) {
        const svfloat32x4_t ya = svld4_f32(pg, y);
        const svfloat32x4_t yb = svld4_f32(pg, y + lanes4);
        const svfloat32_t ya0 = svget4_f32(ya, 0);
        const svfloat32_t ya1 = svget4_f32(ya, 1);
        const svfloat32_t ya2 = svget4_f32(ya, 2);
        const svfloat32_t ya3 = svget4_f32(ya, 3);
        const svfloat32_t yb0 = svget4_f32(yb, 0);
        const svfloat32_t yb1 = svget4_f32(yb, 1);
        const svfloat32_t yb2 = svget4_f32(yb, 2);
        const svfloat32_t yb3 = svget4_f32(yb, 3);
        svfloat32_t y0 = svuzp1(ya0, yb0);
        const svfloat32_t y1 = svuzp1(ya1, yb1);
        svfloat32_t y2 = svuzp1(ya2, yb2);
        const svfloat32_t y3 = svuzp1(ya3, yb3);
        svfloat32_t y4 = svuzp2(ya0, yb0);
        const svfloat32_t y5 = svuzp2(ya1, yb1);
        svfloat32_t y6 = svuzp2(ya2, yb2);
        const svfloat32_t y7 = svuzp2(ya3, yb3);
        y0 = ElementOp::op(pg, x0, y0);
        y2 = ElementOp::op(pg, x2, y2);
        y4 = ElementOp::op(pg, x4, y4);
        y6 = ElementOp::op(pg, x6, y6);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        y2 = ElementOp::merge(pg, y2, x3, y3);
        y4 = ElementOp::merge(pg, y4, x5, y5);
        y6 = ElementOp::merge(pg, y6, x7, y7);
        y0 = svadd_f32_x(pg, y0, y2);
        y4 = svadd_f32_x(pg, y4, y6);
        y0 = svadd_f32_x(pg, y0, y4);
        svst1_f32(pg, dis, y0);
        y += lanes8;
        dis += lanes;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svbool_t pga = svwhilelt_b32_u64(i * 2, ny * 2);
    const svbool_t pgb = svwhilelt_b32_u64(i * 2 + lanes, ny * 2);
    const svfloat32x4_t ya = svld4_f32(pga, y);
    const svfloat32x4_t yb = svld4_f32(pgb, y + lanes4);
    const svfloat32_t ya0 = svget4_f32(ya, 0);
    const svfloat32_t ya1 = svget4_f32(ya, 1);
    const svfloat32_t ya2 = svget4_f32(ya, 2);
    const svfloat32_t ya3 = svget4_f32(ya, 3);
    const svfloat32_t yb0 = svget4_f32(yb, 0);
    const svfloat32_t yb1 = svget4_f32(yb, 1);
    const svfloat32_t yb2 = svget4_f32(yb, 2);
    const svfloat32_t yb3 = svget4_f32(yb, 3);
    svfloat32_t y0 = svuzp1(ya0, yb0);
    const svfloat32_t y1 = svuzp1(ya1, yb1);
    svfloat32_t y2 = svuzp1(ya2, yb2);
    const svfloat32_t y3 = svuzp1(ya3, yb3);
    svfloat32_t y4 = svuzp2(ya0, yb0);
    const svfloat32_t y5 = svuzp2(ya1, yb1);
    svfloat32_t y6 = svuzp2(ya2, yb2);
    const svfloat32_t y7 = svuzp2(ya3, yb3);
    y0 = ElementOp::op(pg0, x0, y0);
    y2 = ElementOp::op(pg0, x2, y2);
    y4 = ElementOp::op(pg0, x4, y4);
    y6 = ElementOp::op(pg0, x6, y6);
    y0 = ElementOp::merge(pg0, y0, x1, y1);
    y2 = ElementOp::merge(pg0, y2, x3, y3);
    y4 = ElementOp::merge(pg0, y4, x5, y5);
    y6 = ElementOp::merge(pg0, y6, x7, y7);
    y0 = svadd_f32_x(pg0, y0, y2);
    y4 = svadd_f32_x(pg0, y4, y6);
    y0 = svadd_f32_x(pg0, y0, y4);
    svst1_f32(pg0, dis, y0);
    y += lanes8;
    dis += lanes;
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes1(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    size_t i = 0;
    for (; i + 3 < ny; i += 4) {
        svfloat32_t y0 = svld1_f32(pg, y);
        svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        svfloat32_t y3 = svld1_f32(pg, y + lanes3);
        y += lanes4;
        y0 = ElementOp::op(pg, x0, y0);
        y1 = ElementOp::op(pg, x0, y1);
        y2 = ElementOp::op(pg, x0, y2);
        y3 = ElementOp::op(pg, x0, y3);
        dis[i] = svaddv_f32(pg, y0);
        dis[i + 1] = svaddv_f32(pg, y1);
        dis[i + 2] = svaddv_f32(pg, y2);
        dis[i + 3] = svaddv_f32(pg, y3);
    }
    for (; i < ny; ++i) {
        svfloat32_t y0 = svld1_f32(pg, y);
        y += lanes;
        y0 = ElementOp::op(pg, x0, y0);
        dis[i] = svaddv_f32(pg, y0);
    }
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes2(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    const svfloat32_t x1 = svld1_f32(pg, x + lanes);
    size_t i = 0;
    for (; i + 1 < ny; i += 2) {
        svfloat32_t y00 = svld1_f32(pg, y);
        const svfloat32_t y01 = svld1_f32(pg, y + lanes);
        svfloat32_t y10 = svld1_f32(pg, y + lanes2);
        const svfloat32_t y11 = svld1_f32(pg, y + lanes3);
        y += lanes4;
        y00 = ElementOp::op(pg, x0, y00);
        y10 = ElementOp::op(pg, x0, y10);
        y00 = ElementOp::merge(pg, y00, x1, y01);
        y10 = ElementOp::merge(pg, y10, x1, y11);
        dis[i] = svaddv_f32(pg, y00);
        dis[i + 1] = svaddv_f32(pg, y10);
    }
    if (i < ny) {
        svfloat32_t y0 = svld1_f32(pg, y);
        const svfloat32_t y1 = svld1_f32(pg, y + lanes);
        y0 = ElementOp::op(pg, x0, y0);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        dis[i] = svaddv_f32(pg, y0);
    }
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes3(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    const svfloat32_t x1 = svld1_f32(pg, x + lanes);
    const svfloat32_t x2 = svld1_f32(pg, x + lanes2);
    for (size_t i = 0; i < ny; ++i) {
        svfloat32_t y0 = svld1_f32(pg, y);
        const svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        y += lanes3;
        y0 = ElementOp::op(pg, x0, y0);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        y0 = ElementOp::merge(pg, y0, x2, y2);
        dis[i] = svaddv_f32(pg, y0);
    }
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes4(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    const svfloat32_t x1 = svld1_f32(pg, x + lanes);
    const svfloat32_t x2 = svld1_f32(pg, x + lanes2);
    const svfloat32_t x3 = svld1_f32(pg, x + lanes3);
    for (size_t i = 0; i < ny; ++i) {
        svfloat32_t y0 = svld1_f32(pg, y);
        const svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        const svfloat32_t y3 = svld1_f32(pg, y + lanes3);
        y += lanes4;
        y0 = ElementOp::op(pg, x0, y0);
        y2 = ElementOp::op(pg, x2, y2);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        y2 = ElementOp::merge(pg, y2, x3, y3);
        y0 = svadd_f32_x(pg, y0, y2);
        dis[i] = svaddv_f32(pg, y0);
    }
}

void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
}

void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_y_transposed_ref(dis, x, y, y_sqlen, d, d_offset, ny);
}

size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed_ref(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

float fvec_L1(const float* x, const float* y, size_t d) {
    return fvec_L1_ref(x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    return fvec_Linf_ref(x, y, d);
}

void fvec_inner_products_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    const size_t lanes = svcntw();
    switch (d) {
        case 1:
            fvec_op_ny_sve_d1<ElementOpIP>(dis, x, y, ny);
            break;
        case 2:
            fvec_op_ny_sve_d2<ElementOpIP>(dis, x, y, ny);
            break;
        case 4:
            fvec_op_ny_sve_d4<ElementOpIP>(dis, x, y, ny);
            break;
        case 8:
            fvec_op_ny_sve_d8<ElementOpIP>(dis, x, y, ny);
            break;
        default:
            if (d == lanes)
                fvec_op_ny_sve_lanes1<ElementOpIP>(dis, x, y, ny);
            else if (d == lanes * 2)
                fvec_op_ny_sve_lanes2<ElementOpIP>(dis, x, y, ny);
            else if (d == lanes * 3)
                fvec_op_ny_sve_lanes3<ElementOpIP>(dis, x, y, ny);
            else if (d == lanes * 4)
                fvec_op_ny_sve_lanes4<ElementOpIP>(dis, x, y, ny);
            else
                fvec_inner_products_ny_ref(dis, x, y, d, ny);
            break;
    }
}

#elif defined(__aarch64__)

// not optimized for ARM
void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
}

void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_y_transposed_ref(dis, x, y, y_sqlen, d, d_offset, ny);
}

size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed_ref(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

float fvec_L1(const float* x, const float* y, size_t d) {
    return fvec_L1_ref(x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    return fvec_Linf_ref(x, y, d);
}

void fvec_inner_products_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny_ref(dis, x, y, d, ny);
}

#else
// scalar implementation

float fvec_L1(const float* x, const float* y, size_t d) {
    return fvec_L1_ref(x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    return fvec_Linf_ref(x, y, d);
}

void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
}

void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_y_transposed_ref(dis, x, y, y_sqlen, d, d_offset, ny);
}

size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed_ref(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

void fvec_inner_products_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny_ref(dis, x, y, d, ny);
}

#endif

/***************************************************************************
 * heavily optimized table computations
 ***************************************************************************/

[[maybe_unused]] static inline void fvec_madd_ref(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + bf * b[i];
}

#if defined(__AVX512F__)

static inline void fvec_madd_avx512(
        const size_t n,
        const float* __restrict a,
        const float bf,
        const float* __restrict b,
        float* __restrict c) {
    const size_t n16 = n / 16;
    const size_t n_for_masking = n % 16;

    const __m512 bfmm = _mm512_set1_ps(bf);

    size_t idx = 0;
    for (idx = 0; idx < n16 * 16; idx += 16) {
        const __m512 ax = _mm512_loadu_ps(a + idx);
        const __m512 bx = _mm512_loadu_ps(b + idx);
        const __m512 abmul = _mm512_fmadd_ps(bfmm, bx, ax);
        _mm512_storeu_ps(c + idx, abmul);
    }

    if (n_for_masking > 0) {
        const __mmask16 mask = (1 << n_for_masking) - 1;

        const __m512 ax = _mm512_maskz_loadu_ps(mask, a + idx);
        const __m512 bx = _mm512_maskz_loadu_ps(mask, b + idx);
        const __m512 abmul = _mm512_fmadd_ps(bfmm, bx, ax);
        _mm512_mask_storeu_ps(c + idx, mask, abmul);
    }
}

#elif defined(__AVX2__)

static inline void fvec_madd_avx2(
        const size_t n,
        const float* __restrict a,
        const float bf,
        const float* __restrict b,
        float* __restrict c) {
    //
    const size_t n8 = n / 8;
    const size_t n_for_masking = n % 8;

    const __m256 bfmm = _mm256_set1_ps(bf);

    size_t idx = 0;
    for (idx = 0; idx < n8 * 8; idx += 8) {
        const __m256 ax = _mm256_loadu_ps(a + idx);
        const __m256 bx = _mm256_loadu_ps(b + idx);
        const __m256 abmul = _mm256_fmadd_ps(bfmm, bx, ax);
        _mm256_storeu_ps(c + idx, abmul);
    }

    if (n_for_masking > 0) {
        __m256i mask;
        switch (n_for_masking) {
            case 1:
                mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
                break;
            case 2:
                mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
                break;
            case 3:
                mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
                break;
            case 4:
                mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
                break;
            case 5:
                mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
                break;
            case 6:
                mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
                break;
            case 7:
                mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
                break;
        }

        const __m256 ax = _mm256_maskload_ps(a + idx, mask);
        const __m256 bx = _mm256_maskload_ps(b + idx, mask);
        const __m256 abmul = _mm256_fmadd_ps(bfmm, bx, ax);
        _mm256_maskstore_ps(c + idx, mask, abmul);
    }
}

#endif

#ifdef __SSE3__

[[maybe_unused]] static inline void fvec_madd_sse(
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

void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c) {
#ifdef __AVX512F__
    fvec_madd_avx512(n, a, bf, b, c);
#elif __AVX2__
    fvec_madd_avx2(n, a, bf, b, c);
#else
    if ((n & 3) == 0 && ((((long)a) | ((long)b) | ((long)c)) & 15) == 0)
        fvec_madd_sse(n, a, bf, b, c);
    else
        fvec_madd_ref(n, a, bf, b, c);
#endif
}

#elif defined(__ARM_FEATURE_SVE)

void fvec_madd(
        const size_t n,
        const float* __restrict a,
        const float bf,
        const float* __restrict b,
        float* __restrict c) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    size_t i = 0;
    for (; i + lanes4 < n; i += lanes4) {
        const auto mask = svptrue_b32();
        const auto ai0 = svld1_f32(mask, a + i);
        const auto ai1 = svld1_f32(mask, a + i + lanes);
        const auto ai2 = svld1_f32(mask, a + i + lanes2);
        const auto ai3 = svld1_f32(mask, a + i + lanes3);
        const auto bi0 = svld1_f32(mask, b + i);
        const auto bi1 = svld1_f32(mask, b + i + lanes);
        const auto bi2 = svld1_f32(mask, b + i + lanes2);
        const auto bi3 = svld1_f32(mask, b + i + lanes3);
        const auto ci0 = svmla_n_f32_x(mask, ai0, bi0, bf);
        const auto ci1 = svmla_n_f32_x(mask, ai1, bi1, bf);
        const auto ci2 = svmla_n_f32_x(mask, ai2, bi2, bf);
        const auto ci3 = svmla_n_f32_x(mask, ai3, bi3, bf);
        svst1_f32(mask, c + i, ci0);
        svst1_f32(mask, c + i + lanes, ci1);
        svst1_f32(mask, c + i + lanes2, ci2);
        svst1_f32(mask, c + i + lanes3, ci3);
    }
    const auto mask0 = svwhilelt_b32_u64(i, n);
    const auto mask1 = svwhilelt_b32_u64(i + lanes, n);
    const auto mask2 = svwhilelt_b32_u64(i + lanes2, n);
    const auto mask3 = svwhilelt_b32_u64(i + lanes3, n);
    const auto ai0 = svld1_f32(mask0, a + i);
    const auto ai1 = svld1_f32(mask1, a + i + lanes);
    const auto ai2 = svld1_f32(mask2, a + i + lanes2);
    const auto ai3 = svld1_f32(mask3, a + i + lanes3);
    const auto bi0 = svld1_f32(mask0, b + i);
    const auto bi1 = svld1_f32(mask1, b + i + lanes);
    const auto bi2 = svld1_f32(mask2, b + i + lanes2);
    const auto bi3 = svld1_f32(mask3, b + i + lanes3);
    const auto ci0 = svmla_n_f32_x(mask0, ai0, bi0, bf);
    const auto ci1 = svmla_n_f32_x(mask1, ai1, bi1, bf);
    const auto ci2 = svmla_n_f32_x(mask2, ai2, bi2, bf);
    const auto ci3 = svmla_n_f32_x(mask3, ai3, bi3, bf);
    svst1_f32(mask0, c + i, ci0);
    svst1_f32(mask1, c + i + lanes, ci1);
    svst1_f32(mask2, c + i + lanes2, ci2);
    svst1_f32(mask3, c + i + lanes3, ci3);
}

#elif defined(__aarch64__)

void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c) {
    const size_t n_simd = n - (n & 3);
    const float32x4_t bfv = vdupq_n_f32(bf);
    size_t i;
    for (i = 0; i < n_simd; i += 4) {
        const float32x4_t ai = vld1q_f32(a + i);
        const float32x4_t bi = vld1q_f32(b + i);
        const float32x4_t ci = vfmaq_f32(ai, bfv, bi);
        vst1q_f32(c + i, ci);
    }
    for (; i < n; ++i)
        c[i] = a[i] + bf * b[i];
}

#else

void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c) {
    fvec_madd_ref(n, a, bf, b, c);
}

#endif

static inline int fvec_madd_and_argmin_ref(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    float vmin = 1e20;
    int imin = -1;

    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = i;
        }
    }
    return imin;
}

#ifdef __SSE3__

static inline int fvec_madd_and_argmin_sse(
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

int fvec_madd_and_argmin(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    if ((n & 3) == 0 && ((((long)a) | ((long)b) | ((long)c)) & 15) == 0)
        return fvec_madd_and_argmin_sse(n, a, bf, b, c);
    else
        return fvec_madd_and_argmin_ref(n, a, bf, b, c);
}

#elif defined(__aarch64__)

int fvec_madd_and_argmin(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    float32x4_t vminv = vdupq_n_f32(1e20);
    uint32x4_t iminv = vdupq_n_u32(static_cast<uint32_t>(-1));
    size_t i;
    {
        const size_t n_simd = n - (n & 3);
        const uint32_t iota[] = {0, 1, 2, 3};
        uint32x4_t iv = vld1q_u32(iota);
        const uint32x4_t incv = vdupq_n_u32(4);
        const float32x4_t bfv = vdupq_n_f32(bf);
        for (i = 0; i < n_simd; i += 4) {
            const float32x4_t ai = vld1q_f32(a + i);
            const float32x4_t bi = vld1q_f32(b + i);
            const float32x4_t ci = vfmaq_f32(ai, bfv, bi);
            vst1q_f32(c + i, ci);
            const uint32x4_t less_than = vcltq_f32(ci, vminv);
            vminv = vminq_f32(ci, vminv);
            iminv = vorrq_u32(
                    vandq_u32(less_than, iv),
                    vandq_u32(vmvnq_u32(less_than), iminv));
            iv = vaddq_u32(iv, incv);
        }
    }
    float vmin = vminvq_f32(vminv);
    uint32_t imin;
    {
        const float32x4_t vminy = vdupq_n_f32(vmin);
        const uint32x4_t equals = vceqq_f32(vminv, vminy);
        imin = vminvq_u32(vorrq_u32(
                vandq_u32(equals, iminv),
                vandq_u32(
                        vmvnq_u32(equals),
                        vdupq_n_u32(std::numeric_limits<uint32_t>::max()))));
    }
    for (; i < n; ++i) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = static_cast<uint32_t>(i);
        }
    }
    return static_cast<int>(imin);
}

#else

int fvec_madd_and_argmin(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    return fvec_madd_and_argmin_ref(n, a, bf, b, c);
}

#endif

/***************************************************************************
 * PQ tables computations
 ***************************************************************************/

namespace {

/// compute the IP for dsub = 2 for 8 centroids and 4 sub-vectors at a time
template <bool is_inner_product>
void pq2_8cents_table(
        const simd8float32 centroids[8],
        const simd8float32 x,
        float* out,
        size_t ldo,
        size_t nout = 4) {
    simd8float32 ips[4];

    for (int i = 0; i < 4; i++) {
        simd8float32 p1, p2;
        if (is_inner_product) {
            p1 = x * centroids[2 * i];
            p2 = x * centroids[2 * i + 1];
        } else {
            p1 = (x - centroids[2 * i]);
            p1 = p1 * p1;
            p2 = (x - centroids[2 * i + 1]);
            p2 = p2 * p2;
        }
        ips[i] = hadd(p1, p2);
    }

    simd8float32 ip02a = geteven(ips[0], ips[1]);
    simd8float32 ip02b = geteven(ips[2], ips[3]);
    simd8float32 ip0 = getlow128(ip02a, ip02b);
    simd8float32 ip2 = gethigh128(ip02a, ip02b);

    simd8float32 ip13a = getodd(ips[0], ips[1]);
    simd8float32 ip13b = getodd(ips[2], ips[3]);
    simd8float32 ip1 = getlow128(ip13a, ip13b);
    simd8float32 ip3 = gethigh128(ip13a, ip13b);

    switch (nout) {
        case 4:
            ip3.storeu(out + 3 * ldo);
            [[fallthrough]];
        case 3:
            ip2.storeu(out + 2 * ldo);
            [[fallthrough]];
        case 2:
            ip1.storeu(out + 1 * ldo);
            [[fallthrough]];
        case 1:
            ip0.storeu(out);
    }
}

simd8float32 load_simd8float32_partial(const float* x, int n) {
    ALIGNED(32) float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float* wp = tmp;
    for (int i = 0; i < n; i++) {
        *wp++ = *x++;
    }
    return simd8float32(tmp);
}

} // anonymous namespace

void compute_PQ_dis_tables_dsub2(
        size_t d,
        size_t ksub,
        const float* all_centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables) {
    size_t M = d / 2;
    FAISS_THROW_IF_NOT(ksub % 8 == 0);

    for (size_t m0 = 0; m0 < M; m0 += 4) {
        int m1 = std::min(M, m0 + 4);
        for (int k0 = 0; k0 < ksub; k0 += 8) {
            simd8float32 centroids[8];
            for (int k = 0; k < 8; k++) {
                ALIGNED(32) float centroid[8];
                size_t wp = 0;
                size_t rp = (m0 * ksub + k + k0) * 2;
                for (int m = m0; m < m1; m++) {
                    centroid[wp++] = all_centroids[rp];
                    centroid[wp++] = all_centroids[rp + 1];
                    rp += 2 * ksub;
                }
                centroids[k] = simd8float32(centroid);
            }
            for (size_t i = 0; i < nx; i++) {
                simd8float32 xi;
                if (m1 == m0 + 4) {
                    xi.loadu(x + i * d + m0 * 2);
                } else {
                    xi = load_simd8float32_partial(
                            x + i * d + m0 * 2, 2 * (m1 - m0));
                }

                if (is_inner_product) {
                    pq2_8cents_table<true>(
                            centroids,
                            xi,
                            dis_tables + (i * M + m0) * ksub + k0,
                            ksub,
                            m1 - m0);
                } else {
                    pq2_8cents_table<false>(
                            centroids,
                            xi,
                            dis_tables + (i * M + m0) * ksub + k0,
                            ksub,
                            m1 - m0);
                }
            }
        }
    }
}

/*********************************************************
 * Vector to vector functions
 *********************************************************/

void fvec_sub(size_t d, const float* a, const float* b, float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32 ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai - bi;
        ci.storeu(c + i);
    }
    // finish non-multiple of 8 remainder
    for (; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

void fvec_add(size_t d, const float* a, const float* b, float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32 ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai + bi;
        ci.storeu(c + i);
    }
    // finish non-multiple of 8 remainder
    for (; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

void fvec_add(size_t d, const float* a, float b, float* c) {
    size_t i;
    simd8float32 bv(b);
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32 ci, ai, bi;
        ai.loadu(a + i);
        ci = ai + bv;
        ci.storeu(c + i);
    }
    // finish non-multiple of 8 remainder
    for (; i < d; i++) {
        c[i] = a[i] + b;
    }
}

} // namespace faiss
