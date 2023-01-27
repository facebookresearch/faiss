/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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

float fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

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

float fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++)
        res += x[i] * y[i];
    return res;
}

float fvec_norm_L2sqr_ref(const float* x, size_t d) {
    size_t i;
    double res = 0;
    for (i = 0; i < d; i++)
        res += x[i] * x[i];
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
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

float fvec_norm_L2sqr(const float* x, size_t d) {
    __m128 mx;
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        mx = _mm_loadu_ps(x);
        x += 4;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));
        d -= 4;
    }

    mx = masked_read(d, x);
    msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

namespace {

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
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

#ifdef __AVX2__

// Specialized versions for AVX2 for any CPUs that support gather/scatter.
// Todo: implement fvec_op_ny_Dxxx in the same way.

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
        _mm_prefetch(y, _MM_HINT_NTA);
        _mm_prefetch(y + 16, _MM_HINT_NTA);

        // m0 = (x[0], x[0], x[0], x[0], x[0], x[0], x[0], x[0])
        const __m256 m0 = _mm256_set1_ps(x[0]);
        // m1 = (x[1], x[1], x[1], x[1], x[1], x[1], x[1], x[1])
        const __m256 m1 = _mm256_set1_ps(x[1]);
        // m2 = (x[2], x[2], x[2], x[2], x[2], x[2], x[2], x[2])
        const __m256 m2 = _mm256_set1_ps(x[2]);
        // m3 = (x[3], x[3], x[3], x[3], x[3], x[3], x[3], x[3])
        const __m256 m3 = _mm256_set1_ps(x[3]);

        const __m256i indices0 =
                _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);

        for (i = 0; i < ny8 * 8; i += 8) {
            _mm_prefetch(y + 32, _MM_HINT_NTA);
            _mm_prefetch(y + 48, _MM_HINT_NTA);

            // collect dim 0 for 8 D4-vectors.
            // v0 = (y[(i * 8 + 0) * 4 + 0], ..., y[(i * 8 + 7) * 4 + 0])
            const __m256 v0 = _mm256_i32gather_ps(y, indices0, 1);
            // collect dim 1 for 8 D4-vectors.
            // v1 = (y[(i * 8 + 0) * 4 + 1], ..., y[(i * 8 + 7) * 4 + 1])
            const __m256 v1 = _mm256_i32gather_ps(y + 1, indices0, 1);
            // collect dim 2 for 8 D4-vectors.
            // v2 = (y[(i * 8 + 0) * 4 + 2], ..., y[(i * 8 + 7) * 4 + 2])
            const __m256 v2 = _mm256_i32gather_ps(y + 2, indices0, 1);
            // collect dim 3 for 8 D4-vectors.
            // v3 = (y[(i * 8 + 0) * 4 + 3], ..., y[(i * 8 + 7) * 4 + 3])
            const __m256 v3 = _mm256_i32gather_ps(y + 3, indices0, 1);

            // compute distances
            __m256 distances = _mm256_mul_ps(m0, v0);
            distances = _mm256_fmadd_ps(m1, v1, distances);
            distances = _mm256_fmadd_ps(m2, v2, distances);
            distances = _mm256_fmadd_ps(m3, v3, distances);

            //   distances[0] = (x[0] * y[(i * 8 + 0) * 4 + 0]) +
            //                  (x[1] * y[(i * 8 + 0) * 4 + 1]) +
            //                  (x[2] * y[(i * 8 + 0) * 4 + 2]) +
            //                  (x[3] * y[(i * 8 + 0) * 4 + 3])
            //   ...
            //   distances[7] = (x[0] * y[(i * 8 + 7) * 4 + 0]) +
            //                  (x[1] * y[(i * 8 + 7) * 4 + 1]) +
            //                  (x[2] * y[(i * 8 + 7) * 4 + 2]) +
            //                  (x[3] * y[(i * 8 + 7) * 4 + 3])
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
            accu = _mm_hadd_ps(accu, accu);
            accu = _mm_hadd_ps(accu, accu);
            dis[i] = _mm_cvtss_f32(accu);
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
        _mm_prefetch(y, _MM_HINT_NTA);
        _mm_prefetch(y + 16, _MM_HINT_NTA);

        // m0 = (x[0], x[0], x[0], x[0], x[0], x[0], x[0], x[0])
        const __m256 m0 = _mm256_set1_ps(x[0]);
        // m1 = (x[1], x[1], x[1], x[1], x[1], x[1], x[1], x[1])
        const __m256 m1 = _mm256_set1_ps(x[1]);
        // m2 = (x[2], x[2], x[2], x[2], x[2], x[2], x[2], x[2])
        const __m256 m2 = _mm256_set1_ps(x[2]);
        // m3 = (x[3], x[3], x[3], x[3], x[3], x[3], x[3], x[3])
        const __m256 m3 = _mm256_set1_ps(x[3]);

        const __m256i indices0 =
                _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);

        for (i = 0; i < ny8 * 8; i += 8) {
            _mm_prefetch(y + 32, _MM_HINT_NTA);
            _mm_prefetch(y + 48, _MM_HINT_NTA);

            // collect dim 0 for 8 D4-vectors.
            // v0 = (y[(i * 8 + 0) * 4 + 0], ..., y[(i * 8 + 7) * 4 + 0])
            const __m256 v0 = _mm256_i32gather_ps(y, indices0, 1);
            // collect dim 1 for 8 D4-vectors.
            // v1 = (y[(i * 8 + 0) * 4 + 1], ..., y[(i * 8 + 7) * 4 + 1])
            const __m256 v1 = _mm256_i32gather_ps(y + 1, indices0, 1);
            // collect dim 2 for 8 D4-vectors.
            // v2 = (y[(i * 8 + 0) * 4 + 2], ..., y[(i * 8 + 7) * 4 + 2])
            const __m256 v2 = _mm256_i32gather_ps(y + 2, indices0, 1);
            // collect dim 3 for 8 D4-vectors.
            // v3 = (y[(i * 8 + 0) * 4 + 3], ..., y[(i * 8 + 7) * 4 + 3])
            const __m256 v3 = _mm256_i32gather_ps(y + 3, indices0, 1);

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

            //   distances[0] = (x[0] - y[(i * 8 + 0) * 4 + 0]) ^ 2 +
            //                  (x[1] - y[(i * 8 + 0) * 4 + 1]) ^ 2 +
            //                  (x[2] - y[(i * 8 + 0) * 4 + 2]) ^ 2 +
            //                  (x[3] - y[(i * 8 + 0) * 4 + 3])
            //   ...
            //   distances[7] = (x[0] - y[(i * 8 + 7) * 4 + 0]) ^ 2 +
            //                  (x[1] - y[(i * 8 + 7) * 4 + 1]) ^ 2 +
            //                  (x[2] - y[(i * 8 + 7) * 4 + 2]) ^ 2 +
            //                  (x[3] - y[(i * 8 + 7) * 4 + 3])
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
            accu = _mm_hadd_ps(accu, accu);
            accu = _mm_hadd_ps(accu, accu);
            dis[i] = _mm_cvtss_f32(accu);
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
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
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

#ifdef __AVX2__
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

        //
        _mm_prefetch(y, _MM_HINT_NTA);
        _mm_prefetch(y + 16, _MM_HINT_NTA);

        // m0 = (x[0], x[0], x[0], x[0], x[0], x[0], x[0], x[0])
        const __m256 m0 = _mm256_set1_ps(x[0]);
        // m1 = (x[1], x[1], x[1], x[1], x[1], x[1], x[1], x[1])
        const __m256 m1 = _mm256_set1_ps(x[1]);
        // m2 = (x[2], x[2], x[2], x[2], x[2], x[2], x[2], x[2])
        const __m256 m2 = _mm256_set1_ps(x[2]);
        // m3 = (x[3], x[3], x[3], x[3], x[3], x[3], x[3], x[3])
        const __m256 m3 = _mm256_set1_ps(x[3]);

        const __m256i indices0 =
                _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);

        for (; i < ny8 * 8; i += 8) {
            _mm_prefetch(y + 32, _MM_HINT_NTA);
            _mm_prefetch(y + 48, _MM_HINT_NTA);

            // collect dim 0 for 8 D4-vectors.
            // v0 = (y[(i * 8 + 0) * 4 + 0], ..., y[(i * 8 + 7) * 4 + 0])
            const __m256 v0 = _mm256_i32gather_ps(y, indices0, 1);
            // collect dim 1 for 8 D4-vectors.
            // v1 = (y[(i * 8 + 0) * 4 + 1], ..., y[(i * 8 + 7) * 4 + 1])
            const __m256 v1 = _mm256_i32gather_ps(y + 1, indices0, 1);
            // collect dim 2 for 8 D4-vectors.
            // v2 = (y[(i * 8 + 0) * 4 + 2], ..., y[(i * 8 + 7) * 4 + 2])
            const __m256 v2 = _mm256_i32gather_ps(y + 2, indices0, 1);
            // collect dim 3 for 8 D4-vectors.
            // v3 = (y[(i * 8 + 0) * 4 + 3], ..., y[(i * 8 + 7) * 4 + 3])
            const __m256 v3 = _mm256_i32gather_ps(y + 3, indices0, 1);

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

            //   distances[0] = (x[0] - y[(i * 8 + 0) * 4 + 0]) ^ 2 +
            //                  (x[1] - y[(i * 8 + 0) * 4 + 1]) ^ 2 +
            //                  (x[2] - y[(i * 8 + 0) * 4 + 2]) ^ 2 +
            //                  (x[3] - y[(i * 8 + 0) * 4 + 3])
            //   ...
            //   distances[7] = (x[0] - y[(i * 8 + 7) * 4 + 0]) ^ 2 +
            //                  (x[1] - y[(i * 8 + 7) * 4 + 1]) ^ 2 +
            //                  (x[2] - y[(i * 8 + 7) * 4 + 2]) ^ 2 +
            //                  (x[3] - y[(i * 8 + 7) * 4 + 3])

            // compare the new distances to the min distances
            // for each of 8 AVX2 components.
            __m256 comparison =
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
            accu = _mm_hadd_ps(accu, accu);
            accu = _mm_hadd_ps(accu, accu);

            const auto distance = _mm_cvtss_f32(accu);

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}
#else
size_t fvec_L2sqr_ny_nearest_D4(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, 4, ny);
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
        DISPATCH(4)
        default:
            return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, d, ny);
    }
#undef DISPATCH
}

#ifdef __AVX2__
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

// reads 0 <= d < 8 floats as __m256
static inline __m256 masked_read_8(int d, const float* x) {
    assert(0 <= d && d < 8);
    if (d < 4) {
        __m256 res = _mm256_setzero_ps();
        res = _mm256_insertf128_ps(res, masked_read(d, x), 0);
        return res;
    } else {
        __m256 res = _mm256_setzero_ps();
        res = _mm256_insertf128_ps(res, _mm_loadu_ps(x), 0);
        res = _mm256_insertf128_ps(res, masked_read(d - 4, x + 4), 1);
        return res;
    }
}

float fvec_inner_product(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1, a_m_b1));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

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

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

float fvec_inner_product(const float* x, const float* y, size_t d) {
    __m128 mx, my;
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        mx = _mm_loadu_ps(x);
        x += 4;
        my = _mm_loadu_ps(y);
        y += 4;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
        d -= 4;
    }

    // add the last 1, 2, or 3 values
    mx = masked_read(d, x);
    my = masked_read(d, y);
    __m128 prod = _mm_mul_ps(mx, my);

    msum1 = _mm_add_ps(msum1, prod);

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

#elif defined(__aarch64__)

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    float32x4_t accux4 = vdupq_n_f32(0);
    const size_t d_simd = d - (d & 3);
    size_t i;
    for (i = 0; i < d_simd; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        float32x4_t sq = vsubq_f32(xi, yi);
        accux4 = vfmaq_f32(accux4, sq, sq);
    }
    float32_t accux1 = vaddvq_f32(accux4);
    for (; i < d; ++i) {
        float32_t xi = x[i];
        float32_t yi = y[i];
        float32_t sq = xi - yi;
        accux1 += sq * sq;
    }
    return accux1;
}

float fvec_inner_product(const float* x, const float* y, size_t d) {
    float32x4_t accux4 = vdupq_n_f32(0);
    const size_t d_simd = d - (d & 3);
    size_t i;
    for (i = 0; i < d_simd; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        accux4 = vfmaq_f32(accux4, xi, yi);
    }
    float32_t accux1 = vaddvq_f32(accux4);
    for (; i < d; ++i) {
        float32_t xi = x[i];
        float32_t yi = y[i];
        accux1 += xi * yi;
    }
    return accux1;
}

float fvec_norm_L2sqr(const float* x, size_t d) {
    float32x4_t accux4 = vdupq_n_f32(0);
    const size_t d_simd = d - (d & 3);
    size_t i;
    for (i = 0; i < d_simd; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        accux4 = vfmaq_f32(accux4, xi, xi);
    }
    float32_t accux1 = vaddvq_f32(accux4);
    for (; i < d; ++i) {
        float32_t xi = x[i];
        accux1 += xi * xi;
    }
    return accux1;
}

// not optimized for ARM
void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
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

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    return fvec_L2sqr_ref(x, y, d);
}

float fvec_L1(const float* x, const float* y, size_t d) {
    return fvec_L1_ref(x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    return fvec_Linf_ref(x, y, d);
}

float fvec_inner_product(const float* x, const float* y, size_t d) {
    return fvec_inner_product_ref(x, y, d);
}

float fvec_norm_L2sqr(const float* x, size_t d) {
    return fvec_norm_L2sqr_ref(x, d);
}

void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
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

static inline void fvec_madd_ref(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + bf * b[i];
}

#ifdef __AVX2__
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

static inline void fvec_madd_sse(
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
#ifdef __AVX2__
    fvec_madd_avx2(n, a, bf, b, c);
#else
    if ((n & 3) == 0 && ((((long)a) | ((long)b) | ((long)c)) & 15) == 0)
        fvec_madd_sse(n, a, bf, b, c);
    else
        fvec_madd_ref(n, a, bf, b, c);
#endif
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
        case 3:
            ip2.storeu(out + 2 * ldo);
        case 2:
            ip1.storeu(out + 1 * ldo);
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
