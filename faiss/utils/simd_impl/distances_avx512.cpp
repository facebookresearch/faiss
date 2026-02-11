/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#include <immintrin.h>

#define AUTOVEC_LEVEL SIMDLevel::AVX512
#include <faiss/utils/simd_impl/distances_autovec-inl.h>
#include <faiss/utils/simd_impl/distances_sse-inl.h>
#include <faiss/utils/transpose/transpose-avx512-inl.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::AVX512>(
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

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::AVX512>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    // optimized for a few special cases
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
            return fvec_L2sqr_ny_transposed<SIMDLevel::NONE>(
                    dis, x, y, y_sqlen, d, d_offset, ny);
    }
#undef DISPATCH
}

struct AVX512ElementOpIP : public ElementOpIP {
    using ElementOpIP::op;
    static __m512 op(__m512 x, __m512 y) {
        return _mm512_mul_ps(x, y);
    }
    static __m256 op(__m256 x, __m256 y) {
        return _mm256_mul_ps(x, y);
    }
};

struct AVX512ElementOpL2 : public ElementOpL2 {
    using ElementOpL2::op;
    static __m512 op(__m512 x, __m512 y) {
        __m512 tmp = _mm512_sub_ps(x, y);
        return _mm512_mul_ps(tmp, tmp);
    }
    static __m256 op(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(x, y);
        return _mm256_mul_ps(tmp, tmp);
    }
};

/// helper function for AVX512
inline float horizontal_sum(const __m512 v) {
    // performs better than adding the high and low parts
    return _mm512_reduce_add_ps(v);
}

inline float horizontal_sum(const __m256 v) {
    // add high and low parts
    const __m128 v0 =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    // perform horizontal sum on v0
    return horizontal_sum(v0);
}

template <>
void fvec_op_ny_D2<AVX512ElementOpIP>(
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
void fvec_op_ny_D2<AVX512ElementOpL2>(
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

template <>
void fvec_op_ny_D4<AVX512ElementOpIP>(
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
            __m128 accu = AVX512ElementOpIP::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D4<AVX512ElementOpL2>(
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
            __m128 accu = AVX512ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D8<AVX512ElementOpIP>(
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
            __m256 accu = AVX512ElementOpIP::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D8<AVX512ElementOpL2>(
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
            __m256 accu = AVX512ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::AVX512>(
        float* ip, /* output inner product */
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny_ref<AVX512ElementOpIP>(ip, x, y, d, ny);
}

template <>
void fvec_L2sqr_ny<SIMDLevel::AVX512>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref<AVX512ElementOpL2>(dis, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_D2<SIMDLevel::AVX512>(
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

template <>
size_t fvec_L2sqr_ny_nearest_D4<SIMDLevel::AVX512>(
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

template <>
size_t fvec_L2sqr_ny_nearest_D8<SIMDLevel::AVX512>(
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
            __m256 accu = AVX512ElementOpL2::op(x0, _mm256_loadu_ps(y));
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

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::AVX512>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_x86<SIMDLevel::AVX512>(
            distances_tmp_buffer,
            x,
            y,
            d,
            ny,
            &fvec_L2sqr_ny_nearest_D2<SIMDLevel::AVX512>,
            &fvec_L2sqr_ny_nearest_D4<SIMDLevel::AVX512>,
            &fvec_L2sqr_ny_nearest_D8<SIMDLevel::AVX512>);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::AVX512>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

// TODO: Following functions are not used in the current codebase. Check AVX2 ,
// respective implementation has been used
template <size_t DIM>
size_t fvec_L2sqr_ny_nearest_y_transposed_D(
        float* /* distances_tmp_buffer */,
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

template <>
int fvec_madd_and_argmin<SIMDLevel::AVX512>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    return fvec_madd_and_argmin_sse(n, a, bf, b, c);
}

} // namespace faiss
