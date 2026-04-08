/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#include <immintrin.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances_fused/distances_fused.h>
#include <faiss/utils/simd_impl/exhaustive_L2sqr_blas_cmax.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

#define THE_SIMD_LEVEL SIMDLevel::AVX2
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/distances_autovec-inl.h>

// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/distances_simdlib256.h>

// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/distances_sse-inl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/transpose/transpose-avx2-inl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::AVX2>(
        const size_t n,
        const float* a,
        const float bf,
        const float* b,
        float* c) {
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

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::AVX2>(
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

namespace {

struct AVX2ElementOpIP : public ElementOpIP {
    using ElementOpIP::op;
    static __m256 op(__m256 x, __m256 y) {
        return _mm256_mul_ps(x, y);
    }
};

struct AVX2ElementOpL2 : public ElementOpL2 {
    using ElementOpL2::op;

    static __m256 op(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(x, y);
        return _mm256_mul_ps(tmp, tmp);
    }
};

} // namespace

/// helper function for AVX2
inline float horizontal_sum(const __m256 v) {
    // add high and low parts
    const __m128 v0 =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    // perform horizontal sum on v0
    return horizontal_sum(v0);
}

template <>
void fvec_op_ny_D2<AVX2ElementOpIP>(
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
void fvec_op_ny_D2<AVX2ElementOpL2>(
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

template <>
void fvec_op_ny_D4<AVX2ElementOpIP>(
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
            __m128 accu = AVX2ElementOpIP::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D4<AVX2ElementOpL2>(
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
            __m128 accu = AVX2ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D8<AVX2ElementOpIP>(
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
            __m256 accu = AVX2ElementOpIP::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_op_ny_D8<AVX2ElementOpL2>(
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
            __m256 accu = AVX2ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::AVX2>(
        float* ip, /* output inner product */
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny_ref<AVX2ElementOpIP>(ip, x, y, d, ny);
}

template <>
void fvec_L2sqr_ny<SIMDLevel::AVX2>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref<AVX2ElementOpL2>(dis, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_D2<SIMDLevel::AVX2>(
        float* /*distances_tmp_buffer*/,
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

template <>
size_t fvec_L2sqr_ny_nearest_D4<SIMDLevel::AVX2>(
        float* /*distances_tmp_buffer*/,
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

template <>
size_t fvec_L2sqr_ny_nearest_D8<SIMDLevel::AVX2>(
        float* /*distances_tmp_buffer*/,
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
            __m256 accu = AVX2ElementOpL2::op(x0, _mm256_loadu_ps(y));
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
size_t fvec_L2sqr_ny_nearest<SIMDLevel::AVX2>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_x86<SIMDLevel::AVX2>(
            distances_tmp_buffer,
            x,
            y,
            d,
            ny,
            &fvec_L2sqr_ny_nearest_D2<SIMDLevel::AVX2>,
            &fvec_L2sqr_ny_nearest_D4<SIMDLevel::AVX2>,
            &fvec_L2sqr_ny_nearest_D8<SIMDLevel::AVX2>);
}

template <size_t DIM>
size_t fvec_L2sqr_ny_nearest_y_transposed_D(
        float* /*distances_tmp_buffer*/,
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

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::AVX2>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
// optimized for a few special cases
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
            return fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::NONE>(
                    distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
    }
#undef DISPATCH
}

template <>
int fvec_madd_and_argmin<SIMDLevel::AVX2>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    return fvec_madd_and_argmin_sse(n, a, bf, b, c);
}

template <>
void exhaustive_L2sqr_blas_cmax<SIMDLevel::AVX2>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0) {
        return;
    }

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx) {
            i1 = nx;
        }

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) {
                j1 = ny;
            }
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                _mm_prefetch((const char*)ip_line, _MM_HINT_NTA);
                _mm_prefetch((const char*)(ip_line + 16), _MM_HINT_NTA);

                // constant
                const __m256 mul_minus2 = _mm256_set1_ps(-2);

                // Track 8 min distances + 8 min indices.
                // All the distances tracked do not take x_norms[i]
                //   into account in order to get rid of extra
                //   _mm256_add_ps(x_norms[i], ...) instructions
                //   is distance computations.
                __m256 min_distances =
                        _mm256_set1_ps(res.dis_tab[i] - x_norms[i]);

                // these indices are local and are relative to j0.
                // so, value 0 means j0.
                __m256i min_indices = _mm256_set1_epi32(0);

                __m256i current_indices =
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i indices_delta = _mm256_set1_epi32(8);

                // current j index
                size_t idx_j = 0;
                size_t count = j1 - j0;

                // process 16 elements per loop
                for (; idx_j < (count / 16) * 16; idx_j += 16, ip_line += 16) {
                    _mm_prefetch((const char*)(ip_line + 32), _MM_HINT_NTA);
                    _mm_prefetch((const char*)(ip_line + 48), _MM_HINT_NTA);

                    // load values for norms
                    const __m256 y_norm_0 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 0);
                    const __m256 y_norm_1 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 8);

                    // load values for dot products
                    const __m256 ip_0 = _mm256_loadu_ps(ip_line + 0);
                    const __m256 ip_1 = _mm256_loadu_ps(ip_line + 8);

                    // compute dis = y_norm[j] - 2 * dot(x_norm[i], y_norm[j]).
                    // x_norm[i] was dropped off because it is a constant for a
                    // given i. We'll deal with it later.
                    __m256 distances_0 =
                            _mm256_fmadd_ps(ip_0, mul_minus2, y_norm_0);
                    __m256 distances_1 =
                            _mm256_fmadd_ps(ip_1, mul_minus2, y_norm_1);

                    // compare the new distances to the min distances
                    // for each of the first group of 8 AVX2 components.
                    const __m256 comparison_0 = _mm256_cmp_ps(
                            min_distances, distances_0, _CMP_LE_OS);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances = _mm256_blendv_ps(
                            distances_0, min_distances, comparison_0);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_0));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);

                    // compare the new distances to the min distances
                    // for each of the second group of 8 AVX2 components.
                    const __m256 comparison_1 = _mm256_cmp_ps(
                            min_distances, distances_1, _CMP_LE_OS);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances = _mm256_blendv_ps(
                            distances_1, min_distances, comparison_1);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_1));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);
                }

                // dump values and find the minimum distance / minimum index
                float min_distances_scalar[8];
                uint32_t min_indices_scalar[8];
                _mm256_storeu_ps(min_distances_scalar, min_distances);
                _mm256_storeu_si256(
                        (__m256i*)(min_indices_scalar), min_indices);

                float current_min_distance = res.dis_tab[i];
                uint32_t current_min_index = res.ids_tab[i];

                // This unusual comparison is needed to maintain the behavior
                // of the original implementation: if two indices are
                // represented with equal distance values, then
                // the index with the min value is returned.
                for (size_t jv = 0; jv < 8; jv++) {
                    // add missing x_norms[i]
                    float distance_candidate =
                            min_distances_scalar[jv] + x_norms[i];

                    // negative values can occur for identical vectors
                    //    due to roundoff errors.
                    if (distance_candidate < 0) {
                        distance_candidate = 0;
                    }

                    int64_t index_candidate = min_indices_scalar[jv] + j0;

                    if (current_min_distance > distance_candidate) {
                        current_min_distance = distance_candidate;
                        current_min_index = index_candidate;
                    } else if (
                            current_min_distance == distance_candidate &&
                            current_min_index > index_candidate) {
                        current_min_index = index_candidate;
                    }
                }

                // process leftovers
                for (; idx_j < count; idx_j++, ip_line++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[idx_j + j0] - 2 * ip;
                    // negative values can occur for identical vectors
                    //    due to roundoff errors.
                    if (dis < 0) {
                        dis = 0;
                    }

                    if (current_min_distance > dis) {
                        current_min_distance = dis;
                        current_min_index = idx_j + j0;
                    }
                }

                //
                res.add_result(i, current_min_distance, current_min_index);
            }
        }
        // Does nothing for SingleBestResultHandler, but
        // keeping the call for the consistency.
        res.end_multiple();
        InterruptCallback::check();
    }
}

} // namespace faiss
