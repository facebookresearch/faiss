/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/nndescent/gram.h>

#include <immintrin.h>

#include <algorithm>

namespace faiss {
namespace nndescent {

namespace {

// tiles of 4 x 6 entries: 24 accumulators and 5 operands of the 32 registers
constexpr int RI = 4, RJ = 6;

/// out[i * ldo + j] = <a[i], b[j]> for i < I, j < J; the tail is a masked
/// load, so there is no scalar loop
template <int I, int J>
void tile(
        const float* const* a,
        const float* const* b,
        size_t d,
        float* out,
        size_t ldo) {
    __m512 acc[I][J];
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            acc[i][j] = _mm512_setzero_ps();
        }
    }
    size_t k = 0;
    for (; k + 16 <= d; k += 16) {
        __m512 x[I];
        for (int i = 0; i < I; i++) {
            x[i] = _mm512_loadu_ps(a[i] + k);
        }
        for (int j = 0; j < J; j++) {
            const __m512 y = _mm512_loadu_ps(b[j] + k);
            for (int i = 0; i < I; i++) {
                acc[i][j] = _mm512_fmadd_ps(x[i], y, acc[i][j]);
            }
        }
    }
    if (k < d) {
        const __mmask16 mask = __mmask16((1u << (d - k)) - 1);
        __m512 x[I];
        for (int i = 0; i < I; i++) {
            x[i] = _mm512_maskz_loadu_ps(mask, a[i] + k);
        }
        for (int j = 0; j < J; j++) {
            const __m512 y = _mm512_maskz_loadu_ps(mask, b[j] + k);
            for (int i = 0; i < I; i++) {
                acc[i][j] = _mm512_fmadd_ps(x[i], y, acc[i][j]);
            }
        }
    }
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            out[i * ldo + j] = _mm512_reduce_add_ps(acc[i][j]);
        }
    }
}

template <int I, int J = RJ>
void run_tile_cols(
        int nj,
        const float* const* a,
        const float* const* b,
        size_t d,
        float* out,
        size_t ldo) {
    if constexpr (J > 1) {
        if (nj < J) {
            run_tile_cols<I, J - 1>(nj, a, b, d, out, ldo);
            return;
        }
    }
    tile<I, J>(a, b, d, out, ldo);
}

template <int I = RI>
void run_tile(
        int ni,
        int nj,
        const float* const* a,
        const float* const* b,
        size_t d,
        float* out,
        size_t ldo) {
    if constexpr (I > 1) {
        if (ni < I) {
            run_tile<I - 1>(ni, nj, a, b, d, out, ldo);
            return;
        }
    }
    run_tile_cols<I>(nj, a, b, d, out, ldo);
}

} // namespace

template <>
void gram_upper_rows<SIMDLevel::AVX512>(
        const float* const* rows,
        size_t r,
        size_t m,
        size_t d,
        float* out) {
    for (size_t i0 = 0; i0 < r; i0 += RI) {
        const int ni = int(std::min<size_t>(RI, r - i0));
        for (size_t j = i0 + 1; j < m; j += RJ) {
            const int nj = int(std::min<size_t>(RJ, m - j));
            run_tile(ni, nj, rows + i0, rows + j, d, out + i0 * m + j, m);
        }
    }
}

} // namespace nndescent
} // namespace faiss
