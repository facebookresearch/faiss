/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/nndescent/gram.h>

#include <arm_neon.h>

#include <algorithm>
#include <cmath>

namespace faiss {
namespace nndescent {

namespace {

// tiles of 4 x 6 entries: 24 accumulators and 5 operands of the 32 registers
constexpr int RI = 4, RJ = 6;

template <int I, int J>
void tile(
        const float* const* a,
        const float* const* b,
        size_t d,
        float* out,
        size_t ldo) {
    float32x4_t acc[I][J];
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            acc[i][j] = vdupq_n_f32(0.0f);
        }
    }
    size_t k = 0;
    for (; k + 4 <= d; k += 4) {
        float32x4_t x[I];
        for (int i = 0; i < I; i++) {
            x[i] = vld1q_f32(a[i] + k);
        }
        for (int j = 0; j < J; j++) {
            const float32x4_t y = vld1q_f32(b[j] + k);
            for (int i = 0; i < I; i++) {
                acc[i][j] = vfmaq_f32(acc[i][j], x[i], y);
            }
        }
    }
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            float v = vaddvq_f32(acc[i][j]);
            for (size_t t = k; t < d; t++) {
                v = std::fma(a[i][t], b[j][t], v);
            }
            out[i * ldo + j] = v;
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
void gram_upper_rows<SIMDLevel::ARM_NEON>(
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
