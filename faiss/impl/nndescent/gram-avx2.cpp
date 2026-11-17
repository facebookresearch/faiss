/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/nndescent/gram.h>

#include <algorithm>
#include <cmath>

#include <faiss/impl/simdlib/simdlib_avx2.h>

namespace faiss {
namespace nndescent {

using VF = simd8float32_tpl<SIMDLevel::AVX2>;
constexpr size_t W = sizeof(VF) / sizeof(float);
// tiles of RI x RJ entries: 12 accumulators and 4 operands fill the 16
// registers
constexpr int RI = 3, RJ = 4;

/** One tile, a block of I x J Gram matrix entries that the kernel computes
 * together: out[i * ldo + j] = <a[i], b[j]> for i < I, j < J. Each vector
 * that the kernel loads serves J or I products. Each product has its own
 * accumulator, and the tail uses std::fma whatever the compiler contracts, so
 * a product depends only on its two rows. */
template <int I, int J>
static void tile(
        const float* const* a,
        const float* const* b,
        size_t d,
        float* out,
        size_t ldo) {
    VF acc[I][J];
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            acc[i][j] = VF(0.0f);
        }
    }
    size_t k = 0;
    for (; k + W <= d; k += W) {
        VF x[I];
        for (int i = 0; i < I; i++) {
            x[i] = VF(a[i] + k);
        }
        for (int j = 0; j < J; j++) {
            const VF y(b[j] + k);
            for (int i = 0; i < I; i++) {
                acc[i][j] = fmadd(x[i], y, acc[i][j]);
            }
        }
    }
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            float v = horizontal_add(acc[i][j]);
            for (size_t t = k; t < d; t++) {
                v = std::fma(a[i][t], b[j][t], v);
            }
            out[i * ldo + j] = v;
        }
    }
}

/// a tile of I x nj entries, nj <= J
template <int I, int J = RJ>
static void run_tile_cols(
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

/// a tile of ni x nj entries, ni <= I, nj <= RJ
template <int I = RI>
static void run_tile(
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

template <>
void gram_upper_rows<SIMDLevel::AVX2>(
        const float* const* rows,
        size_t r,
        size_t m,
        size_t d,
        float* out) {
    // rows i0 .. i0 + RI - 1 against the rows j > i0, tile by tile
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
