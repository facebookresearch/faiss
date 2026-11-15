/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <faiss/utils/simd_levels.h>

namespace faiss {
namespace nndescent {

/** Rows [0, r) of the upper triangle of the Gram matrix of the m vectors
 * rows[0 .. m): out[i * m + j] = <rows[i], rows[j]> for i < r and
 * i < j < m. The Gram matrix of m vectors is the m x m matrix of their inner
 * products. In a local join of DeterministicNNDescent, the first r vectors
 * are the new members, so these are the pairs with at least one new member.
 * The vectors are read in place. The kernel also sets some entries with
 * j <= i. The sums have a fixed order, so the values do not depend on the
 * thread. Defined for AVX2, AVX512 and ARM_NEON. The dispatch uses AVX512 on
 * the higher AVX-512 levels and ARM_NEON on ARM_SVE, and the caller uses BLAS
 * for the other levels. */
template <SIMDLevel SL>
void gram_upper_rows(
        const float* const* rows,
        size_t r,
        size_t m,
        size_t d,
        float* out);

#ifdef COMPILE_SIMD_AVX2
template <>
void gram_upper_rows<SIMDLevel::AVX2>(
        const float* const* rows,
        size_t r,
        size_t m,
        size_t d,
        float* out);
#endif

#ifdef COMPILE_SIMD_AVX512
template <>
void gram_upper_rows<SIMDLevel::AVX512>(
        const float* const* rows,
        size_t r,
        size_t m,
        size_t d,
        float* out);
#endif

#ifdef COMPILE_SIMD_ARM_NEON
template <>
void gram_upper_rows<SIMDLevel::ARM_NEON>(
        const float* const* rows,
        size_t r,
        size_t m,
        size_t d,
        float* out);
#endif

} // namespace nndescent
} // namespace faiss
