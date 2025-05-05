/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#include <immintrin.h>

#define AUTOVEC_LEVEL SIMDLevel::AVX512F
#include <faiss/utils/simd_impl/distances_autovec-inl.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::AVX512F>(
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

} // namespace faiss
