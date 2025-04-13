/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#include <immintrin.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::AVX2>(
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

} // namespace faiss
