/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__x86_64__) && defined(__AVX512F__)

#include <faiss/utils/simd_impl/super_kmeans_kernels.h>

#include <immintrin.h>

namespace faiss {
namespace detail {

template <>
float block_l2<SIMDLevel::AVX512>(const float* x, const float* y, int n) {
    __m512 acc = _mm512_setzero_ps();
    int m = 0;
    for (; m + 16 <= n; m += 16) {
        __m512 xv = _mm512_loadu_ps(x + m);
        __m512 yv = _mm512_loadu_ps(y + m);
        __m512 diff = _mm512_sub_ps(xv, yv);
        acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    // _mm512_reduce_add_ps: on modern AVX-512 SKUs (Cascade Lake+, Sapphire
    // Rapids) GCC/Clang lower this to a shuffle+add tree, ~5-cycle latency.
    // On older AVX-512 SKUs (Skylake-X, Ice Lake) the cross-lane reduction
    // can be ~20 cycles. Acceptable here because n ~ pdx_block_size = 64
    // (4 iterations of 16-wide accumulation), so per-block work dominates
    // the reduction cost. AVX2 uses a manual shuffle+add tree explicitly
    // to avoid `_mm_hadd_ps` overhead, where the ratio is reversed.
    float result = _mm512_reduce_add_ps(acc);
    for (; m < n; ++m) {
        const float d = x[m] - y[m];
        result += d * d;
    }
    return result;
}

} // namespace detail
} // namespace faiss

#endif // __x86_64__ && __AVX512F__
