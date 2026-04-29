/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__x86_64__)

#include <faiss/utils/simd_impl/super_kmeans_kernels.h>

#include <immintrin.h>

namespace faiss {
namespace detail {

namespace {

// Reduce 8 float lanes of an AVX2 register to a scalar sum.
// Uses a shuffle+add tree instead of two _mm_hadd_ps. On Skylake-class
// cores, hadd is 3-cycle latency / 2-uop, while movehdup/movehl/add_ss
// are single-uop, single-cycle ops.
inline float horizontal_sum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);     // 4 lanes
    __m128 shuf = _mm_movehdup_ps(sum128);  // [s1, s1, s3, s3]
    __m128 sums = _mm_add_ps(sum128, shuf); // [s0+s1, _, s2+s3, _]
    shuf = _mm_movehl_ps(shuf, sums);       // [s2+s3, s3, _, _]
    sums = _mm_add_ss(sums, shuf);          // (s0+s1) + (s2+s3)
    return _mm_cvtss_f32(sums);
}

} // namespace

template <>
float block_l2<SIMDLevel::AVX2>(const float* x, const float* y, int n) {
    __m256 acc = _mm256_setzero_ps();
    int m = 0;
    for (; m + 8 <= n; m += 8) {
        __m256 xv = _mm256_loadu_ps(x + m);
        __m256 yv = _mm256_loadu_ps(y + m);
        __m256 diff = _mm256_sub_ps(xv, yv);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float result = horizontal_sum_avx2(acc);
    for (; m < n; ++m) {
        const float d = x[m] - y[m];
        result += d * d;
    }
    return result;
}

} // namespace detail
} // namespace faiss

#endif // __x86_64__
