/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// This file contains transposing kernels for AVX2 for
// tiny float/int32 matrices, such as 8x2.

#ifdef __AVX2__

#include <immintrin.h>

namespace faiss {

// 8x2 -> 2x8
inline void transpose_8x2(
        const __m256 i0,
        const __m256 i1,
        __m256& o0,
        __m256& o1) {
    // say, we have the following as in input:
    // i0:  00 01 10 11 20 21 30 31
    // i1:  40 41 50 51 60 61 70 71

    // 00 01 10 11 40 41 50 51
    const __m256 r0 = _mm256_permute2f128_ps(i0, i1, _MM_SHUFFLE(0, 2, 0, 0));
    // 20 21 30 31 60 61 70 71
    const __m256 r1 = _mm256_permute2f128_ps(i0, i1, _MM_SHUFFLE(0, 3, 0, 1));

    // 00 10 20 30 40 50 60 70
    o0 = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
    // 01 11 21 31 41 51 61 71
    o1 = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
}

// 8x4 -> 4x8
inline void transpose_8x4(
        const __m256 i0,
        const __m256 i1,
        const __m256 i2,
        const __m256 i3,
        __m256& o0,
        __m256& o1,
        __m256& o2,
        __m256& o3) {
    // say, we have the following as an input:
    // i0:  00 01 02 03 10 11 12 13
    // i1:  20 21 22 23 30 31 32 33
    // i2:  40 41 42 43 50 51 52 53
    // i3:  60 61 62 63 70 71 72 73

    // 00 01 02 03 40 41 42 43
    const __m256 r0 = _mm256_permute2f128_ps(i0, i2, _MM_SHUFFLE(0, 2, 0, 0));
    // 20 21 22 23 60 61 62 63
    const __m256 r1 = _mm256_permute2f128_ps(i1, i3, _MM_SHUFFLE(0, 2, 0, 0));
    // 10 11 12 13 50 51 52 53
    const __m256 r2 = _mm256_permute2f128_ps(i0, i2, _MM_SHUFFLE(0, 3, 0, 1));
    // 30 31 32 33 70 71 72 73
    const __m256 r3 = _mm256_permute2f128_ps(i1, i3, _MM_SHUFFLE(0, 3, 0, 1));

    // 00 02 10 12 40 42 50 52
    const __m256 t0 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(2, 0, 2, 0));
    // 01 03 11 13 41 43 51 53
    const __m256 t1 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 1, 3, 1));
    // 20 22 30 32 60 62 70 72
    const __m256 t2 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(2, 0, 2, 0));
    // 21 23 31 33 61 63 71 73
    const __m256 t3 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 1, 3, 1));

    // 00 10 20 30 40 50 60 70
    o0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(2, 0, 2, 0));
    // 01 11 21 31 41 51 61 71
    o1 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(2, 0, 2, 0));
    // 02 12 22 32 42 52 62 72
    o2 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 1, 3, 1));
    // 03 13 23 33 43 53 63 73
    o3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 1, 3, 1));
}

inline void transpose_8x8(
        const __m256 i0,
        const __m256 i1,
        const __m256 i2,
        const __m256 i3,
        const __m256 i4,
        const __m256 i5,
        const __m256 i6,
        const __m256 i7,
        __m256& o0,
        __m256& o1,
        __m256& o2,
        __m256& o3,
        __m256& o4,
        __m256& o5,
        __m256& o6,
        __m256& o7) {
    // say, we have the following as an input:
    // i0:  00 01 02 03 04 05 06 07
    // i1:  10 11 12 13 14 15 16 17
    // i2:  20 21 22 23 24 25 26 27
    // i3:  30 31 32 33 34 35 36 37
    // i4:  40 41 42 43 44 45 46 47
    // i5:  50 51 52 53 54 55 56 57
    // i6:  60 61 62 63 64 65 66 67
    // i7:  70 71 72 73 74 75 76 77

    // 00 10 01 11 04 14 05 15
    const __m256 r0 = _mm256_unpacklo_ps(i0, i1);
    // 02 12 03 13 06 16 07 17
    const __m256 r1 = _mm256_unpackhi_ps(i0, i1);
    // 20 30 21 31 24 34 25 35
    const __m256 r2 = _mm256_unpacklo_ps(i2, i3);
    // 22 32 23 33 26 36 27 37
    const __m256 r3 = _mm256_unpackhi_ps(i2, i3);
    // 40 50 41 51 44 54 45 55
    const __m256 r4 = _mm256_unpacklo_ps(i4, i5);
    // 42 52 43 53 46 56 47 57
    const __m256 r5 = _mm256_unpackhi_ps(i4, i5);
    // 60 70 61 71 64 74 65 75
    const __m256 r6 = _mm256_unpacklo_ps(i6, i7);
    // 62 72 63 73 66 76 67 77
    const __m256 r7 = _mm256_unpackhi_ps(i6, i7);

    // 00 10 20 30 04 14 24 34
    const __m256 rr0 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(1, 0, 1, 0));
    // 01 11 21 31 05 15 25 35
    const __m256 rr1 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 2, 3, 2));
    // 02 12 22 32 06 16 26 36
    const __m256 rr2 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(1, 0, 1, 0));
    // 03 13 23 33 07 17 27 37
    const __m256 rr3 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 2, 3, 2));
    // 40 50 60 70 44 54 64 74
    const __m256 rr4 = _mm256_shuffle_ps(r4, r6, _MM_SHUFFLE(1, 0, 1, 0));
    // 41 51 61 71 45 55 65 75
    const __m256 rr5 = _mm256_shuffle_ps(r4, r6, _MM_SHUFFLE(3, 2, 3, 2));
    // 42 52 62 72 46 56 66 76
    const __m256 rr6 = _mm256_shuffle_ps(r5, r7, _MM_SHUFFLE(1, 0, 1, 0));
    // 43 53 63 73 47 57 67 77
    const __m256 rr7 = _mm256_shuffle_ps(r5, r7, _MM_SHUFFLE(3, 2, 3, 2));

    // 00 10 20 30 40 50 60 70
    o0 = _mm256_permute2f128_ps(rr0, rr4, 0x20);
    // 01 11 21 31 41 51 61 71
    o1 = _mm256_permute2f128_ps(rr1, rr5, 0x20);
    // 02 12 22 32 42 52 62 72
    o2 = _mm256_permute2f128_ps(rr2, rr6, 0x20);
    // 03 13 23 33 43 53 63 73
    o3 = _mm256_permute2f128_ps(rr3, rr7, 0x20);
    // 04 14 24 34 44 54 64 74
    o4 = _mm256_permute2f128_ps(rr0, rr4, 0x31);
    // 05 15 25 35 45 55 65 75
    o5 = _mm256_permute2f128_ps(rr1, rr5, 0x31);
    // 06 16 26 36 46 56 66 76
    o6 = _mm256_permute2f128_ps(rr2, rr6, 0x31);
    // 07 17 27 37 47 57 67 77
    o7 = _mm256_permute2f128_ps(rr3, rr7, 0x31);
}

} // namespace faiss

#endif
