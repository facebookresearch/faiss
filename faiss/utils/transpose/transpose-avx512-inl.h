/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// This file contains transposing kernels for AVX512 for // tiny float/int32
// matrices, such as 16x2.

#ifdef __AVX512F__

#include <immintrin.h>

namespace faiss {

// 16x2 -> 2x16
inline void transpose_16x2(
        const __m512 i0,
        const __m512 i1,
        __m512& o0,
        __m512& o1) {
    // assume we have the following input:
    // i0:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    // i1: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31

    // 0  1  2  3  8  9 10 11 16 17 18 19 24 25 26 27
    const __m512 r0 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(2, 0, 2, 0));
    // 4  5  6  7 12 13 14 15 20 21 22 23 28 29 30 31
    const __m512 r1 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(3, 1, 3, 1));

    // 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30
    o0 = _mm512_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
    // 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31
    o1 = _mm512_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
}

// 16x4 -> 4x16
inline void transpose_16x4(
        const __m512 i0,
        const __m512 i1,
        const __m512 i2,
        const __m512 i3,
        __m512& o0,
        __m512& o1,
        __m512& o2,
        __m512& o3) {
    // assume that we have the following input:
    // i0:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    // i1: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    // i2: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    // i3: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63

    //  0  1  2  3  8  9 10 11 16 17 18 19 24 25 26 27
    const __m512 r0 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(2, 0, 2, 0));
    //  4  5  6  7 12 13 14 15 20 21 22 23 28 29 30 31
    const __m512 r1 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(3, 1, 3, 1));
    // 32 33 34 35 40 41 42 43 48 49 50 51 56 57 58 59
    const __m512 r2 = _mm512_shuffle_f32x4(i2, i3, _MM_SHUFFLE(2, 0, 2, 0));
    // 52 53 54 55 60 61 62 63 52 53 54 55 60 61 62 63
    const __m512 r3 = _mm512_shuffle_f32x4(i2, i3, _MM_SHUFFLE(3, 1, 3, 1));

    //  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30
    const __m512 t0 = _mm512_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
    //  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31
    const __m512 t1 = _mm512_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
    // 32 34 52 54 40 42 60 62 48 50 52 54 56 58 60 62
    const __m512 t2 = _mm512_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
    // 33 35 53 55 41 43 61 63 49 51 53 55 57 59 61 63
    const __m512 t3 = _mm512_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));

    const __m512i idx0 = _mm512_set_epi32(
            30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx1 = _mm512_set_epi32(
            31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    // 0 4  8 12 16 20 24 28 32 52 40 60 48 52 56 60
    o0 = _mm512_permutex2var_ps(t0, idx0, t2);
    // 1 5  9 13 17 21 25 29 33 53 41 61 49 53 57 61
    o1 = _mm512_permutex2var_ps(t1, idx0, t3);
    // 2 6 10 14 18 22 26 30 34 54 42 62 50 54 58 62
    o2 = _mm512_permutex2var_ps(t0, idx1, t2);
    // 3 7 11 15 19 23 27 31 35 55 43 63 51 55 59 63
    o3 = _mm512_permutex2var_ps(t1, idx1, t3);
}

// 16x8 -> 8x16 transpose
inline void transpose_16x8(
        const __m512 i0,
        const __m512 i1,
        const __m512 i2,
        const __m512 i3,
        const __m512 i4,
        const __m512 i5,
        const __m512 i6,
        const __m512 i7,
        __m512& o0,
        __m512& o1,
        __m512& o2,
        __m512& o3,
        __m512& o4,
        __m512& o5,
        __m512& o6,
        __m512& o7) {
    // assume that we have the following input:
    // i0:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    // i1:  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
    // i2:  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
    // i3:  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
    // i4:  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
    // i5:  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
    // i6:  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
    // i7: 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127

    //  0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
    const __m512 r0 = _mm512_unpacklo_ps(i0, i1);
    //  2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    const __m512 r1 = _mm512_unpackhi_ps(i0, i1);
    // 32  48  33  49  36  52  37  53  40  56  41  57  44  60  45  61
    const __m512 r2 = _mm512_unpacklo_ps(i2, i3);
    // 34  50  35  51  38  54  39  55  42  58  43  59  46  62  47  63
    const __m512 r3 = _mm512_unpackhi_ps(i2, i3);
    // 64  80  65  81  68  84  69  85  72  88  73  89  76  92  77  93
    const __m512 r4 = _mm512_unpacklo_ps(i4, i5);
    // 66  82  67  83  70  86  71  87  74  90  75  91  78  94  79  95
    const __m512 r5 = _mm512_unpackhi_ps(i4, i5);
    // 96 112  97 113 100 116 101 117 104 120 105 121 108 124 109 125
    const __m512 r6 = _mm512_unpacklo_ps(i6, i7);
    // 98 114  99 115 102 118 103 119 106 122 107 123 110 126 111 127
    const __m512 r7 = _mm512_unpackhi_ps(i6, i7);

    //  0  16  32  48   4  20  36  52   8  24  40  56  12  28  44  60
    const __m512 t0 = _mm512_shuffle_ps(r0, r2, _MM_SHUFFLE(1, 0, 1, 0));
    //  1  17  33  49   5  21  37  53   9  25  41  57  13  29  45  61
    const __m512 t1 = _mm512_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 2, 3, 2));
    //  2  18  34  50   6  22  38  54  10  26  42  58  14  30  46  62
    const __m512 t2 = _mm512_shuffle_ps(r1, r3, _MM_SHUFFLE(1, 0, 1, 0));
    //  3  19  35  51   7  23  39  55  11  27  43  59  15  31  47  63
    const __m512 t3 = _mm512_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 2, 3, 2));
    // 64  80  96 112  68  84 100 116  72  88 104 120  76  92 108 124
    const __m512 t4 = _mm512_shuffle_ps(r4, r6, _MM_SHUFFLE(1, 0, 1, 0));
    // 65  81  97 113  69  85 101 117  73  89 105 121  77  93 109 125
    const __m512 t5 = _mm512_shuffle_ps(r4, r6, _MM_SHUFFLE(3, 2, 3, 2));
    // 66  82  98 114  70  86 102 118  74  90 106 122  78  94 110 126
    const __m512 t6 = _mm512_shuffle_ps(r5, r7, _MM_SHUFFLE(1, 0, 1, 0));
    // 67  83  99 115  71  87 103 119  75  91 107 123  79  95 111 127
    const __m512 t7 = _mm512_shuffle_ps(r5, r7, _MM_SHUFFLE(3, 2, 3, 2));

    const __m512i idx0 = _mm512_set_epi32(
            27, 19, 26, 18, 25, 17, 24, 16, 11, 3, 10, 2, 9, 1, 8, 0);
    const __m512i idx1 = _mm512_set_epi32(
            31, 23, 30, 22, 29, 21, 28, 20, 15, 7, 14, 6, 13, 5, 12, 4);

    //  0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120
    o0 = _mm512_permutex2var_ps(t0, idx0, t4);
    //  1   9  17  25  33  41  49  57  65  73  81  89  97 105 113 121
    o1 = _mm512_permutex2var_ps(t1, idx0, t5);
    //  2  10  18  26  34  42  50  58  66  74  82  90  98 106 114 122
    o2 = _mm512_permutex2var_ps(t2, idx0, t6);
    //  3  11  19  27  35  43  51  59  67  75  83  91  99 107 115 123
    o3 = _mm512_permutex2var_ps(t3, idx0, t7);
    //  4  12  20  28  36  44  52  60  68  76  84  92 100 108 116 124
    o4 = _mm512_permutex2var_ps(t0, idx1, t4);
    //  5  13  21  29  37  45  53  61  69  77  85  93 101 109 117 125
    o5 = _mm512_permutex2var_ps(t1, idx1, t5);
    //  6  14  22  30  38  46  54  62  70  78  86  94 102 110 118 126
    o6 = _mm512_permutex2var_ps(t2, idx1, t6);
    //  7  15  23  31  39  47  55  63  71  79  87  95 103 111 119 127
    o7 = _mm512_permutex2var_ps(t3, idx1, t7);
}

} // namespace faiss

#endif
