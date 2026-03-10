/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <random>

__m512 flip_with_step(__m512 a, int log2_p) {
    switch (log2_p) {
        case 0:
            // p=1: swap adjacent elements (0,1), (2,3), etc.
            // [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] ->
            // [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14]
            return _mm512_shuffle_ps(a, a, 0b10110001);
        case 1:
            // p=2: swap pairs (0,1)<->(2,3), (4,5)<->(6,7), etc.
            // [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] ->
            // [2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13]
            return _mm512_shuffle_ps(a, a, 0b01001110);
        case 2:
            // p=4: swap 128-bit lanes within each 256-bit half
            // [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] ->
            // [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11]
            return _mm512_shuffle_f32x4(a, a, 0b10110001);
        case 3:
            // p=8: swap 256-bit halves
            // [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] ->
            // [8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7]
            return _mm512_shuffle_f32x4(a, a, 0b01001110);
        default:
            assert(false);
            return a; // keep some compilers happy
    }
}

__mmask16 mask_with_step(int step) {
    switch (step) {
        case 0:
            return 0xAAAA;
        case 1:
            return 0xCCCC;
        case 2:
            return 0xF0F0;
        case 3:
            return 0xFF00;
        case 4:
            return 0x0000;
        default:
            assert(false);
            return 0; // keep some compilers happy
    }
}

__m512 bitonic_merge(__m512 tab, int step, int stepk) {
    __m512 inv_tab = flip_with_step(tab, step);
    __mmask16 mask = _mm512_cmp_ps_mask(tab, inv_tab, _CMP_GT_OQ);

    // for distinct values, the mask is ok, but for others we should make sure
    // there is one 0 and one 1 per pair, so force
    //    mask[i ^ (1<<step)] = 1 - mask[i]
    mask &= ~mask_with_step(step);
    mask |= (mask & mask_with_step(step)) << step;

    mask ^= mask_with_step(step) ^ mask_with_step(stepk);
    return _mm512_mask_blend_ps(mask, tab, inv_tab);
}

__m512 bitonic_sort(__m512 tab) {
    for (int stepk = 1; stepk < 5; stepk++) {
        for (int step = stepk - 1; step >= 0; step--) {
            tab = bitonic_merge(tab, step, stepk);
        }
    }
    return tab;
}

int main() {
    // Generate a random permutation of 0..15
    float values[16];
    std::iota(values, values + 16, 0.0f);
    std::mt19937 rng(1234);
    std::shuffle(values, values + 16, rng);

    printf("Input:  ");
    for (int i = 0; i < 16; i++) {
        printf("%2.0f ", values[i]);
    }
    printf("\n");

    // Load into __m512 and sort
    __m512 vec = _mm512_loadu_ps(values);
    vec = bitonic_sort(vec);

    // Store and display result
    float result[16];
    _mm512_storeu_ps(result, vec);

    printf("Output: ");
    for (int i = 0; i < 16; i++) {
        printf("%2.0f ", result[i]);
    }
    printf("\n");

    return 0;
}
