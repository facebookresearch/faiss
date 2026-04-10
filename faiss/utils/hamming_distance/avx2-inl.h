/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_AVX2_INL_H
#define HAMMING_AVX2_INL_H

// AVX2 version.
// Most code is now in common.h. This file only contains the
// GenHammingComputer classes that leverage SSE/AVX2 intrinsics.

#include <cassert>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

#include <immintrin.h>

namespace faiss {

// I'm not sure whether this version is faster of slower, tbh
// todo: test on different CPUs
struct GenHammingComputer16 {
    __m128i a;

    GenHammingComputer16(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        a = _mm_loadu_si128((const __m128i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m128i b = _mm_loadu_si128((const __m128i_u*)b8);
        const __m128i cmp = _mm_cmpeq_epi8(a, b);
        const auto movemask = _mm_movemask_epi8(cmp);
        return 16 - popcount32(movemask);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

struct GenHammingComputer32 {
    __m256i a;

    GenHammingComputer32(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        a = _mm256_loadu_si256((const __m256i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m256i b = _mm256_loadu_si256((const __m256i_u*)b8);
        const __m256i cmp = _mm256_cmpeq_epi8(a, b);
        const uint32_t movemask = _mm256_movemask_epi8(cmp);
        return 32 - popcount32(movemask);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

// A specialized version might be needed for the very long
// GenHamming code_size. In such a case, one may accumulate
// counts using _mm256_sub_epi8 and then compute a horizontal
// sum (using _mm256_sad_epu8, maybe, in blocks of no larger
// than 256 * 32 bytes).

struct GenHammingComputerM8 {
    const uint64_t* a;
    int n;

    GenHammingComputerM8(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = (uint64_t*)a8;
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        int accu = 0;

        int i = 0;
        int n4 = (n / 4) * 4;
        for (; i < n4; i += 4) {
            const __m256i av = _mm256_loadu_si256((const __m256i_u*)(a + i));
            const __m256i bv = _mm256_loadu_si256((const __m256i_u*)(b + i));
            const __m256i cmp = _mm256_cmpeq_epi8(av, bv);
            const uint32_t movemask = _mm256_movemask_epi8(cmp);
            accu += 32 - popcount32(movemask);
        }

        for (; i < n; i++)
            accu += generalized_hamming_64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};

} // namespace faiss

#endif
