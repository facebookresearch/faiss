/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#include <immintrin.h>

#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simdlib/simdlib.h>

namespace faiss {

/** Simple wrapper around the AVX 256-bit registers
 *
 * The objective is to separate the different interpretations of the same
 * registers (as a vector of uint8, uint16 or uint32), to provide printing
 * functions, and to give more readable names to the AVX intrinsics. It does not
 * pretend to be exhaustive, functions are added as needed.
 */

/// 256-bit representation without interpretation as a vector
template <>
struct simd256bit_tpl<SIMDLevel::AVX2> {
    union {
        __m256i i;
        __m256 f;
    };

    simd256bit_tpl() {}

    explicit simd256bit_tpl(__m256i val) : i(val) {}

    explicit simd256bit_tpl(__m256 val) : f(val) {}

    explicit simd256bit_tpl(const void* x)
            : i(_mm256_loadu_si256((__m256i const*)x)) {}

    void clear() {
        i = _mm256_setzero_si256();
    }

    void storeu(void* ptr) const {
        _mm256_storeu_si256((__m256i*)ptr, i);
    }

    void loadu(const void* ptr) {
        i = _mm256_loadu_si256((__m256i*)ptr);
    }

    void store(void* ptr) const {
        _mm256_store_si256((__m256i*)ptr, i);
    }

    void bin(char bits[257]) const {
        char bytes[32];
        storeu((void*)bytes);
        for (int idx = 0; idx < 256; idx++) {
            bits[idx] = '0' + ((bytes[idx / 8] >> (idx % 8)) & 1);
        }
        bits[256] = 0;
    }

    std::string bin() const {
        char bits[257];
        bin(bits);
        return std::string(bits);
    }

    // Checks whether the other holds exactly the same bytes.
    bool is_same_as(simd256bit_tpl other) const {
        const __m256i pcmp = _mm256_cmpeq_epi32(i, other.i);
        unsigned bitmask = _mm256_movemask_epi8(pcmp);
        return (bitmask == 0xffffffffU);
    }
};

/// vector of 16 elements in uint16
template <>
struct simd16uint16_tpl<SIMDLevel::AVX2> : simd256bit_tpl<SIMDLevel::AVX2> {
    simd16uint16_tpl() {}

    explicit simd16uint16_tpl(__m256i val)
            : simd256bit_tpl<SIMDLevel::AVX2>(val) {}

    explicit simd16uint16_tpl(int x)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_set1_epi16(x)) {}

    explicit simd16uint16_tpl(uint16_t x)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_set1_epi16(x)) {}

    explicit simd16uint16_tpl(simd256bit_tpl<SIMDLevel::AVX2> x)
            : simd256bit_tpl<SIMDLevel::AVX2>(x) {}

    explicit simd16uint16_tpl(const uint16_t* x)
            : simd256bit_tpl<SIMDLevel::AVX2>((const void*)x) {}

    explicit simd16uint16_tpl(
            uint16_t u0,
            uint16_t u1,
            uint16_t u2,
            uint16_t u3,
            uint16_t u4,
            uint16_t u5,
            uint16_t u6,
            uint16_t u7,
            uint16_t u8,
            uint16_t u9,
            uint16_t u10,
            uint16_t u11,
            uint16_t u12,
            uint16_t u13,
            uint16_t u14,
            uint16_t u15)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_setr_epi16(
                      u0,
                      u1,
                      u2,
                      u3,
                      u4,
                      u5,
                      u6,
                      u7,
                      u8,
                      u9,
                      u10,
                      u11,
                      u12,
                      u13,
                      u14,
                      u15)) {}

    std::string elements_to_string(const char* fmt) const {
        uint16_t bytes[16];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
        for (int idx = 0; idx < 16; idx++) {
            ptr += sprintf(ptr, fmt, bytes[idx]);
        }
#pragma GCC diagnostic pop
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }

    std::string dec() const {
        return elements_to_string("%3d,");
    }

    void set1(uint16_t x) {
        i = _mm256_set1_epi16((short)x);
    }

    simd16uint16_tpl operator*(const simd16uint16_tpl& other) const {
        return simd16uint16_tpl(_mm256_mullo_epi16(i, other.i));
    }

    // shift must be known at compile time
    simd16uint16_tpl operator>>(const int shift) const {
        return simd16uint16_tpl(_mm256_srli_epi16(i, shift));
    }

    // shift must be known at compile time
    simd16uint16_tpl operator<<(const int shift) const {
        return simd16uint16_tpl(_mm256_slli_epi16(i, shift));
    }

    simd16uint16_tpl& operator+=(simd16uint16_tpl other) {
        i = _mm256_add_epi16(i, other.i);
        return *this;
    }

    simd16uint16_tpl& operator-=(simd16uint16_tpl other) {
        i = _mm256_sub_epi16(i, other.i);
        return *this;
    }

    simd16uint16_tpl operator+(simd16uint16_tpl other) const {
        return simd16uint16_tpl(_mm256_add_epi16(i, other.i));
    }

    simd16uint16_tpl operator-(simd16uint16_tpl other) const {
        return simd16uint16_tpl(_mm256_sub_epi16(i, other.i));
    }

    simd16uint16_tpl operator&(simd256bit_tpl<SIMDLevel::AVX2> other) const {
        return simd16uint16_tpl(_mm256_and_si256(i, other.i));
    }

    simd16uint16_tpl operator|(simd256bit_tpl<SIMDLevel::AVX2> other) const {
        return simd16uint16_tpl(_mm256_or_si256(i, other.i));
    }

    simd16uint16_tpl operator^(simd256bit_tpl<SIMDLevel::AVX2> other) const {
        return simd16uint16_tpl(_mm256_xor_si256(i, other.i));
    }

    // returns binary masks
    friend simd16uint16_tpl operator==(
            const simd256bit_tpl<SIMDLevel::AVX2> lhs,
            const simd256bit_tpl<SIMDLevel::AVX2> rhs) {
        return simd16uint16_tpl(_mm256_cmpeq_epi16(lhs.i, rhs.i));
    }

    simd16uint16_tpl operator~() const {
        return simd16uint16_tpl(_mm256_xor_si256(i, _mm256_set1_epi32(-1)));
    }

    // get scalar at index 0
    uint16_t get_scalar_0() const {
        return _mm256_extract_epi16(i, 0);
    }

    // mask of elements where this >= thresh
    // 2 bit per component: 16 * 2 = 32 bit
    uint32_t ge_mask(simd16uint16_tpl thresh) const {
        __m256i j = thresh.i;
        __m256i max = _mm256_max_epu16(i, j);
        __m256i ge = _mm256_cmpeq_epi16(i, max);
        return _mm256_movemask_epi8(ge);
    }

    uint32_t le_mask(simd16uint16_tpl thresh) const {
        return thresh.ge_mask(*this);
    }

    uint32_t gt_mask(simd16uint16_tpl thresh) const {
        return ~le_mask(thresh);
    }

    bool all_gt(simd16uint16_tpl thresh) const {
        return le_mask(thresh) == 0;
    }

    // for debugging only
    uint16_t operator[](int idx) const {
        ALIGNED(32) uint16_t tab[16];
        store(tab);
        return tab[idx];
    }

    void accu_min(simd16uint16_tpl incoming) {
        i = _mm256_min_epu16(i, incoming.i);
    }

    void accu_max(simd16uint16_tpl incoming) {
        i = _mm256_max_epu16(i, incoming.i);
    }
};

// not really a std::min because it returns an elementwise min
inline simd16uint16_tpl<SIMDLevel::AVX2> min(
        simd16uint16_tpl<SIMDLevel::AVX2> a,
        simd16uint16_tpl<SIMDLevel::AVX2> b) {
    return simd16uint16_tpl<SIMDLevel::AVX2>(_mm256_min_epu16(a.i, b.i));
}

inline simd16uint16_tpl<SIMDLevel::AVX2> max(
        simd16uint16_tpl<SIMDLevel::AVX2> a,
        simd16uint16_tpl<SIMDLevel::AVX2> b) {
    return simd16uint16_tpl<SIMDLevel::AVX2>(_mm256_max_epu16(a.i, b.i));
}

// decompose in 128-lanes: a = (a0, a1), b = (b0, b1)
// return (a0 + a1, b0 + b1)
// TODO find a better name
inline simd16uint16_tpl<SIMDLevel::AVX2> combine2x2(
        simd16uint16_tpl<SIMDLevel::AVX2> a,
        simd16uint16_tpl<SIMDLevel::AVX2> b) {
    __m256i a1b0 = _mm256_permute2f128_si256(a.i, b.i, 0x21);
    __m256i a0b1 = _mm256_blend_epi32(a.i, b.i, 0xF0);

    return simd16uint16_tpl<SIMDLevel::AVX2>(a1b0) +
            simd16uint16_tpl<SIMDLevel::AVX2>(a0b1);
}

// compare d0 and d1 to thr, return 32 bits corresponding to the concatenation
// of d0 and d1 with thr
inline uint32_t cmp_ge32(
        simd16uint16_tpl<SIMDLevel::AVX2> d0,
        simd16uint16_tpl<SIMDLevel::AVX2> d1,
        simd16uint16_tpl<SIMDLevel::AVX2> thr) {
    __m256i max0 = _mm256_max_epu16(d0.i, thr.i);
    __m256i ge0 = _mm256_cmpeq_epi16(d0.i, max0);

    __m256i max1 = _mm256_max_epu16(d1.i, thr.i);
    __m256i ge1 = _mm256_cmpeq_epi16(d1.i, max1);

    __m256i ge01 = _mm256_packs_epi16(ge0, ge1);

    // easier than manipulating bit fields afterwards
    ge01 = _mm256_permute4x64_epi64(ge01, 0 | (2 << 2) | (1 << 4) | (3 << 6));
    uint32_t ge = _mm256_movemask_epi8(ge01);

    return ge;
}

inline uint32_t cmp_le32(
        simd16uint16_tpl<SIMDLevel::AVX2> d0,
        simd16uint16_tpl<SIMDLevel::AVX2> d1,
        simd16uint16_tpl<SIMDLevel::AVX2> thr) {
    __m256i max0 = _mm256_min_epu16(d0.i, thr.i);
    __m256i ge0 = _mm256_cmpeq_epi16(d0.i, max0);

    __m256i max1 = _mm256_min_epu16(d1.i, thr.i);
    __m256i ge1 = _mm256_cmpeq_epi16(d1.i, max1);

    __m256i ge01 = _mm256_packs_epi16(ge0, ge1);

    // easier than manipulating bit fields afterwards
    ge01 = _mm256_permute4x64_epi64(ge01, 0 | (2 << 2) | (1 << 4) | (3 << 6));
    uint32_t ge = _mm256_movemask_epi8(ge01);

    return ge;
}

inline simd16uint16_tpl<SIMDLevel::AVX2> hadd(
        const simd16uint16_tpl<SIMDLevel::AVX2>& a,
        const simd16uint16_tpl<SIMDLevel::AVX2>& b) {
    return simd16uint16_tpl<SIMDLevel::AVX2>(_mm256_hadd_epi16(a.i, b.i));
}

// Vectorized version of the following code:
//   for (size_t i = 0; i < n; i++) {
//      bool flag = (candidateValues[i] < currentValues[i]);
//      minValues[i] = flag ? candidateValues[i] : currentValues[i];
//      minIndices[i] = flag ? candidateIndices[i] : currentIndices[i];
//      maxValues[i] = !flag ? candidateValues[i] : currentValues[i];
//      maxIndices[i] = !flag ? candidateIndices[i] : currentIndices[i];
//   }
// Max indices evaluation is inaccurate in case of equal values (the index of
// the last equal value is saved instead of the first one), but this behavior
// saves instructions.
//
// Works in i16 mode in order to save instructions. One may
// switch from i16 to u16.
inline void cmplt_min_max_fast(
        const simd16uint16_tpl<SIMDLevel::AVX2> candidateValues,
        const simd16uint16_tpl<SIMDLevel::AVX2> candidateIndices,
        const simd16uint16_tpl<SIMDLevel::AVX2> currentValues,
        const simd16uint16_tpl<SIMDLevel::AVX2> currentIndices,
        simd16uint16_tpl<SIMDLevel::AVX2>& minValues,
        simd16uint16_tpl<SIMDLevel::AVX2>& minIndices,
        simd16uint16_tpl<SIMDLevel::AVX2>& maxValues,
        simd16uint16_tpl<SIMDLevel::AVX2>& maxIndices) {
    // there's no lt instruction, so we'll need to emulate one
    __m256i comparison = _mm256_cmpgt_epi16(currentValues.i, candidateValues.i);
    comparison = _mm256_andnot_si256(comparison, _mm256_set1_epi16(-1));

    minValues.i = _mm256_min_epi16(candidateValues.i, currentValues.i);
    minIndices.i = _mm256_blendv_epi8(
            candidateIndices.i, currentIndices.i, comparison);
    maxValues.i = _mm256_max_epi16(candidateValues.i, currentValues.i);
    maxIndices.i = _mm256_blendv_epi8(
            currentIndices.i, candidateIndices.i, comparison);
}

// vector of 32 unsigned 8-bit integers
template <>
struct simd32uint8_tpl<SIMDLevel::AVX2> : simd256bit_tpl<SIMDLevel::AVX2> {
    simd32uint8_tpl() {}

    explicit simd32uint8_tpl(__m256i val)
            : simd256bit_tpl<SIMDLevel::AVX2>(val) {}

    explicit simd32uint8_tpl(int x)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_set1_epi8(x)) {}

    explicit simd32uint8_tpl(uint8_t x)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_set1_epi8(x)) {}

    template <
            uint8_t _0,
            uint8_t _1,
            uint8_t _2,
            uint8_t _3,
            uint8_t _4,
            uint8_t _5,
            uint8_t _6,
            uint8_t _7,
            uint8_t _8,
            uint8_t _9,
            uint8_t _10,
            uint8_t _11,
            uint8_t _12,
            uint8_t _13,
            uint8_t _14,
            uint8_t _15,
            uint8_t _16,
            uint8_t _17,
            uint8_t _18,
            uint8_t _19,
            uint8_t _20,
            uint8_t _21,
            uint8_t _22,
            uint8_t _23,
            uint8_t _24,
            uint8_t _25,
            uint8_t _26,
            uint8_t _27,
            uint8_t _28,
            uint8_t _29,
            uint8_t _30,
            uint8_t _31>
    static simd32uint8_tpl create() {
        return simd32uint8_tpl(_mm256_setr_epi8(
                (char)_0,
                (char)_1,
                (char)_2,
                (char)_3,
                (char)_4,
                (char)_5,
                (char)_6,
                (char)_7,
                (char)_8,
                (char)_9,
                (char)_10,
                (char)_11,
                (char)_12,
                (char)_13,
                (char)_14,
                (char)_15,
                (char)_16,
                (char)_17,
                (char)_18,
                (char)_19,
                (char)_20,
                (char)_21,
                (char)_22,
                (char)_23,
                (char)_24,
                (char)_25,
                (char)_26,
                (char)_27,
                (char)_28,
                (char)_29,
                (char)_30,
                (char)_31));
    }

    explicit simd32uint8_tpl(simd256bit_tpl<SIMDLevel::AVX2> x)
            : simd256bit_tpl<SIMDLevel::AVX2>(x) {}

    explicit simd32uint8_tpl(const uint8_t* x)
            : simd256bit_tpl<SIMDLevel::AVX2>((const void*)x) {}

    std::string elements_to_string(const char* fmt) const {
        uint8_t bytes[32];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
        for (int idx = 0; idx < 32; idx++) {
            ptr += sprintf(ptr, fmt, bytes[idx]);
        }
#pragma GCC diagnostic pop
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }

    std::string dec() const {
        return elements_to_string("%3d,");
    }

    void set1(uint8_t x) {
        i = _mm256_set1_epi8((char)x);
    }

    simd32uint8_tpl operator&(simd256bit_tpl<SIMDLevel::AVX2> other) const {
        return simd32uint8_tpl(_mm256_and_si256(i, other.i));
    }

    simd32uint8_tpl operator+(simd32uint8_tpl other) const {
        return simd32uint8_tpl(_mm256_add_epi8(i, other.i));
    }

    simd32uint8_tpl lookup_2_lanes(simd32uint8_tpl idx) const {
        return simd32uint8_tpl(_mm256_shuffle_epi8(i, idx.i));
    }

    // extract + 0-extend lane
    // this operation is slow (3 cycles)
    simd16uint16_tpl<SIMDLevel::AVX2> lane0_as_uint16() const {
        __m128i x = _mm256_extracti128_si256(i, 0);
        return simd16uint16_tpl<SIMDLevel::AVX2>(_mm256_cvtepu8_epi16(x));
    }

    simd16uint16_tpl<SIMDLevel::AVX2> lane1_as_uint16() const {
        __m128i x = _mm256_extracti128_si256(i, 1);
        return simd16uint16_tpl<SIMDLevel::AVX2>(_mm256_cvtepu8_epi16(x));
    }

    simd32uint8_tpl& operator+=(simd32uint8_tpl other) {
        i = _mm256_add_epi8(i, other.i);
        return *this;
    }

    // for debugging only
    uint8_t operator[](int idx) const {
        ALIGNED(32) uint8_t tab[32];
        store(tab);
        return tab[idx];
    }
};

// convert with saturation
// careful: this does not cross lanes, so the order is weird
inline simd32uint8_tpl<SIMDLevel::AVX2> uint16_to_uint8_saturate(
        simd16uint16_tpl<SIMDLevel::AVX2> a,
        simd16uint16_tpl<SIMDLevel::AVX2> b) {
    return simd32uint8_tpl<SIMDLevel::AVX2>(_mm256_packs_epi16(a.i, b.i));
}

/// get most significant bit of each byte
inline uint32_t get_MSBs(simd32uint8_tpl<SIMDLevel::AVX2> a) {
    return _mm256_movemask_epi8(a.i);
}

/// use MSB of each byte of mask to select a byte between a and b
inline simd32uint8_tpl<SIMDLevel::AVX2> blendv(
        simd32uint8_tpl<SIMDLevel::AVX2> a,
        simd32uint8_tpl<SIMDLevel::AVX2> b,
        simd32uint8_tpl<SIMDLevel::AVX2> mask) {
    return simd32uint8_tpl<SIMDLevel::AVX2>(
            _mm256_blendv_epi8(a.i, b.i, mask.i));
}

/// vector of 8 unsigned 32-bit integers
template <>
struct simd8uint32_tpl<SIMDLevel::AVX2> : simd256bit_tpl<SIMDLevel::AVX2> {
    simd8uint32_tpl() {}

    explicit simd8uint32_tpl(__m256i val)
            : simd256bit_tpl<SIMDLevel::AVX2>(val) {}

    explicit simd8uint32_tpl(uint32_t x)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_set1_epi32(x)) {}

    explicit simd8uint32_tpl(simd256bit_tpl<SIMDLevel::AVX2> x)
            : simd256bit_tpl<SIMDLevel::AVX2>(x) {}

    explicit simd8uint32_tpl(const uint8_t* x)
            : simd256bit_tpl<SIMDLevel::AVX2>((const void*)x) {}

    explicit simd8uint32_tpl(
            uint32_t u0,
            uint32_t u1,
            uint32_t u2,
            uint32_t u3,
            uint32_t u4,
            uint32_t u5,
            uint32_t u6,
            uint32_t u7)
            : simd256bit_tpl<SIMDLevel::AVX2>(
                      _mm256_setr_epi32(u0, u1, u2, u3, u4, u5, u6, u7)) {}

    simd8uint32_tpl operator+(simd8uint32_tpl other) const {
        return simd8uint32_tpl(_mm256_add_epi32(i, other.i));
    }

    simd8uint32_tpl operator-(simd8uint32_tpl other) const {
        return simd8uint32_tpl(_mm256_sub_epi32(i, other.i));
    }

    simd8uint32_tpl& operator+=(const simd8uint32_tpl& other) {
        i = _mm256_add_epi32(i, other.i);
        return *this;
    }

    bool operator==(simd8uint32_tpl other) const {
        const __m256i pcmp = _mm256_cmpeq_epi32(i, other.i);
        unsigned bitmask = _mm256_movemask_epi8(pcmp);
        return (bitmask == 0xffffffffU);
    }

    bool operator!=(simd8uint32_tpl other) const {
        return !(*this == other);
    }

    std::string elements_to_string(const char* fmt) const {
        uint32_t bytes[8];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
        for (int idx = 0; idx < 8; idx++) {
            ptr += sprintf(ptr, fmt, bytes[idx]);
        }
#pragma GCC diagnostic pop
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%08x,");
    }

    std::string dec() const {
        return elements_to_string("%10d,");
    }

    void set1(uint32_t x) {
        i = _mm256_set1_epi32((int)x);
    }

    simd8uint32_tpl unzip() const {
        return simd8uint32_tpl(_mm256_permutevar8x32_epi32(
                i, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
    }
};

// Vectorized version of the following code:
//   for (size_t i = 0; i < n; i++) {
//      bool flag = (candidateValues[i] < currentValues[i]);
//      minValues[i] = flag ? candidateValues[i] : currentValues[i];
//      minIndices[i] = flag ? candidateIndices[i] : currentIndices[i];
//      maxValues[i] = !flag ? candidateValues[i] : currentValues[i];
//      maxIndices[i] = !flag ? candidateIndices[i] : currentIndices[i];
//   }
// Max indices evaluation is inaccurate in case of equal values (the index of
// the last equal value is saved instead of the first one), but this behavior
// saves instructions.
inline void cmplt_min_max_fast(
        const simd8uint32_tpl<SIMDLevel::AVX2> candidateValues,
        const simd8uint32_tpl<SIMDLevel::AVX2> candidateIndices,
        const simd8uint32_tpl<SIMDLevel::AVX2> currentValues,
        const simd8uint32_tpl<SIMDLevel::AVX2> currentIndices,
        simd8uint32_tpl<SIMDLevel::AVX2>& minValues,
        simd8uint32_tpl<SIMDLevel::AVX2>& minIndices,
        simd8uint32_tpl<SIMDLevel::AVX2>& maxValues,
        simd8uint32_tpl<SIMDLevel::AVX2>& maxIndices) {
    // there's no lt instruction, so we'll need to emulate one
    __m256i comparison = _mm256_cmpgt_epi32(currentValues.i, candidateValues.i);
    comparison = _mm256_andnot_si256(comparison, _mm256_set1_epi32(-1));

    minValues.i = _mm256_min_epi32(candidateValues.i, currentValues.i);
    minIndices.i = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(candidateIndices.i),
            _mm256_castsi256_ps(currentIndices.i),
            _mm256_castsi256_ps(comparison)));
    maxValues.i = _mm256_max_epi32(candidateValues.i, currentValues.i);
    maxIndices.i = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(currentIndices.i),
            _mm256_castsi256_ps(candidateIndices.i),
            _mm256_castsi256_ps(comparison)));
}

template <>
struct simd8float32_tpl<SIMDLevel::AVX2> : simd256bit_tpl<SIMDLevel::AVX2> {
    simd8float32_tpl() {}

    explicit simd8float32_tpl(simd256bit_tpl<SIMDLevel::AVX2> x)
            : simd256bit_tpl<SIMDLevel::AVX2>(x) {}

    explicit simd8float32_tpl(__m256 x) : simd256bit_tpl<SIMDLevel::AVX2>(x) {}

    explicit simd8float32_tpl(float x)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_set1_ps(x)) {}

    explicit simd8float32_tpl(const float* x)
            : simd256bit_tpl<SIMDLevel::AVX2>(_mm256_loadu_ps(x)) {}

    explicit simd8float32_tpl(
            float f0,
            float f1,
            float f2,
            float f3,
            float f4,
            float f5,
            float f6,
            float f7)
            : simd256bit_tpl<SIMDLevel::AVX2>(
                      _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7)) {}

    simd8float32_tpl operator*(simd8float32_tpl other) const {
        return simd8float32_tpl(_mm256_mul_ps(f, other.f));
    }

    simd8float32_tpl operator+(simd8float32_tpl other) const {
        return simd8float32_tpl(_mm256_add_ps(f, other.f));
    }

    simd8float32_tpl operator-(simd8float32_tpl other) const {
        return simd8float32_tpl(_mm256_sub_ps(f, other.f));
    }

    simd8float32_tpl& operator+=(const simd8float32_tpl& other) {
        f = _mm256_add_ps(f, other.f);
        return *this;
    }

    bool operator==(simd8float32_tpl other) const {
        const __m256i pcmp =
                _mm256_castps_si256(_mm256_cmp_ps(f, other.f, _CMP_EQ_OQ));
        unsigned bitmask = _mm256_movemask_epi8(pcmp);
        return (bitmask == 0xffffffffU);
    }

    bool operator!=(simd8float32_tpl other) const {
        return !(*this == other);
    }

    std::string tostring() const {
        float tab[8];
        storeu((void*)tab);
        char res[1000];
        char* ptr = res;
        for (int idx = 0; idx < 8; idx++) {
            ptr += sprintf(ptr, "%g,", tab[idx]);
        }
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }
};

inline simd8float32_tpl<SIMDLevel::AVX2> hadd(
        simd8float32_tpl<SIMDLevel::AVX2> a,
        simd8float32_tpl<SIMDLevel::AVX2> b) {
    return simd8float32_tpl<SIMDLevel::AVX2>(_mm256_hadd_ps(a.f, b.f));
}

inline simd8float32_tpl<SIMDLevel::AVX2> unpacklo(
        simd8float32_tpl<SIMDLevel::AVX2> a,
        simd8float32_tpl<SIMDLevel::AVX2> b) {
    return simd8float32_tpl<SIMDLevel::AVX2>(_mm256_unpacklo_ps(a.f, b.f));
}

inline simd8float32_tpl<SIMDLevel::AVX2> unpackhi(
        simd8float32_tpl<SIMDLevel::AVX2> a,
        simd8float32_tpl<SIMDLevel::AVX2> b) {
    return simd8float32_tpl<SIMDLevel::AVX2>(_mm256_unpackhi_ps(a.f, b.f));
}

// compute a * b + c
inline simd8float32_tpl<SIMDLevel::AVX2> fmadd(
        simd8float32_tpl<SIMDLevel::AVX2> a,
        simd8float32_tpl<SIMDLevel::AVX2> b,
        simd8float32_tpl<SIMDLevel::AVX2> c) {
    return simd8float32_tpl<SIMDLevel::AVX2>(_mm256_fmadd_ps(a.f, b.f, c.f));
}

// The following primitive is a vectorized version of the following code
// snippet:
//   float lowestValue = HUGE_VAL;
//   uint lowestIndex = 0;
//   for (size_t i = 0; i < n; i++) {
//     if (values[i] < lowestValue) {
//       lowestValue = values[i];
//       lowestIndex = i;
//     }
//   }
// Vectorized version can be implemented via two operations: cmp and blend
// with something like this:
//   lowestValues = [HUGE_VAL; 8];
//   lowestIndices = {0, 1, 2, 3, 4, 5, 6, 7};
//   for (size_t i = 0; i < n; i += 8) {
//     auto comparison = cmp(values + i, lowestValues);
//     lowestValues = blend(
//         comparison,
//         values + i,
//         lowestValues);
//     lowestIndices = blend(
//         comparison,
//         i + {0, 1, 2, 3, 4, 5, 6, 7},
//         lowestIndices);
//     lowestIndices += {8, 8, 8, 8, 8, 8, 8, 8};
//   }
// The problem is that blend primitive needs very different instruction
// order for AVX and ARM.
// So, let's introduce a combination of these two in order to avoid
// confusion for ppl who write in low-level SIMD instructions. Additionally,
// these two ops (cmp and blend) are very often used together.
inline void cmplt_and_blend_inplace(
        const simd8float32_tpl<SIMDLevel::AVX2> candidateValues,
        const simd8uint32_tpl<SIMDLevel::AVX2> candidateIndices,
        simd8float32_tpl<SIMDLevel::AVX2>& lowestValues,
        simd8uint32_tpl<SIMDLevel::AVX2>& lowestIndices) {
    const __m256 comparison =
            _mm256_cmp_ps(lowestValues.f, candidateValues.f, _CMP_LE_OS);
    lowestValues.f = _mm256_min_ps(candidateValues.f, lowestValues.f);
    lowestIndices.i = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(candidateIndices.i),
            _mm256_castsi256_ps(lowestIndices.i),
            comparison));
}

// Vectorized version of the following code:
//   for (size_t i = 0; i < n; i++) {
//      bool flag = (candidateValues[i] < currentValues[i]);
//      minValues[i] = flag ? candidateValues[i] : currentValues[i];
//      minIndices[i] = flag ? candidateIndices[i] : currentIndices[i];
//      maxValues[i] = !flag ? candidateValues[i] : currentValues[i];
//      maxIndices[i] = !flag ? candidateIndices[i] : currentIndices[i];
//   }
// Max indices evaluation is inaccurate in case of equal values (the index of
// the last equal value is saved instead of the first one), but this behavior
// saves instructions.
inline void cmplt_min_max_fast(
        const simd8float32_tpl<SIMDLevel::AVX2> candidateValues,
        const simd8uint32_tpl<SIMDLevel::AVX2> candidateIndices,
        const simd8float32_tpl<SIMDLevel::AVX2> currentValues,
        const simd8uint32_tpl<SIMDLevel::AVX2> currentIndices,
        simd8float32_tpl<SIMDLevel::AVX2>& minValues,
        simd8uint32_tpl<SIMDLevel::AVX2>& minIndices,
        simd8float32_tpl<SIMDLevel::AVX2>& maxValues,
        simd8uint32_tpl<SIMDLevel::AVX2>& maxIndices) {
    const __m256 comparison =
            _mm256_cmp_ps(currentValues.f, candidateValues.f, _CMP_LE_OS);
    minValues.f = _mm256_min_ps(candidateValues.f, currentValues.f);
    minIndices.i = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(candidateIndices.i),
            _mm256_castsi256_ps(currentIndices.i),
            comparison));
    maxValues.f = _mm256_max_ps(candidateValues.f, currentValues.f);
    maxIndices.i = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(currentIndices.i),
            _mm256_castsi256_ps(candidateIndices.i),
            comparison));
}

namespace {

// get even float32's of a and b, interleaved
inline simd8float32_tpl<SIMDLevel::AVX2> geteven(
        simd8float32_tpl<SIMDLevel::AVX2> a,
        simd8float32_tpl<SIMDLevel::AVX2> b) {
    return simd8float32_tpl<SIMDLevel::AVX2>(
            _mm256_shuffle_ps(a.f, b.f, 0 << 0 | 2 << 2 | 0 << 4 | 2 << 6));
}

// get odd float32's of a and b, interleaved
inline simd8float32_tpl<SIMDLevel::AVX2> getodd(
        simd8float32_tpl<SIMDLevel::AVX2> a,
        simd8float32_tpl<SIMDLevel::AVX2> b) {
    return simd8float32_tpl<SIMDLevel::AVX2>(
            _mm256_shuffle_ps(a.f, b.f, 1 << 0 | 3 << 2 | 1 << 4 | 3 << 6));
}

// 3 cycles
// if the lanes are a = [a0 a1] and b = [b0 b1], return [a0 b0]
inline simd8float32_tpl<SIMDLevel::AVX2> getlow128(
        simd8float32_tpl<SIMDLevel::AVX2> a,
        simd8float32_tpl<SIMDLevel::AVX2> b) {
    return simd8float32_tpl<SIMDLevel::AVX2>(
            _mm256_permute2f128_ps(a.f, b.f, 0 | 2 << 4));
}

inline simd8float32_tpl<SIMDLevel::AVX2> gethigh128(
        const simd8float32_tpl<SIMDLevel::AVX2>& a,
        const simd8float32_tpl<SIMDLevel::AVX2>& b) {
    return simd8float32_tpl<SIMDLevel::AVX2>(
            _mm256_permute2f128_ps(a.f, b.f, 1 | 3 << 4));
}

// horizontal add: sum all 8 floats in the register
inline float horizontal_add(const simd8float32_tpl<SIMDLevel::AVX2>& a) {
    __m128 sum = _mm_add_ps(
            _mm256_castps256_ps128(a.f), _mm256_extractf128_ps(a.f, 1));
    __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
    __m128 v1 = _mm_add_ps(sum, v0);
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 v3 = _mm_add_ps(v1, v2);
    return _mm_cvtss_f32(v3);
}

} // namespace

} // namespace faiss
