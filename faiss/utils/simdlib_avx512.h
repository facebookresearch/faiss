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

#include <faiss/utils/simdlib_avx2.h>

namespace faiss {

/** Simple wrapper around the AVX 512-bit registers
 *
 * The objective is to separate the different interpretations of the same
 * registers (as a vector of uint8, uint16 or uint32), to provide printing
 * functions, and to give more readable names to the AVX intrinsics. It does not
 * pretend to be exhausitve, functions are added as needed.
 */

/// 512-bit representation without interpretation as a vector
struct simd512bit {
    union {
        __m512i i;
        __m512 f;
    };

    simd512bit() {}

    explicit simd512bit(__m512i i) : i(i) {}

    explicit simd512bit(__m512 f) : f(f) {}

    explicit simd512bit(const void* x)
            : i(_mm512_loadu_si512((__m512i const*)x)) {}

    // sets up a lower half of the register while keeping upper one as zero
    explicit simd512bit(simd256bit lo)
            : simd512bit(_mm512_inserti32x8(
                      _mm512_castsi256_si512(lo.i),
                      _mm256_setzero_si256(),
                      1)) {}

    // constructs from lower and upper halves
    explicit simd512bit(simd256bit lo, simd256bit hi)
            : simd512bit(_mm512_inserti32x8(
                      _mm512_castsi256_si512(lo.i),
                      hi.i,
                      1)) {}

    void clear() {
        i = _mm512_setzero_si512();
    }

    void storeu(void* ptr) const {
        _mm512_storeu_si512((__m512i*)ptr, i);
    }

    void loadu(const void* ptr) {
        i = _mm512_loadu_si512((__m512i*)ptr);
    }

    void store(void* ptr) const {
        _mm512_storeu_si512((__m512i*)ptr, i);
    }

    void bin(char bits[513]) const {
        char bytes[64];
        storeu((void*)bytes);
        for (int i = 0; i < 512; i++) {
            bits[i] = '0' + ((bytes[i / 8] >> (i % 8)) & 1);
        }
        bits[512] = 0;
    }

    std::string bin() const {
        char bits[257];
        bin(bits);
        return std::string(bits);
    }
};

/// vector of 32 elements in uint16
struct simd32uint16 : simd512bit {
    simd32uint16() {}

    explicit simd32uint16(__m512i i) : simd512bit(i) {}

    explicit simd32uint16(int x) : simd512bit(_mm512_set1_epi16(x)) {}

    explicit simd32uint16(uint16_t x) : simd512bit(_mm512_set1_epi16(x)) {}

    explicit simd32uint16(simd512bit x) : simd512bit(x) {}

    explicit simd32uint16(const uint16_t* x) : simd512bit((const void*)x) {}

    // sets up a lower half of the register
    explicit simd32uint16(simd256bit lo) : simd512bit(lo) {}

    // constructs from lower and upper halves
    explicit simd32uint16(simd256bit lo, simd256bit hi) : simd512bit(lo, hi) {}

    std::string elements_to_string(const char* fmt) const {
        uint16_t bytes[32];
        storeu((void*)bytes);
        char res[2000];
        char* ptr = res;
        for (int i = 0; i < 32; i++) {
            ptr += sprintf(ptr, fmt, bytes[i]);
        }
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
        i = _mm512_set1_epi16((short)x);
    }

    simd32uint16 operator*(const simd32uint16& other) const {
        return simd32uint16(_mm512_mullo_epi16(i, other.i));
    }

    // shift must be known at compile time
    simd32uint16 operator>>(const int shift) const {
        return simd32uint16(_mm512_srli_epi16(i, shift));
    }

    // shift must be known at compile time
    simd32uint16 operator<<(const int shift) const {
        return simd32uint16(_mm512_slli_epi16(i, shift));
    }

    simd32uint16 operator+=(simd32uint16 other) {
        i = _mm512_add_epi16(i, other.i);
        return *this;
    }

    simd32uint16 operator-=(simd32uint16 other) {
        i = _mm512_sub_epi16(i, other.i);
        return *this;
    }

    simd32uint16 operator+(simd32uint16 other) const {
        return simd32uint16(_mm512_add_epi16(i, other.i));
    }

    simd32uint16 operator-(simd32uint16 other) const {
        return simd32uint16(_mm512_sub_epi16(i, other.i));
    }

    simd32uint16 operator&(simd512bit other) const {
        return simd32uint16(_mm512_and_si512(i, other.i));
    }

    simd32uint16 operator|(simd512bit other) const {
        return simd32uint16(_mm512_or_si512(i, other.i));
    }

    simd32uint16 operator^(simd512bit other) const {
        return simd32uint16(_mm512_xor_si512(i, other.i));
    }

    simd32uint16 operator~() const {
        return simd32uint16(_mm512_xor_si512(i, _mm512_set1_epi32(-1)));
    }

    simd16uint16 low() const {
        return simd16uint16(_mm512_castsi512_si256(i));
    }

    simd16uint16 high() const {
        return simd16uint16(_mm512_extracti32x8_epi32(i, 1));
    }

    // for debugging only
    uint16_t operator[](int i) const {
        ALIGNED(64) uint16_t tab[32];
        store(tab);
        return tab[i];
    }

    void accu_min(simd32uint16 incoming) {
        i = _mm512_min_epu16(i, incoming.i);
    }

    void accu_max(simd32uint16 incoming) {
        i = _mm512_max_epu16(i, incoming.i);
    }
};

// decompose in 128-lanes: a = (a0, a1, a2, a3), b = (b0, b1, b2, b3)
// return (a0 + a1 + a2 + a3, b0 + b1 + b2 + b3)
inline simd16uint16 combine4x2(simd32uint16 a, simd32uint16 b) {
    return combine2x2(a.low(), b.low()) + combine2x2(a.high(), b.high());
}

// vector of 32 unsigned 8-bit integers
struct simd64uint8 : simd512bit {
    simd64uint8() {}

    explicit simd64uint8(__m512i i) : simd512bit(i) {}

    explicit simd64uint8(int x) : simd512bit(_mm512_set1_epi8(x)) {}

    explicit simd64uint8(uint8_t x) : simd512bit(_mm512_set1_epi8(x)) {}

    // sets up a lower half of the register
    explicit simd64uint8(simd256bit lo) : simd512bit(lo) {}

    // constructs from lower and upper halves
    explicit simd64uint8(simd256bit lo, simd256bit hi) : simd512bit(lo, hi) {}

    explicit simd64uint8(simd512bit x) : simd512bit(x) {}

    explicit simd64uint8(const uint8_t* x) : simd512bit((const void*)x) {}

    std::string elements_to_string(const char* fmt) const {
        uint8_t bytes[64];
        storeu((void*)bytes);
        char res[2000];
        char* ptr = res;
        for (int i = 0; i < 64; i++) {
            ptr += sprintf(ptr, fmt, bytes[i]);
        }
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
        i = _mm512_set1_epi8((char)x);
    }

    simd64uint8 operator&(simd512bit other) const {
        return simd64uint8(_mm512_and_si512(i, other.i));
    }

    simd64uint8 operator+(simd64uint8 other) const {
        return simd64uint8(_mm512_add_epi8(i, other.i));
    }

    simd64uint8 lookup_4_lanes(simd64uint8 idx) const {
        return simd64uint8(_mm512_shuffle_epi8(i, idx.i));
    }

    // extract + 0-extend lane
    // this operation is slow (3 cycles)
    simd32uint16 lane0_as_uint16() const {
        __m256i x = _mm512_extracti32x8_epi32(i, 0);
        return simd32uint16(_mm512_cvtepu8_epi16(x));
    }

    simd32uint16 lane1_as_uint16() const {
        __m256i x = _mm512_extracti32x8_epi32(i, 1);
        return simd32uint16(_mm512_cvtepu8_epi16(x));
    }

    simd64uint8 operator+=(simd64uint8 other) {
        i = _mm512_add_epi8(i, other.i);
        return *this;
    }

    // for debugging only
    uint8_t operator[](int i) const {
        ALIGNED(64) uint8_t tab[64];
        store(tab);
        return tab[i];
    }
};

} // namespace faiss
