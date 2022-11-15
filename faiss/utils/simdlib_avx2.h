/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#include <immintrin.h>

#include <faiss/impl/platform_macros.h>

namespace faiss {

/** Simple wrapper around the AVX 256-bit registers
 *
 * The objective is to separate the different interpretations of the same
 * registers (as a vector of uint8, uint16 or uint32), to provide printing
 * functions, and to give more readable names to the AVX intrinsics. It does not
 * pretend to be exhausitve, functions are added as needed.
 */

/// 256-bit representation without interpretation as a vector
struct simd256bit {
    union {
        __m256i i;
        __m256 f;
    };

    simd256bit() {}

    explicit simd256bit(__m256i i) : i(i) {}

    explicit simd256bit(__m256 f) : f(f) {}

    explicit simd256bit(const void* x)
            : i(_mm256_load_si256((__m256i const*)x)) {}

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
        for (int i = 0; i < 256; i++) {
            bits[i] = '0' + ((bytes[i / 8] >> (i % 8)) & 1);
        }
        bits[256] = 0;
    }

    std::string bin() const {
        char bits[257];
        bin(bits);
        return std::string(bits);
    }
};

/// vector of 16 elements in uint16
struct simd16uint16 : simd256bit {
    simd16uint16() {}

    explicit simd16uint16(__m256i i) : simd256bit(i) {}

    explicit simd16uint16(int x) : simd256bit(_mm256_set1_epi16(x)) {}

    explicit simd16uint16(uint16_t x) : simd256bit(_mm256_set1_epi16(x)) {}

    explicit simd16uint16(simd256bit x) : simd256bit(x) {}

    explicit simd16uint16(const uint16_t* x) : simd256bit((const void*)x) {}

    std::string elements_to_string(const char* fmt) const {
        uint16_t bytes[16];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
        for (int i = 0; i < 16; i++) {
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
        i = _mm256_set1_epi16((short)x);
    }

    simd16uint16 operator*(const simd16uint16& other) const {
        return simd16uint16(_mm256_mullo_epi16(i, other.i));
    }

    // shift must be known at compile time
    simd16uint16 operator>>(const int shift) const {
        return simd16uint16(_mm256_srli_epi16(i, shift));
    }

    // shift must be known at compile time
    simd16uint16 operator<<(const int shift) const {
        return simd16uint16(_mm256_slli_epi16(i, shift));
    }

    simd16uint16 operator+=(simd16uint16 other) {
        i = _mm256_add_epi16(i, other.i);
        return *this;
    }

    simd16uint16 operator-=(simd16uint16 other) {
        i = _mm256_sub_epi16(i, other.i);
        return *this;
    }

    simd16uint16 operator+(simd16uint16 other) const {
        return simd16uint16(_mm256_add_epi16(i, other.i));
    }

    simd16uint16 operator-(simd16uint16 other) const {
        return simd16uint16(_mm256_sub_epi16(i, other.i));
    }

    simd16uint16 operator&(simd256bit other) const {
        return simd16uint16(_mm256_and_si256(i, other.i));
    }

    simd16uint16 operator|(simd256bit other) const {
        return simd16uint16(_mm256_or_si256(i, other.i));
    }

    // returns binary masks
    simd16uint16 operator==(simd256bit other) const {
        return simd16uint16(_mm256_cmpeq_epi16(i, other.i));
    }

    simd16uint16 operator~() const {
        return simd16uint16(_mm256_xor_si256(i, _mm256_set1_epi32(-1)));
    }

    // get scalar at index 0
    uint16_t get_scalar_0() const {
        return _mm256_extract_epi16(i, 0);
    }

    // mask of elements where this >= thresh
    // 2 bit per component: 16 * 2 = 32 bit
    uint32_t ge_mask(simd16uint16 thresh) const {
        __m256i j = thresh.i;
        __m256i max = _mm256_max_epu16(i, j);
        __m256i ge = _mm256_cmpeq_epi16(i, max);
        return _mm256_movemask_epi8(ge);
    }

    uint32_t le_mask(simd16uint16 thresh) const {
        return thresh.ge_mask(*this);
    }

    uint32_t gt_mask(simd16uint16 thresh) const {
        return ~le_mask(thresh);
    }

    bool all_gt(simd16uint16 thresh) const {
        return le_mask(thresh) == 0;
    }

    // for debugging only
    uint16_t operator[](int i) const {
        ALIGNED(32) uint16_t tab[16];
        store(tab);
        return tab[i];
    }

    void accu_min(simd16uint16 incoming) {
        i = _mm256_min_epu16(i, incoming.i);
    }

    void accu_max(simd16uint16 incoming) {
        i = _mm256_max_epu16(i, incoming.i);
    }
};

// not really a std::min because it returns an elementwise min
inline simd16uint16 min(simd16uint16 a, simd16uint16 b) {
    return simd16uint16(_mm256_min_epu16(a.i, b.i));
}

inline simd16uint16 max(simd16uint16 a, simd16uint16 b) {
    return simd16uint16(_mm256_max_epu16(a.i, b.i));
}

// decompose in 128-lanes: a = (a0, a1), b = (b0, b1)
// return (a0 + a1, b0 + b1)
// TODO find a better name
inline simd16uint16 combine2x2(simd16uint16 a, simd16uint16 b) {
    __m256i a1b0 = _mm256_permute2f128_si256(a.i, b.i, 0x21);
    __m256i a0b1 = _mm256_blend_epi32(a.i, b.i, 0xF0);

    return simd16uint16(a1b0) + simd16uint16(a0b1);
}

// compare d0 and d1 to thr, return 32 bits corresponding to the concatenation
// of d0 and d1 with thr
inline uint32_t cmp_ge32(simd16uint16 d0, simd16uint16 d1, simd16uint16 thr) {
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

inline uint32_t cmp_le32(simd16uint16 d0, simd16uint16 d1, simd16uint16 thr) {
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

// vector of 32 unsigned 8-bit integers
struct simd32uint8 : simd256bit {
    simd32uint8() {}

    explicit simd32uint8(__m256i i) : simd256bit(i) {}

    explicit simd32uint8(int x) : simd256bit(_mm256_set1_epi8(x)) {}

    explicit simd32uint8(uint8_t x) : simd256bit(_mm256_set1_epi8(x)) {}

    explicit simd32uint8(simd256bit x) : simd256bit(x) {}

    explicit simd32uint8(const uint8_t* x) : simd256bit((const void*)x) {}

    std::string elements_to_string(const char* fmt) const {
        uint8_t bytes[32];
        storeu((void*)bytes);
        char res[1000];
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

    void set1(uint8_t x) {
        i = _mm256_set1_epi8((char)x);
    }

    simd32uint8 operator&(simd256bit other) const {
        return simd32uint8(_mm256_and_si256(i, other.i));
    }

    simd32uint8 operator+(simd32uint8 other) const {
        return simd32uint8(_mm256_add_epi8(i, other.i));
    }

    simd32uint8 lookup_2_lanes(simd32uint8 idx) const {
        return simd32uint8(_mm256_shuffle_epi8(i, idx.i));
    }

    // extract + 0-extend lane
    // this operation is slow (3 cycles)
    simd16uint16 lane0_as_uint16() const {
        __m128i x = _mm256_extracti128_si256(i, 0);
        return simd16uint16(_mm256_cvtepu8_epi16(x));
    }

    simd16uint16 lane1_as_uint16() const {
        __m128i x = _mm256_extracti128_si256(i, 1);
        return simd16uint16(_mm256_cvtepu8_epi16(x));
    }

    simd32uint8 operator+=(simd32uint8 other) {
        i = _mm256_add_epi8(i, other.i);
        return *this;
    }

    // for debugging only
    uint8_t operator[](int i) const {
        ALIGNED(32) uint8_t tab[32];
        store(tab);
        return tab[i];
    }
};

// convert with saturation
// careful: this does not cross lanes, so the order is weird
inline simd32uint8 uint16_to_uint8_saturate(simd16uint16 a, simd16uint16 b) {
    return simd32uint8(_mm256_packs_epi16(a.i, b.i));
}

/// get most significant bit of each byte
inline uint32_t get_MSBs(simd32uint8 a) {
    return _mm256_movemask_epi8(a.i);
}

/// use MSB of each byte of mask to select a byte between a and b
inline simd32uint8 blendv(simd32uint8 a, simd32uint8 b, simd32uint8 mask) {
    return simd32uint8(_mm256_blendv_epi8(a.i, b.i, mask.i));
}

/// vector of 8 unsigned 32-bit integers
struct simd8uint32 : simd256bit {
    simd8uint32() {}

    explicit simd8uint32(__m256i i) : simd256bit(i) {}

    explicit simd8uint32(uint32_t x) : simd256bit(_mm256_set1_epi32(x)) {}

    explicit simd8uint32(simd256bit x) : simd256bit(x) {}

    explicit simd8uint32(const uint8_t* x) : simd256bit((const void*)x) {}

    explicit simd8uint32(
            uint32_t u0,
            uint32_t u1,
            uint32_t u2,
            uint32_t u3,
            uint32_t u4,
            uint32_t u5,
            uint32_t u6,
            uint32_t u7)
            : simd256bit(_mm256_setr_epi32(u0, u1, u2, u3, u4, u5, u6, u7)) {}

    simd8uint32 operator+(simd8uint32 other) const {
        return simd8uint32(_mm256_add_epi32(i, other.i));
    }

    simd8uint32 operator-(simd8uint32 other) const {
        return simd8uint32(_mm256_sub_epi32(i, other.i));
    }

    bool operator==(simd8uint32 other) const {
        const __m256i pcmp = _mm256_cmpeq_epi32(i, other.i);
        unsigned bitmask = _mm256_movemask_epi8(pcmp);
        return (bitmask == 0xffffffffU);
    }

    bool operator!=(simd8uint32 other) const {
        return !(*this == other);
    }

    std::string elements_to_string(const char* fmt) const {
        uint32_t bytes[8];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
        for (int i = 0; i < 8; i++) {
            ptr += sprintf(ptr, fmt, bytes[i]);
        }
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
};

struct simd8float32 : simd256bit {
    simd8float32() {}

    explicit simd8float32(simd256bit x) : simd256bit(x) {}

    explicit simd8float32(__m256 x) : simd256bit(x) {}

    explicit simd8float32(float x) : simd256bit(_mm256_set1_ps(x)) {}

    explicit simd8float32(const float* x) : simd256bit(_mm256_loadu_ps(x)) {}

    explicit simd8float32(
            float f0,
            float f1,
            float f2,
            float f3,
            float f4,
            float f5,
            float f6,
            float f7)
            : simd256bit(_mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7)) {}

    simd8float32 operator*(simd8float32 other) const {
        return simd8float32(_mm256_mul_ps(f, other.f));
    }

    simd8float32 operator+(simd8float32 other) const {
        return simd8float32(_mm256_add_ps(f, other.f));
    }

    simd8float32 operator-(simd8float32 other) const {
        return simd8float32(_mm256_sub_ps(f, other.f));
    }

    bool operator==(simd8float32 other) const {
        const __m256i pcmp =
                _mm256_castps_si256(_mm256_cmp_ps(f, other.f, _CMP_EQ_OQ));
        unsigned bitmask = _mm256_movemask_epi8(pcmp);
        return (bitmask == 0xffffffffU);
    }

    bool operator!=(simd8float32 other) const {
        return !(*this == other);
    }

    std::string tostring() const {
        float tab[8];
        storeu((void*)tab);
        char res[1000];
        char* ptr = res;
        for (int i = 0; i < 8; i++) {
            ptr += sprintf(ptr, "%g,", tab[i]);
        }
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }
};

inline simd8float32 hadd(simd8float32 a, simd8float32 b) {
    return simd8float32(_mm256_hadd_ps(a.f, b.f));
}

inline simd8float32 unpacklo(simd8float32 a, simd8float32 b) {
    return simd8float32(_mm256_unpacklo_ps(a.f, b.f));
}

inline simd8float32 unpackhi(simd8float32 a, simd8float32 b) {
    return simd8float32(_mm256_unpackhi_ps(a.f, b.f));
}

// compute a * b + c
inline simd8float32 fmadd(simd8float32 a, simd8float32 b, simd8float32 c) {
    return simd8float32(_mm256_fmadd_ps(a.f, b.f, c.f));
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
        const simd8float32 candidateValues,
        const simd8uint32 candidateIndices,
        simd8float32& lowestValues,
        simd8uint32& lowestIndices) {
    auto comparison =
            _mm256_cmp_ps(lowestValues.f, candidateValues.f, _CMP_LE_OS);
    lowestValues.f =
            _mm256_blendv_ps(candidateValues.f, lowestValues.f, comparison);
    lowestIndices.i = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(candidateIndices.i),
            _mm256_castsi256_ps(lowestIndices.i),
            comparison));
}

namespace {

// get even float32's of a and b, interleaved
inline simd8float32 geteven(simd8float32 a, simd8float32 b) {
    return simd8float32(
            _mm256_shuffle_ps(a.f, b.f, 0 << 0 | 2 << 2 | 0 << 4 | 2 << 6));
}

// get odd float32's of a and b, interleaved
inline simd8float32 getodd(simd8float32 a, simd8float32 b) {
    return simd8float32(
            _mm256_shuffle_ps(a.f, b.f, 1 << 0 | 3 << 2 | 1 << 4 | 3 << 6));
}

// 3 cycles
// if the lanes are a = [a0 a1] and b = [b0 b1], return [a0 b0]
inline simd8float32 getlow128(simd8float32 a, simd8float32 b) {
    return simd8float32(_mm256_permute2f128_ps(a.f, b.f, 0 | 2 << 4));
}

inline simd8float32 gethigh128(simd8float32 a, simd8float32 b) {
    return simd8float32(_mm256_permute2f128_ps(a.f, b.f, 1 | 3 << 4));
}

} // namespace

} // namespace faiss
