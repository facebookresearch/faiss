#pragma once

#include <string>
#include <cstdint>

namespace faiss {

#ifdef __AVX2__

#include <immintrin.h>

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

    simd256bit()   {}

    simd256bit(__m256i i): i(i) {}
    simd256bit(__m256 f): f(f) {}

    simd256bit(const void *x):
    i(_mm256_load_si256((__m256i const *)x))
    {}

    void clear() {
        i = _mm256_setzero_si256();
    }

    void storeu(void *ptr) const {
        _mm256_storeu_si256((__m256i *)ptr, i);
    }

    void loadu(const void *ptr) {
        i = _mm256_loadu_si256((__m256i*)ptr);
    }

    void store(void *ptr) const {
        _mm256_store_si256((__m256i *)ptr, i);
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
struct simd16uint16: simd256bit {
    simd16uint16() {}

    simd16uint16(int x): simd256bit(_mm256_set1_epi16(x)) {}

    simd16uint16(uint16_t x): simd256bit(_mm256_set1_epi16(x)) {}

    simd16uint16(simd256bit x): simd256bit(x) {}

    simd16uint16(const uint16_t *x): simd256bit((const void*)x) {}

    std::string elements_to_string(const char * fmt) const {
        uint16_t bytes[16];
        storeu((void*)bytes);
        char res[1000], *ptr = res;
        for(int i = 0; i < 16; i++) {
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

    // shift must be known at compile time
    simd16uint16 operator >> (const int shift) const {
        return simd16uint16(_mm256_srli_epi16(i, shift));
    }

    // shift must be known at compile time
    simd16uint16 operator << (const int shift) const {
        return simd16uint16(_mm256_slli_epi16(i, shift));
    }

    simd16uint16 operator += (simd16uint16 other) {
        i = _mm256_add_epi16(i, other.i);
        return *this;
    }

    simd16uint16 operator -= (simd16uint16 other) {
        i = _mm256_sub_epi16(i, other.i);
        return *this;
    }

    simd16uint16 operator + (simd16uint16 other) const {
        return simd16uint16(_mm256_add_epi16(i, other.i));
    }

    simd16uint16 operator - (simd16uint16 other) const {
        return simd16uint16(_mm256_sub_epi16(i, other.i));
    }

    simd16uint16 operator & (simd256bit other) const {
        return simd16uint16(_mm256_and_si256(i, other.i));
    }

    simd16uint16 operator | (simd256bit other) const {
        return simd16uint16(_mm256_or_si256(i, other.i));
    }

    simd16uint16 operator == (simd256bit other) const {
        return simd16uint16(_mm256_cmpeq_epi16(i, other.i));
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
    uint16_t operator [] (int i) const {
        uint16_t tab[16] __attribute__ ((aligned (32)));
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
struct simd32uint8: simd256bit {

    simd32uint8() {}

    simd32uint8(int x): simd256bit(_mm256_set1_epi8(x)) {}

    simd32uint8(uint8_t x): simd256bit(_mm256_set1_epi8(x)) {}

    simd32uint8(simd256bit x): simd256bit(x) {}

    simd32uint8(const uint8_t *x): simd256bit((const void*)x) {}

    std::string elements_to_string(const char * fmt) const {
        uint8_t bytes[32];
        storeu((void*)bytes);
        char res[1000], *ptr = res;
        for(int i = 0; i < 32; i++) {
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

    simd32uint8 operator & (simd256bit other) const {
        return simd32uint8(_mm256_and_si256(i, other.i));
    }

    simd32uint8 operator + (simd32uint8 other) const {
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

    simd32uint8 operator += (simd32uint8 other) {
        i = _mm256_add_epi8(i, other.i);
        return *this;
    }

    // for debugging only
    uint8_t operator [] (int i) const {
        uint8_t tab[32] __attribute__ ((aligned (32)));
        store(tab);
        return tab[i];
    }

};

/// vector of 8 unsigned 32-bit integers
struct simd8uint32: simd256bit {
    simd8uint32() {}


    simd8uint32(uint32_t x): simd256bit(_mm256_set1_epi32(x)) {}

    simd8uint32(simd256bit x): simd256bit(x) {}

    simd8uint32(const uint8_t *x): simd256bit((const void*)x) {}

    std::string elements_to_string(const char * fmt) const {
        uint32_t bytes[8];
        storeu((void*)bytes);
        char res[1000], *ptr = res;
        for(int i = 0; i < 8; i++) {
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

struct simd8float32: simd256bit {

    simd8float32() {}

    simd8float32(simd256bit x): simd256bit(x) {}

    simd8float32(float x): simd256bit(_mm256_set1_ps(x)) {}

    simd8float32(const float *x): simd256bit(_mm256_load_ps(x)) {}

    simd8float32 operator * (simd8float32 other) const {
        return simd8float32(_mm256_mul_ps(f, other.f));
    }

    simd8float32 operator + (simd8float32 other) const {
        return simd8float32(_mm256_add_ps(f, other.f));
    }

    simd8float32 operator - (simd8float32 other) const {
        return simd8float32(_mm256_sub_ps(f, other.f));
    }

    std::string tostring() const {
        float tab[8];
        storeu((void*)tab);
        char res[1000], *ptr = res;
        for(int i = 0; i < 8; i++) {
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

#else

/*********************************************************************
 * Eventullay it would be good to have a scalar emulation of SIMD
 * registers. Leave empty for now.
 *********************************************************************/


#endif

} // namespace faiss
