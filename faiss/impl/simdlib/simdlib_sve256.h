/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#include <arm_sve.h>

#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simdlib/simdlib.h>
#include <faiss/impl/simdlib/simdlib_neon.h>

typedef svuint8_t uvec8 __attribute__((arm_sve_vector_bits(256)));
typedef svuint16_t uvec16 __attribute__((arm_sve_vector_bits(256)));
typedef svuint32_t uvec32 __attribute__((arm_sve_vector_bits(256)));
typedef svuint64_t uvec64 __attribute__((arm_sve_vector_bits(256)));
typedef svfloat32_t fvec32 __attribute__((arm_sve_vector_bits(256)));

namespace faiss {

template <>
struct simd256bit_tpl<SIMDLevel::ARM_SVE> {
    union {
        uvec8 vect_u8;
        uvec16 vect_u16;
        uvec32 vect_u32;
        fvec32 vect_f32;
    };

    simd256bit_tpl() {}

    explicit simd256bit_tpl(uvec8 val) : vect_u8(val) {}

    explicit simd256bit_tpl(uvec16 val) : vect_u16(val) {}

    explicit simd256bit_tpl(uvec32 val) : vect_u32(val) {}

    explicit simd256bit_tpl(fvec32 val) : vect_f32(val) {}

    explicit simd256bit_tpl(const void* x) {
        this->vect_u32 = svld1_u32(svptrue_pat_b32(SV_VL8), (const uint32_t*)x);
    }

    void clear() {
        vect_u32 = svdup_n_u32(0);
    }

    void storeu(void* ptr) const {
        svst1_u32(svptrue_pat_b32(SV_VL8), (uint32_t*)ptr, vect_u32);
    }

    void loadu(const void* ptr) {
        vect_u32 = svld1_u32(svptrue_pat_b32(SV_VL8), (const uint32_t*)ptr);
    }

    void store(void* ptr) const {
        svst1_u32(svptrue_pat_b32(SV_VL8), (uint32_t*)ptr, vect_u32);
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

    bool is_same_as(simd256bit_tpl other) const {
        auto pg =
                svcmpne_u32(svptrue_pat_b32(SV_VL8), other.vect_u32, vect_u32);
        return !svptest_any(svptrue_pat_b32(SV_VL8), pg);
    }
};

template <>
struct simd16uint16_tpl<SIMDLevel::ARM_SVE>
        : simd256bit_tpl<SIMDLevel::ARM_SVE> {
    simd16uint16_tpl() {}

    explicit simd16uint16_tpl(uvec16 val)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(val) {}

    explicit simd16uint16_tpl(int x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(svdup_n_u16(x)) {}

    explicit simd16uint16_tpl(uint16_t x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(svdup_n_u16(x)) {}

    explicit simd16uint16_tpl(simd256bit_tpl<SIMDLevel::ARM_SVE> x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(x) {}

    explicit simd16uint16_tpl(const uint16_t* x) {
        this->vect_u16 = svld1_u16(svptrue_pat_b16(SV_VL16), x);
    }

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
            uint16_t u15) {
        uint16_t temp[16] = {
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
                u15};
        this->vect_u16 = svld1_u16(svptrue_pat_b16(SV_VL16), temp);
    }

    std::string elements_to_string(const char* fmt) const {
        uint16_t bytes[16];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
        for (int idx = 0; idx < 16; idx++) {
            ptr += snprintf(
                    ptr, (size_t)(res + sizeof(res) - ptr), fmt, bytes[idx]);
        }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }
    std::string dec() const {
        return elements_to_string("%03d,");
    }

    void set1(uint16_t x) {
        vect_u16 = svdup_n_u16(x);
    }

    void loadu(const void* ptr) {
        vect_u16 = svld1_u16(svptrue_pat_b16(SV_VL16), (const uint16_t*)ptr);
    }

    simd16uint16_tpl operator*(const simd16uint16_tpl& other) const {
        return simd16uint16_tpl(svmul_u16_x(
                svptrue_pat_b16(SV_VL16), other.vect_u16, vect_u16));
    }

    simd16uint16_tpl operator>>(const int shift) const {
        return simd16uint16_tpl(svlsr_wide_n_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, (uint64_t)shift));
    }

    simd16uint16_tpl operator<<(const int shift) const {
        return simd16uint16_tpl(svlsl_wide_n_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, (uint64_t)shift));
    }

    simd16uint16_tpl& operator+=(simd16uint16_tpl other) {
        vect_u16 =
                svadd_u16_x(svptrue_pat_b16(SV_VL16), vect_u16, other.vect_u16);
        return *this;
    }

    simd16uint16_tpl& operator-=(simd16uint16_tpl other) {
        vect_u16 =
                svsub_u16_x(svptrue_pat_b16(SV_VL16), vect_u16, other.vect_u16);
        return *this;
    }

    simd16uint16_tpl operator+(simd16uint16_tpl other) const {
        return simd16uint16_tpl(svadd_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, other.vect_u16));
    }

    simd16uint16_tpl operator-(simd16uint16_tpl other) const {
        return simd16uint16_tpl(svsub_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, other.vect_u16));
    }

    simd16uint16_tpl operator&(simd256bit_tpl<SIMDLevel::ARM_SVE> other) const {
        return simd16uint16_tpl(svand_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, other.vect_u16));
    }

    simd16uint16_tpl operator|(simd256bit_tpl<SIMDLevel::ARM_SVE> other) const {
        return simd16uint16_tpl(svorr_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, other.vect_u16));
    }

    simd16uint16_tpl operator^(simd256bit_tpl<SIMDLevel::ARM_SVE> other) const {
        return simd16uint16_tpl(sveor_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, other.vect_u16));
    }

    friend simd16uint16_tpl operator==(
            const simd256bit_tpl<SIMDLevel::ARM_SVE> lhs,
            const simd256bit_tpl<SIMDLevel::ARM_SVE> rhs) {
        return simd16uint16_tpl(svdup_n_u16_z(
                svcmpeq_u16(
                        svptrue_pat_b16(SV_VL16), lhs.vect_u16, rhs.vect_u16),
                0xffff));
    }

    simd16uint16_tpl operator~() const {
        return simd16uint16_tpl(sveor_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, svdup_n_u16(0xffff)));
    }

    uint16_t get_scalar_0() const {
        return svlasta_u16(svpfalse_b(), vect_u16);
    }

    uint32_t ge_mask(simd16uint16_tpl thresh) const {
        auto pg = svcmpge_u16(
                svptrue_pat_b16(SV_VL16), vect_u16, thresh.vect_u16);
        pg = svtrn1_b8(pg, pg);
        uint32_t res = *(uint32_t*)&pg;
        return res;
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

    uint16_t operator[](int idx) const {
        ALIGNED(32) uint16_t tab[16];
        store(tab);
        return tab[idx];
    }

    void accu_min(simd16uint16_tpl incoming) {
        vect_u16 = svmin_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, incoming.vect_u16);
    }

    void accu_max(simd16uint16_tpl incoming) {
        vect_u16 = svmax_u16_x(
                svptrue_pat_b16(SV_VL16), vect_u16, incoming.vect_u16);
    }

    void pq4_accumulate_low_bytes(const simd32uint8_tpl<SIMDLevel::ARM_SVE>& r);

    void pq4_accumulate_high_bytes(
            const simd32uint8_tpl<SIMDLevel::ARM_SVE>& r);
};

inline simd16uint16_tpl<SIMDLevel::ARM_SVE> min(
        simd16uint16_tpl<SIMDLevel::ARM_SVE> a,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> b) {
    return simd16uint16_tpl<SIMDLevel::ARM_SVE>(
            svmin_u16_x(svptrue_pat_b16(SV_VL16), a.vect_u16, b.vect_u16));
}

inline simd16uint16_tpl<SIMDLevel::ARM_SVE> max(
        simd16uint16_tpl<SIMDLevel::ARM_SVE> a,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> b) {
    return simd16uint16_tpl<SIMDLevel::ARM_SVE>(
            svmax_u16_x(svptrue_pat_b16(SV_VL16), a.vect_u16, b.vect_u16));
}

inline simd16uint16_tpl<SIMDLevel::ARM_SVE> combine2x2(
        simd16uint16_tpl<SIMDLevel::ARM_SVE> a,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> b) {
    svuint16_t z = svadd_u16_x(
            svptrue_pat_b16(SV_VL16),
            svzip1_u16(a.vect_u16, b.vect_u16),
            svzip2_u16(a.vect_u16, b.vect_u16));
    svuint16_t lo = svuzp1_u16(z, z); // a[j]+a[j+8], j=0..7
    svuint16_t hi = svuzp2_u16(z, z); // b[j]+b[j+8], j=0..7
    svuint16_t r = svsplice_u16(svptrue_pat_b16(SV_VL8), lo, hi);
    return simd16uint16_tpl<SIMDLevel::ARM_SVE>(r);
}

inline uint32_t cmp_ge32(
        simd16uint16_tpl<SIMDLevel::ARM_SVE> d0,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> d1,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> thr) {
    auto low = svcmpge_u16(svptrue_pat_b16(SV_VL16), d0.vect_u16, thr.vect_u16);
    auto high =
            svcmpge_u16(svptrue_pat_b16(SV_VL16), d1.vect_u16, thr.vect_u16);
    auto pg = svuzp1_b8(low, high);
    uint32_t res = *(uint32_t*)&pg;
    return res;
}

inline uint32_t cmp_le32(
        simd16uint16_tpl<SIMDLevel::ARM_SVE> d0,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> d1,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> thr) {
    auto low = svcmple_u16(svptrue_pat_b16(SV_VL16), d0.vect_u16, thr.vect_u16);
    auto high =
            svcmple_u16(svptrue_pat_b16(SV_VL16), d1.vect_u16, thr.vect_u16);
    auto pg = svuzp1_b8(low, high);
    uint32_t res = *(uint32_t*)&pg;
    return res;
}

inline simd16uint16_tpl<SIMDLevel::ARM_SVE> hadd(
        simd16uint16_tpl<SIMDLevel::ARM_SVE> a,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> b) {
    const svbool_t pg8 = svptrue_pat_b32(SV_VL8);
    svuint32_t au = svreinterpret_u32_u16(a.vect_u16);
    svuint32_t bu = svreinterpret_u32_u16(b.vect_u16);
    svuint32_t ps_a = svadd_u32_x(
            pg8, svand_n_u32_x(pg8, au, 0xffff), svlsr_n_u32_x(pg8, au, 16));
    svuint32_t ps_b = svadd_u32_x(
            pg8, svand_n_u32_x(pg8, bu, 0xffff), svlsr_n_u32_x(pg8, bu, 16));
    svuint16_t c = svuzp1_u16(
            svreinterpret_u16_u32(ps_a), svreinterpret_u16_u32(ps_b));
    static const uint16_t sel[16] = {
            0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};
    return simd16uint16_tpl<SIMDLevel::ARM_SVE>(
            svtbl_u16(c, svld1_u16(svptrue_pat_b16(SV_VL16), sel)));
}

inline void cmplt_min_max_fast(
        const simd16uint16_tpl<SIMDLevel::ARM_SVE> candidateValues,
        const simd16uint16_tpl<SIMDLevel::ARM_SVE> candidateIndices,
        const simd16uint16_tpl<SIMDLevel::ARM_SVE> currentValues,
        const simd16uint16_tpl<SIMDLevel::ARM_SVE> currentIndices,
        simd16uint16_tpl<SIMDLevel::ARM_SVE>& minValue,
        simd16uint16_tpl<SIMDLevel::ARM_SVE>& minIndices,
        simd16uint16_tpl<SIMDLevel::ARM_SVE>& maxValue,
        simd16uint16_tpl<SIMDLevel::ARM_SVE>& maxIndices) {
    auto pg = svcmplt_u16(
            svptrue_pat_b16(SV_VL16),
            candidateValues.vect_u16,
            currentValues.vect_u16);
    minValue = simd16uint16_tpl<SIMDLevel::ARM_SVE>(
            svsel_u16(pg, candidateValues.vect_u16, currentValues.vect_u16));
    minIndices = simd16uint16_tpl<SIMDLevel::ARM_SVE>(
            svsel_u16(pg, candidateIndices.vect_u16, currentIndices.vect_u16));
    maxValue = simd16uint16_tpl<SIMDLevel::ARM_SVE>(
            svsel_u16(pg, currentValues.vect_u16, candidateValues.vect_u16));
    maxIndices = simd16uint16_tpl<SIMDLevel::ARM_SVE>(
            svsel_u16(pg, currentIndices.vect_u16, candidateIndices.vect_u16));
}

template <>
struct simd32uint8_tpl<SIMDLevel::ARM_SVE>
        : simd256bit_tpl<SIMDLevel::ARM_SVE> {
    simd32uint8_tpl() {}

    explicit simd32uint8_tpl(uvec8 val)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(val) {}

    explicit simd32uint8_tpl(uvec16 val)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(val) {}

    explicit simd32uint8_tpl(simd16uint16_tpl<SIMDLevel::ARM_SVE> val)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(val.vect_u16) {}

    explicit simd32uint8_tpl(int x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(svdup_n_u8(x)) {}

    explicit simd32uint8_tpl(uint8_t x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(svdup_n_u8(x)) {}

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
        uint8_t ds[32] = {_0,  _1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9,  _10,
                          _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21,
                          _22, _23, _24, _25, _26, _27, _28, _29, _30, _31};
        return simd32uint8_tpl(svld1_u8(svptrue_pat_b8(SV_VL32), ds));
    }

    explicit simd32uint8_tpl(simd256bit_tpl<SIMDLevel::ARM_SVE> x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(x) {}

    explicit simd32uint8_tpl(const uint8_t* x) {
        this->vect_u8 = svld1_u8(svptrue_pat_b8(SV_VL32), x);
    }

    std::string elements_to_string(const char* fmt) const {
        uint8_t bytes[32];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
        for (int idx = 0; idx < 32; idx++) {
            ptr += snprintf(
                    ptr, (size_t)(res + sizeof(res) - ptr), fmt, bytes[idx]);
        }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }
    std::string dec() const {
        return elements_to_string("%03d,");
    }

    void set1(uint8_t x) {
        vect_u8 = svdup_n_u8(x);
    }

    void loadu(const void* ptr) {
        vect_u8 = svld1_u8(svptrue_pat_b8(SV_VL32), (const uint8_t*)ptr);
    }

    simd32uint8_tpl operator&(simd256bit_tpl<SIMDLevel::ARM_SVE> other) const {
        return simd32uint8_tpl(
                svand_u8_x(svptrue_pat_b8(SV_VL32), vect_u8, other.vect_u8));
    }

    simd32uint8_tpl operator+(simd32uint8_tpl other) const {
        return simd32uint8_tpl(
                svadd_u8_x(svptrue_pat_b8(SV_VL32), vect_u8, other.vect_u8));
    }

    static svuint8_t lookup_2_lanes_bias() {
        return svand_n_u8_x(svptrue_pat_b8(SV_VL32), svindex_u8(0, 1), 0x10);
    }

    simd32uint8_tpl lookup_2_lanes(simd32uint8_tpl idx) const {
        svuint8_t ii = svorr_u8_x(
                svptrue_pat_b8(SV_VL32), idx.vect_u8, lookup_2_lanes_bias());
        return simd32uint8_tpl(svtbl_u8(vect_u8, ii));
    }

    simd32uint8_tpl<SIMDLevel::ARM_SVE> lane0_as_uint16() const {
        return simd32uint8_tpl<SIMDLevel::ARM_SVE>(svunpklo_u16(vect_u8));
    }

    simd32uint8_tpl<SIMDLevel::ARM_SVE> lane1_as_uint16() const {
        return simd32uint8_tpl<SIMDLevel::ARM_SVE>(svunpkhi_u16(vect_u8));
    }

    simd32uint8_tpl& operator+=(simd32uint8_tpl other) {
        vect_u8 = svadd_u8_x(svptrue_pat_b8(SV_VL32), vect_u8, other.vect_u8);
        return *this;
    }

    simd32uint8_tpl shr4() const {
        return simd32uint8_tpl(
                svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), vect_u8, 4));
    }

    uint8_t operator[](int idx) const {
        ALIGNED(32) uint8_t tab[32];
        store(tab);
        return tab[idx];
    }
};

inline void simd16uint16_tpl<SIMDLevel::ARM_SVE>::pq4_accumulate_low_bytes(
        const simd32uint8_tpl<SIMDLevel::ARM_SVE>& r) {
    vect_u16 = svaddwb_u16(vect_u16, r.vect_u8);
}

inline void simd16uint16_tpl<SIMDLevel::ARM_SVE>::pq4_accumulate_high_bytes(
        const simd32uint8_tpl<SIMDLevel::ARM_SVE>& r) {
    vect_u16 = svaddwt_u16(vect_u16, r.vect_u8);
}

inline simd32uint8_tpl<SIMDLevel::ARM_SVE> uint16_to_uint8_saturate(
        simd16uint16_tpl<SIMDLevel::ARM_SVE> a,
        simd16uint16_tpl<SIMDLevel::ARM_SVE> b) {
    svuint16_t sa = svmin_n_u16_x(svptrue_pat_b16(SV_VL16), a.vect_u16, 255);
    svuint16_t sb = svmin_n_u16_x(svptrue_pat_b16(SV_VL16), b.vect_u16, 255);
    svuint8_t z = svuzp1_u8(svreinterpret_u8_u16(sa), svreinterpret_u8_u16(sb));
    static const uint8_t sel[32] = {0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18,
                                    19, 20, 21, 22, 23, 8,  9,  10, 11, 12, 13,
                                    14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    return simd32uint8_tpl<SIMDLevel::ARM_SVE>(
            svtbl_u8(z, svld1_u8(svptrue_pat_b8(SV_VL32), sel)));
}

inline uint32_t get_MSBs(simd32uint8_tpl<SIMDLevel::ARM_SVE> a) {
    auto pg = svcmplt_n_s8(
            svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(a.vect_u8), 0);
    uint32_t res = *(uint32_t*)&pg;
    return res;
}

inline simd32uint8_tpl<SIMDLevel::ARM_SVE> blendv(
        simd32uint8_tpl<SIMDLevel::ARM_SVE> a,
        simd32uint8_tpl<SIMDLevel::ARM_SVE> b,
        simd32uint8_tpl<SIMDLevel::ARM_SVE> mask) {
    auto pg = svcmplt_n_s8(
            svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(mask.vect_u8), 0);
    return simd32uint8_tpl<SIMDLevel::ARM_SVE>(
            svsel_u8(pg, b.vect_u8, a.vect_u8));
}

template <>
struct simd8uint32_tpl<SIMDLevel::ARM_SVE>
        : simd256bit_tpl<SIMDLevel::ARM_SVE> {
    simd8uint32_tpl() {}

    explicit simd8uint32_tpl(uvec32 val)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(val) {}

    explicit simd8uint32_tpl(uint32_t x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(svdup_n_u32(x)) {}

    explicit simd8uint32_tpl(simd256bit_tpl<SIMDLevel::ARM_SVE> x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(x) {}

    explicit simd8uint32_tpl(const uint8_t* x) {
        this->vect_u32 = svld1_u32(svptrue_pat_b32(SV_VL8), (const uint32_t*)x);
    }

    explicit simd8uint32_tpl(
            uint32_t u0,
            uint32_t u1,
            uint32_t u2,
            uint32_t u3,
            uint32_t u4,
            uint32_t u5,
            uint32_t u6,
            uint32_t u7) {
        uint32_t data[8] = {u0, u1, u2, u3, u4, u5, u6, u7};
        this->vect_u32 = svld1_u32(svptrue_pat_b32(SV_VL8), data);
    }

    simd8uint32_tpl operator+(simd8uint32_tpl other) const {
        return simd8uint32_tpl(
                svadd_u32_x(svptrue_pat_b32(SV_VL8), vect_u32, other.vect_u32));
    }

    simd8uint32_tpl operator-(simd8uint32_tpl other) const {
        return simd8uint32_tpl(
                svsub_u32_x(svptrue_pat_b32(SV_VL8), vect_u32, other.vect_u32));
    }

    simd8uint32_tpl& operator+=(const simd8uint32_tpl& other) {
        vect_u32 =
                svadd_u32_x(svptrue_pat_b32(SV_VL8), vect_u32, other.vect_u32);
        return *this;
    }

    bool operator==(simd8uint32_tpl other) const {
        auto pg =
                svcmpne_u32(svptrue_pat_b32(SV_VL8), other.vect_u32, vect_u32);
        return !svptest_any(svptrue_pat_b32(SV_VL8), pg);
    }

    bool operator!=(simd8uint32_tpl other) const {
        return !(*this == other);
    }

    std::string elements_to_string(const char* fmt) const {
        uint32_t bytes[8];
        storeu((void*)bytes);
        char res[1000];
        char* ptr = res;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
        for (int idx = 0; idx < 8; idx++) {
            ptr += snprintf(
                    ptr, (size_t)(res + sizeof(res) - ptr), fmt, bytes[idx]);
        }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
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
        vect_u32 = svdup_n_u32(x);
    }

    void loadu(const void* ptr) {
        vect_u32 = svld1_u32(svptrue_pat_b32(SV_VL8), (const uint32_t*)ptr);
    }

    simd8uint32_tpl unzip() const {
        auto idx = svuzp1_u32(svindex_u32(0, 1), svindex_u32(1, 1));
        return simd8uint32_tpl(svtbl_u32(vect_u32, idx));
    }
};

inline void cmplt_min_max_fast(
        const simd8uint32_tpl<SIMDLevel::ARM_SVE> candidateValues,
        const simd8uint32_tpl<SIMDLevel::ARM_SVE> candidateIndices,
        const simd8uint32_tpl<SIMDLevel::ARM_SVE> currentValues,
        const simd8uint32_tpl<SIMDLevel::ARM_SVE> currentIndices,
        simd8uint32_tpl<SIMDLevel::ARM_SVE>& minValue,
        simd8uint32_tpl<SIMDLevel::ARM_SVE>& minIndices,
        simd8uint32_tpl<SIMDLevel::ARM_SVE>& maxValue,
        simd8uint32_tpl<SIMDLevel::ARM_SVE>& maxIndices) {
    auto pg = svcmplt_u32(
            svptrue_pat_b32(SV_VL8),
            candidateValues.vect_u32,
            currentValues.vect_u32);
    minValue.vect_u32 =
            svsel_u32(pg, candidateValues.vect_u32, currentValues.vect_u32);
    minIndices.vect_u32 =
            svsel_u32(pg, candidateIndices.vect_u32, currentIndices.vect_u32);
    maxValue.vect_u32 =
            svsel_u32(pg, currentValues.vect_u32, candidateValues.vect_u32);
    maxIndices.vect_u32 =
            svsel_u32(pg, currentIndices.vect_u32, candidateIndices.vect_u32);
}

template <>
struct simd8float32_tpl<SIMDLevel::ARM_SVE>
        : simd256bit_tpl<SIMDLevel::ARM_SVE> {
    simd8float32_tpl() {}

    explicit simd8float32_tpl(simd256bit_tpl<SIMDLevel::ARM_SVE> x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(x) {}

    explicit simd8float32_tpl(fvec32 x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(x) {}

    explicit simd8float32_tpl(float x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(svdup_n_f32(x)) {}

    explicit simd8float32_tpl(const float* x)
            : simd256bit_tpl<SIMDLevel::ARM_SVE>(
                      svld1_f32(svptrue_pat_b32(SV_VL8), x)) {}

    explicit simd8float32_tpl(
            float f0,
            float f1,
            float f2,
            float f3,
            float f4,
            float f5,
            float f6,
            float f7) {
        float data[8] = {f0, f1, f2, f3, f4, f5, f6, f7};
        this->vect_f32 = svld1_f32(svptrue_pat_b32(SV_VL8), data);
    }

    simd8float32_tpl operator*(simd8float32_tpl other) const {
        return simd8float32_tpl(
                svmul_f32_x(svptrue_pat_b32(SV_VL8), vect_f32, other.vect_f32));
    }

    simd8float32_tpl operator+(simd8float32_tpl other) const {
        return simd8float32_tpl(
                svadd_f32_x(svptrue_pat_b32(SV_VL8), vect_f32, other.vect_f32));
    }

    simd8float32_tpl operator-(simd8float32_tpl other) const {
        return simd8float32_tpl(
                svsub_f32_x(svptrue_pat_b32(SV_VL8), vect_f32, other.vect_f32));
    }

    simd8float32_tpl& operator+=(const simd8float32_tpl& other) {
        vect_f32 =
                svadd_f32_x(svptrue_pat_b32(SV_VL8), vect_f32, other.vect_f32);
        return *this;
    }

    bool operator==(simd8float32_tpl other) const {
        auto pg =
                svcmpne_f32(svptrue_pat_b32(SV_VL8), other.vect_f32, vect_f32);
        return !svptest_any(svptrue_pat_b32(SV_VL8), pg);
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
            ptr += snprintf(
                    ptr, (size_t)(res + sizeof(res) - ptr), "%g,", tab[idx]);
        }
        ptr[-1] = 0;
        return std::string(res);
    }
};

inline simd8float32_tpl<SIMDLevel::ARM_SVE> hadd(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b) {
    svbool_t pg = svptrue_pat_b32(SV_VL8);
    svfloat32_t pa = svaddp_f32_x(pg, a.vect_f32, a.vect_f32);
    svfloat32_t pb = svaddp_f32_x(pg, b.vect_f32, b.vect_f32);
    svfloat32_t c = svuzp1_f32(pa, pb);
    static const uint32_t sel[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(
            svtbl_f32(c, svld1_u32(pg, sel)));
}

inline simd8float32_tpl<SIMDLevel::ARM_SVE> unpacklo(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b) {
    auto a0 = svzip1_f32(a.vect_f32, b.vect_f32);
    auto a1 = svzip2_f32(a.vect_f32, b.vect_f32);
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(
            svsplice_f32(svptrue_pat_b32(SV_VL4), a0, a1));
}

inline simd8float32_tpl<SIMDLevel::ARM_SVE> unpackhi(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b) {
    auto a0 = svzip1_f32(a.vect_f32, b.vect_f32);
    auto a1 = svzip2_f32(a.vect_f32, b.vect_f32);
    auto pg = sveor_b_z(
            svptrue_pat_b32(SV_VL8),
            svptrue_pat_b32(SV_VL8),
            svptrue_pat_b32(SV_VL4));
    a1 = svcompact_f32(pg, a1);
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(svext_f32(a0, a1, 4));
}

inline simd8float32_tpl<SIMDLevel::ARM_SVE> fmadd(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b,
        simd8float32_tpl<SIMDLevel::ARM_SVE> c) {
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(svmad_f32_x(
            svptrue_pat_b32(SV_VL8), a.vect_f32, b.vect_f32, c.vect_f32));
}

inline void cmplt_and_blend_inplace(
        const simd8float32_tpl<SIMDLevel::ARM_SVE> candidateValues,
        const simd8uint32_tpl<SIMDLevel::ARM_SVE> candidateIndices,
        simd8float32_tpl<SIMDLevel::ARM_SVE>& lowestValues,
        simd8uint32_tpl<SIMDLevel::ARM_SVE>& lowestIndices) {
    auto pg = svcmplt_f32(
            svptrue_pat_b32(SV_VL8),
            candidateValues.vect_f32,
            lowestValues.vect_f32);
    lowestValues.vect_f32 =
            svsel_f32(pg, candidateValues.vect_f32, lowestValues.vect_f32);
    lowestIndices.vect_u32 =
            svsel_u32(pg, candidateIndices.vect_u32, lowestIndices.vect_u32);
}

inline void cmplt_min_max_fast(
        const simd8float32_tpl<SIMDLevel::ARM_SVE> candidateValues,
        const simd8uint32_tpl<SIMDLevel::ARM_SVE> candidateIndices,
        const simd8float32_tpl<SIMDLevel::ARM_SVE> currentValues,
        const simd8uint32_tpl<SIMDLevel::ARM_SVE> currentIndices,
        simd8float32_tpl<SIMDLevel::ARM_SVE>& minValues,
        simd8uint32_tpl<SIMDLevel::ARM_SVE>& minIndices,
        simd8float32_tpl<SIMDLevel::ARM_SVE>& maxValues,
        simd8uint32_tpl<SIMDLevel::ARM_SVE>& maxIndices) {
    auto pg = svcmplt_f32(
            svptrue_pat_b32(SV_VL8),
            candidateValues.vect_f32,
            currentValues.vect_f32);
    minValues.vect_f32 =
            svsel_f32(pg, candidateValues.vect_f32, currentValues.vect_f32);
    minIndices.vect_u32 =
            svsel_u32(pg, candidateIndices.vect_u32, currentIndices.vect_u32);
    maxValues.vect_f32 =
            svsel_f32(pg, currentValues.vect_f32, candidateValues.vect_f32);
    maxIndices.vect_u32 =
            svsel_u32(pg, currentIndices.vect_u32, candidateIndices.vect_u32);
}

namespace {

inline simd8float32_tpl<SIMDLevel::ARM_SVE> geteven(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b) {
    auto a0 = svuzp1_f32(a.vect_f32, a.vect_f32);
    auto b0 = svuzp1_f32(b.vect_f32, b.vect_f32);
    auto a1 = svtrn1_f32(a0, b0);
    auto b1 = svtrn2_f32(a0, b0);
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(svzip1_f32(a1, b1));
}

inline simd8float32_tpl<SIMDLevel::ARM_SVE> getodd(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b) {
    auto a0 = svuzp2_f32(a.vect_f32, a.vect_f32);
    auto b0 = svuzp2_f32(b.vect_f32, b.vect_f32);
    auto a1 = svtrn1_f32(a0, b0);
    auto b1 = svtrn2_f32(a0, b0);
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(svzip1_f32(a1, b1));
}

inline simd8float32_tpl<SIMDLevel::ARM_SVE> getlow128(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b) {
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(
            svsplice_f32(svptrue_pat_b32(SV_VL4), a.vect_f32, b.vect_f32));
}

inline simd8float32_tpl<SIMDLevel::ARM_SVE> gethigh128(
        simd8float32_tpl<SIMDLevel::ARM_SVE> a,
        simd8float32_tpl<SIMDLevel::ARM_SVE> b) {
    auto pg = sveor_b_z(
            svptrue_pat_b32(SV_VL8),
            svptrue_pat_b32(SV_VL8),
            svptrue_pat_b32(SV_VL4));
    auto c = svcompact_f32(pg, b.vect_f32);
    return simd8float32_tpl<SIMDLevel::ARM_SVE>(svext_f32(a.vect_f32, c, 4));
}

} // namespace

} // namespace faiss
