/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <faiss/impl/simdlib/simdlib.h>
#include <lsxintrin.h>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
namespace faiss {
namespace detail {
namespace simdlib_lsx {
inline uint32_t byte_mask(__m128i x) {
    const auto bits = __lsx_vmskltz_b(x);

    return __lsx_vpickve2gr_wu(bits, 0);
}
} // namespace simdlib_lsx
} // namespace detail

// The public simdlib interface has 256 bits, including on LSX.
template <>
struct simd256bit_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[2];
    simd256bit_tpl() = default;
    explicit simd256bit_tpl(const void* x) {
        loadu(x);
    }
    void loadu(const void* x) {
        const auto* bytes = static_cast<const uint8_t*>(x);
        for (int i = 0; i < 2; ++i) {
            data[i] = __lsx_vld(bytes + i * sizeof(__m128i), 0);
        }
    }
    void storeu(void* x) const {
        auto* bytes = static_cast<uint8_t*>(x);
        for (int i = 0; i < 2; ++i) {
            __lsx_vst(data[i], bytes + i * sizeof(__m128i), 0);
        }
    }
    void store(void* x) const {
        storeu(x);
    }
    void clear() {
        for (auto& v : data) {
            v = __lsx_vldi(0);
        }
    }
    bool is_same_as(const simd256bit_tpl& other) const {
        for (int i = 0; i < 2; ++i) {
            if (detail::simdlib_lsx::byte_mask(
                        __lsx_vseq_b(data[i], other.data[i])) != 0xffffu) {
                return false;
            }
        }
        return true;
    }
    void bin(char bits[257]) const {
        uint8_t bytes[32];
        storeu(bytes);
        for (int i = 0; i < 256; ++i) {
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

template <>
struct simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX> {
    simd16uint16_tpl() = default;
    explicit simd16uint16_tpl(const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd16uint16_tpl(const uint16_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd16uint16_tpl(uint16_t x) {
        set1(x);
    }
    explicit simd16uint16_tpl(int x) {
        set1(x);
    }
    explicit simd16uint16_tpl(
            uint16_t x0,
            uint16_t x1,
            uint16_t x2,
            uint16_t x3,
            uint16_t x4,
            uint16_t x5,
            uint16_t x6,
            uint16_t x7,
            uint16_t x8,
            uint16_t x9,
            uint16_t x10,
            uint16_t x11,
            uint16_t x12,
            uint16_t x13,
            uint16_t x14,
            uint16_t x15) {
        const uint16_t values[] = {
                x0,
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                x7,
                x8,
                x9,
                x10,
                x11,
                x12,
                x13,
                x14,
                x15};
        loadu(values);
    }
    void set1(uint16_t x) {
        for (auto& v : data) {
            v = __lsx_vreplgr2vr_h(x);
        }
    }
    std::string elements_to_string(const char* fmt) const {
        uint16_t values[16];
        storeu(values);
        char result[1000], *ptr = result;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
        for (auto v : values) {
            ptr += snprintf(ptr, result + sizeof(result) - ptr, fmt, v);
        }
#pragma GCC diagnostic pop
        ptr[-1] = 0;
        return std::string(result);
    }
    std::string hex() const {
        return elements_to_string("%02x,");
    }
    std::string dec() const {
        return elements_to_string("%3d,");
    }
    uint16_t operator[](int i) const {
        uint16_t values[16];
        storeu(values);
        return values[i];
    }
    simd16uint16_tpl operator+(const simd16uint16_tpl& other) const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vadd_h(data[i], other.data[i]);
        }
        return r;
    }
    simd16uint16_tpl operator-(const simd16uint16_tpl& other) const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vsub_h(data[i], other.data[i]);
        }
        return r;
    }
    simd16uint16_tpl operator*(const simd16uint16_tpl& other) const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vmul_h(data[i], other.data[i]);
        }
        return r;
    }
    simd16uint16_tpl& operator+=(const simd16uint16_tpl& other) {
        *this = *this + other;
        return *this;
    }
    simd16uint16_tpl& operator-=(const simd16uint16_tpl& other) {
        *this = *this - other;
        return *this;
    }
    simd16uint16_tpl operator&(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& other) const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vand_v(data[i], other.data[i]);
        }
        return r;
    }
    simd16uint16_tpl operator|(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& other) const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vor_v(data[i], other.data[i]);
        }
        return r;
    }
    simd16uint16_tpl operator^(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& other) const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vxor_v(data[i], other.data[i]);
        }
        return r;
    }
    simd16uint16_tpl operator>>(int shift) const {
        simd16uint16_tpl r;
        if (shift >= 16) {
            r.clear();
            return r;
        }
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vsrl_h(data[i], __lsx_vreplgr2vr_h(shift));
        }
        return r;
    }
    simd16uint16_tpl operator<<(int shift) const {
        simd16uint16_tpl r;
        if (shift >= 16) {
            r.clear();
            return r;
        }
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vsll_h(data[i], __lsx_vreplgr2vr_h(shift));
        }
        return r;
    }
    simd16uint16_tpl operator==(const simd16uint16_tpl& other) const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vseq_h(data[i], other.data[i]);
        }
        return r;
    }
    simd16uint16_tpl operator~() const {
        simd16uint16_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vnor_v(data[i], data[i]);
        }
        return r;
    }
    uint16_t get_scalar_0() const {
        return __lsx_vpickve2gr_hu(data[0], 0);
    }
    uint32_t ge_mask(const simd16uint16_tpl& other) const {
        uint32_t mask = 0;
        for (int i = 0; i < 2; ++i) {
            mask |= detail::simdlib_lsx::byte_mask(
                            __lsx_vsle_hu(other.data[i], data[i]))
                    << (i * 16);
        }
        return mask;
    }
    uint32_t le_mask(const simd16uint16_tpl& other) const {
        return other.ge_mask(*this);
    }
    uint32_t gt_mask(const simd16uint16_tpl& other) const {
        return ~le_mask(other);
    }
    bool all_gt(const simd16uint16_tpl& other) const {
        return le_mask(other) == 0;
    }
    void accu_min(const simd16uint16_tpl& other) {
        for (int i = 0; i < 2; ++i) {
            data[i] = __lsx_vmin_hu(data[i], other.data[i]);
        }
    }
    void accu_max(const simd16uint16_tpl& other) {
        for (int i = 0; i < 2; ++i) {
            data[i] = __lsx_vmax_hu(data[i], other.data[i]);
        }
    }
};

template <>
struct simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX> {
    simd32uint8_tpl() = default;
    explicit simd32uint8_tpl(const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd32uint8_tpl(const uint8_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd32uint8_tpl(uint8_t x) {
        set1(x);
    }
    explicit simd32uint8_tpl(int x) {
        set1(x);
    }
    template <uint8_t... Values>
    static simd32uint8_tpl create() {
        static_assert(sizeof...(Values) == 32);
        const uint8_t values[] = {Values...};
        return simd32uint8_tpl(values);
    }
    void set1(uint8_t x) {
        for (auto& v : data) {
            v = __lsx_vreplgr2vr_b(x);
        }
    }
    std::string elements_to_string(const char* fmt) const {
        uint8_t values[32];
        storeu(values);
        char result[1000], *ptr = result;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
        for (auto v : values) {
            ptr += snprintf(ptr, result + sizeof(result) - ptr, fmt, v);
        }
#pragma GCC diagnostic pop
        ptr[-1] = 0;
        return std::string(result);
    }
    std::string hex() const {
        return elements_to_string("%02x,");
    }
    std::string dec() const {
        return elements_to_string("%3d,");
    }
    uint8_t operator[](int i) const {
        uint8_t values[32];
        storeu(values);
        return values[i];
    }
    simd32uint8_tpl operator+(const simd32uint8_tpl& other) const {
        simd32uint8_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vadd_b(data[i], other.data[i]);
        }
        return r;
    }
    simd32uint8_tpl& operator+=(const simd32uint8_tpl& other) {
        *this = *this + other;
        return *this;
    }
    simd32uint8_tpl operator&(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& other) const {
        simd32uint8_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vand_v(data[i], other.data[i]);
        }
        return r;
    }
    simd32uint8_tpl lookup_2_lanes(const simd32uint8_tpl& idx) const {
        simd32uint8_tpl r;
        for (int i = 0; i < 2; ++i) {
            const auto control = __lsx_vandi_b(idx.data[i], 15);
            const auto value = __lsx_vshuf_b(data[i], data[i], control);
            r.data[i] = __lsx_vandn_v(
                    __lsx_vslt_b(idx.data[i], __lsx_vldi(0)), value);
        }
        return r;
    }
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> lane0_as_uint16() const {
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> r;
        r.data[0] = __lsx_vilvl_b(__lsx_vldi(0), data[0]);
        r.data[1] = __lsx_vilvh_b(__lsx_vldi(0), data[0]);
        return r;
    }
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> lane1_as_uint16() const {
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> r;
        r.data[0] = __lsx_vilvl_b(__lsx_vldi(0), data[1]);
        r.data[1] = __lsx_vilvh_b(__lsx_vldi(0), data[1]);
        return r;
    }
};

template <>
struct simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX> {
    simd8uint32_tpl() = default;
    explicit simd8uint32_tpl(const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd8uint32_tpl(const uint32_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd8uint32_tpl(uint32_t x) {
        set1(x);
    }
    explicit simd8uint32_tpl(const uint8_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd8uint32_tpl(
            uint32_t x0,
            uint32_t x1,
            uint32_t x2,
            uint32_t x3,
            uint32_t x4,
            uint32_t x5,
            uint32_t x6,
            uint32_t x7) {
        const uint32_t values[] = {x0, x1, x2, x3, x4, x5, x6, x7};
        loadu(values);
    }
    void set1(uint32_t x) {
        for (auto& v : data) {
            v = __lsx_vreplgr2vr_w(x);
        }
    }
    std::string elements_to_string(const char* fmt) const {
        uint32_t values[8];
        storeu(values);
        char result[1000], *ptr = result;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
        for (auto v : values) {
            ptr += snprintf(ptr, result + sizeof(result) - ptr, fmt, v);
        }
#pragma GCC diagnostic pop
        ptr[-1] = 0;
        return std::string(result);
    }
    std::string hex() const {
        return elements_to_string("%02x,");
    }
    std::string dec() const {
        return elements_to_string("%3d,");
    }
    uint32_t operator[](int i) const {
        uint32_t values[8];
        storeu(values);
        return values[i];
    }
    simd8uint32_tpl operator+(const simd8uint32_tpl& other) const {
        simd8uint32_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vadd_w(data[i], other.data[i]);
        }
        return r;
    }
    simd8uint32_tpl operator-(const simd8uint32_tpl& other) const {
        simd8uint32_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = __lsx_vsub_w(data[i], other.data[i]);
        }
        return r;
    }
    simd8uint32_tpl& operator+=(const simd8uint32_tpl& other) {
        *this = *this + other;
        return *this;
    }
    bool operator==(const simd8uint32_tpl& other) const {
        return is_same_as(other);
    }
    simd8uint32_tpl unzip() const {
        simd8uint32_tpl r;
        r.data[0] = __lsx_vpickev_w(data[1], data[0]);
        r.data[1] = __lsx_vpickod_w(data[1], data[0]);
        return r;
    }
    bool operator!=(const simd8uint32_tpl& other) const {
        return !(*this == other);
    }
};

template <>
struct simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX> {
    simd8float32_tpl() = default;
    explicit simd8float32_tpl(const simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd8float32_tpl(const float* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LSX>(x) {}
    explicit simd8float32_tpl(float x) {
        set1(x);
    }
    explicit simd8float32_tpl(
            float x0,
            float x1,
            float x2,
            float x3,
            float x4,
            float x5,
            float x6,
            float x7) {
        const float values[] = {x0, x1, x2, x3, x4, x5, x6, x7};
        loadu(values);
    }
    void set1(float x) {
        for (auto& v : data) {
            v = __lsx_vreplgr2vr_w(std::bit_cast<int32_t>(x));
        }
    }
    std::string elements_to_string(const char* fmt) const {
        float values[8];
        storeu(values);
        char result[1000], *ptr = result;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
        for (auto v : values) {
            ptr += snprintf(ptr, result + sizeof(result) - ptr, fmt, v);
        }
#pragma GCC diagnostic pop
        ptr[-1] = 0;
        return std::string(result);
    }
    std::string tostring() const {
        return elements_to_string("%g,");
    }
    simd8float32_tpl operator+(const simd8float32_tpl& other) const {
        simd8float32_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = (__m128i)__lsx_vfadd_s(
                    (__m128)data[i], (__m128)other.data[i]);
        }
        return r;
    }
    simd8float32_tpl operator-(const simd8float32_tpl& other) const {
        simd8float32_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = (__m128i)__lsx_vfsub_s(
                    (__m128)data[i], (__m128)other.data[i]);
        }
        return r;
    }
    simd8float32_tpl operator*(const simd8float32_tpl& other) const {
        simd8float32_tpl r;
        for (int i = 0; i < 2; ++i) {
            r.data[i] = (__m128i)__lsx_vfmul_s(
                    (__m128)data[i], (__m128)other.data[i]);
        }
        return r;
    }
    simd8float32_tpl& operator+=(const simd8float32_tpl& other) {
        *this = *this + other;
        return *this;
    }
    bool operator==(const simd8float32_tpl& other) const {
        for (int i = 0; i < 2; ++i) {
            if (detail::simdlib_lsx::byte_mask(__lsx_vfcmp_ceq_s(
                        (__m128)data[i], (__m128)other.data[i])) != 0xffffu) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const simd8float32_tpl& other) const {
        return !(*this == other);
    }
};

inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> min(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vmin_hu(a.data[i], b.data[i]);
    }
    return r;
}
inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> max(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vmax_hu(a.data[i], b.data[i]);
    }
    return r;
}
inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> hadd(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vadd_h(
                __lsx_vpickev_h(b.data[i], a.data[i]),
                __lsx_vpickod_h(b.data[i], a.data[i]));
    }
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> hadd(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = (__m128i)__lsx_vfadd_s(
                (__m128)__lsx_vpickev_w(b.data[i], a.data[i]),
                (__m128)__lsx_vpickod_w(b.data[i], a.data[i]));
    }
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> unpacklo(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vilvl_w(b.data[i], a.data[i]);
    }
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> unpackhi(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vilvh_w(b.data[i], a.data[i]);
    }
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> geteven(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vpickev_w(b.data[i], a.data[i]);
    }
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> getodd(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vpickod_w(b.data[i], a.data[i]);
    }
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> getlow128(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    r.data[0] = a.data[0];
    r.data[1] = b.data[0];
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> gethigh128(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    r.data[0] = a.data[1];
    r.data[1] = b.data[1];
    return r;
}
inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> combine2x2(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> r;
    r.data[0] = __lsx_vadd_h(a.data[0], a.data[1]);
    r.data[1] = __lsx_vadd_h(b.data[0], b.data[1]);
    return r;
}
inline simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX> uint16_to_uint8_saturate(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& b) {
    simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vpickev_b(
                __lsx_vmin_hu(b.data[i], __lsx_vreplgr2vr_h(255)),
                __lsx_vmin_hu(a.data[i], __lsx_vreplgr2vr_h(255)));
    }
    return r;
}
inline uint32_t get_MSBs(const simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX>& a) {
    uint32_t mask = 0;
    for (int i = 0; i < 2; ++i) {
        mask |= detail::simdlib_lsx::byte_mask(a.data[i]) << (i * 16);
    }
    return mask;
}
inline simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX> blendv(
        const simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX>& b,
        const simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX>& mask) {
    simd32uint8_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = __lsx_vbitsel_v(
                a.data[i],
                b.data[i],
                __lsx_vslt_b(mask.data[i], __lsx_vldi(0)));
    }
    return r;
}
inline uint32_t cmp_ge32(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& b,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& thr) {
    auto compress = [](uint32_t x) {
        x &= 0x55555555u;
        x = (x | (x >> 1)) & 0x33333333u;
        x = (x | (x >> 2)) & 0x0f0f0f0fu;
        x = (x | (x >> 4)) & 0x00ff00ffu;
        return (x | (x >> 8)) & 0xffffu;
    };
    return compress(a.ge_mask(thr)) | (compress(b.ge_mask(thr)) << 16);
}
inline uint32_t cmp_le32(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& b,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& thr) {
    auto compress = [](uint32_t x) {
        x &= 0x55555555u;
        x = (x | (x >> 1)) & 0x33333333u;
        x = (x | (x >> 2)) & 0x0f0f0f0fu;
        x = (x | (x >> 4)) & 0x00ff00ffu;
        return (x | (x >> 8)) & 0xffffu;
    };
    return compress(a.le_mask(thr)) | (compress(b.le_mask(thr)) << 16);
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> fmadd(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& b,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& c) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> r;
    for (int i = 0; i < 2; ++i) {
        r.data[i] = (__m128i)__lsx_vfmadd_s(
                (__m128)a.data[i], (__m128)b.data[i], (__m128)c.data[i]);
    }
    return r;
}
inline void cmplt_min_max_fast(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> candidateValues,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> candidateIndices,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> currentValues,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX> currentIndices,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& minValues,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& minIndices,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& maxValues,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LSX>& maxIndices) {
    for (int i = 0; i < 2; ++i) {
        const auto mask =
                __lsx_vslt_hu(candidateValues.data[i], currentValues.data[i]);
        minValues.data[i] = __lsx_vbitsel_v(
                currentValues.data[i], candidateValues.data[i], mask);
        minIndices.data[i] = __lsx_vbitsel_v(
                currentIndices.data[i], candidateIndices.data[i], mask);
        maxValues.data[i] = __lsx_vbitsel_v(
                candidateValues.data[i], currentValues.data[i], mask);
        maxIndices.data[i] = __lsx_vbitsel_v(
                candidateIndices.data[i], currentIndices.data[i], mask);
    }
}
inline void cmplt_min_max_fast(
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX> candidateIndices,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX> currentValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX> currentIndices,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>& minValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>& minIndices,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>& maxValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>& maxIndices) {
    for (int i = 0; i < 2; ++i) {
        const auto mask =
                __lsx_vslt_wu(candidateValues.data[i], currentValues.data[i]);
        minValues.data[i] = __lsx_vbitsel_v(
                currentValues.data[i], candidateValues.data[i], mask);
        minIndices.data[i] = __lsx_vbitsel_v(
                currentIndices.data[i], candidateIndices.data[i], mask);
        maxValues.data[i] = __lsx_vbitsel_v(
                candidateValues.data[i], currentValues.data[i], mask);
        maxIndices.data[i] = __lsx_vbitsel_v(
                candidateIndices.data[i], currentIndices.data[i], mask);
    }
}
inline void cmplt_min_max_fast(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX> candidateIndices,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> currentValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX> currentIndices,
        simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& minValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>& minIndices,
        simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& maxValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>& maxIndices) {
    for (int i = 0; i < 2; ++i) {
        const auto mask = __lsx_vfcmp_clt_s(
                (__m128)candidateValues.data[i], (__m128)currentValues.data[i]);
        minValues.data[i] = __lsx_vbitsel_v(
                currentValues.data[i], candidateValues.data[i], mask);
        minIndices.data[i] = __lsx_vbitsel_v(
                currentIndices.data[i], candidateIndices.data[i], mask);
        maxValues.data[i] = __lsx_vbitsel_v(
                candidateValues.data[i], currentValues.data[i], mask);
        maxIndices.data[i] = __lsx_vbitsel_v(
                candidateIndices.data[i], currentIndices.data[i], mask);
    }
}
inline void cmplt_and_blend_inplace(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX> candidateIndices,
        simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& lowestValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>& lowestIndices) {
    for (int i = 0; i < 2; ++i) {
        const auto mask = __lsx_vfcmp_clt_s(
                (__m128)candidateValues.data[i], (__m128)lowestValues.data[i]);
        lowestValues.data[i] = __lsx_vbitsel_v(
                lowestValues.data[i], candidateValues.data[i], mask);
        lowestIndices.data[i] = __lsx_vbitsel_v(
                lowestIndices.data[i], candidateIndices.data[i], mask);
    }
}
inline float horizontal_add(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>& a) {
    __m128 v = __lsx_vfadd_s((__m128)a.data[0], (__m128)a.data[1]);
    v = __lsx_vfadd_s(v, (__m128)__lsx_vshuf4i_w((__m128i)v, 0x4e));
    v = __lsx_vfadd_s(v, (__m128)__lsx_vshuf4i_w((__m128i)v, 0xb1));
    return std::bit_cast<float>(
            static_cast<uint32_t>(__lsx_vpickve2gr_w((__m128i)v, 0)));
}
} // namespace faiss