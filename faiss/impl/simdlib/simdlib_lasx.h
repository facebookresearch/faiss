/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <faiss/impl/simdlib/simdlib.h>
#include <lasxintrin.h>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
namespace faiss {
namespace detail {
namespace simdlib_lasx {
inline uint32_t byte_mask(__m256i x) {
    const auto bits = __lasx_xvmskltz_b(x);

    return static_cast<uint32_t>(__lasx_xvpickve2gr_w(bits, 0)) |
            (static_cast<uint32_t>(__lasx_xvpickve2gr_w(bits, 4)) << 16);
}
} // namespace simdlib_lasx
} // namespace detail

template <>
struct simd256bit_tpl<SIMDLevel::LOONGARCH_LASX> {
    __m256i data;
    simd256bit_tpl() = default;
    explicit simd256bit_tpl(const void* x) {
        loadu(x);
    }
    void loadu(const void* x) {
        data = __lasx_xvld(x, 0);
    }
    void storeu(void* x) const {
        __lasx_xvst(data, x, 0);
    }
    void store(void* x) const {
        storeu(x);
    }
    void clear() {
        data = __lasx_xvldi(0);
    }
    bool is_same_as(const simd256bit_tpl& other) const {
        return detail::simdlib_lasx::byte_mask(
                       __lasx_xvseq_b(data, other.data)) == 0xffffffffu;
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
struct simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX> {
    simd16uint16_tpl() = default;
    explicit simd16uint16_tpl(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
    explicit simd16uint16_tpl(const uint16_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
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
        data = __lasx_xvreplgr2vr_h(x);
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
        r.data = __lasx_xvadd_h(data, other.data);
        return r;
    }
    simd16uint16_tpl operator-(const simd16uint16_tpl& other) const {
        simd16uint16_tpl r;
        r.data = __lasx_xvsub_h(data, other.data);
        return r;
    }
    simd16uint16_tpl operator*(const simd16uint16_tpl& other) const {
        simd16uint16_tpl r;
        r.data = __lasx_xvmul_h(data, other.data);
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
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& other) const {
        simd16uint16_tpl r;
        r.data = __lasx_xvand_v(data, other.data);
        return r;
    }
    simd16uint16_tpl operator|(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& other) const {
        simd16uint16_tpl r;
        r.data = __lasx_xvor_v(data, other.data);
        return r;
    }
    simd16uint16_tpl operator^(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& other) const {
        simd16uint16_tpl r;
        r.data = __lasx_xvxor_v(data, other.data);
        return r;
    }
    simd16uint16_tpl operator>>(int shift) const {
        simd16uint16_tpl r;
        if (shift >= 16) {
            r.clear();
            return r;
        }
        r.data = __lasx_xvsrl_h(data, __lasx_xvreplgr2vr_h(shift));
        return r;
    }
    simd16uint16_tpl operator<<(int shift) const {
        simd16uint16_tpl r;
        if (shift >= 16) {
            r.clear();
            return r;
        }
        r.data = __lasx_xvsll_h(data, __lasx_xvreplgr2vr_h(shift));
        return r;
    }
    simd16uint16_tpl operator==(const simd16uint16_tpl& other) const {
        simd16uint16_tpl r;
        r.data = __lasx_xvseq_h(data, other.data);
        return r;
    }
    simd16uint16_tpl operator~() const {
        simd16uint16_tpl r;
        r.data = __lasx_xvnor_v(data, data);
        return r;
    }
    uint16_t get_scalar_0() const {
        return static_cast<uint16_t>(__lasx_xvpickve2gr_w(data, 0));
    }
    uint32_t ge_mask(const simd16uint16_tpl& other) const {
        return detail::simdlib_lasx::byte_mask(
                __lasx_xvsle_hu(other.data, data));
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
        data = __lasx_xvmin_hu(data, other.data);
    }
    void accu_max(const simd16uint16_tpl& other) {
        data = __lasx_xvmax_hu(data, other.data);
    }
};

template <>
struct simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX> {
    simd32uint8_tpl() = default;
    explicit simd32uint8_tpl(const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
    explicit simd32uint8_tpl(const uint8_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
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
        data = __lasx_xvreplgr2vr_b(x);
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
        r.data = __lasx_xvadd_b(data, other.data);
        return r;
    }
    simd32uint8_tpl& operator+=(const simd32uint8_tpl& other) {
        *this = *this + other;
        return *this;
    }
    simd32uint8_tpl operator&(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& other) const {
        simd32uint8_tpl r;
        r.data = __lasx_xvand_v(data, other.data);
        return r;
    }
    simd32uint8_tpl lookup_2_lanes(const simd32uint8_tpl& idx) const {
        simd32uint8_tpl r;
        const auto control = __lasx_xvandi_b(idx.data, 15);
        const auto value = __lasx_xvshuf_b(data, data, control);
        r.data = __lasx_xvandn_v(
                __lasx_xvslt_b(idx.data, __lasx_xvldi(0)), value);
        return r;
    }
    // 0x50/0xfa place words [0, 1]/[2, 3] in the low 64 bits of each
    // 128-bit lane, so lane-local interleaving widens the selected half.
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> lane0_as_uint16() const {
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> r;
        const auto v = __lasx_xvpermi_d(data, 0x50);
        r.data = __lasx_xvilvl_b(__lasx_xvldi(0), v);
        return r;
    }
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> lane1_as_uint16() const {
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> r;
        const auto v = __lasx_xvpermi_d(data, 0xfa);
        r.data = __lasx_xvilvl_b(__lasx_xvldi(0), v);
        return r;
    }
};

template <>
struct simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX> {
    simd8uint32_tpl() = default;
    explicit simd8uint32_tpl(const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
    explicit simd8uint32_tpl(const uint32_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
    explicit simd8uint32_tpl(uint32_t x) {
        set1(x);
    }
    explicit simd8uint32_tpl(const uint8_t* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
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
        data = __lasx_xvreplgr2vr_w(x);
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
        r.data = __lasx_xvadd_w(data, other.data);
        return r;
    }
    simd8uint32_tpl operator-(const simd8uint32_tpl& other) const {
        simd8uint32_tpl r;
        r.data = __lasx_xvsub_w(data, other.data);
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
        const auto v = __lasx_xvshuf4i_w(data, 0xd8);
        r.data = __lasx_xvpermi_d(v, 0xd8);
        return r;
    }
    bool operator!=(const simd8uint32_tpl& other) const {
        return !(*this == other);
    }
};

template <>
struct simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>
        : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX> {
    simd8float32_tpl() = default;
    explicit simd8float32_tpl(
            const simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>& x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
    explicit simd8float32_tpl(const float* x)
            : simd256bit_tpl<SIMDLevel::LOONGARCH_LASX>(x) {}
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
        data = __lasx_xvreplgr2vr_w(std::bit_cast<int32_t>(x));
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
        r.data = (__m256i)__lasx_xvfadd_s((__m256)data, (__m256)other.data);
        return r;
    }
    simd8float32_tpl operator-(const simd8float32_tpl& other) const {
        simd8float32_tpl r;
        r.data = (__m256i)__lasx_xvfsub_s((__m256)data, (__m256)other.data);
        return r;
    }
    simd8float32_tpl operator*(const simd8float32_tpl& other) const {
        simd8float32_tpl r;
        r.data = (__m256i)__lasx_xvfmul_s((__m256)data, (__m256)other.data);
        return r;
    }
    simd8float32_tpl& operator+=(const simd8float32_tpl& other) {
        *this = *this + other;
        return *this;
    }
    bool operator==(const simd8float32_tpl& other) const {
        return detail::simdlib_lasx::byte_mask(__lasx_xvfcmp_ceq_s(
                       (__m256)data, (__m256)other.data)) == 0xffffffffu;
    }
    bool operator!=(const simd8float32_tpl& other) const {
        return !(*this == other);
    }
};

inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> min(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvmin_hu(a.data, b.data);
    return r;
}
inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> max(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvmax_hu(a.data, b.data);
    return r;
}
inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> hadd(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvadd_h(
            __lasx_xvpickev_h(b.data, a.data),
            __lasx_xvpickod_h(b.data, a.data));
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> hadd(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = (__m256i)__lasx_xvfadd_s(
            (__m256)__lasx_xvpickev_w(b.data, a.data),
            (__m256)__lasx_xvpickod_w(b.data, a.data));
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> unpacklo(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvilvl_w(b.data, a.data);
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> unpackhi(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvilvh_w(b.data, a.data);
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> geteven(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvpickev_w(b.data, a.data);
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> getodd(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvpickod_w(b.data, a.data);
    return r;
}
// With operands (b, a), 0x20/0x31 select [a.low, b.low]/[a.high, b.high].
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> getlow128(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvpermi_q(b.data, a.data, 0x20);
    return r;
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> gethigh128(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvpermi_q(b.data, a.data, 0x31);
    return r;
}
inline simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> combine2x2(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> r;
    const auto lo = __lasx_xvpermi_q(b.data, a.data, 0x20);
    const auto hi = __lasx_xvpermi_q(b.data, a.data, 0x31);
    r.data = __lasx_xvadd_h(lo, hi);
    return r;
}
inline simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX> uint16_to_uint8_saturate(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& b) {
    simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvpickev_b(
            __lasx_xvmin_hu(b.data, __lasx_xvreplgr2vr_h(255)),
            __lasx_xvmin_hu(a.data, __lasx_xvreplgr2vr_h(255)));
    return r;
}
inline uint32_t get_MSBs(const simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX>& a) {
    return detail::simdlib_lasx::byte_mask(a.data);
}
inline simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX> blendv(
        const simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX>& b,
        const simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX>& mask) {
    simd32uint8_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = __lasx_xvbitsel_v(
            a.data, b.data, __lasx_xvslt_b(mask.data, __lasx_xvldi(0)));
    return r;
}
inline uint32_t cmp_ge32(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& b,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& thr) {
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
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& b,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& thr) {
    auto compress = [](uint32_t x) {
        x &= 0x55555555u;
        x = (x | (x >> 1)) & 0x33333333u;
        x = (x | (x >> 2)) & 0x0f0f0f0fu;
        x = (x | (x >> 4)) & 0x00ff00ffu;
        return (x | (x >> 8)) & 0xffffu;
    };
    return compress(a.le_mask(thr)) | (compress(b.le_mask(thr)) << 16);
}
inline simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> fmadd(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& b,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& c) {
    simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> r;
    r.data = (__m256i)__lasx_xvfmadd_s(
            (__m256)a.data, (__m256)b.data, (__m256)c.data);
    return r;
}
inline void cmplt_min_max_fast(
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> candidateValues,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> candidateIndices,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> currentValues,
        const simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX> currentIndices,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& minValues,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& minIndices,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& maxValues,
        simd16uint16_tpl<SIMDLevel::LOONGARCH_LASX>& maxIndices) {
    const auto mask = __lasx_xvslt_hu(candidateValues.data, currentValues.data);
    minValues.data =
            __lasx_xvbitsel_v(currentValues.data, candidateValues.data, mask);
    minIndices.data =
            __lasx_xvbitsel_v(currentIndices.data, candidateIndices.data, mask);
    maxValues.data =
            __lasx_xvbitsel_v(candidateValues.data, currentValues.data, mask);
    maxIndices.data =
            __lasx_xvbitsel_v(candidateIndices.data, currentIndices.data, mask);
}
inline void cmplt_min_max_fast(
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX> candidateIndices,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX> currentValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX> currentIndices,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>& minValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>& minIndices,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>& maxValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>& maxIndices) {
    const auto mask = __lasx_xvslt_wu(candidateValues.data, currentValues.data);
    minValues.data =
            __lasx_xvbitsel_v(currentValues.data, candidateValues.data, mask);
    minIndices.data =
            __lasx_xvbitsel_v(currentIndices.data, candidateIndices.data, mask);
    maxValues.data =
            __lasx_xvbitsel_v(candidateValues.data, currentValues.data, mask);
    maxIndices.data =
            __lasx_xvbitsel_v(candidateIndices.data, currentIndices.data, mask);
}
inline void cmplt_min_max_fast(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX> candidateIndices,
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> currentValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX> currentIndices,
        simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& minValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>& minIndices,
        simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& maxValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>& maxIndices) {
    const auto mask = __lasx_xvfcmp_clt_s(
            (__m256)candidateValues.data, (__m256)currentValues.data);
    minValues.data =
            __lasx_xvbitsel_v(currentValues.data, candidateValues.data, mask);
    minIndices.data =
            __lasx_xvbitsel_v(currentIndices.data, candidateIndices.data, mask);
    maxValues.data =
            __lasx_xvbitsel_v(candidateValues.data, currentValues.data, mask);
    maxIndices.data =
            __lasx_xvbitsel_v(candidateIndices.data, currentIndices.data, mask);
}
inline void cmplt_and_blend_inplace(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX> candidateIndices,
        simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& lowestValues,
        simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>& lowestIndices) {
    const auto mask = __lasx_xvfcmp_clt_s(
            (__m256)candidateValues.data, (__m256)lowestValues.data);
    lowestValues.data =
            __lasx_xvbitsel_v(lowestValues.data, candidateValues.data, mask);
    lowestIndices.data =
            __lasx_xvbitsel_v(lowestIndices.data, candidateIndices.data, mask);
}
inline float horizontal_add(
        const simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>& a) {
    __m256 v = __lasx_xvfadd_s(
            (__m256)a.data, (__m256)__lasx_xvpermi_q(a.data, a.data, 0x01));
    v = __lasx_xvfadd_s(v, (__m256)__lasx_xvshuf4i_w((__m256i)v, 0x4e));
    v = __lasx_xvfadd_s(v, (__m256)__lasx_xvshuf4i_w((__m256i)v, 0xb1));
    return std::bit_cast<float>(
            static_cast<uint32_t>(__lasx_xvpickve2gr_w((__m256i)v, 0)));
}
} // namespace faiss
