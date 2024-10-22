/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// TODO: Support big endian (currently supporting only little endian)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include <arm_neon.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

namespace detail {

namespace simdlib {

static inline uint8x16x2_t reinterpret_u8(const uint8x16x2_t& v) {
    return v;
}

static inline uint8x16x2_t reinterpret_u8(const uint16x8x2_t& v) {
    return {vreinterpretq_u8_u16(v.val[0]), vreinterpretq_u8_u16(v.val[1])};
}

static inline uint8x16x2_t reinterpret_u8(const uint32x4x2_t& v) {
    return {vreinterpretq_u8_u32(v.val[0]), vreinterpretq_u8_u32(v.val[1])};
}

static inline uint8x16x2_t reinterpret_u8(const float32x4x2_t& v) {
    return {vreinterpretq_u8_f32(v.val[0]), vreinterpretq_u8_f32(v.val[1])};
}

static inline uint16x8x2_t reinterpret_u16(const uint8x16x2_t& v) {
    return {vreinterpretq_u16_u8(v.val[0]), vreinterpretq_u16_u8(v.val[1])};
}

static inline uint16x8x2_t reinterpret_u16(const uint16x8x2_t& v) {
    return v;
}

static inline uint16x8x2_t reinterpret_u16(const uint32x4x2_t& v) {
    return {vreinterpretq_u16_u32(v.val[0]), vreinterpretq_u16_u32(v.val[1])};
}

static inline uint16x8x2_t reinterpret_u16(const float32x4x2_t& v) {
    return {vreinterpretq_u16_f32(v.val[0]), vreinterpretq_u16_f32(v.val[1])};
}

static inline uint32x4x2_t reinterpret_u32(const uint8x16x2_t& v) {
    return {vreinterpretq_u32_u8(v.val[0]), vreinterpretq_u32_u8(v.val[1])};
}

static inline uint32x4x2_t reinterpret_u32(const uint16x8x2_t& v) {
    return {vreinterpretq_u32_u16(v.val[0]), vreinterpretq_u32_u16(v.val[1])};
}

static inline uint32x4x2_t reinterpret_u32(const uint32x4x2_t& v) {
    return v;
}

static inline uint32x4x2_t reinterpret_u32(const float32x4x2_t& v) {
    return {vreinterpretq_u32_f32(v.val[0]), vreinterpretq_u32_f32(v.val[1])};
}

static inline float32x4x2_t reinterpret_f32(const uint8x16x2_t& v) {
    return {vreinterpretq_f32_u8(v.val[0]), vreinterpretq_f32_u8(v.val[1])};
}

static inline float32x4x2_t reinterpret_f32(const uint16x8x2_t& v) {
    return {vreinterpretq_f32_u16(v.val[0]), vreinterpretq_f32_u16(v.val[1])};
}

static inline float32x4x2_t reinterpret_f32(const uint32x4x2_t& v) {
    return {vreinterpretq_f32_u32(v.val[0]), vreinterpretq_f32_u32(v.val[1])};
}

static inline float32x4x2_t reinterpret_f32(const float32x4x2_t& v) {
    return v;
}

// Surprisingly, vdupq_n_u16 has the type of
// uint16x8_t (std::uint32_t) , and vdupq_n_u8 also has
// uint8x16_t (std::uint32_t) on **some environments**.
// We want argument type as same as the type of element
// of result vector type (std::uint16_t for uint16x8_t,
// and std::uint8_t for uint8x16_t) instead of
// std::uint32_t due to using set1 function templates,
// so let's fix the argument type here and use these
// overload below.
static inline ::uint16x8_t vdupq_n_u16(std::uint16_t v) {
    return ::vdupq_n_u16(v);
}

static inline ::uint8x16_t vdupq_n_u8(std::uint8_t v) {
    return ::vdupq_n_u8(v);
}

template <
        typename T,
        typename U = decltype(reinterpret_u8(std::declval<T>().data))>
struct is_simd256bit : std::is_same<U, uint8x16x2_t> {};

static inline void bin(const char (&bytes)[32], char bits[257]) {
    for (int i = 0; i < 256; ++i) {
        bits[i] = '0' + ((bytes[i / 8] >> (i % 8)) & 1);
    }
    bits[256] = 0;
}

template <typename T, size_t N, typename S>
static inline void bin(const S& simd, char bits[257]) {
    static_assert(
            std::is_same<void (S::*)(T*) const, decltype(&S::store)>::value,
            "invalid T");
    T ds[N];
    simd.store(ds);
    char bytes[32];
    std::memcpy(bytes, ds, sizeof(char) * 32);
    bin(bytes, bits);
}

template <typename S>
static inline std::string bin(const S& simd) {
    char bits[257];
    simd.bin(bits);
    return std::string(bits);
}

template <typename T>
using remove_cv_ref_t =
        typename std::remove_reference<typename std::remove_cv<T>::type>::type;

template <typename D, typename T>
struct set1_impl {
    D& d;
    T t;
    template <remove_cv_ref_t<decltype(std::declval<D>().val[0])> (*F)(T)>
    inline void call() {
        const auto v = F(t);
        d.val[0] = v;
        d.val[1] = v;
    }
};

template <typename D, typename T>
static inline set1_impl<remove_cv_ref_t<D>, T> set1(D& d, T t) {
    return {d, t};
}

template <typename T, size_t N, typename S>
static inline std::string elements_to_string(const char* fmt, const S& simd) {
    static_assert(
            std::is_same<void (S::*)(T*) const, decltype(&S::store)>::value,
            "invalid T");
    T bytes[N];
    simd.store(bytes);
    char res[1000], *ptr = res;
    for (size_t i = 0; i < N; ++i) {
        int bytesWritten =
                snprintf(ptr, sizeof(res) - (ptr - res), fmt, bytes[i]);
        ptr += bytesWritten;
    }
    // The format usually contains a ',' separator so this is to remove the last
    // separator.
    ptr[-1] = 0;
    return std::string(res);
}

template <typename T, typename U>
struct unary_func_impl {
    const U& a;
    using Telem = remove_cv_ref_t<decltype(std::declval<T>().val[0])>;
    using Uelem = remove_cv_ref_t<decltype(std::declval<U>().val[0])>;
    template <Telem (*F)(Uelem)>
    inline T call() {
        T t;
        t.val[0] = F(a.val[0]);
        t.val[1] = F(a.val[1]);
        return t;
    }
};

template <typename T>
static inline unary_func_impl<remove_cv_ref_t<T>, remove_cv_ref_t<T>> unary_func(
        const T& a) {
    return {a};
}

template <typename T, typename U>
static inline unary_func_impl<remove_cv_ref_t<T>, remove_cv_ref_t<U>> unary_func(
        const U& a) {
    return {a};
}

template <typename T, typename U>
struct binary_func_impl {
    const U& a;
    const U& b;
    using Telem = remove_cv_ref_t<decltype(std::declval<T>().val[0])>;
    using Uelem = remove_cv_ref_t<decltype(std::declval<U>().val[0])>;
    template <Telem (*F)(Uelem, Uelem)>
    inline T call() {
        T t;
        t.val[0] = F(a.val[0], b.val[0]);
        t.val[1] = F(a.val[1], b.val[1]);
        return t;
    }
};

template <typename T>
static inline binary_func_impl<remove_cv_ref_t<T>, remove_cv_ref_t<T>>
binary_func(const T& a, const T& b) {
    return {a, b};
}

template <typename T, typename U>
static inline binary_func_impl<remove_cv_ref_t<T>, remove_cv_ref_t<U>>
binary_func(const U& a, const U& b) {
    return {a, b};
}

static inline uint16_t vmovmask_u8(const uint8x16_t& v) {
    uint8_t d[16];
    const auto v2 = vreinterpretq_u16_u8(vshrq_n_u8(v, 7));
    const auto v3 = vreinterpretq_u32_u16(vsraq_n_u16(v2, v2, 7));
    const auto v4 = vreinterpretq_u64_u32(vsraq_n_u32(v3, v3, 14));
    vst1q_u8(d, vreinterpretq_u8_u64(vsraq_n_u64(v4, v4, 28)));
    return d[0] | static_cast<uint16_t>(d[8]) << 8u;
}

template <uint16x8_t (*F)(uint16x8_t, uint16x8_t)>
static inline uint32_t cmp_xe32(
        const uint16x8x2_t& d0,
        const uint16x8x2_t& d1,
        const uint16x8x2_t& thr) {
    const auto d0_thr = detail::simdlib::binary_func(d0, thr).call<F>();
    const auto d1_thr = detail::simdlib::binary_func(d1, thr).call<F>();
    const auto d0_mask = vmovmask_u8(
            vmovn_high_u16(vmovn_u16(d0_thr.val[0]), d0_thr.val[1]));
    const auto d1_mask = vmovmask_u8(
            vmovn_high_u16(vmovn_u16(d1_thr.val[0]), d1_thr.val[1]));
    return d0_mask | static_cast<uint32_t>(d1_mask) << 16;
}

template <std::uint8_t Shift>
static inline uint16x8_t vshlq(uint16x8_t vec) {
    return vshlq_n_u16(vec, Shift);
}

template <std::uint8_t Shift>
static inline uint16x8_t vshrq(uint16x8_t vec) {
    return vshrq_n_u16(vec, Shift);
}

} // namespace simdlib

} // namespace detail

/// vector of 16 elements in uint16
struct simd16uint16 {
    uint16x8x2_t data;

    simd16uint16() = default;

    explicit simd16uint16(int x) : data{vdupq_n_u16(x), vdupq_n_u16(x)} {}

    explicit simd16uint16(uint16_t x) : data{vdupq_n_u16(x), vdupq_n_u16(x)} {}

    explicit simd16uint16(const uint16x8x2_t& v) : data{v} {}

    explicit simd16uint16(
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
        data.val[0] = vld1q_u16(temp);
        data.val[1] = vld1q_u16(temp + 8);
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    explicit simd16uint16(const T& x)
            : data{detail::simdlib::reinterpret_u16(x.data)} {}

    explicit simd16uint16(const uint16_t* x)
            : data{vld1q_u16(x), vld1q_u16(x + 8)} {}

    void clear() {
        detail::simdlib::set1(data, static_cast<uint16_t>(0))
                .call<&detail::simdlib::vdupq_n_u16>();
    }

    void storeu(uint16_t* ptr) const {
        vst1q_u16(ptr, data.val[0]);
        vst1q_u16(ptr + 8, data.val[1]);
    }

    void loadu(const uint16_t* ptr) {
        data.val[0] = vld1q_u16(ptr);
        data.val[1] = vld1q_u16(ptr + 8);
    }

    void store(uint16_t* ptr) const {
        storeu(ptr);
    }

    void bin(char bits[257]) const {
        detail::simdlib::bin<uint16_t, 16u>(*this, bits);
    }

    std::string bin() const {
        return detail::simdlib::bin(*this);
    }

    std::string elements_to_string(const char* fmt) const {
        return detail::simdlib::elements_to_string<uint16_t, 16u>(fmt, *this);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }

    std::string dec() const {
        return elements_to_string("%3d,");
    }

    void set1(uint16_t x) {
        detail::simdlib::set1(data, x).call<&detail::simdlib::vdupq_n_u16>();
    }

    simd16uint16 operator*(const simd16uint16& other) const {
        return simd16uint16{detail::simdlib::binary_func(data, other.data)
                                    .call<&vmulq_u16>()};
    }

    // shift must be known at compile time
    simd16uint16 operator>>(const int shift) const {
        switch (shift) {
            case 0:
                return *this;
            case 1:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<1>>()};
            case 2:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<2>>()};
            case 3:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<3>>()};
            case 4:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<4>>()};
            case 5:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<5>>()};
            case 6:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<6>>()};
            case 7:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<7>>()};
            case 8:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<8>>()};
            case 9:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshrq<9>>()};
            case 10:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshrq<10>>()};
            case 11:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshrq<11>>()};
            case 12:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshrq<12>>()};
            case 13:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshrq<13>>()};
            case 14:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshrq<14>>()};
            case 15:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshrq<15>>()};
            default:
                FAISS_THROW_FMT("Invalid shift %d", shift);
        }
    }

    // shift must be known at compile time
    simd16uint16 operator<<(const int shift) const {
        switch (shift) {
            case 0:
                return *this;
            case 1:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<1>>()};
            case 2:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<2>>()};
            case 3:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<3>>()};
            case 4:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<4>>()};
            case 5:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<5>>()};
            case 6:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<6>>()};
            case 7:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<7>>()};
            case 8:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<8>>()};
            case 9:
                return simd16uint16{detail::simdlib::unary_func(data)
                                            .call<detail::simdlib::vshlq<9>>()};
            case 10:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshlq<10>>()};
            case 11:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshlq<11>>()};
            case 12:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshlq<12>>()};
            case 13:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshlq<13>>()};
            case 14:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshlq<14>>()};
            case 15:
                return simd16uint16{
                        detail::simdlib::unary_func(data)
                                .call<detail::simdlib::vshlq<15>>()};
            default:
                FAISS_THROW_FMT("Invalid shift %d", shift);
        }
    }

    simd16uint16 operator+=(const simd16uint16& other) {
        *this = *this + other;
        return *this;
    }

    simd16uint16 operator-=(const simd16uint16& other) {
        *this = *this - other;
        return *this;
    }

    simd16uint16 operator+(const simd16uint16& other) const {
        return simd16uint16{detail::simdlib::binary_func(data, other.data)
                                    .call<&vaddq_u16>()};
    }

    simd16uint16 operator-(const simd16uint16& other) const {
        return simd16uint16{detail::simdlib::binary_func(data, other.data)
                                    .call<&vsubq_u16>()};
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    simd16uint16 operator&(const T& other) const {
        return simd16uint16{
                detail::simdlib::binary_func(
                        data, detail::simdlib::reinterpret_u16(other.data))
                        .template call<&vandq_u16>()};
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    simd16uint16 operator|(const T& other) const {
        return simd16uint16{
                detail::simdlib::binary_func(
                        data, detail::simdlib::reinterpret_u16(other.data))
                        .template call<&vorrq_u16>()};
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    simd16uint16 operator^(const T& other) const {
        return simd16uint16{
                detail::simdlib::binary_func(
                        data, detail::simdlib::reinterpret_u16(other.data))
                        .template call<&veorq_u16>()};
    }

    // returns binary masks
    simd16uint16 operator==(const simd16uint16& other) const {
        return simd16uint16{detail::simdlib::binary_func(data, other.data)
                                    .call<&vceqq_u16>()};
    }

    // Checks whether the other holds exactly the same bytes.
    template <typename T>
    bool is_same_as(T other) const {
        const auto o = detail::simdlib::reinterpret_u16(other.data);
        const auto equals = detail::simdlib::binary_func(data, o)
                                    .template call<&vceqq_u16>();
        const auto equal = vandq_u16(equals.val[0], equals.val[1]);
        return vminvq_u16(equal) == 0xffffu;
    }

    simd16uint16 operator~() const {
        return simd16uint16{
                detail::simdlib::unary_func(data).call<&vmvnq_u16>()};
    }

    // get scalar at index 0
    uint16_t get_scalar_0() const {
        return vgetq_lane_u16(data.val[0], 0);
    }

    // mask of elements where this >= thresh
    // 2 bit per component: 16 * 2 = 32 bit
    uint32_t ge_mask(const simd16uint16& thresh) const {
        const auto input = detail::simdlib::binary_func(data, thresh.data)
                                   .call<&vcgeq_u16>();
        const auto vmovmask_u16 = [](uint16x8_t v) -> uint16_t {
            uint16_t d[8];
            const auto v2 = vreinterpretq_u32_u16(vshrq_n_u16(v, 14));
            const auto v3 = vreinterpretq_u64_u32(vsraq_n_u32(v2, v2, 14));
            vst1q_u16(d, vreinterpretq_u16_u64(vsraq_n_u64(v3, v3, 28)));
            return d[0] | d[4] << 8u;
        };
        return static_cast<uint32_t>(vmovmask_u16(input.val[1])) << 16u |
                vmovmask_u16(input.val[0]);
    }

    uint32_t le_mask(const simd16uint16& thresh) const {
        return thresh.ge_mask(*this);
    }

    uint32_t gt_mask(const simd16uint16& thresh) const {
        return ~le_mask(thresh);
    }

    bool all_gt(const simd16uint16& thresh) const {
        return le_mask(thresh) == 0;
    }

    // for debugging only
    uint16_t operator[](int i) const {
        uint16_t tab[8];
        const bool high = i >= 8;
        vst1q_u16(tab, data.val[high]);
        return tab[i - high * 8];
    }

    void accu_min(const simd16uint16& incoming) {
        data = detail::simdlib::binary_func(incoming.data, data)
                       .call<&vminq_u16>();
    }

    void accu_max(const simd16uint16& incoming) {
        data = detail::simdlib::binary_func(incoming.data, data)
                       .call<&vmaxq_u16>();
    }
};

// not really a std::min because it returns an elementwise min
inline simd16uint16 min(const simd16uint16& av, const simd16uint16& bv) {
    return simd16uint16{
            detail::simdlib::binary_func(av.data, bv.data).call<&vminq_u16>()};
}

inline simd16uint16 max(const simd16uint16& av, const simd16uint16& bv) {
    return simd16uint16{
            detail::simdlib::binary_func(av.data, bv.data).call<&vmaxq_u16>()};
}

// decompose in 128-lanes: a = (a0, a1), b = (b0, b1)
// return (a0 + a1, b0 + b1)
// TODO find a better name
inline simd16uint16 combine2x2(const simd16uint16& a, const simd16uint16& b) {
    return simd16uint16{uint16x8x2_t{
            vaddq_u16(a.data.val[0], a.data.val[1]),
            vaddq_u16(b.data.val[0], b.data.val[1])}};
}

// compare d0 and d1 to thr, return 32 bits corresponding to the concatenation
// of d0 and d1 with thr
inline uint32_t cmp_ge32(
        const simd16uint16& d0,
        const simd16uint16& d1,
        const simd16uint16& thr) {
    return detail::simdlib::cmp_xe32<&vcgeq_u16>(d0.data, d1.data, thr.data);
}

inline uint32_t cmp_le32(
        const simd16uint16& d0,
        const simd16uint16& d1,
        const simd16uint16& thr) {
    return detail::simdlib::cmp_xe32<&vcleq_u16>(d0.data, d1.data, thr.data);
}

// hadd does not cross lanes
inline simd16uint16 hadd(const simd16uint16& a, const simd16uint16& b) {
    return simd16uint16{
            detail::simdlib::binary_func(a.data, b.data).call<&vpaddq_u16>()};
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
        const simd16uint16 candidateValues,
        const simd16uint16 candidateIndices,
        const simd16uint16 currentValues,
        const simd16uint16 currentIndices,
        simd16uint16& minValues,
        simd16uint16& minIndices,
        simd16uint16& maxValues,
        simd16uint16& maxIndices) {
    const uint16x8x2_t comparison =
            detail::simdlib::binary_func(
                    candidateValues.data, currentValues.data)
                    .call<&vcltq_u16>();

    minValues = min(candidateValues, currentValues);
    minIndices.data = uint16x8x2_t{
            vbslq_u16(
                    comparison.val[0],
                    candidateIndices.data.val[0],
                    currentIndices.data.val[0]),
            vbslq_u16(
                    comparison.val[1],
                    candidateIndices.data.val[1],
                    currentIndices.data.val[1])};

    maxValues = max(candidateValues, currentValues);
    maxIndices.data = uint16x8x2_t{
            vbslq_u16(
                    comparison.val[0],
                    currentIndices.data.val[0],
                    candidateIndices.data.val[0]),
            vbslq_u16(
                    comparison.val[1],
                    currentIndices.data.val[1],
                    candidateIndices.data.val[1])};
}

// vector of 32 unsigned 8-bit integers
struct simd32uint8 {
    uint8x16x2_t data;

    simd32uint8() = default;

    explicit simd32uint8(int x) : data{vdupq_n_u8(x), vdupq_n_u8(x)} {}

    explicit simd32uint8(uint8_t x) : data{vdupq_n_u8(x), vdupq_n_u8(x)} {}

    explicit simd32uint8(const uint8x16x2_t& v) : data{v} {}

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
    static simd32uint8 create() {
        constexpr uint8_t ds[32] = {_0,  _1,  _2,  _3,  _4,  _5,  _6,  _7,
                                    _8,  _9,  _10, _11, _12, _13, _14, _15,
                                    _16, _17, _18, _19, _20, _21, _22, _23,
                                    _24, _25, _26, _27, _28, _29, _30, _31};
        return simd32uint8{ds};
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    explicit simd32uint8(const T& x)
            : data{detail::simdlib::reinterpret_u8(x.data)} {}

    explicit simd32uint8(const uint8_t* x)
            : data{vld1q_u8(x), vld1q_u8(x + 16)} {}

    void clear() {
        detail::simdlib::set1(data, static_cast<uint8_t>(0))
                .call<&detail::simdlib::vdupq_n_u8>();
    }

    void storeu(uint8_t* ptr) const {
        vst1q_u8(ptr, data.val[0]);
        vst1q_u8(ptr + 16, data.val[1]);
    }

    void loadu(const uint8_t* ptr) {
        data.val[0] = vld1q_u8(ptr);
        data.val[1] = vld1q_u8(ptr + 16);
    }

    void store(uint8_t* ptr) const {
        storeu(ptr);
    }

    void bin(char bits[257]) const {
        uint8_t bytes[32];
        store(bytes);
        detail::simdlib::bin(
                const_cast<const char(&)[32]>(
                        reinterpret_cast<char(&)[32]>(bytes)),
                bits);
    }

    std::string bin() const {
        return detail::simdlib::bin(*this);
    }

    std::string elements_to_string(const char* fmt) const {
        return detail::simdlib::elements_to_string<uint8_t, 32u>(fmt, *this);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }

    std::string dec() const {
        return elements_to_string("%3d,");
    }

    void set1(uint8_t x) {
        detail::simdlib::set1(data, x).call<&detail::simdlib::vdupq_n_u8>();
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    simd32uint8 operator&(const T& other) const {
        return simd32uint8{
                detail::simdlib::binary_func(
                        data, detail::simdlib::reinterpret_u8(other.data))
                        .template call<&vandq_u8>()};
    }

    simd32uint8 operator+(const simd32uint8& other) const {
        return simd32uint8{detail::simdlib::binary_func(data, other.data)
                                   .call<&vaddq_u8>()};
    }

    // The very important operation that everything relies on
    simd32uint8 lookup_2_lanes(const simd32uint8& idx) const {
        return simd32uint8{detail::simdlib::binary_func(data, idx.data)
                                   .call<&vqtbl1q_u8>()};
    }

    simd32uint8 operator+=(const simd32uint8& other) {
        *this = *this + other;
        return *this;
    }

    // for debugging only
    uint8_t operator[](int i) const {
        uint8_t tab[16];
        const bool high = i >= 16;
        vst1q_u8(tab, data.val[high]);
        return tab[i - high * 16];
    }

    // Checks whether the other holds exactly the same bytes.
    template <typename T>
    bool is_same_as(T other) const {
        const auto o = detail::simdlib::reinterpret_u8(other.data);
        const auto equals = detail::simdlib::binary_func(data, o)
                                    .template call<&vceqq_u8>();
        const auto equal = vandq_u8(equals.val[0], equals.val[1]);
        return vminvq_u8(equal) == 0xffu;
    }
};

// convert with saturation
// careful: this does not cross lanes, so the order is weird
inline simd32uint8 uint16_to_uint8_saturate(
        const simd16uint16& a,
        const simd16uint16& b) {
    return simd32uint8{uint8x16x2_t{
            vqmovn_high_u16(vqmovn_u16(a.data.val[0]), b.data.val[0]),
            vqmovn_high_u16(vqmovn_u16(a.data.val[1]), b.data.val[1])}};
}

/// get most significant bit of each byte
inline uint32_t get_MSBs(const simd32uint8& a) {
    using detail::simdlib::vmovmask_u8;
    return vmovmask_u8(a.data.val[0]) |
            static_cast<uint32_t>(vmovmask_u8(a.data.val[1])) << 16u;
}

/// use MSB of each byte of mask to select a byte between a and b
inline simd32uint8 blendv(
        const simd32uint8& a,
        const simd32uint8& b,
        const simd32uint8& mask) {
    const auto msb = vdupq_n_u8(0x80);
    const uint8x16x2_t msb_mask = {
            vtstq_u8(mask.data.val[0], msb), vtstq_u8(mask.data.val[1], msb)};
    const uint8x16x2_t selected = {
            vbslq_u8(msb_mask.val[0], b.data.val[0], a.data.val[0]),
            vbslq_u8(msb_mask.val[1], b.data.val[1], a.data.val[1])};
    return simd32uint8{selected};
}

/// vector of 8 unsigned 32-bit integers
struct simd8uint32 {
    uint32x4x2_t data;

    simd8uint32() = default;

    explicit simd8uint32(uint32_t x) : data{vdupq_n_u32(x), vdupq_n_u32(x)} {}

    explicit simd8uint32(const uint32x4x2_t& v) : data{v} {}

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    explicit simd8uint32(const T& x)
            : data{detail::simdlib::reinterpret_u32(x.data)} {}

    explicit simd8uint32(const uint8_t* x) : simd8uint32(simd32uint8(x)) {}

    explicit simd8uint32(
            uint32_t u0,
            uint32_t u1,
            uint32_t u2,
            uint32_t u3,
            uint32_t u4,
            uint32_t u5,
            uint32_t u6,
            uint32_t u7) {
        uint32_t temp[8] = {u0, u1, u2, u3, u4, u5, u6, u7};
        data.val[0] = vld1q_u32(temp);
        data.val[1] = vld1q_u32(temp + 4);
    }

    simd8uint32 operator+(simd8uint32 other) const {
        return simd8uint32{detail::simdlib::binary_func(data, other.data)
                                   .call<&vaddq_u32>()};
    }

    simd8uint32 operator-(simd8uint32 other) const {
        return simd8uint32{detail::simdlib::binary_func(data, other.data)
                                   .call<&vsubq_u32>()};
    }

    simd8uint32& operator+=(const simd8uint32& other) {
        data.val[0] = vaddq_u32(data.val[0], other.data.val[0]);
        data.val[1] = vaddq_u32(data.val[1], other.data.val[1]);
        return *this;
    }

    simd8uint32 operator==(simd8uint32 other) const {
        return simd8uint32{detail::simdlib::binary_func(data, other.data)
                                   .call<&vceqq_u32>()};
    }

    simd8uint32 operator~() const {
        return simd8uint32{
                detail::simdlib::unary_func(data).call<&vmvnq_u32>()};
    }

    simd8uint32 operator!=(simd8uint32 other) const {
        return ~(*this == other);
    }

    // Checks whether the other holds exactly the same bytes.
    template <typename T>
    bool is_same_as(T other) const {
        const auto o = detail::simdlib::reinterpret_u32(other.data);
        const auto equals = detail::simdlib::binary_func(data, o)
                                    .template call<&vceqq_u32>();
        const auto equal = vandq_u32(equals.val[0], equals.val[1]);
        return vminvq_u32(equal) == 0xffffffffu;
    }

    void clear() {
        detail::simdlib::set1(data, static_cast<uint32_t>(0))
                .call<&vdupq_n_u32>();
    }

    void storeu(uint32_t* ptr) const {
        vst1q_u32(ptr, data.val[0]);
        vst1q_u32(ptr + 4, data.val[1]);
    }

    void loadu(const uint32_t* ptr) {
        data.val[0] = vld1q_u32(ptr);
        data.val[1] = vld1q_u32(ptr + 4);
    }

    void store(uint32_t* ptr) const {
        storeu(ptr);
    }

    void bin(char bits[257]) const {
        detail::simdlib::bin<uint32_t, 8u>(*this, bits);
    }

    std::string bin() const {
        return detail::simdlib::bin(*this);
    }

    std::string elements_to_string(const char* fmt) const {
        return detail::simdlib::elements_to_string<uint32_t, 8u>(fmt, *this);
    }

    std::string hex() const {
        return elements_to_string("%08x,");
    }

    std::string dec() const {
        return elements_to_string("%10d,");
    }

    void set1(uint32_t x) {
        detail::simdlib::set1(data, x).call<&vdupq_n_u32>();
    }

    simd8uint32 unzip() const {
        return simd8uint32{uint32x4x2_t{
                vuzp1q_u32(data.val[0], data.val[1]),
                vuzp2q_u32(data.val[0], data.val[1])}};
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
        const simd8uint32 candidateValues,
        const simd8uint32 candidateIndices,
        const simd8uint32 currentValues,
        const simd8uint32 currentIndices,
        simd8uint32& minValues,
        simd8uint32& minIndices,
        simd8uint32& maxValues,
        simd8uint32& maxIndices) {
    const uint32x4x2_t comparison =
            detail::simdlib::binary_func(
                    candidateValues.data, currentValues.data)
                    .call<&vcltq_u32>();

    minValues.data = detail::simdlib::binary_func(
                             candidateValues.data, currentValues.data)
                             .call<&vminq_u32>();
    minIndices.data = uint32x4x2_t{
            vbslq_u32(
                    comparison.val[0],
                    candidateIndices.data.val[0],
                    currentIndices.data.val[0]),
            vbslq_u32(
                    comparison.val[1],
                    candidateIndices.data.val[1],
                    currentIndices.data.val[1])};

    maxValues.data = detail::simdlib::binary_func(
                             candidateValues.data, currentValues.data)
                             .call<&vmaxq_u32>();
    maxIndices.data = uint32x4x2_t{
            vbslq_u32(
                    comparison.val[0],
                    currentIndices.data.val[0],
                    candidateIndices.data.val[0]),
            vbslq_u32(
                    comparison.val[1],
                    currentIndices.data.val[1],
                    candidateIndices.data.val[1])};
}

struct simd8float32 {
    float32x4x2_t data;

    simd8float32() = default;

    explicit simd8float32(float x) : data{vdupq_n_f32(x), vdupq_n_f32(x)} {}

    explicit simd8float32(const float32x4x2_t& v) : data{v} {}

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    explicit simd8float32(const T& x)
            : data{detail::simdlib::reinterpret_f32(x.data)} {}

    explicit simd8float32(const float* x)
            : data{vld1q_f32(x), vld1q_f32(x + 4)} {}

    explicit simd8float32(
            float f0,
            float f1,
            float f2,
            float f3,
            float f4,
            float f5,
            float f6,
            float f7) {
        float temp[8] = {f0, f1, f2, f3, f4, f5, f6, f7};
        data.val[0] = vld1q_f32(temp);
        data.val[1] = vld1q_f32(temp + 4);
    }

    void clear() {
        detail::simdlib::set1(data, 0.f).call<&vdupq_n_f32>();
    }

    void storeu(float* ptr) const {
        vst1q_f32(ptr, data.val[0]);
        vst1q_f32(ptr + 4, data.val[1]);
    }

    void loadu(const float* ptr) {
        data.val[0] = vld1q_f32(ptr);
        data.val[1] = vld1q_f32(ptr + 4);
    }

    void store(float* ptr) const {
        storeu(ptr);
    }

    void bin(char bits[257]) const {
        detail::simdlib::bin<float, 8u>(*this, bits);
    }

    std::string bin() const {
        return detail::simdlib::bin(*this);
    }

    simd8float32 operator*(const simd8float32& other) const {
        return simd8float32{detail::simdlib::binary_func(data, other.data)
                                    .call<&vmulq_f32>()};
    }

    simd8float32 operator+(const simd8float32& other) const {
        return simd8float32{detail::simdlib::binary_func(data, other.data)
                                    .call<&vaddq_f32>()};
    }

    simd8float32 operator-(const simd8float32& other) const {
        return simd8float32{detail::simdlib::binary_func(data, other.data)
                                    .call<&vsubq_f32>()};
    }

    simd8float32& operator+=(const simd8float32& other) {
        // In this context, it is more compiler friendly to write intrinsics
        // directly instead of using binary_func
        data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return *this;
    }

    simd8uint32 operator==(simd8float32 other) const {
        return simd8uint32{
                detail::simdlib::binary_func<::uint32x4x2_t>(data, other.data)
                        .call<&vceqq_f32>()};
    }

    simd8uint32 operator!=(simd8float32 other) const {
        return ~(*this == other);
    }

    // Checks whether the other holds exactly the same bytes.
    template <typename T>
    bool is_same_as(T other) const {
        const auto o = detail::simdlib::reinterpret_f32(other.data);
        const auto equals =
                detail::simdlib::binary_func<::uint32x4x2_t>(data, o)
                        .template call<&vceqq_f32>();
        const auto equal = vandq_u32(equals.val[0], equals.val[1]);
        return vminvq_u32(equal) == 0xffffffffu;
    }

    std::string tostring() const {
        return detail::simdlib::elements_to_string<float, 8u>("%g,", *this);
    }
};

// hadd does not cross lanes
inline simd8float32 hadd(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data).call<&vpaddq_f32>()};
}

inline simd8float32 unpacklo(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data).call<&vzip1q_f32>()};
}

inline simd8float32 unpackhi(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data).call<&vzip2q_f32>()};
}

// compute a * b + c
inline simd8float32 fmadd(
        const simd8float32& a,
        const simd8float32& b,
        const simd8float32& c) {
    return simd8float32{float32x4x2_t{
            vfmaq_f32(c.data.val[0], a.data.val[0], b.data.val[0]),
            vfmaq_f32(c.data.val[1], a.data.val[1], b.data.val[1])}};
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
    const auto comparison = detail::simdlib::binary_func<::uint32x4x2_t>(
                                    candidateValues.data, lowestValues.data)
                                    .call<&vcltq_f32>();

    lowestValues.data = float32x4x2_t{
            vbslq_f32(
                    comparison.val[0],
                    candidateValues.data.val[0],
                    lowestValues.data.val[0]),
            vbslq_f32(
                    comparison.val[1],
                    candidateValues.data.val[1],
                    lowestValues.data.val[1])};
    lowestIndices.data = uint32x4x2_t{
            vbslq_u32(
                    comparison.val[0],
                    candidateIndices.data.val[0],
                    lowestIndices.data.val[0]),
            vbslq_u32(
                    comparison.val[1],
                    candidateIndices.data.val[1],
                    lowestIndices.data.val[1])};
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
        const simd8float32 candidateValues,
        const simd8uint32 candidateIndices,
        const simd8float32 currentValues,
        const simd8uint32 currentIndices,
        simd8float32& minValues,
        simd8uint32& minIndices,
        simd8float32& maxValues,
        simd8uint32& maxIndices) {
    const uint32x4x2_t comparison =
            detail::simdlib::binary_func<::uint32x4x2_t>(
                    candidateValues.data, currentValues.data)
                    .call<&vcltq_f32>();

    minValues.data = detail::simdlib::binary_func(
                             candidateValues.data, currentValues.data)
                             .call<&vminq_f32>();
    minIndices.data = uint32x4x2_t{
            vbslq_u32(
                    comparison.val[0],
                    candidateIndices.data.val[0],
                    currentIndices.data.val[0]),
            vbslq_u32(
                    comparison.val[1],
                    candidateIndices.data.val[1],
                    currentIndices.data.val[1])};

    maxValues.data = detail::simdlib::binary_func(
                             candidateValues.data, currentValues.data)
                             .call<&vmaxq_f32>();
    maxIndices.data = uint32x4x2_t{
            vbslq_u32(
                    comparison.val[0],
                    currentIndices.data.val[0],
                    candidateIndices.data.val[0]),
            vbslq_u32(
                    comparison.val[1],
                    currentIndices.data.val[1],
                    candidateIndices.data.val[1])};
}

namespace {

// get even float32's of a and b, interleaved
simd8float32 geteven(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data).call<&vuzp1q_f32>()};
}

// get odd float32's of a and b, interleaved
simd8float32 getodd(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data).call<&vuzp2q_f32>()};
}

// 3 cycles
// if the lanes are a = [a0 a1] and b = [b0 b1], return [a0 b0]
simd8float32 getlow128(const simd8float32& a, const simd8float32& b) {
    return simd8float32{float32x4x2_t{a.data.val[0], b.data.val[0]}};
}

simd8float32 gethigh128(const simd8float32& a, const simd8float32& b) {
    return simd8float32{float32x4x2_t{a.data.val[1], b.data.val[1]}};
}

} // namespace

} // namespace faiss
