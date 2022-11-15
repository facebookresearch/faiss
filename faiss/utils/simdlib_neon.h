/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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

template <typename D, typename F, typename T>
static inline void set1(D& d, F&& f, T t) {
    const auto v = f(t);
    d.val[0] = v;
    d.val[1] = v;
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
        ptr += sprintf(ptr, fmt, bytes[i]);
    }
    // strip last ,
    ptr[-1] = 0;
    return std::string(res);
}

template <typename T, typename F>
static inline T unary_func(const T& a, F&& f) {
    T t;
    t.val[0] = f(a.val[0]);
    t.val[1] = f(a.val[1]);
    return t;
}

template <typename T, typename F>
static inline T binary_func(const T& a, const T& b, F&& f) {
    T t;
    t.val[0] = f(a.val[0], b.val[0]);
    t.val[1] = f(a.val[1], b.val[1]);
    return t;
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
    const auto d0_thr = detail::simdlib::binary_func(d0, thr, F);
    const auto d1_thr = detail::simdlib::binary_func(d1, thr, F);
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
        detail::simdlib::set1(data, &vdupq_n_u16, static_cast<uint16_t>(0));
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
        detail::simdlib::set1(data, &vdupq_n_u16, x);
    }

    simd16uint16 operator*(const simd16uint16& other) const {
        return simd16uint16{
                detail::simdlib::binary_func(data, other.data, &vmulq_u16)};
    }

    // shift must be known at compile time
    simd16uint16 operator>>(const int shift) const {
        switch (shift) {
            case 0:
                return *this;
            case 1:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<1>)};
            case 2:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<2>)};
            case 3:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<3>)};
            case 4:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<4>)};
            case 5:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<5>)};
            case 6:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<6>)};
            case 7:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<7>)};
            case 8:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<8>)};
            case 9:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<9>)};
            case 10:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<10>)};
            case 11:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<11>)};
            case 12:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<12>)};
            case 13:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<13>)};
            case 14:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<14>)};
            case 15:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshrq<15>)};
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
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<1>)};
            case 2:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<2>)};
            case 3:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<3>)};
            case 4:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<4>)};
            case 5:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<5>)};
            case 6:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<6>)};
            case 7:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<7>)};
            case 8:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<8>)};
            case 9:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<9>)};
            case 10:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<10>)};
            case 11:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<11>)};
            case 12:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<12>)};
            case 13:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<13>)};
            case 14:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<14>)};
            case 15:
                return simd16uint16{detail::simdlib::unary_func(
                        data, detail::simdlib::vshlq<15>)};
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
        return simd16uint16{
                detail::simdlib::binary_func(data, other.data, &vaddq_u16)};
    }

    simd16uint16 operator-(const simd16uint16& other) const {
        return simd16uint16{
                detail::simdlib::binary_func(data, other.data, &vsubq_u16)};
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    simd16uint16 operator&(const T& other) const {
        return simd16uint16{detail::simdlib::binary_func(
                data,
                detail::simdlib::reinterpret_u16(other.data),
                &vandq_u16)};
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    simd16uint16 operator|(const T& other) const {
        return simd16uint16{detail::simdlib::binary_func(
                data,
                detail::simdlib::reinterpret_u16(other.data),
                &vorrq_u16)};
    }

    // returns binary masks
    simd16uint16 operator==(const simd16uint16& other) const {
        return simd16uint16{
                detail::simdlib::binary_func(data, other.data, &vceqq_u16)};
    }

    simd16uint16 operator~() const {
        return simd16uint16{detail::simdlib::unary_func(data, &vmvnq_u16)};
    }

    // get scalar at index 0
    uint16_t get_scalar_0() const {
        return vgetq_lane_u16(data.val[0], 0);
    }

    // mask of elements where this >= thresh
    // 2 bit per component: 16 * 2 = 32 bit
    uint32_t ge_mask(const simd16uint16& thresh) const {
        const auto input =
                detail::simdlib::binary_func(data, thresh.data, &vcgeq_u16);
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
        data = detail::simdlib::binary_func(incoming.data, data, &vminq_u16);
    }

    void accu_max(const simd16uint16& incoming) {
        data = detail::simdlib::binary_func(incoming.data, data, &vmaxq_u16);
    }
};

// not really a std::min because it returns an elementwise min
inline simd16uint16 min(const simd16uint16& av, const simd16uint16& bv) {
    return simd16uint16{
            detail::simdlib::binary_func(av.data, bv.data, &vminq_u16)};
}

inline simd16uint16 max(const simd16uint16& av, const simd16uint16& bv) {
    return simd16uint16{
            detail::simdlib::binary_func(av.data, bv.data, &vmaxq_u16)};
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

// vector of 32 unsigned 8-bit integers
struct simd32uint8 {
    uint8x16x2_t data;

    simd32uint8() = default;

    explicit simd32uint8(int x) : data{vdupq_n_u8(x), vdupq_n_u8(x)} {}

    explicit simd32uint8(uint8_t x) : data{vdupq_n_u8(x), vdupq_n_u8(x)} {}

    explicit simd32uint8(const uint8x16x2_t& v) : data{v} {}

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
        detail::simdlib::set1(data, &vdupq_n_u8, static_cast<uint8_t>(0));
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
        detail::simdlib::set1(data, &vdupq_n_u8, x);
    }

    template <
            typename T,
            typename std::enable_if<
                    detail::simdlib::is_simd256bit<T>::value,
                    std::nullptr_t>::type = nullptr>
    simd32uint8 operator&(const T& other) const {
        return simd32uint8{detail::simdlib::binary_func(
                data, detail::simdlib::reinterpret_u8(other.data), &vandq_u8)};
    }

    simd32uint8 operator+(const simd32uint8& other) const {
        return simd32uint8{
                detail::simdlib::binary_func(data, other.data, &vaddq_u8)};
    }

    // The very important operation that everything relies on
    simd32uint8 lookup_2_lanes(const simd32uint8& idx) const {
        return simd32uint8{
                detail::simdlib::binary_func(data, idx.data, &vqtbl1q_u8)};
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
        return simd8uint32{uint32x4x2_t{
                vaddq_u32(data.val[0], other.data.val[0]),
                vaddq_u32(data.val[1], other.data.val[1])}};
    }

    simd8uint32 operator-(simd8uint32 other) const {
        return simd8uint32{uint32x4x2_t{
                vsubq_u32(data.val[0], other.data.val[0]),
                vsubq_u32(data.val[1], other.data.val[1])}};
    }

    bool operator==(simd8uint32 other) const {
        const bool equal0 =
                (vminvq_u32(vceqq_u32(data.val[0], other.data.val[0])) ==
                 0xffffffff);
        const bool equal1 =
                (vminvq_u32(vceqq_u32(data.val[1], other.data.val[1])) ==
                 0xffffffff);

        return equal0 && equal1;
    }

    bool operator!=(simd8uint32 other) const {
        return !(*this == other);
    }

    void clear() {
        detail::simdlib::set1(data, &vdupq_n_u32, static_cast<uint32_t>(0));
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
        detail::simdlib::set1(data, &vdupq_n_u32, x);
    }
};

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
        detail::simdlib::set1(data, &vdupq_n_f32, 0.f);
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
        return simd8float32{
                detail::simdlib::binary_func(data, other.data, &vmulq_f32)};
    }

    simd8float32 operator+(const simd8float32& other) const {
        return simd8float32{
                detail::simdlib::binary_func(data, other.data, &vaddq_f32)};
    }

    simd8float32 operator-(const simd8float32& other) const {
        return simd8float32{
                detail::simdlib::binary_func(data, other.data, &vsubq_f32)};
    }

    bool operator==(simd8float32 other) const {
        const bool equal0 =
                (vminvq_u32(vceqq_f32(data.val[0], other.data.val[0])) ==
                 0xffffffff);
        const bool equal1 =
                (vminvq_u32(vceqq_f32(data.val[1], other.data.val[1])) ==
                 0xffffffff);

        return equal0 && equal1;
    }

    bool operator!=(simd8float32 other) const {
        return !(*this == other);
    }

    std::string tostring() const {
        return detail::simdlib::elements_to_string<float, 8u>("%g,", *this);
    }
};

// hadd does not cross lanes
inline simd8float32 hadd(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data, &vpaddq_f32)};
}

inline simd8float32 unpacklo(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data, &vzip1q_f32)};
}

inline simd8float32 unpackhi(const simd8float32& a, const simd8float32& b) {
    return simd8float32{
            detail::simdlib::binary_func(a.data, b.data, &vzip2q_f32)};
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
    uint32x4x2_t comparison = uint32x4x2_t{
            vcltq_f32(candidateValues.data.val[0], lowestValues.data.val[0]),
            vcltq_f32(candidateValues.data.val[1], lowestValues.data.val[1])};

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

namespace {

// get even float32's of a and b, interleaved
simd8float32 geteven(const simd8float32& a, const simd8float32& b) {
    return simd8float32{float32x4x2_t{
            vuzp1q_f32(a.data.val[0], b.data.val[0]),
            vuzp1q_f32(a.data.val[1], b.data.val[1])}};
}

// get odd float32's of a and b, interleaved
simd8float32 getodd(const simd8float32& a, const simd8float32& b) {
    return simd8float32{float32x4x2_t{
            vuzp2q_f32(a.data.val[0], b.data.val[0]),
            vuzp2q_f32(a.data.val[1], b.data.val[1])}};
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
