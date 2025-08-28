/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace faiss {

struct simd256bit {
    union {
        uint8_t u8[32];
        uint16_t u16[16];
        uint32_t u32[8];
        float f32[8];
    };

    simd256bit() {}

    explicit simd256bit(const void* x) {
        memcpy(u8, x, 32);
    }

    void clear() {
        memset(u8, 0, 32);
    }

    void storeu(void* ptr) const {
        memcpy(ptr, u8, 32);
    }

    void loadu(const void* ptr) {
        memcpy(u8, ptr, 32);
    }

    void store(void* ptr) const {
        storeu(ptr);
    }

    void bin(char bits[257]) const {
        const char* bytes = (char*)this->u8;
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

    // Checks whether the other holds exactly the same bytes.
    bool is_same_as(simd256bit other) const {
        for (size_t i = 0; i < 8; i++) {
            if (u32[i] != other.u32[i]) {
                return false;
            }
        }

        return true;
    }
};

/// vector of 16 elements in uint16
struct simd16uint16 : simd256bit {
    simd16uint16() {}

    explicit simd16uint16(int x) {
        set1(x);
    }

    explicit simd16uint16(uint16_t x) {
        set1(x);
    }

    explicit simd16uint16(const simd256bit& x) : simd256bit(x) {}

    explicit simd16uint16(const uint16_t* x) : simd256bit((const void*)x) {}

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
        this->u16[0] = u0;
        this->u16[1] = u1;
        this->u16[2] = u2;
        this->u16[3] = u3;
        this->u16[4] = u4;
        this->u16[5] = u5;
        this->u16[6] = u6;
        this->u16[7] = u7;
        this->u16[8] = u8;
        this->u16[9] = u9;
        this->u16[10] = u10;
        this->u16[11] = u11;
        this->u16[12] = u12;
        this->u16[13] = u13;
        this->u16[14] = u14;
        this->u16[15] = u15;
    }

    std::string elements_to_string(const char* fmt) const {
        char res[1000], *ptr = res;
        for (int i = 0; i < 16; i++) {
            ptr += sprintf(ptr, fmt, u16[i]);
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

    template <typename F>
    static simd16uint16 unary_func(const simd16uint16& a, F&& f) {
        simd16uint16 c;
        for (int j = 0; j < 16; j++) {
            c.u16[j] = f(a.u16[j]);
        }
        return c;
    }

    template <typename F>
    static simd16uint16 binary_func(
            const simd16uint16& a,
            const simd16uint16& b,
            F&& f) {
        simd16uint16 c;
        for (int j = 0; j < 16; j++) {
            c.u16[j] = f(a.u16[j], b.u16[j]);
        }
        return c;
    }

    void set1(uint16_t x) {
        for (int i = 0; i < 16; i++) {
            u16[i] = x;
        }
    }

    simd16uint16 operator*(const simd16uint16& other) const {
        return binary_func(
                *this, other, [](uint16_t a, uint16_t b) { return a * b; });
    }

    // shift must be known at compile time
    simd16uint16 operator>>(const int shift) const {
        return unary_func(*this, [shift](uint16_t a) { return a >> shift; });
    }

    // shift must be known at compile time
    simd16uint16 operator<<(const int shift) const {
        return unary_func(*this, [shift](uint16_t a) { return a << shift; });
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
        return binary_func(
                *this, other, [](uint16_t a, uint16_t b) { return a + b; });
    }

    simd16uint16 operator-(const simd16uint16& other) const {
        return binary_func(
                *this, other, [](uint16_t a, uint16_t b) { return a - b; });
    }

    simd16uint16 operator&(const simd256bit& other) const {
        return binary_func(
                *this, simd16uint16(other), [](uint16_t a, uint16_t b) {
                    return a & b;
                });
    }

    simd16uint16 operator|(const simd256bit& other) const {
        return binary_func(
                *this, simd16uint16(other), [](uint16_t a, uint16_t b) {
                    return a | b;
                });
    }

    simd16uint16 operator^(const simd256bit& other) const {
        return binary_func(
                *this, simd16uint16(other), [](uint16_t a, uint16_t b) {
                    return a ^ b;
                });
    }

    // returns binary masks
    simd16uint16 operator==(const simd16uint16& other) const {
        return binary_func(*this, other, [](uint16_t a, uint16_t b) {
            return a == b ? 0xffff : 0;
        });
    }

    simd16uint16 operator~() const {
        return unary_func(*this, [](uint16_t a) { return ~a; });
    }

    // get scalar at index 0
    uint16_t get_scalar_0() const {
        return u16[0];
    }

    // mask of elements where this >= thresh
    // 2 bit per component: 16 * 2 = 32 bit
    uint32_t ge_mask(const simd16uint16& thresh) const {
        uint32_t gem = 0;
        for (int j = 0; j < 16; j++) {
            if (u16[j] >= thresh.u16[j]) {
                gem |= 3 << (j * 2);
            }
        }
        return gem;
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
        return u16[i];
    }

    void accu_min(const simd16uint16& incoming) {
        for (int j = 0; j < 16; j++) {
            if (incoming.u16[j] < u16[j]) {
                u16[j] = incoming.u16[j];
            }
        }
    }

    void accu_max(const simd16uint16& incoming) {
        for (int j = 0; j < 16; j++) {
            if (incoming.u16[j] > u16[j]) {
                u16[j] = incoming.u16[j];
            }
        }
    }
};

// not really a std::min because it returns an elementwise min
inline simd16uint16 min(const simd16uint16& av, const simd16uint16& bv) {
    return simd16uint16::binary_func(
            av, bv, [](uint16_t a, uint16_t b) { return std::min(a, b); });
}

inline simd16uint16 max(const simd16uint16& av, const simd16uint16& bv) {
    return simd16uint16::binary_func(
            av, bv, [](uint16_t a, uint16_t b) { return std::max(a, b); });
}

// decompose in 128-lanes: a = (a0, a1), b = (b0, b1)
// return (a0 + a1, b0 + b1)
// TODO find a better name
inline simd16uint16 combine2x2(const simd16uint16& a, const simd16uint16& b) {
    simd16uint16 c;
    for (int j = 0; j < 8; j++) {
        c.u16[j] = a.u16[j] + a.u16[j + 8];
        c.u16[j + 8] = b.u16[j] + b.u16[j + 8];
    }
    return c;
}

// compare d0 and d1 to thr, return 32 bits corresponding to the concatenation
// of d0 and d1 with thr
inline uint32_t cmp_ge32(
        const simd16uint16& d0,
        const simd16uint16& d1,
        const simd16uint16& thr) {
    uint32_t gem = 0;
    for (int j = 0; j < 16; j++) {
        if (d0.u16[j] >= thr.u16[j]) {
            gem |= 1 << j;
        }
        if (d1.u16[j] >= thr.u16[j]) {
            gem |= 1 << (j + 16);
        }
    }
    return gem;
}

inline uint32_t cmp_le32(
        const simd16uint16& d0,
        const simd16uint16& d1,
        const simd16uint16& thr) {
    uint32_t gem = 0;
    for (int j = 0; j < 16; j++) {
        if (d0.u16[j] <= thr.u16[j]) {
            gem |= 1 << j;
        }
        if (d1.u16[j] <= thr.u16[j]) {
            gem |= 1 << (j + 16);
        }
    }
    return gem;
}

// hadd does not cross lanes
inline simd16uint16 hadd(const simd16uint16& a, const simd16uint16& b) {
    simd16uint16 c;
    c.u16[0] = a.u16[0] + a.u16[1];
    c.u16[1] = a.u16[2] + a.u16[3];
    c.u16[2] = a.u16[4] + a.u16[5];
    c.u16[3] = a.u16[6] + a.u16[7];
    c.u16[4] = b.u16[0] + b.u16[1];
    c.u16[5] = b.u16[2] + b.u16[3];
    c.u16[6] = b.u16[4] + b.u16[5];
    c.u16[7] = b.u16[6] + b.u16[7];

    c.u16[8] = a.u16[8] + a.u16[9];
    c.u16[9] = a.u16[10] + a.u16[11];
    c.u16[10] = a.u16[12] + a.u16[13];
    c.u16[11] = a.u16[14] + a.u16[15];
    c.u16[12] = b.u16[8] + b.u16[9];
    c.u16[13] = b.u16[10] + b.u16[11];
    c.u16[14] = b.u16[12] + b.u16[13];
    c.u16[15] = b.u16[14] + b.u16[15];

    return c;
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
    for (size_t i = 0; i < 16; i++) {
        bool flag = (candidateValues.u16[i] < currentValues.u16[i]);
        minValues.u16[i] = flag ? candidateValues.u16[i] : currentValues.u16[i];
        minIndices.u16[i] =
                flag ? candidateIndices.u16[i] : currentIndices.u16[i];
        maxValues.u16[i] =
                !flag ? candidateValues.u16[i] : currentValues.u16[i];
        maxIndices.u16[i] =
                !flag ? candidateIndices.u16[i] : currentIndices.u16[i];
    }
}

// vector of 32 unsigned 8-bit integers
struct simd32uint8 : simd256bit {
    simd32uint8() {}

    explicit simd32uint8(int x) {
        set1(x);
    }

    explicit simd32uint8(uint8_t x) {
        set1(x);
    }
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
        simd32uint8 ret;
        ret.u8[0] = _0;
        ret.u8[1] = _1;
        ret.u8[2] = _2;
        ret.u8[3] = _3;
        ret.u8[4] = _4;
        ret.u8[5] = _5;
        ret.u8[6] = _6;
        ret.u8[7] = _7;
        ret.u8[8] = _8;
        ret.u8[9] = _9;
        ret.u8[10] = _10;
        ret.u8[11] = _11;
        ret.u8[12] = _12;
        ret.u8[13] = _13;
        ret.u8[14] = _14;
        ret.u8[15] = _15;
        ret.u8[16] = _16;
        ret.u8[17] = _17;
        ret.u8[18] = _18;
        ret.u8[19] = _19;
        ret.u8[20] = _20;
        ret.u8[21] = _21;
        ret.u8[22] = _22;
        ret.u8[23] = _23;
        ret.u8[24] = _24;
        ret.u8[25] = _25;
        ret.u8[26] = _26;
        ret.u8[27] = _27;
        ret.u8[28] = _28;
        ret.u8[29] = _29;
        ret.u8[30] = _30;
        ret.u8[31] = _31;
        return ret;
    }

    explicit simd32uint8(const simd256bit& x) : simd256bit(x) {}

    explicit simd32uint8(const uint8_t* x) : simd256bit((const void*)x) {}

    std::string elements_to_string(const char* fmt) const {
        char res[1000], *ptr = res;
        for (int i = 0; i < 32; i++) {
            ptr += sprintf(ptr, fmt, u8[i]);
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
        for (int j = 0; j < 32; j++) {
            u8[j] = x;
        }
    }

    template <typename F>
    static simd32uint8 binary_func(
            const simd32uint8& a,
            const simd32uint8& b,
            F&& f) {
        simd32uint8 c;
        for (int j = 0; j < 32; j++) {
            c.u8[j] = f(a.u8[j], b.u8[j]);
        }
        return c;
    }

    simd32uint8 operator&(const simd256bit& other) const {
        return binary_func(*this, simd32uint8(other), [](uint8_t a, uint8_t b) {
            return a & b;
        });
    }

    simd32uint8 operator+(const simd32uint8& other) const {
        return binary_func(
                *this, other, [](uint8_t a, uint8_t b) { return a + b; });
    }

    // The very important operation that everything relies on
    simd32uint8 lookup_2_lanes(const simd32uint8& idx) const {
        simd32uint8 c;
        // The original for loop:
        // for (int j = 0; j < 32; j++) {
        //     if (idx.u8[j] & 0x80) {
        //         c.u8[j] = 0;
        //     } else {
        //         uint8_t i = idx.u8[j] & 15;
        //         if (j < 16) {
        //             c.u8[j] = u8[i];
        //         } else {
        //             c.u8[j] = u8[16 + i];
        //         }
        //     }

        // The following function was re-written for Power 10
        // The loop was unrolled to remove the if (j < 16) statement by doing
        // the j and j + 16 iterations in parallel.  The additional unrolling
        // for j + 1 and j + 17, reduces the execution time on Power 10 by
        // about 50% as the instruction scheduling allows on average 2X more
        // instructions to be issued per cycle.

        for (int j = 0; j < 16; j = j + 2) {
            // j < 16, unrolled to depth of 2
            if (idx.u8[j] & 0x80) {
                c.u8[j] = 0;
            } else {
                uint8_t i = idx.u8[j] & 15;
                c.u8[j] = u8[i];
            }

            if (idx.u8[j + 1] & 0x80) {
                c.u8[j + 1] = 0;
            } else {
                uint8_t i = idx.u8[j + 1] & 15;
                c.u8[j + 1] = u8[i];
            }

            // j >= 16, unrolled to depth of 2
            if (idx.u8[j + 16] & 0x80) {
                c.u8[j + 16] = 0;
            } else {
                uint8_t i = idx.u8[j + 16] & 15;
                c.u8[j + 16] = u8[i + 16];
            }

            if (idx.u8[j + 17] & 0x80) {
                c.u8[j + 17] = 0;
            } else {
                uint8_t i = idx.u8[j + 17] & 15;
                c.u8[j + 17] = u8[i + 16];
            }
        }
        return c;
    }

    // extract + 0-extend lane
    // this operation is slow (3 cycles)

    simd32uint8 operator+=(const simd32uint8& other) {
        *this = *this + other;
        return *this;
    }

    // for debugging only
    uint8_t operator[](int i) const {
        return u8[i];
    }
};

// convert with saturation
// careful: this does not cross lanes, so the order is weird
inline simd32uint8 uint16_to_uint8_saturate(
        const simd16uint16& a,
        const simd16uint16& b) {
    simd32uint8 c;

    auto saturate_16_to_8 = [](uint16_t x) { return x >= 256 ? 0xff : x; };

    for (int i = 0; i < 8; i++) {
        c.u8[i] = saturate_16_to_8(a.u16[i]);
        c.u8[8 + i] = saturate_16_to_8(b.u16[i]);
        c.u8[16 + i] = saturate_16_to_8(a.u16[8 + i]);
        c.u8[24 + i] = saturate_16_to_8(b.u16[8 + i]);
    }
    return c;
}

/// get most significant bit of each byte
inline uint32_t get_MSBs(const simd32uint8& a) {
    uint32_t res = 0;
    for (int i = 0; i < 32; i++) {
        if (a.u8[i] & 0x80) {
            res |= 1 << i;
        }
    }
    return res;
}

/// use MSB of each byte of mask to select a byte between a and b
inline simd32uint8 blendv(
        const simd32uint8& a,
        const simd32uint8& b,
        const simd32uint8& mask) {
    simd32uint8 c;
    for (int i = 0; i < 32; i++) {
        if (mask.u8[i] & 0x80) {
            c.u8[i] = b.u8[i];
        } else {
            c.u8[i] = a.u8[i];
        }
    }
    return c;
}

/// vector of 8 unsigned 32-bit integers
struct simd8uint32 : simd256bit {
    simd8uint32() {}

    explicit simd8uint32(uint32_t x) {
        set1(x);
    }

    explicit simd8uint32(const simd256bit& x) : simd256bit(x) {}

    explicit simd8uint32(const uint32_t* x) : simd256bit((const void*)x) {}

    explicit simd8uint32(
            uint32_t u0,
            uint32_t u1,
            uint32_t u2,
            uint32_t u3,
            uint32_t u4,
            uint32_t u5,
            uint32_t u6,
            uint32_t u7) {
        u32[0] = u0;
        u32[1] = u1;
        u32[2] = u2;
        u32[3] = u3;
        u32[4] = u4;
        u32[5] = u5;
        u32[6] = u6;
        u32[7] = u7;
    }

    simd8uint32 operator+(simd8uint32 other) const {
        simd8uint32 result;
        for (int i = 0; i < 8; i++) {
            result.u32[i] = u32[i] + other.u32[i];
        }
        return result;
    }

    simd8uint32 operator-(simd8uint32 other) const {
        simd8uint32 result;
        for (int i = 0; i < 8; i++) {
            result.u32[i] = u32[i] - other.u32[i];
        }
        return result;
    }

    simd8uint32& operator+=(const simd8uint32& other) {
        for (int i = 0; i < 8; i++) {
            u32[i] += other.u32[i];
        }
        return *this;
    }

    bool operator==(simd8uint32 other) const {
        for (size_t i = 0; i < 8; i++) {
            if (u32[i] != other.u32[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(simd8uint32 other) const {
        return !(*this == other);
    }

    std::string elements_to_string(const char* fmt) const {
        char res[1000], *ptr = res;
        for (int i = 0; i < 8; i++) {
            ptr += sprintf(ptr, fmt, u32[i]);
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
        for (int i = 0; i < 8; i++) {
            u32[i] = x;
        }
    }

    simd8uint32 unzip() const {
        const uint32_t ret[] = {
                u32[0], u32[2], u32[4], u32[6], u32[1], u32[3], u32[5], u32[7]};
        return simd8uint32{ret};
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
    for (size_t i = 0; i < 8; i++) {
        bool flag = (candidateValues.u32[i] < currentValues.u32[i]);
        minValues.u32[i] = flag ? candidateValues.u32[i] : currentValues.u32[i];
        minIndices.u32[i] =
                flag ? candidateIndices.u32[i] : currentIndices.u32[i];
        maxValues.u32[i] =
                !flag ? candidateValues.u32[i] : currentValues.u32[i];
        maxIndices.u32[i] =
                !flag ? candidateIndices.u32[i] : currentIndices.u32[i];
    }
}

struct simd8float32 : simd256bit {
    simd8float32() {}

    explicit simd8float32(const simd256bit& x) : simd256bit(x) {}

    explicit simd8float32(float x) {
        set1(x);
    }

    explicit simd8float32(const float* x) {
        loadu((void*)x);
    }

    void set1(float x) {
        for (int i = 0; i < 8; i++) {
            f32[i] = x;
        }
    }

    explicit simd8float32(
            float f0,
            float f1,
            float f2,
            float f3,
            float f4,
            float f5,
            float f6,
            float f7) {
        f32[0] = f0;
        f32[1] = f1;
        f32[2] = f2;
        f32[3] = f3;
        f32[4] = f4;
        f32[5] = f5;
        f32[6] = f6;
        f32[7] = f7;
    }

    template <typename F>
    static simd8float32 binary_func(
            const simd8float32& a,
            const simd8float32& b,
            F&& f) {
        simd8float32 c;
        for (int j = 0; j < 8; j++) {
            c.f32[j] = f(a.f32[j], b.f32[j]);
        }
        return c;
    }

    simd8float32 operator*(const simd8float32& other) const {
        return binary_func(
                *this, other, [](float a, float b) { return a * b; });
    }

    simd8float32 operator+(const simd8float32& other) const {
        return binary_func(
                *this, other, [](float a, float b) { return a + b; });
    }

    simd8float32 operator-(const simd8float32& other) const {
        return binary_func(
                *this, other, [](float a, float b) { return a - b; });
    }

    simd8float32& operator+=(const simd8float32& other) {
        for (size_t i = 0; i < 8; i++) {
            f32[i] += other.f32[i];
        }

        return *this;
    }

    bool operator==(simd8float32 other) const {
        for (size_t i = 0; i < 8; i++) {
            if (f32[i] != other.f32[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(simd8float32 other) const {
        return !(*this == other);
    }

    std::string tostring() const {
        char res[1000], *ptr = res;
        for (int i = 0; i < 8; i++) {
            ptr += sprintf(ptr, "%g,", f32[i]);
        }
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }
};

// hadd does not cross lanes
inline simd8float32 hadd(const simd8float32& a, const simd8float32& b) {
    simd8float32 c;
    c.f32[0] = a.f32[0] + a.f32[1];
    c.f32[1] = a.f32[2] + a.f32[3];
    c.f32[2] = b.f32[0] + b.f32[1];
    c.f32[3] = b.f32[2] + b.f32[3];

    c.f32[4] = a.f32[4] + a.f32[5];
    c.f32[5] = a.f32[6] + a.f32[7];
    c.f32[6] = b.f32[4] + b.f32[5];
    c.f32[7] = b.f32[6] + b.f32[7];

    return c;
}

inline simd8float32 unpacklo(const simd8float32& a, const simd8float32& b) {
    simd8float32 c;
    c.f32[0] = a.f32[0];
    c.f32[1] = b.f32[0];
    c.f32[2] = a.f32[1];
    c.f32[3] = b.f32[1];

    c.f32[4] = a.f32[4];
    c.f32[5] = b.f32[4];
    c.f32[6] = a.f32[5];
    c.f32[7] = b.f32[5];

    return c;
}

inline simd8float32 unpackhi(const simd8float32& a, const simd8float32& b) {
    simd8float32 c;
    c.f32[0] = a.f32[2];
    c.f32[1] = b.f32[2];
    c.f32[2] = a.f32[3];
    c.f32[3] = b.f32[3];

    c.f32[4] = a.f32[6];
    c.f32[5] = b.f32[6];
    c.f32[6] = a.f32[7];
    c.f32[7] = b.f32[7];

    return c;
}

// compute a * b + c
inline simd8float32 fmadd(
        const simd8float32& a,
        const simd8float32& b,
        const simd8float32& c) {
    simd8float32 res;
    for (int i = 0; i < 8; i++) {
        res.f32[i] = a.f32[i] * b.f32[i] + c.f32[i];
    }
    return res;
}

namespace {

// get even float32's of a and b, interleaved
simd8float32 geteven(const simd8float32& a, const simd8float32& b) {
    simd8float32 c;

    c.f32[0] = a.f32[0];
    c.f32[1] = a.f32[2];
    c.f32[2] = b.f32[0];
    c.f32[3] = b.f32[2];

    c.f32[4] = a.f32[4];
    c.f32[5] = a.f32[6];
    c.f32[6] = b.f32[4];
    c.f32[7] = b.f32[6];

    return c;
}

// get odd float32's of a and b, interleaved
simd8float32 getodd(const simd8float32& a, const simd8float32& b) {
    simd8float32 c;

    c.f32[0] = a.f32[1];
    c.f32[1] = a.f32[3];
    c.f32[2] = b.f32[1];
    c.f32[3] = b.f32[3];

    c.f32[4] = a.f32[5];
    c.f32[5] = a.f32[7];
    c.f32[6] = b.f32[5];
    c.f32[7] = b.f32[7];

    return c;
}

// 3 cycles
// if the lanes are a = [a0 a1] and b = [b0 b1], return [a0 b0]
simd8float32 getlow128(const simd8float32& a, const simd8float32& b) {
    simd8float32 c;

    c.f32[0] = a.f32[0];
    c.f32[1] = a.f32[1];
    c.f32[2] = a.f32[2];
    c.f32[3] = a.f32[3];

    c.f32[4] = b.f32[0];
    c.f32[5] = b.f32[1];
    c.f32[6] = b.f32[2];
    c.f32[7] = b.f32[3];

    return c;
}

simd8float32 gethigh128(const simd8float32& a, const simd8float32& b) {
    simd8float32 c;

    c.f32[0] = a.f32[4];
    c.f32[1] = a.f32[5];
    c.f32[2] = a.f32[6];
    c.f32[3] = a.f32[7];

    c.f32[4] = b.f32[4];
    c.f32[5] = b.f32[5];
    c.f32[6] = b.f32[6];
    c.f32[7] = b.f32[7];

    return c;
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
    for (size_t j = 0; j < 8; j++) {
        bool comparison = (candidateValues.f32[j] < lowestValues.f32[j]);
        if (comparison) {
            lowestValues.f32[j] = candidateValues.f32[j];
            lowestIndices.u32[j] = candidateIndices.u32[j];
        }
    }
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
    for (size_t i = 0; i < 8; i++) {
        bool flag = (candidateValues.f32[i] < currentValues.f32[i]);
        minValues.f32[i] = flag ? candidateValues.f32[i] : currentValues.f32[i];
        minIndices.u32[i] =
                flag ? candidateIndices.u32[i] : currentIndices.u32[i];
        maxValues.f32[i] =
                !flag ? candidateValues.f32[i] : currentValues.f32[i];
        maxIndices.u32[i] =
                !flag ? candidateIndices.u32[i] : currentIndices.u32[i];
    }
}

} // namespace

} // namespace faiss
