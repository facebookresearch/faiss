/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace faiss {

// non-intrinsic FP16 <-> FP32 code adapted from
// https://github.com/ispc/ispc/blob/master/stdlib.ispc

namespace {

inline float floatbits(uint32_t x) {
    void* xptr = &x;
    return *(float*)xptr;
}

inline uint32_t intbits(float f) {
    void* fptr = &f;
    return *(uint32_t*)fptr;
}

inline uint16_t encode_fp16(float f) {
    // Rounds to nearest even, as the F16C and Arm conversions do, so that
    // every SIMD level encodes the same bits. Via Fabian "ryg" Giesen
    // (float_to_half_fast3_rtne): https://gist.github.com/2156668
    const uint32_t f32infty = 255u << 23;
    const uint32_t f16max = (127u + 16u) << 23;
    const uint32_t denorm_magic = ((127u - 15u) + (23u - 10u) + 1u) << 23;
    const uint32_t sign_mask = 0x80000000u;

    uint32_t fint = intbits(f);
    const uint32_t sign = fint & sign_mask;
    fint ^= sign;

    uint32_t o;
    if (fint >= f16max) {
        // Inf->Inf. NaN->qNaN with the top bits of the payload, as F16C.
        o = (fint > f32infty) ? 0x7e00u | ((fint >> 13) & 0x3ffu) : 0x7c00u;
    } else if (fint < (113u << 23)) {
        // The result is subnormal or zero. Adding a magic value aligns the 10
        // mantissa bits at the bottom of the float, and the float addition
        // rounds to nearest even.
        o = intbits(floatbits(fint) + floatbits(denorm_magic)) - denorm_magic;
    } else {
        const uint32_t mant_odd = (fint >> 13) & 1;
        // rebias the exponent, and round to nearest even
        fint += ((15u - 127u) << 23) + 0xfffu + mant_odd;
        o = fint >> 13;
    }
    return static_cast<uint16_t>(o | (sign >> 16));
}

inline float decode_fp16(uint16_t h) {
    // https://gist.github.com/2144712
    // Fabian "ryg" Giesen.

    const uint32_t shifted_exp = 0x7c00u << 13; // exponent mask after shift

    int32_t o = ((int32_t)(h & 0x7fffu)) << 13; // exponent/mantissa bits
    int32_t exp = shifted_exp & o;              // just the exponent
    o += (int32_t)(127 - 15) << 23;             // exponent adjust

    int32_t infnan_val = o + ((int32_t)(128 - 16) << 23);
    int32_t zerodenorm_val =
            intbits(floatbits(o + (1u << 23)) - floatbits(113u << 23));
    int32_t reg_val = (exp == 0) ? zerodenorm_val : o;

    int32_t sign_bit = ((int32_t)(h & 0x8000u)) << 16;
    return floatbits(((exp == shifted_exp) ? infnan_val : reg_val) | sign_bit);
}

} // namespace

} // namespace faiss
