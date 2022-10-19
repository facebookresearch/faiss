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

} // namespace

inline uint16_t encode_fp16(float f) {
    // via Fabian "ryg" Giesen.
    // https://gist.github.com/2156668
    uint32_t sign_mask = 0x80000000u;
    int32_t o;

    uint32_t fint = intbits(f);
    uint32_t sign = fint & sign_mask;
    fint ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code (since
    // there's no unsigned PCMPGTD).

    // Inf or NaN (all exponent bits set)
    // NaN->qNaN and Inf->Inf
    // unconditional assignment here, will override with right value for
    // the regular case below.
    uint32_t f32infty = 255u << 23;
    o = (fint > f32infty) ? 0x7e00u : 0x7c00u;

    // (De)normalized number or zero
    // update fint unconditionally to save the blending; we don't need it
    // anymore for the Inf/NaN case anyway.

    const uint32_t round_mask = ~0xfffu;
    const uint32_t magic = 15u << 23;

    // Shift exponent down, denormalize if necessary.
    // NOTE This represents half-float denormals using single
    // precision denormals.  The main reason to do this is that
    // there's no shift with per-lane variable shifts in SSE*, which
    // we'd otherwise need. It has some funky side effects though:
    // - This conversion will actually respect the FTZ (Flush To Zero)
    //   flag in MXCSR - if it's set, no half-float denormals will be
    //   generated. I'm honestly not sure whether this is good or
    //   bad. It's definitely interesting.
    // - If the underlying HW doesn't support denormals (not an issue
    //   with Intel CPUs, but might be a problem on GPUs or PS3 SPUs),
    //   you will always get flush-to-zero behavior. This is bad,
    //   unless you're on a CPU where you don't care.
    // - Denormals tend to be slow. FP32 denormals are rare in
    //   practice outside of things like recursive filters in DSP -
    //   not a typical half-float application. Whether FP16 denormals
    //   are rare in practice, I don't know. Whatever slow path your
    //   HW may or may not have for denormals, this may well hit it.
    float fscale = floatbits(fint & round_mask) * floatbits(magic);
    fscale = std::min(fscale, floatbits((31u << 23) - 0x1000u));
    int32_t fint2 = intbits(fscale) - round_mask;

    if (fint < f32infty)
        o = fint2 >> 13; // Take the bits!

    return (o | (sign >> 16));
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

} // namespace faiss
