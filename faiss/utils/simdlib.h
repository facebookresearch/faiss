/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/utils/simd_levels.h>

namespace faiss {

// 256-bit representations

template <SIMDLevel SL>
struct simd256bit {};

template <SIMDLevel SL>
struct simd16uint16 : simd256bit<SL> {};
template <SIMDLevel SL>
struct simd32uint8 : simd256bit<SL> {};
template <SIMDLevel SL>
struct simd8uint32 : simd256bit<SL> {};
template <SIMDLevel SL>
struct simd8float32 : simd256bit<SL> {};

// 512-bit representations (currently only supported on AVX512)
template <SIMDLevel SL>
struct simd512bit {};

template <SIMDLevel SL>
struct simd32uint16 : simd512bit<SL> {};
template <SIMDLevel SL>
struct simd64uint8 : simd512bit<SL> {};
template <SIMDLevel SL>
struct simd16float32 : simd512bit<SL> {};

} // namespace faiss

/** Abstractions for 256-bit registers
 *
 * The objective is to separate the different interpretations of the same
 * registers (as a vector of uint8, uint16 or uint32), to provide printing
 * functions.
 */

#if defined(__AVX512F__)

// simdlib_avx2.h is included in simdlib_avx512.h
#include <faiss/utils/simd_impl/simdlib_avx512.h>

#elif defined(__AVX2__)

#include <faiss/utils/simd_impl/simdlib_avx2.h>

#elif defined(__aarch64__)

#include <faiss/utils/simd_impl/simdlib_neon.h>

#elif defined(__PPC64__)

#include <faiss/utils/simd_impl/simdlib_ppc64.h>

#else

// FIXME: make a SSE version
// is this ever going to happen? We will probably rather implement AVX512

#endif

// emulated = all operations are implemented as scalars
#include <faiss/utils/simd_impl/simdlib_emulated.h>
