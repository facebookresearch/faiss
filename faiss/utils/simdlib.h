/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Abstractions for 256-bit and 512-bit SIMD registers.
 *
 * The objective is to separate the different interpretations of the same
 * registers (as a vector of uint8, uint16 or uint32), to provide printing
 * functions.
 *
 * The types are templatized on SIMDLevel. Each platform header provides
 * explicit specializations for the appropriate level. Code without explicit
 * SL context uses SINGLE_SIMD_LEVEL (see simd_levels.h).
 */

#include <faiss/utils/simd_levels.h>

namespace faiss {

// 256-bit primary templates
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

// 512-bit primary templates
template <SIMDLevel SL>
struct simd512bit {};
template <SIMDLevel SL>
struct simd32uint16 : simd512bit<SL> {};
template <SIMDLevel SL>
struct simd64uint8 : simd512bit<SL> {};
template <SIMDLevel SL>
struct simd16float32 : simd512bit<SL> {};

} // namespace faiss

// Platform specializations — guarded by COMPILE_SIMD_* AND compiler macros.
// In DD mode: COMPILE_SIMD_* are target-wide, compiler macros are per-file.
// Only per-SIMD TUs (compiled with -mavx2 etc.) see the platform
// specializations. In static mode: only the compiled-in level is available.

#if defined(COMPILE_SIMD_AVX512) && defined(__AVX512F__)

// AVX512 includes AVX2 (simdlib_avx512.h includes simdlib_avx2.h)
#include <faiss/utils/simdlib_avx512.h>

#elif defined(COMPILE_SIMD_AVX2) && defined(__AVX2__)

#include <faiss/utils/simdlib_avx2.h>

#elif defined(COMPILE_SIMD_ARM_NEON) && defined(__aarch64__)

#include <faiss/utils/simdlib_neon.h>

#endif

// NONE specialization — always included.
// Provides simd16uint16<NONE> etc. (scalar fallback).
// On PPC64: uses PPC-optimized scalar code (hand-tuned loop unrolling).
// Elsewhere: generic scalar implementation.
#if defined(__PPC64__)
#include <faiss/utils/simdlib_ppc64.h>
#else
#include <faiss/utils/simdlib_emulated.h>
#endif
