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
struct simd256bit_tpl {};
template <SIMDLevel SL>
struct simd16uint16_tpl : simd256bit_tpl<SL> {};
template <SIMDLevel SL>
struct simd32uint8_tpl : simd256bit_tpl<SL> {};
template <SIMDLevel SL>
struct simd8uint32_tpl : simd256bit_tpl<SL> {};
template <SIMDLevel SL>
struct simd8float32_tpl : simd256bit_tpl<SL> {};

// 512-bit primary templates
template <SIMDLevel SL>
struct simd512bit_tpl {};
template <SIMDLevel SL>
struct simd32uint16_tpl : simd512bit_tpl<SL> {};
template <SIMDLevel SL>
struct simd64uint8_tpl : simd512bit_tpl<SL> {};
template <SIMDLevel SL>
struct simd16float32_tpl : simd512bit_tpl<SL> {};

} // namespace faiss

// NONE specialization — always included.
// Provides simd16uint16_tpl<NONE> etc. (scalar fallback).
// On PPC64: uses PPC-optimized scalar code (hand-tuned loop unrolling).
// Elsewhere: generic scalar implementation.
#if defined(__PPC64__)
#include <faiss/impl/simdlib/simdlib_ppc64.h>
#else
#include <faiss/impl/simdlib/simdlib_emulated.h>
#endif
