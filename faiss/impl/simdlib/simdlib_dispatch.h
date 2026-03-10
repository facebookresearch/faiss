/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Includes simdlib.h (primary templates + NONE/emulated specialization)
 * plus the platform specialization for the current compilation context.
 *
 * Generic code should include this header.
 * Per-SIMD TUs should include the concrete header directly
 * (e.g., simdlib_avx2.h).
 */

#include <faiss/impl/simdlib/simdlib.h>

// Platform specializations — guarded by COMPILE_SIMD_* AND compiler macros.
// In DD mode: COMPILE_SIMD_* are target-wide, compiler macros are per-file.
// Only per-SIMD TUs (compiled with -mavx2 etc.) see the platform
// specializations. In static mode: only the compiled-in level is available.

#if defined(COMPILE_SIMD_AVX512) && defined(__AVX512F__)

// AVX512 includes AVX2 (simdlib_avx512.h includes simdlib_avx2.h)
#include <faiss/impl/simdlib/simdlib_avx512.h>

#elif defined(COMPILE_SIMD_AVX2) && defined(__AVX2__)

#include <faiss/impl/simdlib/simdlib_avx2.h>

#elif defined(COMPILE_SIMD_ARM_NEON) && defined(__aarch64__)

#include <faiss/impl/simdlib/simdlib_neon.h>

#endif

// Convenience aliases: bare names resolve to the current TU's SIMD level.
// Generic code uses SINGLE_SIMD_LEVEL (= NONE in DD, compiled-in in static).
// Per-SIMD TUs should define their own aliases with the concrete level.

namespace faiss {

// 256-bit
using simd256bit = simd256bit_tpl<SINGLE_SIMD_LEVEL_256>;
using simd16uint16 = simd16uint16_tpl<SINGLE_SIMD_LEVEL_256>;
using simd32uint8 = simd32uint8_tpl<SINGLE_SIMD_LEVEL_256>;
using simd8uint32 = simd8uint32_tpl<SINGLE_SIMD_LEVEL_256>;
using simd8float32 = simd8float32_tpl<SINGLE_SIMD_LEVEL_256>;

// 512-bit
using simd512bit = simd512bit_tpl<SINGLE_SIMD_LEVEL>;
using simd32uint16 = simd32uint16_tpl<SINGLE_SIMD_LEVEL>;
using simd64uint8 = simd64uint8_tpl<SINGLE_SIMD_LEVEL>;
using simd16float32 = simd16float32_tpl<SINGLE_SIMD_LEVEL>;

} // namespace faiss
