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

#if (defined(COMPILE_SIMD_AVX512) || defined(COMPILE_SIMD_AVX512_SPR)) && \
        defined(__AVX512F__)

// AVX512 includes AVX2 (simdlib_avx512.h includes simdlib_avx2.h)
#include <faiss/impl/simdlib/simdlib_avx512.h>

#elif defined(COMPILE_SIMD_AVX2) && defined(__AVX2__)

#include <faiss/impl/simdlib/simdlib_avx2.h>

// MSVC ARM64 is intentionally excluded: MSVC NEON builtins cannot be used as
// non-type template parameters, and MSVC collapses distinct NEON struct types
// to the same underlying type causing overload ambiguity. See issue #4993.
#elif defined(COMPILE_SIMD_ARM_NEON) && \
        (defined(__aarch64__) || defined(_M_ARM64)) && !defined(_MSC_VER)

#include <faiss/impl/simdlib/simdlib_neon.h>

#endif

// No global bare-name aliases (simd16uint16, simd32uint8, etc.) — each file
// that needs them must declare its own `using` with an explicit SIMD level.
// This prevents per-ISA TUs from accidentally picking up SINGLE_SIMD_LEVEL
// (= NONE in DD mode) when they should use THE_SIMD_LEVEL.
