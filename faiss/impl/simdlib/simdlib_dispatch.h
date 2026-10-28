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
#include <faiss/utils/simd_levels.h>

// Platform specializations — guarded by FAISS_SIMD_TU, so that only per-SIMD
// TUs (compiled with -mavx2 etc.) see them. In static mode: only the
// compiled-in level is available.

#if FAISS_SIMD_TU(AVX512) || FAISS_SIMD_TU(AVX512_SPR)

// AVX512 includes AVX2 (simdlib_avx512.h includes simdlib_avx2.h)
#include <faiss/impl/simdlib/simdlib_avx512.h>

#elif FAISS_SIMD_TU(AVX2)

#include <faiss/impl/simdlib/simdlib_avx2.h>

#elif FAISS_SIMD_TU(ARM_NEON)

#include <faiss/impl/simdlib/simdlib_neon.h>

#endif

// No global bare-name aliases (simd16uint16, simd32uint8, etc.) — each file
// that needs them must declare its own `using` with an explicit SIMD level.
// This prevents per-ISA TUs from accidentally picking up SINGLE_SIMD_LEVEL
// (= NONE in DD mode) when they should use THE_SIMD_LEVEL.
