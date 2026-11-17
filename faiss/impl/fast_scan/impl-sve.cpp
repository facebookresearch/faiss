/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_SVE

// This TU is compiled with -msve-vector-bits=256 (see FAISS_SIMD_SVE256_SRC),
// so simdlib_dispatch.h selects simdlib_sve256.h and the 256-bit kernels
// instantiated from dispatching.h use the native SVE types instead of the
// NEON fallback that impl-neon.cpp forwards to. It is reached at runtime
// through AVAILABLE_SIMD_LEVELS_BASE_WITH_SVE in fast_scan.cpp.
#define THE_LEVEL_TO_DISPATCH SIMDLevel::ARM_SVE
#include <faiss/impl/fast_scan/dispatching.h> // IWYU pragma: keep

#endif // COMPILE_SIMD_ARM_SVE
