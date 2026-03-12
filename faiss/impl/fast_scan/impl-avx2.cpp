/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX2
#include <faiss/impl/fast_scan/dispatching.h> // IWYU pragma: keep

#endif // COMPILE_SIMD_AVX2
