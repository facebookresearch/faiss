/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512
#include <faiss/impl/pq_4bit/dispatching.h>        // IWYU pragma: keep
#include <faiss/impl/pq_4bit/rabitq_dispatching.h> // IWYU pragma: keep

#endif // COMPILE_SIMD_AVX512
