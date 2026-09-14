/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

namespace faiss::fp16_linear_transform {

#if defined(COMPILE_SIMD_ARM_NEON) || defined(COMPILE_SIMD_AVX512_SPR)
FAISS_API bool supported();

/** Return true after applying the transform. Return false without writing the
 * output when an input value cannot be safely converted to finite FP16.
 */
FAISS_API bool apply(
        const uint16_t* matrix,
        size_t rows,
        size_t columns,
        const float* input,
        float* output);
#endif

} // namespace faiss::fp16_linear_transform
