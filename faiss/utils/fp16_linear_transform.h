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
#include <faiss/utils/simd_levels.h>

namespace faiss::fp16_linear_transform {

constexpr int SIMD_LEVELS = (1 << int(SIMDLevel::NONE)) |
        (1 << int(SIMDLevel::ARM_NEON)) |
        (1 << int(SIMDLevel::AVX512_SPR));

template <SIMDLevel SL>
bool supported();

/** Return true after applying the transform. Return false without writing the
 * output when an input value cannot be safely converted to finite FP16.
 */
template <SIMDLevel SL>
bool apply(
        const uint16_t* matrix,
        size_t rows,
        size_t columns,
        const float* input,
        float* output);

// Declarations prevent callers from attempting to instantiate the undefined
// primary templates. Definitions live in ISA-specific translation units.
template <>
FAISS_API bool supported<SIMDLevel::ARM_NEON>();
template <>
FAISS_API bool apply<SIMDLevel::ARM_NEON>(
        const uint16_t* matrix,
        size_t rows,
        size_t columns,
        const float* input,
        float* output);

template <>
FAISS_API bool supported<SIMDLevel::AVX512_SPR>();
template <>
FAISS_API bool apply<SIMDLevel::AVX512_SPR>(
        const uint16_t* matrix,
        size_t rows,
        size_t columns,
        const float* input,
        float* output);

} // namespace faiss::fp16_linear_transform
