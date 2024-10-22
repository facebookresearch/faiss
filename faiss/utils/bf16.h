/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace faiss {

namespace {

union fp32_bits {
    uint32_t as_u32;
    float as_f32;
};

} // namespace

inline uint16_t encode_bf16(const float f) {
    // Round off
    fp32_bits fp;
    fp.as_f32 = f;
    return static_cast<uint16_t>((fp.as_u32 + 0x8000) >> 16);
}

inline float decode_bf16(const uint16_t v) {
    fp32_bits fp;
    fp.as_u32 = (uint32_t(v) << 16);
    return fp.as_f32;
}

} // namespace faiss
