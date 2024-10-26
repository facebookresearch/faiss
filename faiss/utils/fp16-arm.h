/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arm_neon.h>
#include <cstdint>

namespace faiss {

inline uint16_t encode_fp16(float x) {
    float32x4_t fx4 = vdupq_n_f32(x);
    float16x4_t f16x4 = vcvt_f16_f32(fx4);
    uint16x4_t ui16x4 = vreinterpret_u16_f16(f16x4);
    return vduph_lane_u16(ui16x4, 3);
}

inline float decode_fp16(uint16_t x) {
    uint16x4_t ui16x4 = vdup_n_u16(x);
    float16x4_t f16x4 = vreinterpret_f16_u16(ui16x4);
    float32x4_t fx4 = vcvt_f32_f16(f16x4);
    return vdups_laneq_f32(fx4, 3);
}

} // namespace faiss
