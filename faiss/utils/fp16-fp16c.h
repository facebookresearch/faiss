/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <immintrin.h>
#include <cstdint>

namespace faiss {

inline uint16_t encode_fp16(float x) {
    __m128 xf = _mm_set1_ps(x);
    __m128i xi =
            _mm_cvtps_ph(xf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_cvtsi128_si32(xi) & 0xffff;
}

inline float decode_fp16(uint16_t x) {
    __m128i xi = _mm_set1_epi16(x);
    __m128 xf = _mm_cvtph_ps(xi);
    return _mm_cvtss_f32(xf);
}

} // namespace faiss
