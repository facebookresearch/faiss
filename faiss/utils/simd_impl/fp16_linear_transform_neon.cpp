/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/fp16_linear_transform.h>

#include <arm_neon.h>
#include <vector>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace faiss::fp16_linear_transform {

template <>
bool supported<SIMDLevel::ARM_NEON>() {
#if defined(__linux__) && defined(HWCAP_FPHP) && defined(HWCAP_ASIMDFHM)
    const unsigned long capabilities = getauxval(AT_HWCAP);
    return (capabilities & HWCAP_FPHP) != 0 &&
            (capabilities & HWCAP_ASIMDFHM) != 0;
#else
    return false;
#endif
}

template <>
bool apply<SIMDLevel::ARM_NEON>(
        const uint16_t* matrix,
        size_t rows,
        size_t columns,
        const float* input,
        float* output) {
    std::vector<uint16_t> input_fp16(columns);
    const float32x4_t max_fp16 = vdupq_n_f32(65504.0f);
    size_t column = 0;
    for (; column + 8 <= columns; column += 8) {
        const float32x4_t input0 = vld1q_f32(input + column);
        const float32x4_t input1 = vld1q_f32(input + column + 4);
        if (vminvq_u32(vcleq_f32(vabsq_f32(input0), max_fp16)) == 0 ||
            vminvq_u32(vcleq_f32(vabsq_f32(input1), max_fp16)) == 0) {
            return false;
        }
        const float16x8_t converted =
                vcombine_f16(vcvt_f16_f32(input0), vcvt_f16_f32(input1));
        vst1q_u16(input_fp16.data() + column, vreinterpretq_u16_f16(converted));
    }
    for (; column < columns; ++column) {
        if (!(input[column] >= -65504.0f && input[column] <= 65504.0f)) {
            return false;
        }
        const float32x4_t value = vdupq_n_f32(input[column]);
        input_fp16[column] =
                vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(value)), 0);
    }

    for (size_t row = 0; row < rows; ++row) {
        const uint16_t* weights = matrix + row * columns;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        column = 0;
        for (; column + 8 <= columns; column += 8) {
            const float16x8_t w =
                    vreinterpretq_f16_u16(vld1q_u16(weights + column));
            const float16x8_t q = vreinterpretq_f16_u16(
                    vld1q_u16(input_fp16.data() + column));
            acc0 = vfmlalq_low_f16(acc0, w, q);
            acc1 = vfmlalq_high_f16(acc1, w, q);
        }
        float sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
        for (; column < columns; ++column) {
            const float16x4_t w =
                    vreinterpret_f16_u16(vdup_n_u16(weights[column]));
            const float16x4_t q =
                    vreinterpret_f16_u16(vdup_n_u16(input_fp16[column]));
            sum += vgetq_lane_f32(vcvt_f32_f16(w), 0) *
                    vgetq_lane_f32(vcvt_f32_f16(q), 0);
        }
        output[row] = sum;
    }
    return true;
}

} // namespace faiss::fp16_linear_transform
