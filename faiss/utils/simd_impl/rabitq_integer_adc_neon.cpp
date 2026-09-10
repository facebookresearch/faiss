/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>

#include <arm_neon.h>
#include <algorithm>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace faiss::rabitq_integer_adc {

bool arm_dotprod_supported() {
#if defined(__linux__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

namespace {

// Reduce at bounded intervals so the int32 lanes cannot overflow even for
// unusually high dimensions. One lane receives at most 1024 products per
// 4096-dimensional chunk, well below INT32_MAX for signed bytes.
constexpr size_t kDotChunk = 4096;

} // namespace

int64_t dot_product_arm(const int8_t* query, const int8_t* levels, size_t d) {
    int64_t result = 0;
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        for (; j + 32 <= end; j += 32) {
            acc0 = vdotq_s32(acc0, vld1q_s8(query + j), vld1q_s8(levels + j));
            acc1 = vdotq_s32(
                    acc1, vld1q_s8(query + j + 16), vld1q_s8(levels + j + 16));
        }
        if (j + 16 <= end) {
            acc0 = vdotq_s32(acc0, vld1q_s8(query + j), vld1q_s8(levels + j));
            j += 16;
        }
        result += static_cast<int64_t>(vaddvq_s32(acc0)) +
                static_cast<int64_t>(vaddvq_s32(acc1));
    }
    return result + dot_product_scalar(query + j, levels + j, d - j);
}

void dot_product_batch_4_arm(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3) {
    dot0 = dot1 = dot2 = dot3 = 0;
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            acc0 = vdotq_s32(acc0, q, vld1q_s8(levels0 + j));
            acc1 = vdotq_s32(acc1, q, vld1q_s8(levels1 + j));
            acc2 = vdotq_s32(acc2, q, vld1q_s8(levels2 + j));
            acc3 = vdotq_s32(acc3, q, vld1q_s8(levels3 + j));
        }
        dot0 += vaddvq_s32(acc0);
        dot1 += vaddvq_s32(acc1);
        dot2 += vaddvq_s32(acc2);
        dot3 += vaddvq_s32(acc3);
    }
    int64_t tail0, tail1, tail2, tail3;
    dot_product_batch_4_scalar(
            query + j,
            levels0 + j,
            levels1 + j,
            levels2 + j,
            levels3 + j,
            d - j,
            tail0,
            tail1,
            tail2,
            tail3);
    dot0 += tail0;
    dot1 += tail1;
    dot2 += tail2;
    dot3 += tail3;
}

} // namespace faiss::rabitq_integer_adc
