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

void dot_product_batch_8_arm(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t dots[8]) {
    for (size_t k = 0; k < 8; ++k) {
        dots[k] = 0;
    }
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            acc0 = vdotq_s32(acc0, q, vld1q_s8(levels[0] + j));
            acc1 = vdotq_s32(acc1, q, vld1q_s8(levels[1] + j));
            acc2 = vdotq_s32(acc2, q, vld1q_s8(levels[2] + j));
            acc3 = vdotq_s32(acc3, q, vld1q_s8(levels[3] + j));
            acc4 = vdotq_s32(acc4, q, vld1q_s8(levels[4] + j));
            acc5 = vdotq_s32(acc5, q, vld1q_s8(levels[5] + j));
            acc6 = vdotq_s32(acc6, q, vld1q_s8(levels[6] + j));
            acc7 = vdotq_s32(acc7, q, vld1q_s8(levels[7] + j));
        }
        dots[0] += vaddvq_s32(acc0);
        dots[1] += vaddvq_s32(acc1);
        dots[2] += vaddvq_s32(acc2);
        dots[3] += vaddvq_s32(acc3);
        dots[4] += vaddvq_s32(acc4);
        dots[5] += vaddvq_s32(acc5);
        dots[6] += vaddvq_s32(acc6);
        dots[7] += vaddvq_s32(acc7);
    }
    const int8_t* tail_levels[8];
    for (size_t k = 0; k < 8; ++k) {
        tail_levels[k] = levels[k] + j;
    }
    int64_t tails[8];
    dot_product_batch_8_scalar(query + j, tail_levels, d - j, tails);
    for (size_t k = 0; k < 8; ++k) {
        dots[k] += tails[k];
    }
}

void dot_product_batch_16_arm(
        const int8_t* query,
        const int8_t* const levels[16],
        size_t d,
        int64_t dots[16]) {
    for (size_t k = 0; k < 16; ++k) {
        dots[k] = 0;
    }
    size_t j = 0;
    while (j + 16 <= d) {
        const size_t end = std::min(d - (d - j) % 16, j + kDotChunk);
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        int32x4_t acc8 = vdupq_n_s32(0);
        int32x4_t acc9 = vdupq_n_s32(0);
        int32x4_t acc10 = vdupq_n_s32(0);
        int32x4_t acc11 = vdupq_n_s32(0);
        int32x4_t acc12 = vdupq_n_s32(0);
        int32x4_t acc13 = vdupq_n_s32(0);
        int32x4_t acc14 = vdupq_n_s32(0);
        int32x4_t acc15 = vdupq_n_s32(0);
        for (; j + 16 <= end; j += 16) {
            const int8x16_t q = vld1q_s8(query + j);
            acc0 = vdotq_s32(acc0, q, vld1q_s8(levels[0] + j));
            acc1 = vdotq_s32(acc1, q, vld1q_s8(levels[1] + j));
            acc2 = vdotq_s32(acc2, q, vld1q_s8(levels[2] + j));
            acc3 = vdotq_s32(acc3, q, vld1q_s8(levels[3] + j));
            acc4 = vdotq_s32(acc4, q, vld1q_s8(levels[4] + j));
            acc5 = vdotq_s32(acc5, q, vld1q_s8(levels[5] + j));
            acc6 = vdotq_s32(acc6, q, vld1q_s8(levels[6] + j));
            acc7 = vdotq_s32(acc7, q, vld1q_s8(levels[7] + j));
            acc8 = vdotq_s32(acc8, q, vld1q_s8(levels[8] + j));
            acc9 = vdotq_s32(acc9, q, vld1q_s8(levels[9] + j));
            acc10 = vdotq_s32(acc10, q, vld1q_s8(levels[10] + j));
            acc11 = vdotq_s32(acc11, q, vld1q_s8(levels[11] + j));
            acc12 = vdotq_s32(acc12, q, vld1q_s8(levels[12] + j));
            acc13 = vdotq_s32(acc13, q, vld1q_s8(levels[13] + j));
            acc14 = vdotq_s32(acc14, q, vld1q_s8(levels[14] + j));
            acc15 = vdotq_s32(acc15, q, vld1q_s8(levels[15] + j));
        }
        dots[0] += vaddvq_s32(acc0);
        dots[1] += vaddvq_s32(acc1);
        dots[2] += vaddvq_s32(acc2);
        dots[3] += vaddvq_s32(acc3);
        dots[4] += vaddvq_s32(acc4);
        dots[5] += vaddvq_s32(acc5);
        dots[6] += vaddvq_s32(acc6);
        dots[7] += vaddvq_s32(acc7);
        dots[8] += vaddvq_s32(acc8);
        dots[9] += vaddvq_s32(acc9);
        dots[10] += vaddvq_s32(acc10);
        dots[11] += vaddvq_s32(acc11);
        dots[12] += vaddvq_s32(acc12);
        dots[13] += vaddvq_s32(acc13);
        dots[14] += vaddvq_s32(acc14);
        dots[15] += vaddvq_s32(acc15);
    }
    const int8_t* tail_levels[16];
    for (size_t k = 0; k < 16; ++k) {
        tail_levels[k] = levels[k] + j;
    }
    int64_t tails[16];
    dot_product_batch_16_scalar(query + j, tail_levels, d - j, tails);
    for (size_t k = 0; k < 16; ++k) {
        dots[k] += tails[k];
    }
}

} // namespace faiss::rabitq_integer_adc
