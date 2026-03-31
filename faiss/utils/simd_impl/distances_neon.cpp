/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NEON-optimized distance computations

#ifdef COMPILE_SIMD_ARM_NEON

#include <arm_neon.h>
#include <faiss/utils/distances.h>

namespace faiss {

static inline float fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 16;
    size_t i;
    float res;

    if (d >= multi_round) {
        __builtin_prefetch(x + multi_round, 0, 0);
        __builtin_prefetch(y + multi_round, 0, 0);
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t x8_1 = vld1q_f32(x + 4);
        float32x4_t x8_2 = vld1q_f32(x + 8);
        float32x4_t x8_3 = vld1q_f32(x + 12);

        float32x4_t y8_0 = vld1q_f32(y);
        float32x4_t y8_1 = vld1q_f32(y + 4);
        float32x4_t y8_2 = vld1q_f32(y + 8);
        float32x4_t y8_3 = vld1q_f32(y + 12);

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        float32x4_t d8_1 = vsubq_f32(x8_1, y8_1);
        d8_1 = vmulq_f32(d8_1, d8_1);
        float32x4_t d8_2 = vsubq_f32(x8_2, y8_2);
        d8_2 = vmulq_f32(d8_2, d8_2);
        float32x4_t d8_3 = vsubq_f32(x8_3, y8_3);
        d8_3 = vmulq_f32(d8_3, d8_3);

        for (i = multi_round; i <= d - multi_round; i += multi_round) {
            __builtin_prefetch(x + i + multi_round, 0, 0);
            __builtin_prefetch(y + i + multi_round, 0, 0);
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);

            x8_1 = vld1q_f32(x + i + 4);
            y8_1 = vld1q_f32(y + i + 4);
            const float32x4_t q8_1 = vsubq_f32(x8_1, y8_1);
            d8_1 = vmlaq_f32(d8_1, q8_1, q8_1);

            x8_2 = vld1q_f32(x + i + 8);
            y8_2 = vld1q_f32(y + i + 8);
            const float32x4_t q8_2 = vsubq_f32(x8_2, y8_2);
            d8_2 = vmlaq_f32(d8_2, q8_2, q8_2);

            x8_3 = vld1q_f32(x + i + 12);
            y8_3 = vld1q_f32(y + i + 12);
            const float32x4_t q8_3 = vsubq_f32(x8_3, y8_3);
            d8_3 = vmlaq_f32(d8_3, q8_3, q8_3);
        }

        for (; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }

        d8_0 = vaddq_f32(d8_0, d8_1);
        d8_2 = vaddq_f32(d8_2, d8_3);
        d8_0 = vaddq_f32(d8_0, d8_2);
        res = vaddvq_f32(d8_0);
    } else if (d >= single_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t y8_0 = vld1q_f32(y);

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        for (i = single_round; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }
        res = vaddvq_f32(d8_0);
    } else {
        res = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

static inline float fvec_inner_product_neon(const float* x, const float* y, size_t d) {
    size_t i;
    float res;
    constexpr size_t single_round = 16;

    if (d >= single_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t x8_1 = vld1q_f32(x + 4);
        float32x4_t x8_2 = vld1q_f32(x + 8);
        float32x4_t x8_3 = vld1q_f32(x + 12);

        float32x4_t y8_0 = vld1q_f32(y);
        float32x4_t y8_1 = vld1q_f32(y + 4);
        float32x4_t y8_2 = vld1q_f32(y + 8);
        float32x4_t y8_3 = vld1q_f32(y + 12);

        float32x4_t d8_0 = vmulq_f32(x8_0, y8_0);
        float32x4_t d8_1 = vmulq_f32(x8_1, y8_1);
        float32x4_t d8_2 = vmulq_f32(x8_2, y8_2);
        float32x4_t d8_3 = vmulq_f32(x8_3, y8_3);

        for (i = single_round; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            d8_0 = vmlaq_f32(d8_0, x8_0, y8_0);

            x8_1 = vld1q_f32(x + i + 4);
            y8_1 = vld1q_f32(y + i + 4);
            d8_1 = vmlaq_f32(d8_1, x8_1, y8_1);

            x8_2 = vld1q_f32(x + i + 8);
            y8_2 = vld1q_f32(y + i + 8);
            d8_2 = vmlaq_f32(d8_2, x8_2, y8_2);

            x8_3 = vld1q_f32(x + i + 12);
            y8_3 = vld1q_f32(y + i + 12);
            d8_3 = vmlaq_f32(d8_3, x8_3, y8_3);
        }

        d8_0 = vaddq_f32(d8_0, d8_1);
        d8_2 = vaddq_f32(d8_2, d8_3);
        d8_0 = vaddq_f32(d8_0, d8_2);
        res = vaddvq_f32(d8_0);
    } else {
        i = 0;
        res = 0;
    }

    for (; i < d; i++) {
        const float tmp = x[i] * y[i];
        res += tmp;
    }
    return res;
}

static inline void fvec_L2sqr_batch2_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 8;

    if (d >= single_round) {
        float32x4_t x_0 = vld1q_f32(x);
        float32x4_t x_1 = vld1q_f32(x + 4);

        float32x4_t y0_0 = vld1q_f32(y);
        float32x4_t y0_1 = vld1q_f32(y + 4);
        float32x4_t y1_0 = vld1q_f32(y + d);
        float32x4_t y1_1 = vld1q_f32(y + d + 4);

        float32x4_t d0_0 = vsubq_f32(x_0, y0_0);
        d0_0 = vmulq_f32(d0_0, d0_0);
        float32x4_t d0_1 = vsubq_f32(x_1, y0_1);
        d0_1 = vmulq_f32(d0_1, d0_1);
        float32x4_t d1_0 = vsubq_f32(x_0, y1_0);
        d1_0 = vmulq_f32(d1_0, d1_0);
        float32x4_t d1_1 = vsubq_f32(x_1, y1_1);
        d1_1 = vmulq_f32(d1_1, d1_1);

        for (i = single_round; i <= d - single_round; i += single_round) {
            x_0 = vld1q_f32(x + i);
            y0_0 = vld1q_f32(y + i);
            y1_0 = vld1q_f32(y + d + i);
            const float32x4_t q0_0 = vsubq_f32(x_0, y0_0);
            const float32x4_t q1_0 = vsubq_f32(x_0, y1_0);
            d0_0 = vmlaq_f32(d0_0, q0_0, q0_0);
            d1_0 = vmlaq_f32(d1_0, q1_0, q1_0);

            x_1 = vld1q_f32(x + i + 4);
            y0_1 = vld1q_f32(y + i + 4);
            y1_1 = vld1q_f32(y + d + i + 4);
            const float32x4_t q0_1 = vsubq_f32(x_1, y0_1);
            const float32x4_t q1_1 = vsubq_f32(x_1, y1_1);
            d0_1 = vmlaq_f32(d0_1, q0_1, q0_1);
            d1_1 = vmlaq_f32(d1_1, q1_1, q1_1);
        }

        d0_0 = vaddq_f32(d0_0, d0_1);
        d1_0 = vaddq_f32(d1_0, d1_1);
        dis[0] = vaddvq_f32(d0_0);
        dis[1] = vaddvq_f32(d1_0);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp0 = x[i] - *(y + i);
        const float tmp1 = x[i] - *(y + d + i);
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}

static inline void fvec_L2sqr_batch4_neon(const float* x, const float* y, size_t d, float* dis) {
    constexpr size_t single_round = 4;
    size_t i;
    if (d >= single_round) {
        float32x4_t b = vld1q_f32(x);

        float32x4_t q0 = vld1q_f32(y);
        float32x4_t q1 = vld1q_f32(y + d);
        float32x4_t q2 = vld1q_f32(y + 2 * d);
        float32x4_t q3 = vld1q_f32(y + 3 * d);

        q0 = vsubq_f32(q0, b);
        q1 = vsubq_f32(q1, b);
        q2 = vsubq_f32(q2, b);
        q3 = vsubq_f32(q3, b);

        float32x4_t res0 = vmulq_f32(q0, q0);
        float32x4_t res1 = vmulq_f32(q1, q1);
        float32x4_t res2 = vmulq_f32(q2, q2);
        float32x4_t res3 = vmulq_f32(q3, q3);

        for (i = single_round; i <= d - single_round; i += single_round) {
            b = vld1q_f32(x + i);

            q0 = vld1q_f32(y + i);
            q1 = vld1q_f32(y + d + i);
            q2 = vld1q_f32(y + 2 * d + i);
            q3 = vld1q_f32(y + 3 * d + i);

            q0 = vsubq_f32(q0, b);
            q1 = vsubq_f32(q1, b);
            q2 = vsubq_f32(q2, b);
            q3 = vsubq_f32(q3, b);

            res0 = vmlaq_f32(res0, q0, q0);
            res1 = vmlaq_f32(res1, q1, q1);
            res2 = vmlaq_f32(res2, q2, q2);
            res3 = vmlaq_f32(res3, q3, q3);
        }
        dis[0] = vaddvq_f32(res0);
        dis[1] = vaddvq_f32(res1);
        dis[2] = vaddvq_f32(res2);
        dis[3] = vaddvq_f32(res3);
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (d > i) {
        float q0 = x[i] - *(y + i);
        float q1 = x[i] - *(y + d + i);
        float q2 = x[i] - *(y + 2 * d + i);
        float q3 = x[i] - *(y + 3 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        for (i++; i < d; ++i) {
            float q0 = x[i] - *(y + i);
            float q1 = x[i] - *(y + d + i);
            float q2 = x[i] - *(y + 2 * d + i);
            float q3 = x[i] - *(y + 3 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}

static inline void fvec_inner_product_batch2_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 8;

    if (d >= single_round) {
        float32x4_t x_0 = vld1q_f32(x);
        float32x4_t x_1 = vld1q_f32(x + 4);

        float32x4_t y0_0 = vld1q_f32(y);
        float32x4_t y0_1 = vld1q_f32(y + 4);
        float32x4_t y1_0 = vld1q_f32(y + d);
        float32x4_t y1_1 = vld1q_f32(y + d + 4);

        float32x4_t d0_0 = vmulq_f32(x_0, y0_0);
        float32x4_t d0_1 = vmulq_f32(x_1, y0_1);
        float32x4_t d1_0 = vmulq_f32(x_0, y1_0);
        float32x4_t d1_1 = vmulq_f32(x_1, y1_1);

        for (i = single_round; i <= d - single_round; i += single_round) {
            x_0 = vld1q_f32(x + i);
            y0_0 = vld1q_f32(y + i);
            y1_0 = vld1q_f32(y + d + i);
            d0_0 = vmlaq_f32(d0_0, x_0, y0_0);
            d1_0 = vmlaq_f32(d1_0, x_0, y1_0);

            x_1 = vld1q_f32(x + i + 4);
            y0_1 = vld1q_f32(y + i + 4);
            y1_1 = vld1q_f32(y + d + i + 4);
            d0_1 = vmlaq_f32(d0_1, x_1, y0_1);
            d1_1 = vmlaq_f32(d1_1, x_1, y1_1);
        }

        d0_0 = vaddq_f32(d0_0, d0_1);
        d1_0 = vaddq_f32(d1_0, d1_1);
        dis[0] = vaddvq_f32(d0_0);
        dis[1] = vaddvq_f32(d1_0);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp0 = x[i] * *(y + i);
        const float tmp1 = x[i] * *(y + d + i);
        dis[0] += tmp0;
        dis[1] += tmp1;
    }
}

static inline void fvec_inner_product_batch4_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 4;

    if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}

static inline void fvec_L2sqr_batch8_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 4;
    if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);

        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_base8);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);

            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float q0 = x[i] - *(y + i);
        float q1 = x[i] - *(y + d + i);
        float q2 = x[i] - *(y + 2 * d + i);
        float q3 = x[i] - *(y + 3 * d + i);
        float q4 = x[i] - *(y + 4 * d + i);
        float q5 = x[i] - *(y + 5 * d + i);
        float q6 = x[i] - *(y + 6 * d + i);
        float q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
    }
}

static inline void fvec_inner_product_batch8_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 4;

    if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_query);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);
        }

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
            d4 += x[i] * *(y + 4 * d + i);
            d5 += x[i] * *(y + 5 * d + i);
            d6 += x[i] * *(y + 6 * d + i);
            d7 += x[i] * *(y + 7 * d + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
    }
}

static inline void fvec_L2sqr_batch24_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 4;
    if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 4 * d);
        neon_base2 = vld1q_f32(y + 5 * d);
        neon_base3 = vld1q_f32(y + 6 * d);
        neon_base4 = vld1q_f32(y + 7 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res6 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res7 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res8 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 8 * d);
        neon_base2 = vld1q_f32(y + 9 * d);
        neon_base3 = vld1q_f32(y + 10 * d);
        neon_base4 = vld1q_f32(y + 11 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 12 * d);
        neon_base2 = vld1q_f32(y + 13 * d);
        neon_base3 = vld1q_f32(y + 14 * d);
        neon_base4 = vld1q_f32(y + 15 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res13 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res14 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res15 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res16 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 16 * d);
        neon_base2 = vld1q_f32(y + 17 * d);
        neon_base3 = vld1q_f32(y + 18 * d);
        neon_base4 = vld1q_f32(y + 19 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res17 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res18 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res19 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res20 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 20 * d);
        neon_base2 = vld1q_f32(y + 21 * d);
        neon_base3 = vld1q_f32(y + 22 * d);
        neon_base4 = vld1q_f32(y + 23 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res21 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res22 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res23 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res24 = vmulq_f32(neon_base4, neon_base4);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 4 * d + i);
            neon_base2 = vld1q_f32(y + 5 * d + i);
            neon_base3 = vld1q_f32(y + 6 * d + i);
            neon_base4 = vld1q_f32(y + 7 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base1, neon_base1);
            neon_res6 = vmlaq_f32(neon_res6, neon_base2, neon_base2);
            neon_res7 = vmlaq_f32(neon_res7, neon_base3, neon_base3);
            neon_res8 = vmlaq_f32(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 8 * d + i);
            neon_base2 = vld1q_f32(y + 9 * d + i);
            neon_base3 = vld1q_f32(y + 10 * d + i);
            neon_base4 = vld1q_f32(y + 11 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 12 * d + i);
            neon_base2 = vld1q_f32(y + 13 * d + i);
            neon_base3 = vld1q_f32(y + 14 * d + i);
            neon_base4 = vld1q_f32(y + 15 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base1, neon_base1);
            neon_res14 = vmlaq_f32(neon_res14, neon_base2, neon_base2);
            neon_res15 = vmlaq_f32(neon_res15, neon_base3, neon_base3);
            neon_res16 = vmlaq_f32(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 16 * d + i);
            neon_base2 = vld1q_f32(y + 17 * d + i);
            neon_base3 = vld1q_f32(y + 18 * d + i);
            neon_base4 = vld1q_f32(y + 19 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res17 = vmlaq_f32(neon_res17, neon_base1, neon_base1);
            neon_res18 = vmlaq_f32(neon_res18, neon_base2, neon_base2);
            neon_res19 = vmlaq_f32(neon_res19, neon_base3, neon_base3);
            neon_res20 = vmlaq_f32(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 20 * d + i);
            neon_base2 = vld1q_f32(y + 21 * d + i);
            neon_base3 = vld1q_f32(y + 22 * d + i);
            neon_base4 = vld1q_f32(y + 23 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res21 = vmlaq_f32(neon_res21, neon_base1, neon_base1);
            neon_res22 = vmlaq_f32(neon_res22, neon_base2, neon_base2);
            neon_res23 = vmlaq_f32(neon_res23, neon_base3, neon_base3);
            neon_res24 = vmlaq_f32(neon_res24, neon_base4, neon_base4);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        dis[8] = vaddvq_f32(neon_res9);
        dis[9] = vaddvq_f32(neon_res10);
        dis[10] = vaddvq_f32(neon_res11);
        dis[11] = vaddvq_f32(neon_res12);
        dis[12] = vaddvq_f32(neon_res13);
        dis[13] = vaddvq_f32(neon_res14);
        dis[14] = vaddvq_f32(neon_res15);
        dis[15] = vaddvq_f32(neon_res16);
        dis[16] = vaddvq_f32(neon_res17);
        dis[17] = vaddvq_f32(neon_res18);
        dis[18] = vaddvq_f32(neon_res19);
        dis[19] = vaddvq_f32(neon_res20);
        dis[20] = vaddvq_f32(neon_res21);
        dis[21] = vaddvq_f32(neon_res22);
        dis[22] = vaddvq_f32(neon_res23);
        dis[23] = vaddvq_f32(neon_res24);
    } else {
        for (int j = 0; j < 24; j++) {
            dis[j] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float q0 = x[i] - *(y + i);
        float q1 = x[i] - *(y + d + i);
        float q2 = x[i] - *(y + 2 * d + i);
        float q3 = x[i] - *(y + 3 * d + i);
        float q4 = x[i] - *(y + 4 * d + i);
        float q5 = x[i] - *(y + 5 * d + i);
        float q6 = x[i] - *(y + 6 * d + i);
        float q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        float d8 = q0 * q0;
        float d9 = q1 * q1;
        float d10 = q2 * q2;
        float d11 = q3 * q3;
        float d12 = q4 * q4;
        float d13 = q5 * q5;
        float d14 = q6 * q6;
        float d15 = q7 * q7;
        q0 = x[i] - *(y + 16 * d + i);
        q1 = x[i] - *(y + 17 * d + i);
        q2 = x[i] - *(y + 18 * d + i);
        q3 = x[i] - *(y + 19 * d + i);
        q4 = x[i] - *(y + 20 * d + i);
        q5 = x[i] - *(y + 21 * d + i);
        q6 = x[i] - *(y + 22 * d + i);
        q7 = x[i] - *(y + 23 * d + i);
        float d16 = q0 * q0;
        float d17 = q1 * q1;
        float d18 = q2 * q2;
        float d19 = q3 * q3;
        float d20 = q4 * q4;
        float d21 = q5 * q5;
        float d22 = q6 * q6;
        float d23 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q0 = x[i] - *(y + 8 * d + i);
            q1 = x[i] - *(y + 9 * d + i);
            q2 = x[i] - *(y + 10 * d + i);
            q3 = x[i] - *(y + 11 * d + i);
            q4 = x[i] - *(y + 12 * d + i);
            q5 = x[i] - *(y + 13 * d + i);
            q6 = x[i] - *(y + 14 * d + i);
            q7 = x[i] - *(y + 15 * d + i);
            d8 += q0 * q0;
            d9 += q1 * q1;
            d10 += q2 * q2;
            d11 += q3 * q3;
            d12 += q4 * q4;
            d13 += q5 * q5;
            d14 += q6 * q6;
            d15 += q7 * q7;
            q0 = x[i] - *(y + 16 * d + i);
            q1 = x[i] - *(y + 17 * d + i);
            q2 = x[i] - *(y + 18 * d + i);
            q3 = x[i] - *(y + 19 * d + i);
            q4 = x[i] - *(y + 20 * d + i);
            q5 = x[i] - *(y + 21 * d + i);
            q6 = x[i] - *(y + 22 * d + i);
            q7 = x[i] - *(y + 23 * d + i);
            d16 += q0 * q0;
            d17 += q1 * q1;
            d18 += q2 * q2;
            d19 += q3 * q3;
            d20 += q4 * q4;
            d21 += q5 * q5;
            d22 += q6 * q6;
            d23 += q7 * q7;
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
        dis[16] += d16;
        dis[17] += d17;
        dis[18] += d18;
        dis[19] += d19;
        dis[20] += d20;
        dis[21] += d21;
        dis[22] += d22;
        dis[23] += d23;
    }
}

static inline void fvec_L2sqr_batch16_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 4;
    if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);

        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_base8);

        neon_base1 = vld1q_f32(y + 8 * d);
        neon_base2 = vld1q_f32(y + 9 * d);
        neon_base3 = vld1q_f32(y + 10 * d);
        neon_base4 = vld1q_f32(y + 11 * d);
        neon_base5 = vld1q_f32(y + 12 * d);
        neon_base6 = vld1q_f32(y + 13 * d);
        neon_base7 = vld1q_f32(y + 14 * d);
        neon_base8 = vld1q_f32(y + 15 * d);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res13 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res14 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res15 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res16 = vmulq_f32(neon_base8, neon_base8);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f32(y + 8 * d + i);
            neon_base2 = vld1q_f32(y + 9 * d + i);
            neon_base3 = vld1q_f32(y + 10 * d + i);
            neon_base4 = vld1q_f32(y + 11 * d + i);
            neon_base5 = vld1q_f32(y + 12 * d + i);
            neon_base6 = vld1q_f32(y + 13 * d + i);
            neon_base7 = vld1q_f32(y + 14 * d + i);
            neon_base8 = vld1q_f32(y + 15 * d + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_base5);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_base6);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_base7);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_base8);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        dis[8] = vaddvq_f32(neon_res9);
        dis[9] = vaddvq_f32(neon_res10);
        dis[10] = vaddvq_f32(neon_res11);
        dis[11] = vaddvq_f32(neon_res12);
        dis[12] = vaddvq_f32(neon_res13);
        dis[13] = vaddvq_f32(neon_res14);
        dis[14] = vaddvq_f32(neon_res15);
        dis[15] = vaddvq_f32(neon_res16);
    } else {
        for (int j = 0; j < 16; j++) {
            dis[j] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float q0 = x[i] - *(y + i);
        float q1 = x[i] - *(y + d + i);
        float q2 = x[i] - *(y + 2 * d + i);
        float q3 = x[i] - *(y + 3 * d + i);
        float q4 = x[i] - *(y + 4 * d + i);
        float q5 = x[i] - *(y + 5 * d + i);
        float q6 = x[i] - *(y + 6 * d + i);
        float q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        float q8 = x[i] - *(y + 8 * d + i);
        float q9 = x[i] - *(y + 9 * d + i);
        float q10 = x[i] - *(y + 10 * d + i);
        float q11 = x[i] - *(y + 11 * d + i);
        float q12 = x[i] - *(y + 12 * d + i);
        float q13 = x[i] - *(y + 13 * d + i);
        float q14 = x[i] - *(y + 14 * d + i);
        float q15 = x[i] - *(y + 15 * d + i);
        float d8 = q8 * q8;
        float d9 = q9 * q9;
        float d10 = q10 * q10;
        float d11 = q11 * q11;
        float d12 = q12 * q12;
        float d13 = q13 * q13;
        float d14 = q14 * q14;
        float d15 = q15 * q15;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q8 = x[i] - *(y + 8 * d + i);
            q9 = x[i] - *(y + 9 * d + i);
            q10 = x[i] - *(y + 10 * d + i);
            q11 = x[i] - *(y + 11 * d + i);
            q12 = x[i] - *(y + 12 * d + i);
            q13 = x[i] - *(y + 13 * d + i);
            q14 = x[i] - *(y + 14 * d + i);
            q15 = x[i] - *(y + 15 * d + i);
            d8 += q8 * q8;
            d9 += q9 * q9;
            d10 += q10 * q10;
            d11 += q11 * q11;
            d12 += q12 * q12;
            d13 += q13 * q13;
            d14 += q14 * q14;
            d15 += q15 * q15;
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
    }
}

// Template specializations for Faiss distance functions
template <>
void fvec_L2sqr_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    for (; i + 24 <= ny; i += 24) {
        fvec_L2sqr_batch24_neon(x, y + i * d, d, dis + i);
    }
    if (i + 16 <= ny) {
        fvec_L2sqr_batch16_neon(x, y + i * d, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        fvec_L2sqr_batch8_neon(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        fvec_L2sqr_batch4_neon(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        fvec_L2sqr_batch2_neon(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        dis[ny - 1] = fvec_L2sqr_neon(x, y + (ny - 1) * d, d);
    }
}

static inline void fvec_inner_product_batch16_neon(const float* x, const float* y, size_t d, float* dis) {
    size_t i;
    constexpr size_t single_round = 4;

    if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_query);

        neon_base1 = vld1q_f32(y + 8 * d);
        neon_base2 = vld1q_f32(y + 9 * d);
        neon_base3 = vld1q_f32(y + 10 * d);
        neon_base4 = vld1q_f32(y + 11 * d);
        neon_base5 = vld1q_f32(y + 12 * d);
        neon_base6 = vld1q_f32(y + 13 * d);
        neon_base7 = vld1q_f32(y + 14 * d);
        neon_base8 = vld1q_f32(y + 15 * d);

        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res13 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res14 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res15 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res16 = vmulq_f32(neon_base8, neon_query);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_f32(y + 8 * d + i);
            neon_base2 = vld1q_f32(y + 9 * d + i);
            neon_base3 = vld1q_f32(y + 10 * d + i);
            neon_base4 = vld1q_f32(y + 11 * d + i);
            neon_base5 = vld1q_f32(y + 12 * d + i);
            neon_base6 = vld1q_f32(y + 13 * d + i);
            neon_base7 = vld1q_f32(y + 14 * d + i);
            neon_base8 = vld1q_f32(y + 15 * d + i);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_query);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_query);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_query);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_query);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_query);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_query);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_query);
        }

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        dis[8] = vaddvq_f32(neon_res9);
        dis[9] = vaddvq_f32(neon_res10);
        dis[10] = vaddvq_f32(neon_res11);
        dis[11] = vaddvq_f32(neon_res12);
        dis[12] = vaddvq_f32(neon_res13);
        dis[13] = vaddvq_f32(neon_res14);
        dis[14] = vaddvq_f32(neon_res15);
        dis[15] = vaddvq_f32(neon_res16);
    } else {
        for (int j = 0; j < 16; j++) {
            dis[j] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);
        float d8 = x[i] * *(y + 8 * d + i);
        float d9 = x[i] * *(y + 9 * d + i);
        float d10 = x[i] * *(y + 10 * d + i);
        float d11 = x[i] * *(y + 11 * d + i);
        float d12 = x[i] * *(y + 12 * d + i);
        float d13 = x[i] * *(y + 13 * d + i);
        float d14 = x[i] * *(y + 14 * d + i);
        float d15 = x[i] * *(y + 15 * d + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
            d4 += x[i] * *(y + 4 * d + i);
            d5 += x[i] * *(y + 5 * d + i);
            d6 += x[i] * *(y + 6 * d + i);
            d7 += x[i] * *(y + 7 * d + i);
            d8 += x[i] * *(y + 8 * d + i);
            d9 += x[i] * *(y + 9 * d + i);
            d10 += x[i] * *(y + 10 * d + i);
            d11 += x[i] * *(y + 11 * d + i);
            d12 += x[i] * *(y + 12 * d + i);
            d13 += x[i] * *(y + 13 * d + i);
            d14 += x[i] * *(y + 14 * d + i);
            d15 += x[i] * *(y + 15 * d + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    for (; i + 16 <= ny; i += 16) {
        fvec_inner_product_batch16_neon(x, y + i * d, d, dis + i);
    }
    if (ny & 8) {
        fvec_inner_product_batch8_neon(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        fvec_inner_product_batch4_neon(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        fvec_inner_product_batch2_neon(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        dis[ny - 1] = fvec_inner_product_neon(x, y + (ny - 1) * d, d);
    }
}

// Continuous transpose kernels for PQ precomputed tables

static void inner_product_continuous_transpose_16_neon(
        float* dis,
        const float* x,
        const float* y,
        size_t d) {
    float32x4_t res[4];
    float32x4_t q = vdupq_n_f32(x[0]);
    res[0] = vmulq_f32(vld1q_f32(y), q);
    res[1] = vmulq_f32(vld1q_f32(y + 4), q);
    res[2] = vmulq_f32(vld1q_f32(y + 8), q);
    res[3] = vmulq_f32(vld1q_f32(y + 12), q);
    for (size_t i = 1; i < d; ++i) {
        q = vdupq_n_f32(x[i]);
        res[0] = vmlaq_f32(res[0], vld1q_f32(y + 16 * i), q);
        res[1] = vmlaq_f32(res[1], vld1q_f32(y + 16 * i + 4), q);
        res[2] = vmlaq_f32(res[2], vld1q_f32(y + 16 * i + 8), q);
        res[3] = vmlaq_f32(res[3], vld1q_f32(y + 16 * i + 12), q);
    }
    vst1q_f32(dis, res[0]);
    vst1q_f32(dis + 4, res[1]);
    vst1q_f32(dis + 8, res[2]);
    vst1q_f32(dis + 12, res[3]);
}

static void inner_product_continuous_transpose_32_neon(
        float* dis,
        const float* x,
        const float* y,
        size_t d) {
    float32x4_t res[8];
    float32x4_t q = vdupq_n_f32(x[0]);
    for (int j = 0; j < 8; j++) {
        res[j] = vmulq_f32(vld1q_f32(y + j * 4), q);
    }
    for (size_t i = 1; i < d; ++i) {
        q = vdupq_n_f32(x[i]);
        for (int j = 0; j < 8; j++) {
            res[j] = vmlaq_f32(res[j], vld1q_f32(y + 32 * i + j * 4), q);
        }
    }
    for (int j = 0; j < 8; j++) {
        vst1q_f32(dis + j * 4, res[j]);
    }
}

static void inner_product_continuous_transpose_64_neon(
        float* dis,
        const float* x,
        const float* y,
        size_t d) {
    float32x4_t res[16];
    float32x4_t q = vdupq_n_f32(x[0]);
    for (int j = 0; j < 16; j++) {
        res[j] = vmulq_f32(vld1q_f32(y + j * 4), q);
    }
    for (size_t i = 1; i < d; ++i) {
        q = vdupq_n_f32(x[i]);
        for (int j = 0; j < 16; j++) {
            res[j] = vmlaq_f32(res[j], vld1q_f32(y + 64 * i + j * 4), q);
        }
    }
    for (int j = 0; j < 16; j++) {
        vst1q_f32(dis + j * 4, res[j]);
    }
}

} // namespace faiss

#endif // COMPILE_SIMD_ARM_NEON
