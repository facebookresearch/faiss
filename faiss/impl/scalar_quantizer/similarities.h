/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

namespace scalar_quantizer {

template <int SIMDWIDTH>
struct SimilarityL2 {};

template <>
struct SimilarityL2<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}

    /******* scalar accumulator *******/

    float accu;

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        float tmp = x1 - x2;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

#if defined(__AVX512F__)

template <>
struct SimilarityL2<16> {
    static constexpr int simdwidth = 16;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}
    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(simd16float32 x) {
        __m512 yiv = _mm512_loadu_ps(yi);
        yi += 16;
        __m512 tmp = _mm512_sub_ps(yiv, x.f);
        accu16 = simd16float32(_mm512_fmadd_ps(tmp, tmp, accu16.f));
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(
            simd16float32 x,
            simd16float32 y_2) {
        __m512 tmp = _mm512_sub_ps(y_2.f, x.f);
        accu16 = simd16float32(_mm512_fmadd_ps(tmp, tmp, accu16.f));
    }

    FAISS_ALWAYS_INLINE float result_16() {
        // performs better than dividing into _mm256 and adding
        return _mm512_reduce_add_ps(accu16.f);
    }
};

#elif defined(__AVX2__)

template <>
struct SimilarityL2<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}
    simd8float32 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        __m256 tmp = _mm256_sub_ps(yiv, x.f);
        accu8 = simd8float32(_mm256_fmadd_ps(tmp, tmp, accu8.f));
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x,
            simd8float32 y_2) {
        __m256 tmp = _mm256_sub_ps(y_2.f, x.f);
        accu8 = simd8float32(_mm256_fmadd_ps(tmp, tmp, accu8.f));
    }

    FAISS_ALWAYS_INLINE float result_8() {
        const __m128 sum = _mm_add_ps(
                _mm256_castps256_ps128(accu8.f),
                _mm256_extractf128_ps(accu8.f, 1));
        const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
        const __m128 v1 = _mm_add_ps(sum, v0);
        __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
        const __m128 v3 = _mm_add_ps(v1, v2);
        return _mm_cvtss_f32(v3);
    }
};

#endif

#ifdef USE_NEON
template <>
struct SimilarityL2<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;
    explicit SimilarityL2(const float* y) : y(y) {}
    simd8float32 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        float32x4x2_t yiv = vld1q_f32_x2(yi);
        yi += 8;

        float32x4_t sub0 = vsubq_f32(yiv.val[0], x.val[0]);
        float32x4_t sub1 = vsubq_f32(yiv.val[1], x.val[1]);

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], sub0, sub0);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], sub1, sub1);

        accu8 = simd8float32({accu8_0, accu8_1});
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x,
            simd8float32 y) {
        float32x4_t sub0 = vsubq_f32(y.val[0], x.val[0]);
        float32x4_t sub1 = vsubq_f32(y.val[1], x.val[1]);

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], sub0, sub0);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], sub1, sub1);

        accu8 = simd8float32({accu8_0, accu8_1});
    }

    FAISS_ALWAYS_INLINE float result_8() {
        float32x4_t sum_0 = vpaddq_f32(accu8.data.val[0], accu8.data.val[0]);
        float32x4_t sum_1 = vpaddq_f32(accu8.data.val[1], accu8.data.val[1]);

        float32x4_t sum2_0 = vpaddq_f32(sum_0, sum_0);
        float32x4_t sum2_1 = vpaddq_f32(sum_1, sum_1);
        return vgetq_lane_f32(sum2_0, 0) + vgetq_lane_f32(sum2_1, 0);
    }
};
#endif

template <int SIMDWIDTH>
struct SimilarityIP {};

template <>
struct SimilarityIP<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        accu += *yi++ * x;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        accu += x1 * x2;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

#if defined(__AVX512F__)

template <>
struct SimilarityIP<16> {
    static constexpr int simdwidth = 16;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(__m512 x) {
        __m512 yiv = _mm512_loadu_ps(yi);
        yi += 16;
        accu16.f = _mm512_fmadd_ps(yiv, x, accu16.f);
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(__m512 x1, __m512 x2) {
        accu16.f = _mm512_fmadd_ps(x1, x2, accu16.f);
    }

    FAISS_ALWAYS_INLINE float result_16() {
        // performs better than dividing into _mm256 and adding
        return _mm512_reduce_add_ps(accu16.f);
    }
};

#elif defined(__AVX2__)

template <>
struct SimilarityIP<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    simd8float32 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(simd8float32 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        accu8.f = _mm256_fmadd_ps(yiv, x.f, accu8.f);
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            simd8float32 x1,
            simd8float32 x2) {
        accu8.f = _mm256_fmadd_ps(x1.f, x2.f, accu8.f);
    }

    FAISS_ALWAYS_INLINE float result_8() {
        const __m128 sum = _mm_add_ps(
                _mm256_castps256_ps128(accu8.f),
                _mm256_extractf128_ps(accu8.f, 1));
        const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
        const __m128 v1 = _mm_add_ps(sum, v0);
        __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
        const __m128 v3 = _mm_add_ps(v1, v2);
        return _mm_cvtss_f32(v3);
    }
};
#endif

#ifdef USE_NEON

template <>
struct SimilarityIP<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    explicit SimilarityIP(const float* y) : y(y) {}
    float32x4x2_t accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(float32x4x2_t x) {
        float32x4x2_t yiv = vld1q_f32_x2(yi);
        yi += 8;

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], yiv.val[0], x.val[0]);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], yiv.val[1], x.val[1]);
        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            float32x4x2_t x1,
            float32x4x2_t x2) {
        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], x1.val[0], x2.val[0]);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], x1.val[1], x2.val[1]);
        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE float result_8() {
        float32x4x2_t sum = {
                vpaddq_f32(accu8.val[0], accu8.val[0]),
                vpaddq_f32(accu8.val[1], accu8.val[1])};

        float32x4x2_t sum2 = {
                vpaddq_f32(sum.val[0], sum.val[0]),
                vpaddq_f32(sum.val[1], sum.val[1])};
        return vgetq_lane_f32(sum2.val[0], 0) + vgetq_lane_f32(sum2.val[1], 0);
    }
};
#endif

} // namespace scalar_quantizer
} // namespace faiss
