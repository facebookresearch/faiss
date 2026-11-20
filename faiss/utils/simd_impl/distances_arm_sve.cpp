/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_sve.h>

#include <faiss/utils/distances.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances_fused/distances_fused.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/simd_impl/exhaustive_L2sqr_blas_cmax.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

template <>
float fvec_norm_L2sqr<SIMDLevel::ARM_SVE>(const float* x, size_t d) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const svbool_t pg = svptrue_b32();

    svfloat32_t sum0 = svdup_n_f32(0.0f);
    svfloat32_t sum1 = svdup_n_f32(0.0f);
    svfloat32_t sum2 = svdup_n_f32(0.0f);
    svfloat32_t sum3 = svdup_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * lanes <= d; i += 4 * lanes) {
        const svfloat32_t x0 = svld1_f32(pg, x + i);
        const svfloat32_t x1 = svld1_f32(pg, x + i + lanes);
        const svfloat32_t x2 = svld1_f32(pg, x + i + 2 * lanes);
        const svfloat32_t x3 = svld1_f32(pg, x + i + 3 * lanes);
        sum0 = svmla_f32_x(pg, sum0, x0, x0);
        sum1 = svmla_f32_x(pg, sum1, x1, x1);
        sum2 = svmla_f32_x(pg, sum2, x2, x2);
        sum3 = svmla_f32_x(pg, sum3, x3, x3);
    }

    for (; i + lanes <= d; i += lanes) {
        const svfloat32_t x0 = svld1_f32(pg, x + i);
        sum0 = svmla_f32_x(pg, sum0, x0, x0);
    }

    if (i < d) {
        const svbool_t pg_tail = svwhilelt_b32_u64(i, d);
        const svfloat32_t x0 = svld1_f32(pg_tail, x + i);
        sum0 = svmla_f32_m(pg_tail, sum0, x0, x0);
    }

    sum0 = svadd_f32_x(pg, sum0, sum1);
    sum2 = svadd_f32_x(pg, sum2, sum3);
    sum0 = svadd_f32_x(pg, sum0, sum2);
    return svaddv_f32(pg, sum0);
}

template <>
float fvec_L2sqr<SIMDLevel::ARM_SVE>(const float* x, const float* y, size_t d) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const svbool_t pg = svptrue_b32();

    svfloat32_t sum0 = svdup_n_f32(0.0f);
    svfloat32_t sum1 = svdup_n_f32(0.0f);
    svfloat32_t sum2 = svdup_n_f32(0.0f);
    svfloat32_t sum3 = svdup_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * lanes <= d; i += 4 * lanes) {
        const svfloat32_t d0 =
                svsub_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
        const svfloat32_t d1 = svsub_f32_x(
                pg, svld1_f32(pg, x + i + lanes), svld1_f32(pg, y + i + lanes));
        const svfloat32_t d2 = svsub_f32_x(
                pg,
                svld1_f32(pg, x + i + 2 * lanes),
                svld1_f32(pg, y + i + 2 * lanes));
        const svfloat32_t d3 = svsub_f32_x(
                pg,
                svld1_f32(pg, x + i + 3 * lanes),
                svld1_f32(pg, y + i + 3 * lanes));
        sum0 = svmla_f32_x(pg, sum0, d0, d0);
        sum1 = svmla_f32_x(pg, sum1, d1, d1);
        sum2 = svmla_f32_x(pg, sum2, d2, d2);
        sum3 = svmla_f32_x(pg, sum3, d3, d3);
    }

    for (; i + lanes <= d; i += lanes) {
        const svfloat32_t d0 =
                svsub_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
        sum0 = svmla_f32_x(pg, sum0, d0, d0);
    }

    if (i < d) {
        const svbool_t pg_tail = svwhilelt_b32_u64(i, d);
        const svfloat32_t d0 = svsub_f32_z(
                pg_tail, svld1_f32(pg_tail, x + i), svld1_f32(pg_tail, y + i));
        sum0 = svmla_f32_m(pg_tail, sum0, d0, d0);
    }

    sum0 = svadd_f32_x(pg, sum0, sum1);
    sum2 = svadd_f32_x(pg, sum2, sum3);
    sum0 = svadd_f32_x(pg, sum0, sum2);
    return svaddv_f32(pg, sum0);
}

template <>
float fvec_inner_product<SIMDLevel::ARM_SVE>(
        const float* x,
        const float* y,
        size_t d) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const svbool_t pg = svptrue_b32();

    svfloat32_t sum0 = svdup_n_f32(0.0f);
    svfloat32_t sum1 = svdup_n_f32(0.0f);
    svfloat32_t sum2 = svdup_n_f32(0.0f);
    svfloat32_t sum3 = svdup_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * lanes <= d; i += 4 * lanes) {
        sum0 = svmla_f32_x(
                pg, sum0, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
        sum1 = svmla_f32_x(
                pg,
                sum1,
                svld1_f32(pg, x + i + lanes),
                svld1_f32(pg, y + i + lanes));
        sum2 = svmla_f32_x(
                pg,
                sum2,
                svld1_f32(pg, x + i + 2 * lanes),
                svld1_f32(pg, y + i + 2 * lanes));
        sum3 = svmla_f32_x(
                pg,
                sum3,
                svld1_f32(pg, x + i + 3 * lanes),
                svld1_f32(pg, y + i + 3 * lanes));
    }

    for (; i + lanes <= d; i += lanes) {
        sum0 = svmla_f32_x(
                pg, sum0, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
    }

    if (i < d) {
        const svbool_t pg_tail = svwhilelt_b32_u64(i, d);
        sum0 = svmla_f32_m(
                pg_tail,
                sum0,
                svld1_f32(pg_tail, x + i),
                svld1_f32(pg_tail, y + i));
    }

    sum0 = svadd_f32_x(pg, sum0, sum1);
    sum2 = svadd_f32_x(pg, sum2, sum3);
    sum0 = svadd_f32_x(pg, sum0, sum2);
    return svaddv_f32(pg, sum0);
}

template <>
float fvec_L1<SIMDLevel::ARM_SVE>(const float* x, const float* y, size_t d) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const svbool_t pg = svptrue_b32();

    svfloat32_t sum0 = svdup_n_f32(0.0f);
    svfloat32_t sum1 = svdup_n_f32(0.0f);
    svfloat32_t sum2 = svdup_n_f32(0.0f);
    svfloat32_t sum3 = svdup_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * lanes <= d; i += 4 * lanes) {
        sum0 = svadd_f32_x(
                pg,
                sum0,
                svabd_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i)));
        sum1 = svadd_f32_x(
                pg,
                sum1,
                svabd_f32_x(
                        pg,
                        svld1_f32(pg, x + i + lanes),
                        svld1_f32(pg, y + i + lanes)));
        sum2 = svadd_f32_x(
                pg,
                sum2,
                svabd_f32_x(
                        pg,
                        svld1_f32(pg, x + i + 2 * lanes),
                        svld1_f32(pg, y + i + 2 * lanes)));
        sum3 = svadd_f32_x(
                pg,
                sum3,
                svabd_f32_x(
                        pg,
                        svld1_f32(pg, x + i + 3 * lanes),
                        svld1_f32(pg, y + i + 3 * lanes)));
    }

    for (; i + lanes <= d; i += lanes) {
        sum0 = svadd_f32_x(
                pg,
                sum0,
                svabd_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i)));
    }

    if (i < d) {
        const svbool_t pg_tail = svwhilelt_b32_u64(i, d);
        sum0 = svadd_f32_m(
                pg_tail,
                sum0,
                svabd_f32_z(
                        pg_tail,
                        svld1_f32(pg_tail, x + i),
                        svld1_f32(pg_tail, y + i)));
    }

    sum0 = svadd_f32_x(pg, sum0, sum1);
    sum2 = svadd_f32_x(pg, sum2, sum3);
    sum0 = svadd_f32_x(pg, sum0, sum2);
    return svaddv_f32(pg, sum0);
}

template <>
float fvec_Linf<SIMDLevel::ARM_SVE>(const float* x, const float* y, size_t d) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const svbool_t pg = svptrue_b32();

    svfloat32_t max0 = svdup_n_f32(0.0f);
    svfloat32_t max1 = svdup_n_f32(0.0f);
    svfloat32_t max2 = svdup_n_f32(0.0f);
    svfloat32_t max3 = svdup_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * lanes <= d; i += 4 * lanes) {
        max0 = svmax_f32_x(
                pg,
                max0,
                svabd_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i)));
        max1 = svmax_f32_x(
                pg,
                max1,
                svabd_f32_x(
                        pg,
                        svld1_f32(pg, x + i + lanes),
                        svld1_f32(pg, y + i + lanes)));
        max2 = svmax_f32_x(
                pg,
                max2,
                svabd_f32_x(
                        pg,
                        svld1_f32(pg, x + i + 2 * lanes),
                        svld1_f32(pg, y + i + 2 * lanes)));
        max3 = svmax_f32_x(
                pg,
                max3,
                svabd_f32_x(
                        pg,
                        svld1_f32(pg, x + i + 3 * lanes),
                        svld1_f32(pg, y + i + 3 * lanes)));
    }

    for (; i + lanes <= d; i += lanes) {
        max0 = svmax_f32_x(
                pg,
                max0,
                svabd_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i)));
    }

    if (i < d) {
        const svbool_t pg_tail = svwhilelt_b32_u64(i, d);
        max0 = svmax_f32_m(
                pg_tail,
                max0,
                svabd_f32_z(
                        pg_tail,
                        svld1_f32(pg_tail, x + i),
                        svld1_f32(pg_tail, y + i)));
    }

    max0 = svmax_f32_x(pg, max0, max1);
    max2 = svmax_f32_x(pg, max2, max3);
    max0 = svmax_f32_x(pg, max0, max2);
    return svmaxv_f32(pg, max0);
}

template <>
void fvec_inner_product_batch_4<SIMDLevel::ARM_SVE>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const svbool_t pg = svptrue_b32();

    svfloat32_t acc0_0 = svdup_n_f32(0.0f);
    svfloat32_t acc0_1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_1 = svdup_n_f32(0.0f);

    size_t i = 0;
    for (; i + 2 * lanes <= d; i += 2 * lanes) {
        const svfloat32_t vx0 = svld1_f32(pg, x + i);
        acc0_0 = svmla_f32_x(pg, acc0_0, vx0, svld1_f32(pg, y0 + i));
        acc1_0 = svmla_f32_x(pg, acc1_0, vx0, svld1_f32(pg, y1 + i));
        acc2_0 = svmla_f32_x(pg, acc2_0, vx0, svld1_f32(pg, y2 + i));
        acc3_0 = svmla_f32_x(pg, acc3_0, vx0, svld1_f32(pg, y3 + i));

        const svfloat32_t vx1 = svld1_f32(pg, x + i + lanes);
        acc0_1 = svmla_f32_x(pg, acc0_1, vx1, svld1_f32(pg, y0 + i + lanes));
        acc1_1 = svmla_f32_x(pg, acc1_1, vx1, svld1_f32(pg, y1 + i + lanes));
        acc2_1 = svmla_f32_x(pg, acc2_1, vx1, svld1_f32(pg, y2 + i + lanes));
        acc3_1 = svmla_f32_x(pg, acc3_1, vx1, svld1_f32(pg, y3 + i + lanes));
    }

    for (; i + lanes <= d; i += lanes) {
        const svfloat32_t vx0 = svld1_f32(pg, x + i);
        acc0_0 = svmla_f32_x(pg, acc0_0, vx0, svld1_f32(pg, y0 + i));
        acc1_0 = svmla_f32_x(pg, acc1_0, vx0, svld1_f32(pg, y1 + i));
        acc2_0 = svmla_f32_x(pg, acc2_0, vx0, svld1_f32(pg, y2 + i));
        acc3_0 = svmla_f32_x(pg, acc3_0, vx0, svld1_f32(pg, y3 + i));
    }

    if (i < d) {
        const svbool_t pg_tail = svwhilelt_b32_u64(i, d);
        const svfloat32_t vx = svld1_f32(pg_tail, x + i);
        acc0_0 = svmla_f32_m(pg_tail, acc0_0, vx, svld1_f32(pg_tail, y0 + i));
        acc1_0 = svmla_f32_m(pg_tail, acc1_0, vx, svld1_f32(pg_tail, y1 + i));
        acc2_0 = svmla_f32_m(pg_tail, acc2_0, vx, svld1_f32(pg_tail, y2 + i));
        acc3_0 = svmla_f32_m(pg_tail, acc3_0, vx, svld1_f32(pg_tail, y3 + i));
    }

    acc0_0 = svadd_f32_x(pg, acc0_0, acc0_1);
    acc1_0 = svadd_f32_x(pg, acc1_0, acc1_1);
    acc2_0 = svadd_f32_x(pg, acc2_0, acc2_1);
    acc3_0 = svadd_f32_x(pg, acc3_0, acc3_1);

    dis0 = svaddv_f32(pg, acc0_0);
    dis1 = svaddv_f32(pg, acc1_0);
    dis2 = svaddv_f32(pg, acc2_0);
    dis3 = svaddv_f32(pg, acc3_0);
}

template <>
void fvec_L2sqr_batch_4<SIMDLevel::ARM_SVE>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const svbool_t pg = svptrue_b32();

    svfloat32_t acc0_0 = svdup_n_f32(0.0f);
    svfloat32_t acc0_1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_1 = svdup_n_f32(0.0f);

    size_t i = 0;
    for (; i + 2 * lanes <= d; i += 2 * lanes) {
        const svfloat32_t vx0 = svld1_f32(pg, x + i);
        const svfloat32_t d00 = svsub_f32_x(pg, vx0, svld1_f32(pg, y0 + i));
        const svfloat32_t d10 = svsub_f32_x(pg, vx0, svld1_f32(pg, y1 + i));
        const svfloat32_t d20 = svsub_f32_x(pg, vx0, svld1_f32(pg, y2 + i));
        const svfloat32_t d30 = svsub_f32_x(pg, vx0, svld1_f32(pg, y3 + i));
        acc0_0 = svmla_f32_x(pg, acc0_0, d00, d00);
        acc1_0 = svmla_f32_x(pg, acc1_0, d10, d10);
        acc2_0 = svmla_f32_x(pg, acc2_0, d20, d20);
        acc3_0 = svmla_f32_x(pg, acc3_0, d30, d30);

        const svfloat32_t vx1 = svld1_f32(pg, x + i + lanes);
        const svfloat32_t d01 =
                svsub_f32_x(pg, vx1, svld1_f32(pg, y0 + i + lanes));
        const svfloat32_t d11 =
                svsub_f32_x(pg, vx1, svld1_f32(pg, y1 + i + lanes));
        const svfloat32_t d21 =
                svsub_f32_x(pg, vx1, svld1_f32(pg, y2 + i + lanes));
        const svfloat32_t d31 =
                svsub_f32_x(pg, vx1, svld1_f32(pg, y3 + i + lanes));
        acc0_1 = svmla_f32_x(pg, acc0_1, d01, d01);
        acc1_1 = svmla_f32_x(pg, acc1_1, d11, d11);
        acc2_1 = svmla_f32_x(pg, acc2_1, d21, d21);
        acc3_1 = svmla_f32_x(pg, acc3_1, d31, d31);
    }

    for (; i + lanes <= d; i += lanes) {
        const svfloat32_t vx0 = svld1_f32(pg, x + i);
        const svfloat32_t d00 = svsub_f32_x(pg, vx0, svld1_f32(pg, y0 + i));
        const svfloat32_t d10 = svsub_f32_x(pg, vx0, svld1_f32(pg, y1 + i));
        const svfloat32_t d20 = svsub_f32_x(pg, vx0, svld1_f32(pg, y2 + i));
        const svfloat32_t d30 = svsub_f32_x(pg, vx0, svld1_f32(pg, y3 + i));
        acc0_0 = svmla_f32_x(pg, acc0_0, d00, d00);
        acc1_0 = svmla_f32_x(pg, acc1_0, d10, d10);
        acc2_0 = svmla_f32_x(pg, acc2_0, d20, d20);
        acc3_0 = svmla_f32_x(pg, acc3_0, d30, d30);
    }

    if (i < d) {
        const svbool_t pg_tail = svwhilelt_b32_u64(i, d);
        const svfloat32_t vx = svld1_f32(pg_tail, x + i);
        const svfloat32_t d0 =
                svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y0 + i));
        const svfloat32_t d1 =
                svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y1 + i));
        const svfloat32_t d2 =
                svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y2 + i));
        const svfloat32_t d3 =
                svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y3 + i));
        acc0_0 = svmla_f32_m(pg_tail, acc0_0, d0, d0);
        acc1_0 = svmla_f32_m(pg_tail, acc1_0, d1, d1);
        acc2_0 = svmla_f32_m(pg_tail, acc2_0, d2, d2);
        acc3_0 = svmla_f32_m(pg_tail, acc3_0, d3, d3);
    }

    acc0_0 = svadd_f32_x(pg, acc0_0, acc0_1);
    acc1_0 = svadd_f32_x(pg, acc1_0, acc1_1);
    acc2_0 = svadd_f32_x(pg, acc2_0, acc2_1);
    acc3_0 = svadd_f32_x(pg, acc3_0, acc3_1);

    dis0 = svaddv_f32(pg, acc0_0);
    dis1 = svaddv_f32(pg, acc1_0);
    dis2 = svaddv_f32(pg, acc2_0);
    dis3 = svaddv_f32(pg, acc3_0);
}

template <>
float VectorDistance<METRIC_L2, SIMDLevel::ARM_SVE>::operator()(
        const float* x,
        const float* y) const {
    return fvec_L2sqr<SIMDLevel::ARM_SVE>(x, y, this->d);
}

template <>
float VectorDistance<METRIC_INNER_PRODUCT, SIMDLevel::ARM_SVE>::operator()(
        const float* x,
        const float* y) const {
    return fvec_inner_product<SIMDLevel::ARM_SVE>(x, y, this->d);
}

template <>
float VectorDistance<METRIC_L1, SIMDLevel::ARM_SVE>::operator()(
        const float* x,
        const float* y) const {
    return fvec_L1<SIMDLevel::ARM_SVE>(x, y, this->d);
}

template <>
float VectorDistance<METRIC_Linf, SIMDLevel::ARM_SVE>::operator()(
        const float* x,
        const float* y) const {
    return fvec_Linf<SIMDLevel::ARM_SVE>(x, y, this->d);
}

#define DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(metric)               \
    template <>                                                   \
    float VectorDistance<metric, SIMDLevel::ARM_SVE>::operator()( \
            const float* x, const float* y) const {               \
        return VectorDistance<metric, SIMDLevel::NONE>(           \
                this->d, this->metric_arg)(x, y);                 \
    }

DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(METRIC_Lp)
DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(METRIC_Canberra)
DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(METRIC_BrayCurtis)
DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(METRIC_JensenShannon)
DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(METRIC_Jaccard)
DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(METRIC_NaNEuclidean)
DEFINE_VECTOR_DISTANCE_SVE_FALLBACK(METRIC_GOWER)

#undef DEFINE_VECTOR_DISTANCE_SVE_FALLBACK

} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::ARM_SVE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::ARM_SVE>(
        const size_t n,
        const float* a,
        const float bf,
        const float* b,
        float* c) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    size_t i = 0;
    for (; i + lanes4 < n; i += lanes4) {
        const auto mask = svptrue_b32();
        const auto ai0 = svld1_f32(mask, a + i);
        const auto ai1 = svld1_f32(mask, a + i + lanes);
        const auto ai2 = svld1_f32(mask, a + i + lanes2);
        const auto ai3 = svld1_f32(mask, a + i + lanes3);
        const auto bi0 = svld1_f32(mask, b + i);
        const auto bi1 = svld1_f32(mask, b + i + lanes);
        const auto bi2 = svld1_f32(mask, b + i + lanes2);
        const auto bi3 = svld1_f32(mask, b + i + lanes3);
        const auto ci0 = svmla_n_f32_x(mask, ai0, bi0, bf);
        const auto ci1 = svmla_n_f32_x(mask, ai1, bi1, bf);
        const auto ci2 = svmla_n_f32_x(mask, ai2, bi2, bf);
        const auto ci3 = svmla_n_f32_x(mask, ai3, bi3, bf);
        svst1_f32(mask, c + i, ci0);
        svst1_f32(mask, c + i + lanes, ci1);
        svst1_f32(mask, c + i + lanes2, ci2);
        svst1_f32(mask, c + i + lanes3, ci3);
    }
    const auto mask0 = svwhilelt_b32_u64(i, n);
    const auto mask1 = svwhilelt_b32_u64(i + lanes, n);
    const auto mask2 = svwhilelt_b32_u64(i + lanes2, n);
    const auto mask3 = svwhilelt_b32_u64(i + lanes3, n);
    const auto ai0 = svld1_f32(mask0, a + i);
    const auto ai1 = svld1_f32(mask1, a + i + lanes);
    const auto ai2 = svld1_f32(mask2, a + i + lanes2);
    const auto ai3 = svld1_f32(mask3, a + i + lanes3);
    const auto bi0 = svld1_f32(mask0, b + i);
    const auto bi1 = svld1_f32(mask1, b + i + lanes);
    const auto bi2 = svld1_f32(mask2, b + i + lanes2);
    const auto bi3 = svld1_f32(mask3, b + i + lanes3);
    const auto ci0 = svmla_n_f32_x(mask0, ai0, bi0, bf);
    const auto ci1 = svmla_n_f32_x(mask1, ai1, bi1, bf);
    const auto ci2 = svmla_n_f32_x(mask2, ai2, bi2, bf);
    const auto ci3 = svmla_n_f32_x(mask3, ai3, bi3, bf);
    svst1_f32(mask0, c + i, ci0);
    svst1_f32(mask1, c + i + lanes, ci1);
    svst1_f32(mask2, c + i + lanes2, ci2);
    svst1_f32(mask3, c + i + lanes3, ci3);
}

template <>
int fvec_madd_and_argmin<SIMDLevel::ARM_SVE>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    float vmin = 1e20;
    int imin = -1;

    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = i;
        }
    }
    return imin;
}

struct ElementOpIP {
    static svfloat32_t op(svbool_t pg, svfloat32_t x, svfloat32_t y) {
        return svmul_f32_x(pg, x, y);
    }
    static svfloat32_t merge(
            svbool_t pg,
            svfloat32_t z,
            svfloat32_t x,
            svfloat32_t y) {
        return svmla_f32_x(pg, z, x, y);
    }
};

struct ElementOpL2 {
    static svfloat32_t op(svbool_t pg, svfloat32_t x, svfloat32_t y) {
        const svfloat32_t diff = svsub_f32_x(pg, x, y);
        return svmul_f32_x(pg, diff, diff);
    }

    static svfloat32_t merge(
            svbool_t pg,
            svfloat32_t z,
            svfloat32_t x,
            svfloat32_t y) {
        const svfloat32_t diff = svsub_f32_x(pg, x, y);
        return svmla_f32_x(pg, z, diff, diff);
    }
};

template <typename ElementOp>
void fvec_op_ny_sve_d1(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    size_t i = 0;
    for (; i + lanes4 < ny; i += lanes4) {
        svfloat32_t y0 = svld1_f32(pg, y);
        svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        svfloat32_t y3 = svld1_f32(pg, y + lanes3);
        y0 = ElementOp::op(pg, x0, y0);
        y1 = ElementOp::op(pg, x0, y1);
        y2 = ElementOp::op(pg, x0, y2);
        y3 = ElementOp::op(pg, x0, y3);
        svst1_f32(pg, dis, y0);
        svst1_f32(pg, dis + lanes, y1);
        svst1_f32(pg, dis + lanes2, y2);
        svst1_f32(pg, dis + lanes3, y3);
        y += lanes4;
        dis += lanes4;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svbool_t pg1 = svwhilelt_b32_u64(i + lanes, ny);
    const svbool_t pg2 = svwhilelt_b32_u64(i + lanes2, ny);
    const svbool_t pg3 = svwhilelt_b32_u64(i + lanes3, ny);
    svfloat32_t y0 = svld1_f32(pg0, y);
    svfloat32_t y1 = svld1_f32(pg1, y + lanes);
    svfloat32_t y2 = svld1_f32(pg2, y + lanes2);
    svfloat32_t y3 = svld1_f32(pg3, y + lanes3);
    y0 = ElementOp::op(pg0, x0, y0);
    y1 = ElementOp::op(pg1, x0, y1);
    y2 = ElementOp::op(pg2, x0, y2);
    y3 = ElementOp::op(pg3, x0, y3);
    svst1_f32(pg0, dis, y0);
    svst1_f32(pg1, dis + lanes, y1);
    svst1_f32(pg2, dis + lanes2, y2);
    svst1_f32(pg3, dis + lanes3, y3);
}

template <typename ElementOp>
void fvec_op_ny_sve_d2(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    const svfloat32_t x1 = svdup_n_f32(x[1]);
    size_t i = 0;
    for (; i + lanes2 < ny; i += lanes2) {
        const svfloat32x2_t y0 = svld2_f32(pg, y);
        const svfloat32x2_t y1 = svld2_f32(pg, y + lanes2);
        svfloat32_t y00 = svget2_f32(y0, 0);
        const svfloat32_t y01 = svget2_f32(y0, 1);
        svfloat32_t y10 = svget2_f32(y1, 0);
        const svfloat32_t y11 = svget2_f32(y1, 1);
        y00 = ElementOp::op(pg, x0, y00);
        y10 = ElementOp::op(pg, x0, y10);
        y00 = ElementOp::merge(pg, y00, x1, y01);
        y10 = ElementOp::merge(pg, y10, x1, y11);
        svst1_f32(pg, dis, y00);
        svst1_f32(pg, dis + lanes, y10);
        y += lanes4;
        dis += lanes2;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svbool_t pg1 = svwhilelt_b32_u64(i + lanes, ny);
    const svfloat32x2_t y0 = svld2_f32(pg0, y);
    const svfloat32x2_t y1 = svld2_f32(pg1, y + lanes2);
    svfloat32_t y00 = svget2_f32(y0, 0);
    const svfloat32_t y01 = svget2_f32(y0, 1);
    svfloat32_t y10 = svget2_f32(y1, 0);
    const svfloat32_t y11 = svget2_f32(y1, 1);
    y00 = ElementOp::op(pg0, x0, y00);
    y10 = ElementOp::op(pg1, x0, y10);
    y00 = ElementOp::merge(pg0, y00, x1, y01);
    y10 = ElementOp::merge(pg1, y10, x1, y11);
    svst1_f32(pg0, dis, y00);
    svst1_f32(pg1, dis + lanes, y10);
}

template <typename ElementOp>
void fvec_op_ny_sve_d4(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    const svfloat32_t x1 = svdup_n_f32(x[1]);
    const svfloat32_t x2 = svdup_n_f32(x[2]);
    const svfloat32_t x3 = svdup_n_f32(x[3]);
    size_t i = 0;
    for (; i + lanes < ny; i += lanes) {
        const svfloat32x4_t y0 = svld4_f32(pg, y);
        svfloat32_t y00 = svget4_f32(y0, 0);
        const svfloat32_t y01 = svget4_f32(y0, 1);
        svfloat32_t y02 = svget4_f32(y0, 2);
        const svfloat32_t y03 = svget4_f32(y0, 3);
        y00 = ElementOp::op(pg, x0, y00);
        y02 = ElementOp::op(pg, x2, y02);
        y00 = ElementOp::merge(pg, y00, x1, y01);
        y02 = ElementOp::merge(pg, y02, x3, y03);
        y00 = svadd_f32_x(pg, y00, y02);
        svst1_f32(pg, dis, y00);
        y += lanes4;
        dis += lanes;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svfloat32x4_t y0 = svld4_f32(pg0, y);
    svfloat32_t y00 = svget4_f32(y0, 0);
    const svfloat32_t y01 = svget4_f32(y0, 1);
    svfloat32_t y02 = svget4_f32(y0, 2);
    const svfloat32_t y03 = svget4_f32(y0, 3);
    y00 = ElementOp::op(pg0, x0, y00);
    y02 = ElementOp::op(pg0, x2, y02);
    y00 = ElementOp::merge(pg0, y00, x1, y01);
    y02 = ElementOp::merge(pg0, y02, x3, y03);
    y00 = svadd_f32_x(pg0, y00, y02);
    svst1_f32(pg0, dis, y00);
}

template <typename ElementOp>
void fvec_op_ny_sve_d8(float* dis, const float* x, const float* y, size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes4 = lanes * 4;
    const size_t lanes8 = lanes * 8;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svdup_n_f32(x[0]);
    const svfloat32_t x1 = svdup_n_f32(x[1]);
    const svfloat32_t x2 = svdup_n_f32(x[2]);
    const svfloat32_t x3 = svdup_n_f32(x[3]);
    const svfloat32_t x4 = svdup_n_f32(x[4]);
    const svfloat32_t x5 = svdup_n_f32(x[5]);
    const svfloat32_t x6 = svdup_n_f32(x[6]);
    const svfloat32_t x7 = svdup_n_f32(x[7]);
    size_t i = 0;
    for (; i + lanes < ny; i += lanes) {
        const svfloat32x4_t ya = svld4_f32(pg, y);
        const svfloat32x4_t yb = svld4_f32(pg, y + lanes4);
        const svfloat32_t ya0 = svget4_f32(ya, 0);
        const svfloat32_t ya1 = svget4_f32(ya, 1);
        const svfloat32_t ya2 = svget4_f32(ya, 2);
        const svfloat32_t ya3 = svget4_f32(ya, 3);
        const svfloat32_t yb0 = svget4_f32(yb, 0);
        const svfloat32_t yb1 = svget4_f32(yb, 1);
        const svfloat32_t yb2 = svget4_f32(yb, 2);
        const svfloat32_t yb3 = svget4_f32(yb, 3);
        svfloat32_t y0 = svuzp1(ya0, yb0);
        const svfloat32_t y1 = svuzp1(ya1, yb1);
        svfloat32_t y2 = svuzp1(ya2, yb2);
        const svfloat32_t y3 = svuzp1(ya3, yb3);
        svfloat32_t y4 = svuzp2(ya0, yb0);
        const svfloat32_t y5 = svuzp2(ya1, yb1);
        svfloat32_t y6 = svuzp2(ya2, yb2);
        const svfloat32_t y7 = svuzp2(ya3, yb3);
        y0 = ElementOp::op(pg, x0, y0);
        y2 = ElementOp::op(pg, x2, y2);
        y4 = ElementOp::op(pg, x4, y4);
        y6 = ElementOp::op(pg, x6, y6);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        y2 = ElementOp::merge(pg, y2, x3, y3);
        y4 = ElementOp::merge(pg, y4, x5, y5);
        y6 = ElementOp::merge(pg, y6, x7, y7);
        y0 = svadd_f32_x(pg, y0, y2);
        y4 = svadd_f32_x(pg, y4, y6);
        y0 = svadd_f32_x(pg, y0, y4);
        svst1_f32(pg, dis, y0);
        y += lanes8;
        dis += lanes;
    }
    const svbool_t pg0 = svwhilelt_b32_u64(i, ny);
    const svbool_t pga = svwhilelt_b32_u64(i * 2, ny * 2);
    const svbool_t pgb = svwhilelt_b32_u64(i * 2 + lanes, ny * 2);
    const svfloat32x4_t ya = svld4_f32(pga, y);
    const svfloat32x4_t yb = svld4_f32(pgb, y + lanes4);
    const svfloat32_t ya0 = svget4_f32(ya, 0);
    const svfloat32_t ya1 = svget4_f32(ya, 1);
    const svfloat32_t ya2 = svget4_f32(ya, 2);
    const svfloat32_t ya3 = svget4_f32(ya, 3);
    const svfloat32_t yb0 = svget4_f32(yb, 0);
    const svfloat32_t yb1 = svget4_f32(yb, 1);
    const svfloat32_t yb2 = svget4_f32(yb, 2);
    const svfloat32_t yb3 = svget4_f32(yb, 3);
    svfloat32_t y0 = svuzp1(ya0, yb0);
    const svfloat32_t y1 = svuzp1(ya1, yb1);
    svfloat32_t y2 = svuzp1(ya2, yb2);
    const svfloat32_t y3 = svuzp1(ya3, yb3);
    svfloat32_t y4 = svuzp2(ya0, yb0);
    const svfloat32_t y5 = svuzp2(ya1, yb1);
    svfloat32_t y6 = svuzp2(ya2, yb2);
    const svfloat32_t y7 = svuzp2(ya3, yb3);
    y0 = ElementOp::op(pg0, x0, y0);
    y2 = ElementOp::op(pg0, x2, y2);
    y4 = ElementOp::op(pg0, x4, y4);
    y6 = ElementOp::op(pg0, x6, y6);
    y0 = ElementOp::merge(pg0, y0, x1, y1);
    y2 = ElementOp::merge(pg0, y2, x3, y3);
    y4 = ElementOp::merge(pg0, y4, x5, y5);
    y6 = ElementOp::merge(pg0, y6, x7, y7);
    y0 = svadd_f32_x(pg0, y0, y2);
    y4 = svadd_f32_x(pg0, y4, y6);
    y0 = svadd_f32_x(pg0, y0, y4);
    svst1_f32(pg0, dis, y0);
    y += lanes8;
    dis += lanes;
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes1(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    size_t i = 0;
    for (; i + 3 < ny; i += 4) {
        svfloat32_t y0 = svld1_f32(pg, y);
        svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        svfloat32_t y3 = svld1_f32(pg, y + lanes3);
        y += lanes4;
        y0 = ElementOp::op(pg, x0, y0);
        y1 = ElementOp::op(pg, x0, y1);
        y2 = ElementOp::op(pg, x0, y2);
        y3 = ElementOp::op(pg, x0, y3);
        dis[i] = svaddv_f32(pg, y0);
        dis[i + 1] = svaddv_f32(pg, y1);
        dis[i + 2] = svaddv_f32(pg, y2);
        dis[i + 3] = svaddv_f32(pg, y3);
    }
    for (; i < ny; ++i) {
        svfloat32_t y0 = svld1_f32(pg, y);
        y += lanes;
        y0 = ElementOp::op(pg, x0, y0);
        dis[i] = svaddv_f32(pg, y0);
    }
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes2(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    const svfloat32_t x1 = svld1_f32(pg, x + lanes);
    size_t i = 0;
    for (; i + 1 < ny; i += 2) {
        svfloat32_t y00 = svld1_f32(pg, y);
        const svfloat32_t y01 = svld1_f32(pg, y + lanes);
        svfloat32_t y10 = svld1_f32(pg, y + lanes2);
        const svfloat32_t y11 = svld1_f32(pg, y + lanes3);
        y += lanes4;
        y00 = ElementOp::op(pg, x0, y00);
        y10 = ElementOp::op(pg, x0, y10);
        y00 = ElementOp::merge(pg, y00, x1, y01);
        y10 = ElementOp::merge(pg, y10, x1, y11);
        dis[i] = svaddv_f32(pg, y00);
        dis[i + 1] = svaddv_f32(pg, y10);
    }
    if (i < ny) {
        svfloat32_t y0 = svld1_f32(pg, y);
        const svfloat32_t y1 = svld1_f32(pg, y + lanes);
        y0 = ElementOp::op(pg, x0, y0);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        dis[i] = svaddv_f32(pg, y0);
    }
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes3(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    const svfloat32_t x1 = svld1_f32(pg, x + lanes);
    const svfloat32_t x2 = svld1_f32(pg, x + lanes2);
    for (size_t i = 0; i < ny; ++i) {
        svfloat32_t y0 = svld1_f32(pg, y);
        const svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        y += lanes3;
        y0 = ElementOp::op(pg, x0, y0);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        y0 = ElementOp::merge(pg, y0, x2, y2);
        dis[i] = svaddv_f32(pg, y0);
    }
}

template <typename ElementOp>
void fvec_op_ny_sve_lanes4(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    const size_t lanes2 = lanes * 2;
    const size_t lanes3 = lanes * 3;
    const size_t lanes4 = lanes * 4;
    const svbool_t pg = svptrue_b32();
    const svfloat32_t x0 = svld1_f32(pg, x);
    const svfloat32_t x1 = svld1_f32(pg, x + lanes);
    const svfloat32_t x2 = svld1_f32(pg, x + lanes2);
    const svfloat32_t x3 = svld1_f32(pg, x + lanes3);
    for (size_t i = 0; i < ny; ++i) {
        svfloat32_t y0 = svld1_f32(pg, y);
        const svfloat32_t y1 = svld1_f32(pg, y + lanes);
        svfloat32_t y2 = svld1_f32(pg, y + lanes2);
        const svfloat32_t y3 = svld1_f32(pg, y + lanes3);
        y += lanes4;
        y0 = ElementOp::op(pg, x0, y0);
        y2 = ElementOp::op(pg, x2, y2);
        y0 = ElementOp::merge(pg, y0, x1, y1);
        y2 = ElementOp::merge(pg, y2, x3, y3);
        y0 = svadd_f32_x(pg, y0, y2);
        dis[i] = svaddv_f32(pg, y0);
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::ARM_SVE>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    const size_t lanes = svcntw();
    switch (d) {
        case 1:
            fvec_op_ny_sve_d1<ElementOpIP>(dis, x, y, ny);
            break;
        case 2:
            fvec_op_ny_sve_d2<ElementOpIP>(dis, x, y, ny);
            break;
        case 4:
            fvec_op_ny_sve_d4<ElementOpIP>(dis, x, y, ny);
            break;
        case 8:
            fvec_op_ny_sve_d8<ElementOpIP>(dis, x, y, ny);
            break;
        default:
            if (d == lanes)
                fvec_op_ny_sve_lanes1<ElementOpIP>(dis, x, y, ny);
            else if (d == lanes * 2)
                fvec_op_ny_sve_lanes2<ElementOpIP>(dis, x, y, ny);
            else if (d == lanes * 3)
                fvec_op_ny_sve_lanes3<ElementOpIP>(dis, x, y, ny);
            else if (d == lanes * 4)
                fvec_op_ny_sve_lanes4<ElementOpIP>(dis, x, y, ny);
            else {
                size_t i = 0;
                for (; i + 4 <= ny; i += 4) {
                    fvec_inner_product_batch_4<SIMDLevel::ARM_SVE>(
                            x,
                            y,
                            y + d,
                            y + 2 * d,
                            y + 3 * d,
                            d,
                            dis[i],
                            dis[i + 1],
                            dis[i + 2],
                            dis[i + 3]);
                    y += 4 * d;
                }
                for (; i < ny; i++) {
                    dis[i] = fvec_inner_product<SIMDLevel::ARM_SVE>(x, y, d);
                    y += d;
                }
            }
            break;
    }
}

template <>
void fvec_L2sqr_ny<SIMDLevel::ARM_SVE>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    const size_t lanes = static_cast<size_t>(svcntw());

    switch (d) {
        case 1:
            fvec_op_ny_sve_d1<ElementOpL2>(dis, x, y, ny);
            break;

        case 2:
            fvec_op_ny_sve_d2<ElementOpL2>(dis, x, y, ny);
            break;

        case 4:
            fvec_op_ny_sve_d4<ElementOpL2>(dis, x, y, ny);
            break;

        case 8:
            fvec_op_ny_sve_d8<ElementOpL2>(dis, x, y, ny);
            break;

        default:
            if (d == lanes)
                fvec_op_ny_sve_lanes1<ElementOpL2>(dis, x, y, ny);
            else if (d == lanes * 2)
                fvec_op_ny_sve_lanes2<ElementOpL2>(dis, x, y, ny);
            else if (d == lanes * 3)
                fvec_op_ny_sve_lanes3<ElementOpL2>(dis, x, y, ny);
            else if (d == lanes * 4)
                fvec_op_ny_sve_lanes4<ElementOpL2>(dis, x, y, ny);
            else {
                size_t i = 0;
                for (; i + 4 <= ny; i += 4) {
                    fvec_L2sqr_batch_4<SIMDLevel::ARM_SVE>(
                            x,
                            y,
                            y + d,
                            y + 2 * d,
                            y + 3 * d,
                            d,
                            dis[i],
                            dis[i + 1],
                            dis[i + 2],
                            dis[i + 3]);
                    y += 4 * d;
                }
                for (; i < ny; i++) {
                    dis[i] = fvec_L2sqr<SIMDLevel::ARM_SVE>(x, y, d);
                    y += d;
                }
            }
            break;
    }
}

namespace {

/// Low-dimensional L2sqr nearest (D in {2,4,8}). The data is row-major
/// (centroid c is y[c*D .. c*D+D]). Each SVE lane tracks one centroid across
/// batches, keeping the lane-local minimum and index in registers. The scratch
/// buffer is not written, matching the AVX2/AVX512 D2/D4/D8 implementations.
template <int D>
size_t fvec_L2sqr_ny_nearest_lowdim(
        float* /*distances_tmp_buffer*/,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t lanes = svcntw();
    svfloat32_t global_mins = svdup_n_f32(HUGE_VALF);
    svuint32_t global_ids = svdup_n_u32(0);
    svuint32_t current_ids = svindex_u32(0, 1);

    for (size_t c = 0; c < ny; c += lanes) {
        const svbool_t pg = svwhilelt_b32_u64(c, ny);
        svfloat32_t distances = svdup_n_f32(0.0f);
        for (uint32_t j = 0; j < D; ++j) {
            const svuint32_t offsets = svindex_u32(j, D);
            const svfloat32_t yv =
                    svld1_gather_u32index_f32(pg, y + c * D, offsets);
            const svfloat32_t diff = svsub_n_f32_x(pg, yv, x[j]);
            distances = svmla_f32_m(pg, distances, diff, diff);
        }

        const svbool_t closer = svcmplt_f32(pg, distances, global_mins);
        global_mins = svsel_f32(closer, distances, global_mins);
        global_ids = svsel_u32(closer, current_ids, global_ids);
        current_ids =
                svadd_n_u32_x(pg, current_ids, static_cast<uint32_t>(lanes));
    }

    const svbool_t all = svptrue_b32();
    const float global_min = svminv_f32(all, global_mins);
    return svminv_u32(svcmpeq_n_f32(all, global_mins, global_min), global_ids);
}

} // namespace

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::ARM_SVE>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    switch (d) {
        case 2:
            return fvec_L2sqr_ny_nearest_lowdim<2>(
                    distances_tmp_buffer, x, y, ny);
        case 4:
            return fvec_L2sqr_ny_nearest_lowdim<4>(
                    distances_tmp_buffer, x, y, ny);
        case 8:
            return fvec_L2sqr_ny_nearest_lowdim<8>(
                    distances_tmp_buffer, x, y, ny);
    }

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        float d0, d1, d2, d3;
        fvec_L2sqr_batch_4<SIMDLevel::ARM_SVE>(
                x, y, y + d, y + 2 * d, y + 3 * d, d, d0, d1, d2, d3);
        y += 4 * d;
        distances_tmp_buffer[i] = d0;
        distances_tmp_buffer[i + 1] = d1;
        distances_tmp_buffer[i + 2] = d2;
        distances_tmp_buffer[i + 3] = d3;
        if (d0 < min_dis) {
            min_dis = d0;
            nearest_idx = i;
        }
        if (d1 < min_dis) {
            min_dis = d1;
            nearest_idx = i + 1;
        }
        if (d2 < min_dis) {
            min_dis = d2;
            nearest_idx = i + 2;
        }
        if (d3 < min_dis) {
            min_dis = d3;
            nearest_idx = i + 3;
        }
    }
    for (; i < ny; ++i) {
        const float dist = fvec_L2sqr<SIMDLevel::ARM_SVE>(x, y, d);
        y += d;
        distances_tmp_buffer[i] = dist;
        if (dist < min_dis) {
            min_dis = dist;
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::ARM_SVE>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    const size_t lanes = static_cast<size_t>(svcntw());
    const float x_sq = fvec_norm_L2sqr(x, d);

    for (size_t k = 0; k < ny; k += lanes) {
        svbool_t pg = svwhilelt_b32_u64(k, ny);
        svfloat32_t acc = svdup_n_f32(0.0f);

        for (size_t j = 0; j < d; ++j) {
            svfloat32_t ychunk = svld1_f32(pg, y + j * d_offset + k);
            svfloat32_t xj = svdup_n_f32(x[j]);
            acc = svmla_f32_x(pg, acc, xj, ychunk);
        }

        svfloat32_t ysq = svld1_f32(pg, y_sqlen + k);
        svfloat32_t two_acc = svmul_f32_x(pg, acc, svdup_n_f32(2.0f));
        svfloat32_t sum = svadd_f32_x(pg, svdup_n_f32(x_sq), ysq);
        svfloat32_t res = svsub_f32_x(pg, sum, two_acc);
        svst1_f32(pg, dis + k, res);
    }
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::ARM_SVE>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    const size_t lanes = svcntw();
    const float x_sq = fvec_norm_L2sqr(x, d);

    size_t current_min_idx = 0;
    svfloat32_t current_min_v = svdup_n_f32(HUGE_VALF);

    // Max SVE vector length is 2048 bits → 64 fp32 lanes
    float tmp_buf[64];

    for (size_t k = 0; k < ny; k += lanes) {
        svbool_t pg = svwhilelt_b32_u64(k, ny);
        svfloat32_t acc = svdup_n_f32(0.0f);

        for (size_t j = 0; j < d; ++j) {
            svfloat32_t ychunk = svld1_f32(pg, y + j * d_offset + k);
            svfloat32_t xj = svdup_n_f32(x[j]);
            acc = svmla_f32_x(pg, acc, xj, ychunk);
        }

        svfloat32_t ysq = svld1_f32(pg, y_sqlen + k);
        svfloat32_t two_acc = svmul_f32_x(pg, acc, svdup_n_f32(2.0f));
        svfloat32_t sum = svadd_f32_x(pg, svdup_n_f32(x_sq), ysq);
        svfloat32_t res = svsub_f32_x(pg, sum, two_acc);

        svst1_f32(pg, distances_tmp_buffer + k, res);

        svbool_t less_mask = svcmplt_f32(pg, res, current_min_v);

        if (svptest_any(pg, less_mask)) {
            float vec_min = svminv_f32(pg, res);

            current_min_v =
                    svmin_f32_x(pg, current_min_v, svdup_n_f32(vec_min));

            svst1_f32(pg, tmp_buf, res);

            size_t cnt = (size_t)svcntw();
            for (size_t lane = 0; lane < cnt && (k + lane) < ny; ++lane) {
                if (tmp_buf[lane] == vec_min) {
                    current_min_idx = k + lane;
                    break;
                }
            }
        }
    }

    return current_min_idx;
}

namespace {

template <class QueryTiles>
void exhaustive_L2sqr_blas_cmax_sve(
        QueryTiles& queries,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    queries.prepare_norms(nx, x_norms.get());

    const size_t lanes = svcntw();

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        const float* xt = queries.tile(i0, i1, x_norms.get());

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       xt,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
#pragma omp parallel for schedule(static) if ((i1 - i0) >= 16)
            for (int64_t i = static_cast<int64_t>(i0);
                 i < static_cast<int64_t>(i1);
                 i++) {
                const size_t count = j1 - j0;
                float* ip_line = ip_block.get() + (i - i0) * count;

                svprfw(svwhilelt_b32_u64(0, count), ip_line, SV_PLDL1KEEP);
                svprfw(svwhilelt_b32_u64(lanes, count),
                       ip_line + lanes,
                       SV_PLDL1KEEP);

                // Track lanes min distances + lanes min indices.
                // All the distances tracked do not take x_norms[i]
                //   into account in order to get rid of extra
                //   vaddq_f32(x_norms[i], ...) instructions
                //   is distance computations.
                auto min_distances = svdup_n_f32(res.dis_tab[i] - x_norms[i]);

                // these indices are local and are relative to j0.
                // so, value 0 means j0.
                auto min_indices = svdup_n_u32(0u);

                auto current_indices = svindex_u32(0u, 1u);

                // process lanes * 2 elements per loop
                for (size_t idx_j = 0; idx_j < count;
                     idx_j += lanes * 2, ip_line += lanes * 2) {
                    svprfw(svwhilelt_b32_u64(idx_j + lanes * 2, count),
                           ip_line + lanes * 2,
                           SV_PLDL1KEEP);
                    svprfw(svwhilelt_b32_u64(idx_j + lanes * 3, count),
                           ip_line + lanes * 3,
                           SV_PLDL1KEEP);

                    // mask
                    const auto mask_0 = svwhilelt_b32_u64(idx_j, count);
                    const auto mask_1 = svwhilelt_b32_u64(idx_j + lanes, count);

                    // load values for norms
                    const auto y_norm_0 =
                            svld1_f32(mask_0, y_norms + idx_j + j0 + 0);
                    const auto y_norm_1 =
                            svld1_f32(mask_1, y_norms + idx_j + j0 + lanes);

                    // load values for dot products
                    const auto ip_0 = svld1_f32(mask_0, ip_line + 0);
                    const auto ip_1 = svld1_f32(mask_1, ip_line + lanes);

                    // compute dis = y_norm[j] - 2 * dot(x_norm[i], y_norm[j]).
                    // x_norm[i] was dropped off because it is a constant for a
                    // given i. We'll deal with it later.
                    const auto distances_0 =
                            svmla_n_f32_z(mask_0, y_norm_0, ip_0, -2.f);
                    const auto distances_1 =
                            svmla_n_f32_z(mask_1, y_norm_1, ip_1, -2.f);

                    // compare the new distances to the min distances
                    // for each of the first group of 4 ARM SIMD components.
                    auto comparison =
                            svcmpgt_f32(mask_0, min_distances, distances_0);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances =
                            svsel_f32(comparison, distances_0, min_distances);
                    min_indices =
                            svsel_u32(comparison, current_indices, min_indices);
                    current_indices = svadd_n_u32_x(
                            mask_0,
                            current_indices,
                            static_cast<uint32_t>(lanes));

                    // compare the new distances to the min distances
                    // for each of the second group of 4 ARM SIMD components.
                    comparison =
                            svcmpgt_f32(mask_1, min_distances, distances_1);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances =
                            svsel_f32(comparison, distances_1, min_distances);
                    min_indices =
                            svsel_u32(comparison, current_indices, min_indices);
                    current_indices = svadd_n_u32_x(
                            mask_1,
                            current_indices,
                            static_cast<uint32_t>(lanes));
                }

                // add missing x_norms[i]
                // negative values can occur for identical vectors
                //    due to roundoff errors.
                auto mask = svwhilelt_b32_u64(0, count);
                min_distances = svadd_n_f32_z(
                        svcmpge_n_f32(mask, min_distances, -x_norms[i]),
                        min_distances,
                        x_norms[i]);
                min_indices = svadd_n_u32_x(
                        mask, min_indices, static_cast<uint32_t>(j0));
                mask = svcmple_n_f32(mask, min_distances, res.dis_tab[i]);
                if (svcntp_b32(svptrue_b32(), mask) == 0)
                    res.add_result(i, res.dis_tab[i], res.ids_tab[i]);
                else {
                    const auto min_distance = svminv_f32(mask, min_distances);
                    const auto min_index = svminv_u32(
                            svcmpeq_n_f32(mask, min_distances, min_distance),
                            min_indices);
                    res.add_result(i, min_distance, min_index);
                }
            }
        }
        // Does nothing for SingleBestResultHandler, but
        // keeping the call for the consistency.
        res.end_multiple();
        InterruptCallback::check();
    }
}

} // namespace

template <>
void exhaustive_L2sqr_blas_cmax<SIMDLevel::ARM_SVE>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    Fp32QueryTiles queries{x, d};
    exhaustive_L2sqr_blas_cmax_sve(queries, y, d, nx, ny, res, y_norms);
}

template <>
void exhaustive_L2sqr_blas_cmax_fp16<SIMDLevel::ARM_SVE>(
        const uint16_t* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    Fp16QueryTiles queries(x, d, nx);
    exhaustive_L2sqr_blas_cmax_sve(queries, y, d, nx, ny, res, y_norms);
}

} // namespace faiss
