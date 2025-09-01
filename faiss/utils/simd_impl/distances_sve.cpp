/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#ifdef COMPILE_SIMD_SVE

#include <arm_sve.h>

#define AUTOVEC_LEVEL SIMDLevel::ARM_SVE
#include <faiss/utils/simd_impl/distances_autovec-inl.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::ARM_SVE>(
        const size_t n,
        const float* __restrict a,
        const float bf,
        const float* __restrict b,
        float* __restrict c) {
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

void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
}

void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_y_transposed_ref(dis, x, y, y_sqlen, d, d_offset, ny);
}

size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_ref(distances_tmp_buffer, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed_ref(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

float fvec_L1(const float* x, const float* y, size_t d) {
    return fvec_L1_ref(x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    return fvec_Linf_ref(x, y, d);
}

void fvec_inner_products_ny(
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
            else
                fvec_inner_products_ny_ref(dis, x, y, d, ny);
            break;
    }
}

} // namespace faiss

#endif // COMPILE_SIMD_SVE
