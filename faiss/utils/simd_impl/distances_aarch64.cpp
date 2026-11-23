/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>
#include <faiss/utils/simd_impl/masked_sum.h>

#include <arm_neon.h>
#include <limits>

#define THE_SIMD_LEVEL SIMDLevel::ARM_NEON
#include <faiss/utils/simd_impl/distances_autovec-inl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

#include <faiss/utils/simd_impl/distances_simdlib256.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::ARM_NEON>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    const size_t n_simd = n - (n & 3);
    const float32x4_t bfv = vdupq_n_f32(bf);
    size_t i;
    for (i = 0; i < n_simd; i += 4) {
        const float32x4_t ai = vld1q_f32(a + i);
        const float32x4_t bi = vld1q_f32(b + i);
        const float32x4_t ci = vfmaq_f32(ai, bfv, bi);
        vst1q_f32(c + i, ci);
    }
    for (; i < n; ++i)
        c[i] = a[i] + bf * b[i];
}

template <>
float fvec_masked_sum<SIMDLevel::ARM_NEON>(
        const float* x,
        const uint8_t* bits,
        size_t d) {
    return masked_sum<SIMDLevel::ARM_NEON>(x, bits, d);
}

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    // Use autovectorized implementation
    fvec_L2sqr_ny_transposed<SIMDLevel::NONE>(
            dis, x, y, y_sqlen, d, d_offset, ny);
}

struct NEONElementOpIP {
    static float op(float x, float y) {
        return x * y;
    }

    static float32x4_t op(float32x4_t x, float32x4_t y) {
        return vmulq_f32(x, y);
    }

    static float32x4_t merge(float32x4_t sum, float32x4_t x, float32x4_t y) {
        return vfmaq_f32(sum, x, y);
    }
};

struct NEONElementOpL2 {
    static float op(float x, float y) {
        const float diff = x - y;
        return diff * diff;
    }

    static float32x4_t op(float32x4_t x, float32x4_t y) {
        const float32x4_t diff = vsubq_f32(x, y);
        return vmulq_f32(diff, diff);
    }

    static float32x4_t merge(float32x4_t sum, float32x4_t x, float32x4_t y) {
        const float32x4_t diff = vsubq_f32(x, y);
        return vfmaq_f32(sum, diff, diff);
    }
};

template <class ElementOp>
static void fvec_op_ny_D1(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    if (ny == 0) {
        return;
    }
    const float x0 = x[0];
    const float32x4_t xv = vdupq_n_f32(x0);
    const size_t ny4 = ny - (ny & 3);
    size_t i = 0;
    for (; i < ny4; i += 4) {
        vst1q_f32(dis + i, ElementOp::op(xv, vld1q_f32(y + i)));
    }
    for (; i < ny; ++i) {
        dis[i] = ElementOp::op(x0, y[i]);
    }
}

template <class ElementOp>
static void fvec_op_ny_D2(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny4 = ny - (ny & 3);
    size_t i = 0;
    if (ny4 > 0) {
        const float32x4_t x0 = vdupq_n_f32(x[0]);
        const float32x4_t x1 = vdupq_n_f32(x[1]);
        for (; i < ny4; i += 4) {
            const float32x4x2_t yv = vld2q_f32(y);
            float32x4_t sum = ElementOp::op(x0, yv.val[0]);
            sum = ElementOp::merge(sum, x1, yv.val[1]);
            vst1q_f32(dis + i, sum);
            y += 8;
        }
    }
    for (; i < ny; ++i) {
        dis[i] = ElementOp::op(x[0], y[0]) + ElementOp::op(x[1], y[1]);
        y += 2;
    }
}

template <class ElementOp>
static void fvec_op_ny_D4(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    const size_t ny4 = ny - (ny & 3);
    size_t i = 0;
    if (ny4 > 0) {
        const float32x4_t x0 = vdupq_n_f32(x[0]);
        const float32x4_t x1 = vdupq_n_f32(x[1]);
        const float32x4_t x2 = vdupq_n_f32(x[2]);
        const float32x4_t x3 = vdupq_n_f32(x[3]);
        for (; i < ny4; i += 4) {
            const float32x4x4_t yv = vld4q_f32(y);
            float32x4_t sum = ElementOp::op(x0, yv.val[0]);
            sum = ElementOp::merge(sum, x1, yv.val[1]);
            sum = ElementOp::merge(sum, x2, yv.val[2]);
            sum = ElementOp::merge(sum, x3, yv.val[3]);
            vst1q_f32(dis + i, sum);
            y += 16;
        }
    }
    for (; i < ny; ++i) {
        dis[i] = vaddvq_f32(ElementOp::op(vld1q_f32(x), vld1q_f32(y)));
        y += 4;
    }
}

template <class ElementOp>
static void fvec_op_ny_D8(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    if (ny == 0) {
        return;
    }
    const float32x4_t x0 = vld1q_f32(x);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const auto partial_distance = [&](const float* row) {
        return ElementOp::merge(
                ElementOp::op(x0, vld1q_f32(row)), x1, vld1q_f32(row + 4));
    };

    const size_t ny4 = ny - (ny & 3);
    size_t i = 0;
    for (; i < ny4; i += 4) {
        const float32x4_t s0 = partial_distance(y);
        const float32x4_t s1 = partial_distance(y + 8);
        const float32x4_t s2 = partial_distance(y + 16);
        const float32x4_t s3 = partial_distance(y + 24);
        vst1q_f32(dis + i, vpaddq_f32(vpaddq_f32(s0, s1), vpaddq_f32(s2, s3)));
        y += 32;
    }
    for (; i < ny; ++i) {
        dis[i] = vaddvq_f32(partial_distance(y));
        y += 8;
    }
}

template <class ElementOp>
static void fvec_op_ny_D12(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    if (ny == 0) {
        return;
    }
    const float32x4_t x0 = vld1q_f32(x);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);
    for (size_t i = 0; i < ny; ++i) {
        float32x4_t sum = ElementOp::op(x0, vld1q_f32(y));
        sum = ElementOp::merge(sum, x1, vld1q_f32(y + 4));
        sum = ElementOp::merge(sum, x2, vld1q_f32(y + 8));
        dis[i] = vaddvq_f32(sum);
        y += 12;
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
#define DISPATCH(dval)                                      \
    case dval:                                              \
        fvec_op_ny_D##dval<NEONElementOpIP>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        default:
            fvec_inner_products_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}

template <>
void fvec_L2sqr_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
#define DISPATCH(dval)                                      \
    case dval:                                              \
        fvec_op_ny_D##dval<NEONElementOpL2>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        default:
            fvec_L2sqr_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::ARM_NEON>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::ARM_NEON>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

template <>
int fvec_madd_and_argmin<SIMDLevel::ARM_NEON>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    float32x4_t vminv = vdupq_n_f32(1e20);
    uint32x4_t iminv = vdupq_n_u32(static_cast<uint32_t>(-1));
    size_t i;
    {
        const size_t n_simd = n - (n & 3);
        const uint32_t iota[] = {0, 1, 2, 3};
        uint32x4_t iv = vld1q_u32(iota);
        const uint32x4_t incv = vdupq_n_u32(4);
        const float32x4_t bfv = vdupq_n_f32(bf);
        for (i = 0; i < n_simd; i += 4) {
            const float32x4_t ai = vld1q_f32(a + i);
            const float32x4_t bi = vld1q_f32(b + i);
            const float32x4_t ci = vfmaq_f32(ai, bfv, bi);
            vst1q_f32(c + i, ci);
            const uint32x4_t less_than = vcltq_f32(ci, vminv);
            vminv = vminq_f32(ci, vminv);
            iminv = vorrq_u32(
                    vandq_u32(less_than, iv),
                    vandq_u32(vmvnq_u32(less_than), iminv));
            iv = vaddq_u32(iv, incv);
        }
    }
    float vmin = vminvq_f32(vminv);
    uint32_t imin;
    {
        const float32x4_t vminy = vdupq_n_f32(vmin);
        const uint32x4_t equals = vceqq_f32(vminv, vminy);
        imin = vminvq_u32(vorrq_u32(
                vandq_u32(equals, iminv),
                vandq_u32(
                        vmvnq_u32(equals),
                        vdupq_n_u32(std::numeric_limits<uint32_t>::max()))));
    }
    for (; i < n; ++i) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = static_cast<uint32_t>(i);
        }
    }
    return static_cast<int>(imin);
}

} // namespace faiss
