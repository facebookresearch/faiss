/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#ifdef __aarch64__

#include <arm_neon.h>
#include <limits>

#define THE_SIMD_LEVEL SIMDLevel::ARM_NEON
#include <faiss/utils/simd_impl/distances_autovec-inl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

#include <faiss/utils/simd_impl/distances_simdlib256.h>

namespace faiss {

namespace {

/// Component-wise operations between x and y for the small-d kernels below.
struct NeonElementOpIP {
    static float op(float x, float y) {
        return x * y;
    }

    static float32x4_t op(float32x4_t x, float32x4_t y) {
        return vmulq_f32(x, y);
    }
};

struct NeonElementOpL2 {
    static float op(float x, float y) {
        const float tmp = x - y;
        return tmp * tmp;
    }

    static float32x4_t op(float32x4_t x, float32x4_t y) {
        const float32x4_t tmp = vsubq_f32(x, y);
        return vmulq_f32(tmp, tmp);
    }
};

/// Returns {sum(a), sum(b), sum(c), sum(d)}.
inline float32x4_t horizontal_sum_4x4(
        float32x4_t a,
        float32x4_t b,
        float32x4_t c,
        float32x4_t d) {
    return vpaddq_f32(vpaddq_f32(a, b), vpaddq_f32(c, d));
}

template <class ElementOp>
void fvec_op_ny_neon_d1(float* dis, const float* x, const float* y, size_t ny) {
    const float x0s = x[0];
    const float32x4_t x0 = vdupq_n_f32(x0s);
    size_t i = 0;
    for (; i + 16 <= ny; i += 16) {
        vst1q_f32(dis + i, ElementOp::op(x0, vld1q_f32(y + i)));
        vst1q_f32(dis + i + 4, ElementOp::op(x0, vld1q_f32(y + i + 4)));
        vst1q_f32(dis + i + 8, ElementOp::op(x0, vld1q_f32(y + i + 8)));
        vst1q_f32(dis + i + 12, ElementOp::op(x0, vld1q_f32(y + i + 12)));
    }
    for (; i + 4 <= ny; i += 4) {
        vst1q_f32(dis + i, ElementOp::op(x0, vld1q_f32(y + i)));
    }
    for (; i < ny; i++) {
        dis[i] = ElementOp::op(x0s, y[i]);
    }
}

template <class ElementOp>
void fvec_op_ny_neon_d2(float* dis, const float* x, const float* y, size_t ny) {
    // {x0, x1, x0, x1}: one register covers two rows of y
    const float32x4_t x01 = vcombine_f32(vld1_f32(x), vld1_f32(x));
    size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        const float32x4_t a = ElementOp::op(x01, vld1q_f32(y));
        const float32x4_t b = ElementOp::op(x01, vld1q_f32(y + 4));
        vst1q_f32(dis + i, vpaddq_f32(a, b));
        y += 8;
    }
    for (; i < ny; i++) {
        dis[i] = ElementOp::op(x[0], y[0]) + ElementOp::op(x[1], y[1]);
        y += 2;
    }
}

/// d = 4 * NV: each row of y is NV registers, reduced to one register per
/// row and then across four rows so that four results leave in one store.
template <class ElementOp, size_t NV>
void fvec_op_ny_neon_d4n(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {
    constexpr size_t d = 4 * NV;
    float32x4_t xv[NV];
    for (size_t k = 0; k < NV; k++) {
        xv[k] = vld1q_f32(x + 4 * k);
    }
    auto row = [&](const float* yr) {
        float32x4_t acc = ElementOp::op(xv[0], vld1q_f32(yr));
        for (size_t k = 1; k < NV; k++) {
            acc = vaddq_f32(acc, ElementOp::op(xv[k], vld1q_f32(yr + 4 * k)));
        }
        return acc;
    };
    size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        vst1q_f32(
                dis + i,
                horizontal_sum_4x4(
                        row(y), row(y + d), row(y + 2 * d), row(y + 3 * d)));
        y += 4 * d;
    }
    for (; i < ny; i++) {
        dis[i] = vaddvq_f32(row(y));
        y += d;
    }
}

/// Dedicated kernels for the sub-quantizer dimensions that PQ table
/// construction uses most (the x86 path has 1, 2, 4, 8, 12; SVE at 128 bits
/// has 1, 2, 4, 8, 12, 16). Returns false when d has none.
template <class ElementOp>
bool fvec_op_ny_neon_small_d(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    switch (d) {
        case 1:
            fvec_op_ny_neon_d1<ElementOp>(dis, x, y, ny);
            return true;
        case 2:
            fvec_op_ny_neon_d2<ElementOp>(dis, x, y, ny);
            return true;
        case 4:
            fvec_op_ny_neon_d4n<ElementOp, 1>(dis, x, y, ny);
            return true;
        case 8:
            fvec_op_ny_neon_d4n<ElementOp, 2>(dis, x, y, ny);
            return true;
        case 12:
            fvec_op_ny_neon_d4n<ElementOp, 3>(dis, x, y, ny);
            return true;
        case 16:
            fvec_op_ny_neon_d4n<ElementOp, 4>(dis, x, y, ny);
            return true;
        default:
            return false;
    }
}

} // namespace

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

template <>
void fvec_inner_products_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    if (fvec_op_ny_neon_small_d<NeonElementOpIP>(dis, x, y, d, ny)) {
        return;
    }
    // call the NEON kernel directly rather than the dispatching
    // fvec_inner_product() once per row
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_inner_product<SIMDLevel::ARM_NEON>(x, y, d);
        y += d;
    }
}

template <>
void fvec_L2sqr_ny<SIMDLevel::ARM_NEON>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    if (fvec_op_ny_neon_small_d<NeonElementOpL2>(dis, x, y, d, ny)) {
        return;
    }
    // call the NEON kernel directly rather than the dispatching fvec_L2sqr()
    // once per row
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr<SIMDLevel::ARM_NEON>(x, y, d);
        y += d;
    }
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

#endif // __aarch64__
