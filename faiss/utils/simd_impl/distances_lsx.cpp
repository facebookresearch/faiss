/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#include <lsxintrin.h>
#include <bit>
#include <limits>

namespace faiss {
namespace {
struct FloatVec {
    static constexpr size_t width = 4;
    __m128 data;
    explicit FloatVec(__m128 value) : data(value) {}
    explicit FloatVec(float value)
            : data((__m128)__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(value))) {}
    explicit FloatVec(const float* ptr) : data((__m128)__lsx_vld(ptr, 0)) {}
    void store(float* ptr) const {
        __lsx_vst((__m128i)data, ptr, 0);
    }
    static FloatVec madd(FloatVec a, FloatVec b, FloatVec c) {
        return FloatVec(__lsx_vfmadd_s(a.data, b.data, c.data));
    }
    FloatVec min(FloatVec other) const {
        const auto mask = __lsx_vfcmp_clt_s(other.data, data);
        return FloatVec((__m128)__lsx_vbitsel_v(
                (__m128i)data, (__m128i)other.data, mask));
    }
    float scalar() const {
        return std::bit_cast<float>(
                static_cast<uint32_t>(__lsx_vpickve2gr_w((__m128i)data, 0)));
    }
    uint32_t equal_mask(FloatVec other) const {
        const auto mask = __lsx_vmskltz_b(__lsx_vfcmp_ceq_s(data, other.data));
        return __lsx_vpickve2gr_wu(mask, 0);
    }
    float reduce_min() const {
        FloatVec v = *this;
        v = v.min(FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0x4e)));
        v = v.min(FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0xb1)));
        return v.scalar();
    }
};
} // namespace
} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LSX
#include <faiss/utils/simd_impl/distances_autovec-inl.h>

#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

#include <faiss/utils/simd_impl/distances_simdlib256.h>

#include <faiss/utils/simd_impl/masked_sum.h>

namespace faiss {
namespace {
size_t loongarch_argmin(const float* values, size_t n, float limit) {
    FloatVec minima(limit);
    size_t i = 0;
    for (; i + FloatVec::width <= n; i += FloatVec::width) {
        minima = minima.min(FloatVec(values + i));
    }
    float best = minima.reduce_min();
    for (; i < n; ++i) {
        if (values[i] < best)
            best = values[i];
    }
    if (!(best < limit)) {
        return n;
    }
    i = 0;
    const FloatVec target(best);
    for (; i + FloatVec::width <= n; i += FloatVec::width) {
        const uint32_t mask = FloatVec(values + i).equal_mask(target);
        if (mask) {
            return i + __builtin_ctz(mask) / 4;
        }
    }
    for (; i < n; ++i) {
        if (values[i] == best)
            return i;
    }
    return n;
}
} // namespace

template <>
float fvec_masked_sum<THE_SIMD_LEVEL>(
        const float* x,
        const uint8_t* bits,
        size_t d) {
    return masked_sum<THE_SIMD_LEVEL>(x, bits, d);
}

template <>
void fvec_L2sqr_ny_transposed<THE_SIMD_LEVEL>(
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
void fvec_inner_products_ny<THE_SIMD_LEVEL>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
}

template <>
void fvec_L2sqr_ny<THE_SIMD_LEVEL>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny<SIMDLevel::NONE>(dis, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest<THE_SIMD_LEVEL>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<THE_SIMD_LEVEL>(
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
void fvec_madd<THE_SIMD_LEVEL>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    size_t i = 0;
    const FloatVec factor(bf);
    for (; i + FloatVec::width <= n; i += FloatVec::width)
        FloatVec::madd(factor, FloatVec(b + i), FloatVec(a + i)).store(c + i);
    for (; i < n; ++i)
        c[i] = a[i] + bf * b[i];
}

template <>
int fvec_madd_and_argmin<THE_SIMD_LEVEL>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    fvec_madd<THE_SIMD_LEVEL>(n, a, bf, b, c);
    const size_t i = loongarch_argmin(c, n, 1e20f);
    return i < n ? static_cast<int>(i) : -1;
}
} // namespace faiss
#undef THE_SIMD_LEVEL
