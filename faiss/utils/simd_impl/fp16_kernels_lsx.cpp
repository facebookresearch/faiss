/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_impl/fp16_kernels.h>

#include <lsxintrin.h>
#include <bit>

namespace faiss {
namespace detail {

template <>
void fp16_madd<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        const uint16_t* x,
        float w,
        float* c) {
    const __m128 wv = (__m128)__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(w));
    size_t j = 0;
    for (; j + 4 <= d; j += 4) {
        const __m128 xv = __lsx_vfcvtl_s_h(__lsx_vldrepl_d(x + j, 0));
        __lsx_vst(
                (__m128i)__lsx_vfadd_s(
                        (__m128)__lsx_vld(c + j, 0), __lsx_vfmul_s(xv, wv)),
                c + j,
                0);
    }
    for (; j < d; ++j) {
        c[j] += decode_fp16(x[j]) * w;
    }
}

template <>
void fp16_to_fp32_kernel<SIMDLevel::LOONGARCH_LSX>(
        size_t n,
        const uint16_t* x,
        float* out) {
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        __lsx_vst(
                (__m128i)__lsx_vfcvtl_s_h(__lsx_vldrepl_d(x + j, 0)),
                out + j,
                0);
    }
    for (; j < n; ++j) {
        out[j] = decode_fp16(x[j]);
    }
}

} // namespace detail
} // namespace faiss
