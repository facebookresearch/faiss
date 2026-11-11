/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_impl/fp16_kernels.h>

#include <lasxintrin.h>
#include <bit>

namespace faiss {
namespace detail {

template <>
void fp16_madd<SIMDLevel::LOONGARCH_LASX>(
        size_t d,
        const uint16_t* x,
        float w,
        float* c) {
    const __m256 wv = (__m256)__lasx_xvreplgr2vr_w(std::bit_cast<int32_t>(w));
    size_t j = 0;
    for (; j + 8 <= d; j += 8) {
        // FCVTL reads four half values from each 128-bit lane.
        const __m256 xv = __lasx_xvfcvtl_s_h(__lasx_xvpermi_q(
                __lasx_xvldrepl_d(x + j, 8),
                __lasx_xvldrepl_d(x + j, 0),
                0x20));
        __lasx_xvst(
                (__m256i)__lasx_xvfadd_s(
                        (__m256)__lasx_xvld(c + j, 0), __lasx_xvfmul_s(xv, wv)),
                c + j,
                0);
    }
    for (; j < d; ++j) {
        c[j] += decode_fp16(x[j]) * w;
    }
}

template <>
void fp16_to_fp32_kernel<SIMDLevel::LOONGARCH_LASX>(
        size_t n,
        const uint16_t* x,
        float* out) {
    size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        __lasx_xvst(
                (__m256i)__lasx_xvfcvtl_s_h(__lasx_xvpermi_q(
                        __lasx_xvldrepl_d(x + j, 8),
                        __lasx_xvldrepl_d(x + j, 0),
                        0x20)),
                out + j,
                0);
    }
    for (; j < n; ++j) {
        out[j] = decode_fp16(x[j]);
    }
}

} // namespace detail
} // namespace faiss
