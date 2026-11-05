/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Definitions of the SIMDLevel-templatized accum_and_*_tab functions.
// Included by the per-ISA .cpp files (avx2.cpp, neon.cpp, rvv.cpp) and by
// the RVV benchmark/test TUs. Do NOT include this from other common
// translation units.
//
// Common TUs include rq_beam_search_tab.h (declarations only).

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/approx_topk/rq_beam_search_tab.h>
#include <faiss/impl/simdlib/simdlib.h>

namespace faiss {

// Native RVV kernel entry points (defined in impl/approx_topk/rvv.cpp).
// The primary templates below dispatch every SIMDLevel::RISCV_RVV
// instantiation to them; the RVV kernels strip-mine K with a runtime
// vsetvl and therefore do not depend on NK. Every other level keeps the
// generic simdlib body. Declared here so all translation units see one
// declaration; the rvv.cpp definitions are the only non-inline
// definitions.
#if defined(COMPILE_SIMD_RISCV_RVV)
template <size_t M>
void accum_tab_rvv_store(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output);

template <size_t M>
void accum_tab_rvv_add(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output);

template <size_t M>
void accum_tab_rvv_finalize(
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        const float* const __restrict distances_i,
        const float* const __restrict cd_common,
        float* const __restrict output);
#endif // COMPILE_SIMD_RISCV_RVV

template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_store_tab(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output) {
#if defined(COMPILE_SIMD_RISCV_RVV)
    if constexpr (SL == SIMDLevel::RISCV_RVV) {
        accum_tab_rvv_store<M>(
                m_offset,
                codebook_cross_norms,
                codebook_offsets,
                codes_i,
                b,
                ldc,
                K,
                output);
    } else
#endif
    {
        // The generic simdlib body is discarded for RISCV_RVV, so this TU
        // does not need a 256-bit RVV specialization.
        using simd_float = simd8float32_tpl<SL>;

        const float* cbs[M];
        for (size_t ij = 0; ij < M; ij++) {
            const size_t code = static_cast<size_t>(codes_i[b * m_offset + ij]);
            cbs[ij] =
                    &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
        }

        const size_t K8 = (K / (8 * NK)) * (8 * NK);

        for (size_t kk = 0; kk < K8; kk += 8 * NK) {
            simd_float regs[NK];
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] = simd_float(cbs[0] + kk + ik * 8);
            }

            for (size_t ij = 1; ij < M; ij++) {
                for (size_t ik = 0; ik < NK; ik++) {
                    regs[ik] += simd_float(cbs[ij] + kk + ik * 8);
                }
            }

            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik].storeu(output + kk + ik * 8);
            }
        }

        for (size_t kk = K8; kk < K; kk++) {
            float reg = cbs[0][kk];
            for (size_t ij = 1; ij < M; ij++) {
                reg += cbs[ij][kk];
            }
            output[kk] = reg;
        }
    }
}

template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_add_tab(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output) {
#if defined(COMPILE_SIMD_RISCV_RVV)
    if constexpr (SL == SIMDLevel::RISCV_RVV) {
        accum_tab_rvv_add<M>(
                m_offset,
                codebook_cross_norms,
                codebook_offsets,
                codes_i,
                b,
                ldc,
                K,
                output);
    } else
#endif
    {
        using simd_float = simd8float32_tpl<SL>;

        const float* cbs[M];
        for (size_t ij = 0; ij < M; ij++) {
            const size_t code = static_cast<size_t>(codes_i[b * m_offset + ij]);
            cbs[ij] =
                    &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
        }

        const size_t K8 = (K / (8 * NK)) * (8 * NK);

        for (size_t kk = 0; kk < K8; kk += 8 * NK) {
            simd_float regs[NK];
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] = simd_float(cbs[0] + kk + ik * 8);
            }

            for (size_t ij = 1; ij < M; ij++) {
                for (size_t ik = 0; ik < NK; ik++) {
                    regs[ik] += simd_float(cbs[ij] + kk + ik * 8);
                }
            }

            for (size_t ik = 0; ik < NK; ik++) {
                simd_float existing(output + kk + ik * 8);
                existing += regs[ik];
                existing.storeu(output + kk + ik * 8);
            }
        }

        for (size_t kk = K8; kk < K; kk++) {
            float reg = cbs[0][kk];
            for (size_t ij = 1; ij < M; ij++) {
                reg += cbs[ij][kk];
            }
            output[kk] += reg;
        }
    }
}

template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_finalize_tab(
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        const float* const __restrict distances_i,
        const float* const __restrict cd_common,
        float* const __restrict output) {
#if defined(COMPILE_SIMD_RISCV_RVV)
    if constexpr (SL == SIMDLevel::RISCV_RVV) {
        accum_tab_rvv_finalize<M>(
                codebook_cross_norms,
                codebook_offsets,
                codes_i,
                b,
                ldc,
                K,
                distances_i,
                cd_common,
                output);
    } else
#endif
    {
        using simd_float = simd8float32_tpl<SL>;

        const float* cbs[M];
        for (size_t ij = 0; ij < M; ij++) {
            const size_t code = static_cast<size_t>(codes_i[b * M + ij]);
            cbs[ij] =
                    &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
        }

        const size_t K8 = (K / (8 * NK)) * (8 * NK);

        for (size_t kk = 0; kk < K8; kk += 8 * NK) {
            simd_float regs[NK];
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] = simd_float(cbs[0] + kk + ik * 8);
            }

            for (size_t ij = 1; ij < M; ij++) {
                for (size_t ik = 0; ik < NK; ik++) {
                    regs[ik] += simd_float(cbs[ij] + kk + ik * 8);
                }
            }

            simd_float two(2.0f);
            for (size_t ik = 0; ik < NK; ik++) {
                simd_float common_v(cd_common + kk + ik * 8);
                common_v = fmadd(two, regs[ik], common_v);
                common_v += simd_float(distances_i[b]);
                common_v.storeu(output + b * K + kk + ik * 8);
            }
        }

        for (size_t kk = K8; kk < K; kk++) {
            float reg = cbs[0][kk];
            for (size_t ij = 1; ij < M; ij++) {
                reg += cbs[ij][kk];
            }
            // Keep the same operation order as the vector body above
            // (fmadd(two, regs, common) then += dist) so the scalar tail
            // and the vector blocks round identically. The native RVV
            // kernel evaluates the whole K with dynamic VL and relies on
            // this for bit-exact parity.
            output[b * K + kk] = (2 * reg + cd_common[kk]) + distances_i[b];
        }
    }
}

} // namespace faiss
