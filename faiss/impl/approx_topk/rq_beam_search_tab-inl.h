/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Definitions of the SIMDLevel-templatized accum_and_*_tab functions.
// Only included by per-ISA .cpp files (avx2.cpp, neon.cpp).
// Do NOT include this from common translation units.
//
// Common TUs include rq_beam_search_tab.h (declarations only).

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/approx_topk/rq_beam_search_tab.h>
#include <faiss/impl/simdlib/simdlib.h>

namespace faiss {

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
    using simd_float = simd8float32_tpl<SL>;

    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * m_offset + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
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
    using simd_float = simd8float32_tpl<SL>;

    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * m_offset + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
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
    using simd_float = simd8float32_tpl<SL>;

    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * M + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
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
        output[b * K + kk] = distances_i[b] + cd_common[kk] + 2 * reg;
    }
}

} // namespace faiss
