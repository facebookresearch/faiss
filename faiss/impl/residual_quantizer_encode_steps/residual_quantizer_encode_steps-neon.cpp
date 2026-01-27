/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#ifdef COMPILE_SIMD_NEON

#include <faiss/impl/residual_quantizer_encode_steps/residual_quantizer_encode_steps.h>

namespace faiss {

template void beam_search_encode_step_tab<SIMDLevel::ARM_NEON>(
        size_t K,
        size_t n,
        size_t beam_size,                  // input sizes
        const float* codebook_cross_norms, // size K * ldc
        size_t ldc,                        // >= K
        const uint64_t* codebook_offsets,  // m
        const float* query_cp,             // size n * ldqc
        size_t ldqc,                       // >= K
        const float* cent_norms_i,         // size K
        size_t m,
        const int32_t* codes,   // n * beam_size * m
        const float* distances, // n * beam_size
        size_t new_beam_size,
        int32_t* new_codes,   // n * new_beam_size * (m + 1)
        float* new_distances, // n * new_beam_size
        ApproxTopK_mode_t approx_topk_mode = ApproxTopK_mode_t::EXACT_TOPK);

namespace rq_encode_steps {

// use_beam_LUT == 1
template void compute_codes_add_centroids_mp_lut1<SIMDLevel::ARM_NEON>(
        const ResidualQuantizer& rq,
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids,
        ComputeCodesAddCentroidsLUT1MemoryPool& pool);

template void refine_beam_LUT_mp<SIMDLevel::ARM_NEON>(
        const ResidualQuantizer& rq,
        size_t n,
        const float* query_norms, // size n
        const float* query_cp,    //
        int out_beam_size,
        int32_t* out_codes,
        float* out_distances,
        RefineBeamLUTMemoryPool& pool);

} // namespace rq_encode_steps

} // namespace faiss

#endif // COMPILE_SIMD_NEON
