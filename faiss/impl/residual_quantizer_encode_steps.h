/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Index.h>
#include <faiss/utils/approx_topk/mode.h>

namespace faiss {

/********************************************************************
 * Single step of encoding
 ********************************************************************/

/** Encode a residual by sampling from a centroid table.
 *
 * This is a single encoding step the residual quantizer.
 * It allows low-level access to the encoding function, exposed mainly for unit
 * tests.
 *
 * @param n              number of vectors to handle
 * @param residuals      vectors to encode, size (n, beam_size, d)
 * @param cent           centroids, size (K, d)
 * @param beam_size      input beam size
 * @param m              size of the codes for the previous encoding steps
 * @param codes          code array for the previous steps of the beam (n,
 * beam_size, m)
 * @param new_beam_size  output beam size (should be <= K * beam_size)
 * @param new_codes      output codes, size (n, new_beam_size, m + 1)
 * @param new_residuals  output residuals, size (n, new_beam_size, d)
 * @param new_distances  output distances, size (n, new_beam_size)
 * @param assign_index   if non-NULL, will be used to perform assignment
 */
void beam_search_encode_step(
        size_t d,
        size_t K,
        const float* cent,
        size_t n,
        size_t beam_size,
        const float* residuals,
        size_t m,
        const int32_t* codes,
        size_t new_beam_size,
        int32_t* new_codes,
        float* new_residuals,
        float* new_distances,
        Index* assign_index = nullptr,
        ApproxTopK_mode_t approx_topk = ApproxTopK_mode_t::EXACT_TOPK);

/** Encode a set of vectors using their dot products with the codebooks
 *
 * @param K           number of vectors in the codebook
 * @param n           nb of vectors to encode
 * @param beam_size   input beam size
 * @param codebook_cross_norms inner product of this codebook with the m
 *                             previously encoded codebooks
 * @param codebook_offsets     offsets into codebook_cross_norms for each
 *                             previous codebook
 * @param query_cp    dot products of query vectors with ???
 * @param cent_norms_i  norms of centroids
 */
void beam_search_encode_step_tab(
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
        ApproxTopK_mode_t approx_topk = ApproxTopK_mode_t::EXACT_TOPK);

/********************************************************************
 * Multiple encoding steps
 *
 * The following functions take buffer objects that they use as temp
 * memory (allocated within the functions). The buffers are intended
 * to be re-used over batches of points to encode.
 ********************************************************************/

struct ResidualQuantizer;

namespace rq_encode_steps {

// Preallocated memory chunk for refine_beam_mp() call
struct RefineBeamMemoryPool {
    std::vector<int32_t> new_codes;
    std::vector<float> new_residuals;

    std::vector<float> residuals;
    std::vector<int32_t> codes;
    std::vector<float> distances;
};

void refine_beam_mp(
        const ResidualQuantizer& rq,
        size_t n,
        size_t beam_size,
        const float* x,
        int out_beam_size,
        int32_t* out_codes,
        float* out_residuals,
        float* out_distances,
        RefineBeamMemoryPool& pool);

// Preallocated memory chunk for refine_beam_LUT_mp() call
struct RefineBeamLUTMemoryPool {
    std::vector<int32_t> new_codes;
    std::vector<float> new_distances;

    std::vector<int32_t> codes;
    std::vector<float> distances;
};

void refine_beam_LUT_mp(
        const ResidualQuantizer& rq,
        size_t n,
        const float* query_norms, // size n
        const float* query_cp,    //
        int out_beam_size,
        int32_t* out_codes,
        float* out_distances,
        RefineBeamLUTMemoryPool& pool);

// this is for use_beam_LUT == 0 in compute_codes_add_centroids_mp_lut0() call
struct ComputeCodesAddCentroidsLUT0MemoryPool {
    std::vector<int32_t> codes;
    std::vector<float> norms;
    std::vector<float> distances;
    std::vector<float> residuals;
    RefineBeamMemoryPool refine_beam_pool;
};

void compute_codes_add_centroids_mp_lut0(
        const ResidualQuantizer& rq,
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids,
        ComputeCodesAddCentroidsLUT0MemoryPool& pool);

// this is for use_beam_LUT == 1 in compute_codes_add_centroids_mp_lut1() call
struct ComputeCodesAddCentroidsLUT1MemoryPool {
    std::vector<int32_t> codes;
    std::vector<float> distances;
    std::vector<float> query_norms;
    std::vector<float> query_cp;
    std::vector<float> residuals;
    RefineBeamLUTMemoryPool refine_beam_lut_pool;
};

void compute_codes_add_centroids_mp_lut1(
        const ResidualQuantizer& rq,
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids,
        ComputeCodesAddCentroidsLUT1MemoryPool& pool);

} // namespace rq_encode_steps

} // namespace faiss
