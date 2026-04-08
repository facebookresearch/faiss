/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// @file rq_beam_search_tab.h
/// @brief Declarations for SIMDLevel-templatized codebook accumulation
/// functions.
///
/// These functions accumulate codebook cross-norm tables for beam search
/// encoding in the Residual Quantizer. They compute the distance
/// contributions from previously encoded codebooks using SIMD-accelerated
/// register accumulation.
///
/// Definitions are in rq_beam_search_tab-inl.h (only included by per-ISA
/// .cpp files). The common TU only sees these declarations, so no extern
/// template suppression is needed — the linker resolves to the explicit
/// instantiations in avx2.cpp / neon.cpp.

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/utils/simd_levels.h>

namespace faiss {

/// Accumulate cross-norms for M codebooks and store the result.
///
/// Loads M codebook rows (selected by codes_i) and sums them using
/// NK×8-wide SIMD chunks, writing the result to output. Used to
/// initialize the temporary buffer in the m≥8 path.
///
/// @tparam M      number of codebook rows to accumulate
/// @tparam NK     number of 8-float SIMD chunks per loop iteration
/// @tparam SL     SIMD level (AVX2, ARM_NEON, etc.)
/// @param m_offset  stride between beam entries in codes_i
/// @param codebook_cross_norms  cross-norm table, shape (total_codes, ldc)
/// @param codebook_offsets      per-codebook offset into cross-norm table
/// @param codes_i               code indices for the current query
/// @param b                     beam index
/// @param ldc                   leading dimension of cross-norm table (≥ K)
/// @param K                     number of centroids in the current codebook
/// @param output                output buffer, size K (overwritten)
template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_store_tab(
        size_t m_offset,
        const float* __restrict codebook_cross_norms,
        const uint64_t* __restrict codebook_offsets,
        const int32_t* __restrict codes_i,
        size_t b,
        size_t ldc,
        size_t K,
        float* __restrict output);

/// Accumulate cross-norms for M codebooks and add to existing output.
///
/// Like accum_and_store_tab, but adds the accumulated result to the
/// existing values in output (output[k] += sum). Used for subsequent
/// chunks of 8 codebooks in the m≥8 path.
///
/// @tparam M      number of codebook rows to accumulate
/// @tparam NK     number of 8-float SIMD chunks per loop iteration
/// @tparam SL     SIMD level (AVX2, ARM_NEON, etc.)
/// @param m_offset  stride between beam entries in codes_i
/// @param codebook_cross_norms  cross-norm table
/// @param codebook_offsets      per-codebook offset
/// @param codes_i               code indices
/// @param b                     beam index
/// @param ldc                   leading dimension of cross-norm table
/// @param K                     number of centroids
/// @param output                output buffer, size K (accumulated into)
template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_add_tab(
        size_t m_offset,
        const float* __restrict codebook_cross_norms,
        const uint64_t* __restrict codebook_offsets,
        const int32_t* __restrict codes_i,
        size_t b,
        size_t ldc,
        size_t K,
        float* __restrict output);

/// Accumulate cross-norms for M codebooks and finalize distances.
///
/// Accumulates M codebook rows, then computes the final centroid distance:
///   output[b*K + k] = distances_i[b] + cd_common[k] + 2 * sum[k]
/// Used for m=1..7 where the entire accumulation fits in registers.
///
/// @tparam M      number of codebook rows to accumulate (equals m)
/// @tparam NK     number of 8-float SIMD chunks per loop iteration
/// @tparam SL     SIMD level (AVX2, ARM_NEON, etc.)
/// @param codebook_cross_norms  cross-norm table
/// @param codebook_offsets      per-codebook offset
/// @param codes_i               code indices (stride is M)
/// @param b                     beam index
/// @param ldc                   leading dimension of cross-norm table
/// @param K                     number of centroids
/// @param distances_i           per-beam input distances, size beam_size
/// @param cd_common             common distance term, size K
/// @param output                output centroid distances (b*K offset)
template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_finalize_tab(
        const float* __restrict codebook_cross_norms,
        const uint64_t* __restrict codebook_offsets,
        const int32_t* __restrict codes_i,
        size_t b,
        size_t ldc,
        size_t K,
        const float* __restrict distances_i,
        const float* __restrict cd_common,
        float* __restrict output);

} // namespace faiss
