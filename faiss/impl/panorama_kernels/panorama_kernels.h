/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file panorama_kernels.h
 * @brief Panorama search kernels with SIMD-dispatched implementations.
 *
 * The three core kernels of the Panorama progressive filtering search:
 * - process_level: accumulate PQ distance table lookups over chunks
 * - process_filtering: Cauchy-Schwarz lower bound pruning with stream
 *   compaction
 * - process_code_compression: byte-level stream compaction of PQ codes
 */

#include <cstddef>
#include <cstdint>
#include <utility>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {
namespace panorama_kernels {

template <SIMDLevel SL>
void process_level_impl(
        size_t level_width_bytes,
        size_t max_batch_size,
        size_t num_active,
        float* sim_table,
        uint8_t* compressed_codes,
        float* exact_distances);

template <SIMDLevel SL>
std::pair<uint8_t*, size_t> process_code_compression_impl(
        size_t next_num_active,
        size_t max_batch_size,
        size_t level_width_bytes,
        uint8_t* compressed_codes_begin,
        uint8_t* bitset,
        const uint8_t* codes);

/// Accumulate PQ distance table lookups over chunks.
///
/// For each chunk, looks up `sim_table[compressed_codes[i]]` and
/// accumulates into `exact_distances[i]` for all active elements.
/// Iterates chunks first to keep the LUT slice in L1 cache.
/// The AVX2/AVX-512 versions unroll 4 chunks at a time.
FAISS_API void process_level(
        size_t level_width_bytes,
        size_t max_batch_size,
        size_t num_active,
        float* sim_table,
        uint8_t* compressed_codes,
        float* exact_distances);

/// Filter active elements using Cauchy-Schwarz lower bound pruning.
///
/// Computes a lower bound on the true distance for each active element
/// and removes elements that cannot improve the current heap top.
/// Uses stream compaction to pack surviving elements contiguously.
/// Updates the bitset to reflect which elements were removed.
FAISS_API size_t process_filtering(
        size_t num_active,
        float* exact_distances,
        uint32_t* active_indices,
        float* cum_sums,
        uint8_t* bitset,
        size_t batch_offset,
        float dis0,
        float query_cum_norm,
        float heap_max);

/// Byte-level stream compaction of PQ codes using the active bitset.
///
/// An important optimization is to skip the compression if all points
/// are active, as we can just use the original codes pointer.
///
/// Compress the codes: here we don't need to process remainders
/// as long as `max_batch_size` is a multiple of 64 (which we
/// assert in the constructor). Conveniently, compressed_codes is
/// allocated to `max_batch_size` * `level_width_bytes` elements.
/// `num_active` is guaranteed to always be less than or equal to
/// `max_batch_size`. Only the last batch may be smaller than
/// `max_batch_size`, the caller ensures that the batch and
/// bitset are padded with zeros.
FAISS_API std::pair<uint8_t*, size_t> process_code_compression(
        size_t next_num_active,
        size_t max_batch_size,
        size_t level_width_bytes,
        uint8_t* compressed_codes_begin,
        uint8_t* bitset,
        const uint8_t* codes);

} // namespace panorama_kernels
} // namespace faiss
