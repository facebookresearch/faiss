/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Scalar implementations of Panorama kernels.
// Compiled only when no SIMD variant (AVX2/AVX-512) is available.

#if !defined(COMPILE_SIMD_AVX2) && !defined(COMPILE_SIMD_AVX512)

#include <faiss/impl/panorama_kernels/panorama_kernels.h>

#include <cstring>

#ifdef __BMI2__
#include <immintrin.h>
#endif

namespace faiss {
namespace panorama_kernels {

void process_chunks(
        size_t chunk_size,
        size_t max_batch_size,
        size_t num_active,
        float* sim_table,
        uint8_t* compressed_codes,
        float* exact_distances) {
    for (size_t chunk_idx = 0; chunk_idx < chunk_size; chunk_idx++) {
        size_t chunk_offset = chunk_idx * max_batch_size;
        float* chunk_sim = sim_table + chunk_idx * 256;
        for (size_t i = 0; i < num_active; i++) {
            exact_distances[i] +=
                    chunk_sim[compressed_codes[chunk_offset + i]];
        }
    }
}

size_t process_filtering(
        size_t num_active,
        float* exact_distances,
        uint32_t* active_indices,
        float* cum_sums,
        uint8_t* bitset,
        size_t batch_offset,
        float dis0,
        float query_cum_norm,
        float epsilon,
        float heap_max) {
    size_t next_num_active = 0;
    for (size_t i = 0; i < num_active; i++) {
        float exact_distance = exact_distances[i];
        float cum_sum = cum_sums[active_indices[i] - batch_offset];
        float lower_bound =
                exact_distance + dis0 - cum_sum * query_cum_norm * epsilon;

        bool keep = heap_max > lower_bound;
        active_indices[next_num_active] = active_indices[i];
        exact_distances[next_num_active] = exact_distance;
        bitset[active_indices[i] - batch_offset] = keep;
        next_num_active += keep;
    }
    return next_num_active;
}

std::pair<uint8_t*, size_t> process_code_compression(
        size_t next_num_active,
        size_t max_batch_size,
        size_t chunk_size,
        uint8_t* compressed_codes_begin,
        uint8_t* bitset,
        const uint8_t* codes) {
    uint8_t* compressed_codes = compressed_codes_begin;
    size_t num_active = 0;

    // An important optimization is to skip the compression if all points
    // are active, as we can just use the compressed_codes_begin pointer.
    if (next_num_active < max_batch_size) {
        compressed_codes = compressed_codes_begin;
        for (size_t point_idx = 0; point_idx < max_batch_size;
             point_idx += 64) {
            // Build a 64-bit mask from the byteset: each byte is
            // 0 or 1, collect into a single bitmask.
            uint64_t mask = 0;
#ifdef __BMI2__
            for (int g = 0; g < 8; g++) {
                uint64_t bytes;
                memcpy(&bytes, bitset + point_idx + g * 8, 8);
                uint8_t bits = (uint8_t)_pext_u64(
                        bytes, 0x0101010101010101ULL);
                mask |= ((uint64_t)bits << (g * 8));
            }
#else
            for (int b = 0; b < 64; b++) {
                if (bitset[point_idx + b])
                    mask |= (1ULL << b);
            }
#endif

            // Byte-level stream compaction.
#ifdef __BMI2__
            // PEXT/PDEP path: process 8 bytes at a time. PDEP
            // expands the per-byte mask bits into a per-byte lane
            // mask, then PEXT extracts only the selected bytes.
            for (size_t ci = 0; ci < chunk_size; ci++) {
                size_t chunk_offset = ci * max_batch_size;
                const uint8_t* src = codes + chunk_offset + point_idx;
                uint8_t* dst = compressed_codes + chunk_offset + num_active;
                int write_pos = 0;
                for (int g = 0; g < 8; g++) {
                    uint64_t src_val;
                    memcpy(&src_val, src + g * 8, 8);
                    uint8_t submask = (uint8_t)((mask >> (g * 8)) & 0xFF);
                    uint64_t byte_mask =
                            _pdep_u64(submask, 0x0101010101010101ULL) *
                            0xFF;
                    uint64_t compressed_val = _pext_u64(src_val, byte_mask);
                    int count = __builtin_popcount(submask);
                    memcpy(dst + write_pos, &compressed_val, 8);
                    write_pos += count;
                }
            }
#else
            // Scalar fallback: scan set bits one by one and copy
            // the corresponding code byte.
            for (size_t ci = 0; ci < chunk_size; ci++) {
                size_t chunk_offset = ci * max_batch_size;
                const uint8_t* src = codes + chunk_offset + point_idx;
                uint8_t* dst = compressed_codes + chunk_offset + num_active;
                int write_pos = 0;
                uint64_t m = mask;
                while (m) {
                    int bit = __builtin_ctzll(m);
                    dst[write_pos++] = src[bit];
                    m &= m - 1;
                }
            }
#endif

            num_active += __builtin_popcountll(mask);
        }
    } else {
        num_active = next_num_active;
        compressed_codes = const_cast<uint8_t*>(codes);
    }

    return std::make_pair(compressed_codes, num_active);
}

} // namespace panorama_kernels
} // namespace faiss

#endif // !COMPILE_SIMD_AVX2 && !COMPILE_SIMD_AVX512
