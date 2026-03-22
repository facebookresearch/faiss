/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <immintrin.h>

#include <faiss/impl/panorama_kernels/panorama_kernels.h>

#include <cstring>

namespace faiss {
namespace panorama_kernels {

void process_chunks(
        size_t level_width_bytes,
        size_t max_batch_size,
        size_t num_active,
        float* sim_table,
        uint8_t* compressed_codes,
        float* exact_distances) {
    size_t byte_idx = 0;

    // Process 4 chunks at a time to amortize loop overhead and keep
    // the accumulator in registers across chunks.
    for (; byte_idx + 3 < level_width_bytes; byte_idx += 4) {
        size_t byte_offset0 = (byte_idx + 0) * max_batch_size;
        size_t byte_offset1 = (byte_idx + 1) * max_batch_size;
        size_t byte_offset2 = (byte_idx + 2) * max_batch_size;
        size_t byte_offset3 = (byte_idx + 3) * max_batch_size;

        float* sim_table0 = sim_table + (byte_idx + 0) * 256;
        float* sim_table1 = sim_table + (byte_idx + 1) * 256;
        float* sim_table2 = sim_table + (byte_idx + 2) * 256;
        float* sim_table3 = sim_table + (byte_idx + 3) * 256;

        size_t batch_idx = 0;
        for (; batch_idx + 15 < num_active; batch_idx += 16) {
            __m512 acc = _mm512_loadu_ps(exact_distances + batch_idx);

            __m128i comp0 = _mm_loadu_si128(
                    (__m128i*)(compressed_codes + byte_offset0 + batch_idx));
            __m512i codes0 = _mm512_cvtepu8_epi32(comp0);
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(codes0, sim_table0, sizeof(float)));

            __m128i comp1 = _mm_loadu_si128(
                    (__m128i*)(compressed_codes + byte_offset1 + batch_idx));
            __m512i codes1 = _mm512_cvtepu8_epi32(comp1);
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(codes1, sim_table1, sizeof(float)));

            __m128i comp2 = _mm_loadu_si128(
                    (__m128i*)(compressed_codes + byte_offset2 + batch_idx));
            __m512i codes2 = _mm512_cvtepu8_epi32(comp2);
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(codes2, sim_table2, sizeof(float)));

            __m128i comp3 = _mm_loadu_si128(
                    (__m128i*)(compressed_codes + byte_offset3 + batch_idx));
            __m512i codes3 = _mm512_cvtepu8_epi32(comp3);
            acc = _mm512_add_ps(
                    acc,
                    _mm512_i32gather_ps(codes3, sim_table3, sizeof(float)));

            _mm512_storeu_ps(exact_distances + batch_idx, acc);
        }

        for (; batch_idx < num_active; batch_idx += 1) {
            float acc = exact_distances[batch_idx];
            acc += sim_table0[compressed_codes[byte_offset0 + batch_idx]];
            acc += sim_table1[compressed_codes[byte_offset1 + batch_idx]];
            acc += sim_table2[compressed_codes[byte_offset2 + batch_idx]];
            acc += sim_table3[compressed_codes[byte_offset3 + batch_idx]];
            exact_distances[batch_idx] = acc;
        }
    }

    for (; byte_idx < level_width_bytes; byte_idx++) {
        size_t byte_offset = byte_idx * max_batch_size;
        float* sim_table_ptr = sim_table + byte_idx * 256;

        size_t batch_idx = 0;
        for (; batch_idx + 15 < num_active; batch_idx += 16) {
            __m512 acc = _mm512_loadu_ps(exact_distances + batch_idx);
            __m128i comp = _mm_loadu_si128(
                    (__m128i*)(compressed_codes + byte_offset + batch_idx));
            __m512i codes = _mm512_cvtepu8_epi32(comp);
            __m512 m_dist =
                    _mm512_i32gather_ps(codes, sim_table_ptr, sizeof(float));
            acc = _mm512_add_ps(acc, m_dist);
            _mm512_storeu_ps(exact_distances + batch_idx, acc);
        }

        for (; batch_idx < num_active; batch_idx += 1) {
            exact_distances[batch_idx] +=
                    sim_table_ptr[compressed_codes[byte_offset + batch_idx]];
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
        float heap_max) {
    size_t next_num_active = 0;
    for (size_t i = 0; i < num_active; i++) {
        float exact_distance = exact_distances[i];
        float cum_sum = cum_sums[active_indices[i] - batch_offset];
        float lower_bound = exact_distance + dis0 - cum_sum * query_cum_norm;

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
        size_t level_width_bytes,
        uint8_t* compressed_codes_begin,
        uint8_t* bitset,
        const uint8_t* codes) {
    uint8_t* compressed_codes = compressed_codes_begin;
    size_t num_active = 0;

    // An important optimization is to skip the compression if all points
    // are active, as we can just use the compressed_codes_begin pointer.
    if (next_num_active < max_batch_size) {
        // Compress the codes: here we don't need to process remainders
        // as long as `max_batch_size` is a multiple of 64 (which we
        // assert in the constructor). Conveniently, compressed_codes is
        // allocated to `max_batch_size` * `level_width_bytes` elements.
        // `num_active` is guaranteed to always be less than or equal to
        // `max_batch_size`. Only the last batch may be smaller than
        // `max_batch_size`, the caller ensures that the batch and
        // bitset are padded with zeros.
        compressed_codes = compressed_codes_begin;
        for (size_t point_idx = 0; point_idx < max_batch_size;
             point_idx += 64) {
            // Build a 64-bit mask from the byteset: each byte is
            // 0 or 1, collect into a single bitmask.
            uint64_t mask = 0;
#ifdef __BMI2__
            // PEXT path: extract the LSB of each byte into a
            // single bit, producing a 64-bit bitmask.
            for (int g = 0; g < 8; g++) {
                uint64_t bytes;
                memcpy(&bytes, bitset + point_idx + g * 8, 8);
                uint8_t bits = (uint8_t)_pext_u64(bytes, 0x0101010101010101ULL);
                mask |= ((uint64_t)bits << (g * 8));
            }
#else
            for (int b = 0; b < 64; b++) {
                if (bitset[point_idx + b])
                    mask |= (1ULL << b);
            }
#endif

            // Byte-level stream compaction (replaces
            // _mm512_maskz_compress_epi8 which requires VBMI2).
#ifdef __BMI2__
            // PEXT/PDEP path: process 8 bytes at a time. PDEP
            // expands the per-byte mask bits into a per-byte lane
            // mask, then PEXT extracts only the selected bytes.
            for (size_t ci = 0; ci < level_width_bytes; ci++) {
                size_t byte_offset = ci * max_batch_size;
                const uint8_t* src = codes + byte_offset + point_idx;
                uint8_t* dst = compressed_codes + byte_offset + num_active;
                int write_pos = 0;
                for (int g = 0; g < 8; g++) {
                    uint64_t src_val;
                    memcpy(&src_val, src + g * 8, 8);
                    uint8_t submask = (uint8_t)((mask >> (g * 8)) & 0xFF);
                    uint64_t byte_mask =
                            _pdep_u64(submask, 0x0101010101010101ULL) * 0xFF;
                    uint64_t compressed_val = _pext_u64(src_val, byte_mask);
                    int count = __builtin_popcount(submask);
                    memcpy(dst + write_pos, &compressed_val, 8);
                    write_pos += count;
                }
            }
#else
            // Scalar fallback: scan set bits one by one and copy
            // the corresponding code byte.
            for (size_t ci = 0; ci < level_width_bytes; ci++) {
                size_t byte_offset = ci * max_batch_size;
                const uint8_t* src = codes + byte_offset + point_idx;
                uint8_t* dst = compressed_codes + byte_offset + num_active;
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

#endif // COMPILE_SIMD_AVX512
