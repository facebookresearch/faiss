/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/Panorama.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace faiss {

/**************************************************************
 * Panorama structure implementation
 **************************************************************/

Panorama::Panorama(size_t code_size, size_t n_levels, size_t batch_size)
        : code_size(code_size), n_levels(n_levels), batch_size(batch_size) {
    set_derived_values();
}

void Panorama::set_derived_values() {
    this->d = code_size / sizeof(float);
    this->level_width_floats = ((d + n_levels - 1) / n_levels);
    this->level_width = this->level_width_floats * sizeof(float);
}

/**
 * @brief Copy codes to level-oriented layout
 * @param codes The base pointer to codes
 * @param offset Where to start writing new data (in number of vectors)
 * @param n_entry The number of new vectors to write
 * @param code The new vector data
 */
void Panorama::copy_codes_to_level_layout(
        uint8_t* codes,
        size_t offset,
        size_t n_entry,
        const uint8_t* code) {
    for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
        size_t current_pos = offset + entry_idx;

        // Determine which batch we're in and position within that batch.
        size_t batch_no = current_pos / batch_size;
        size_t pos_in_batch = current_pos % batch_size;

        // Copy entry into level-oriented layout for this batch.
        size_t batch_offset = batch_no * batch_size * code_size;
        for (size_t level = 0; level < n_levels; level++) {
            size_t level_offset = level * level_width * batch_size;
            size_t start_byte = level * level_width;
            size_t actual_level_width =
                    std::min(level_width, code_size - level * level_width);

            const uint8_t* src = code + entry_idx * code_size + start_byte;
            uint8_t* dest = codes + batch_offset + level_offset +
                    pos_in_batch * actual_level_width;

            memcpy(dest, src, actual_level_width);
        }
    }
}

void Panorama::compute_cumulative_sums(
        float* cumsum_base,
        size_t offset,
        size_t n_entry,
        const float* vectors) {
    std::vector<float> suffix_sums(d + 1);

    for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
        size_t current_pos = offset + entry_idx;
        size_t batch_no = current_pos / batch_size;
        size_t pos_in_batch = current_pos % batch_size;

        const float* vector = vectors + entry_idx * d;

        // Compute suffix sums of squared values.
        suffix_sums[d] = 0.0f;
        for (int j = d - 1; j >= 0; j--) {
            float squared_val = vector[j] * vector[j];
            suffix_sums[j] = suffix_sums[j + 1] + squared_val;
        }

        // Store cumulative sums in batch-oriented layout.
        size_t cumsum_batch_offset = batch_no * batch_size * (n_levels + 1);

        for (size_t level = 0; level < n_levels; level++) {
            size_t start_idx = level * level_width_floats;
            size_t cumsum_offset =
                    cumsum_batch_offset + level * batch_size + pos_in_batch;
            if (start_idx < d) {
                cumsum_base[cumsum_offset] = std::sqrt(suffix_sums[start_idx]);
            } else {
                cumsum_base[cumsum_offset] = 0.0f;
            }
        }

        // Last level sum is always 0.
        size_t cumsum_offset =
                cumsum_batch_offset + n_levels * batch_size + pos_in_batch;
        cumsum_base[cumsum_offset] = 0.0f;
    }
}

void Panorama::compute_query_cum_sums(const float* query, float* query_cum_sums)
        const {
    std::vector<float> suffix_sums(d + 1);
    suffix_sums[d] = 0.0f;

    for (int j = d - 1; j >= 0; j--) {
        float squared_val = query[j] * query[j];
        suffix_sums[j] = suffix_sums[j + 1] + squared_val;
    }

    for (size_t level = 0; level < n_levels; level++) {
        size_t start_idx = level * level_width_floats;
        if (start_idx < d) {
            query_cum_sums[level] = std::sqrt(suffix_sums[start_idx]);
        } else {
            query_cum_sums[level] = 0.0f;
        }
    }

    query_cum_sums[n_levels] = 0.0f;
}

void Panorama::reconstruct(idx_t key, float* recons, const uint8_t* codes_base)
        const {
    uint8_t* recons_buffer = reinterpret_cast<uint8_t*>(recons);

    size_t batch_no = key / batch_size;
    size_t pos_in_batch = key % batch_size;
    size_t batch_offset = batch_no * batch_size * code_size;

    for (size_t level = 0; level < n_levels; level++) {
        size_t level_offset = level * level_width * batch_size;
        const uint8_t* src = codes_base + batch_offset + level_offset +
                pos_in_batch * level_width;
        uint8_t* dest = recons_buffer + level * level_width;
        size_t copy_size =
                std::min(level_width, code_size - level * level_width);
        memcpy(dest, src, copy_size);
    }
}

void Panorama::copy_entry(
        uint8_t* dest_codes,
        uint8_t* src_codes,
        float* dest_cum_sums,
        float* src_cum_sums,
        size_t dest_idx,
        size_t src_idx) const {
    // Calculate positions
    size_t src_batch_no = src_idx / batch_size;
    size_t src_pos_in_batch = src_idx % batch_size;
    size_t dest_batch_no = dest_idx / batch_size;
    size_t dest_pos_in_batch = dest_idx % batch_size;

    // Calculate offsets
    size_t src_batch_offset = src_batch_no * batch_size * code_size;
    size_t dest_batch_offset = dest_batch_no * batch_size * code_size;
    size_t src_cumsum_batch_offset = src_batch_no * batch_size * (n_levels + 1);
    size_t dest_cumsum_batch_offset =
            dest_batch_no * batch_size * (n_levels + 1);

    for (size_t level = 0; level < n_levels; level++) {
        // Copy code
        size_t level_offset = level * level_width * batch_size;
        size_t actual_level_width =
                std::min(level_width, code_size - level * level_width);

        const uint8_t* src = src_codes + src_batch_offset + level_offset +
                src_pos_in_batch * actual_level_width;
        uint8_t* dest = dest_codes + dest_batch_offset + level_offset +
                dest_pos_in_batch * actual_level_width;
        memcpy(dest, src, actual_level_width);

        // Copy cum_sums
        size_t cumsum_level_offset = level * batch_size;

        const size_t src_offset = src_cumsum_batch_offset +
                cumsum_level_offset + src_pos_in_batch;
        size_t dest_offset = dest_cumsum_batch_offset + cumsum_level_offset +
                dest_pos_in_batch;
        dest_cum_sums[dest_offset] = src_cum_sums[src_offset];
    }
}
} // namespace faiss
