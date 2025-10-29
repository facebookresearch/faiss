/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/Panorama.h>
#include <faiss/utils/distances.h>

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
    this->level_width = ((d + n_levels - 1) / n_levels) * sizeof(float);
    this->level_width_floats = this->level_width / sizeof(float);
}

void Panorama::copy_codes_to_level_layout(
        uint8_t* codes,
        size_t offset,
        size_t n_entry,
        const uint8_t* code) {
    size_t current_pos = offset;
    for (size_t entry_idx = 0; entry_idx < n_entry;) {
        // Determine which batch we're in and position within that batch.
        size_t batch_no = current_pos / batch_size;
        size_t pos_in_batch = current_pos % batch_size;
        size_t entries_in_this_batch =
                std::min(n_entry - entry_idx, batch_size - pos_in_batch);

        // Copy entries into level-oriented layout for this batch.
        size_t batch_offset = batch_no * batch_size * code_size;
        for (size_t level = 0; level < n_levels; level++) {
            size_t level_offset = level * level_width * batch_size;
            size_t start_byte = level * level_width;
            size_t copy_size =
                    std::min(level_width, code_size - level * level_width);

            for (size_t i = 0; i < entries_in_this_batch; i++) {
                const uint8_t* src =
                        code + (entry_idx + i) * code_size + start_byte;
                uint8_t* dest = codes + batch_offset + level_offset +
                        (pos_in_batch + i) * level_width;

                memcpy(dest, src, copy_size);
            }
        }

        entry_idx += entries_in_this_batch;
        current_pos += entries_in_this_batch;
    }
}

void Panorama::compute_cumulative_sums(
        float* cumsum_base,
        size_t offset,
        size_t n_entry,
        const uint8_t* code) {
    // Cast to float* is safe here as we guarantee codes are always float
    // vectors for `IndexIVFFlatPanorama` (verified by the constructor).
    const float* vectors = reinterpret_cast<const float*>(code);
    const size_t d = code_size / sizeof(float);

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

        const size_t level_width_floats = level_width / sizeof(float);
        for (size_t level = 0; level < n_levels; level++) {
            size_t start_idx = level * level_width_floats;
            size_t cumsum_offset =
                    cumsum_batch_offset + level * batch_size + pos_in_batch;
            if (start_idx < d) {
                cumsum_base[cumsum_offset] = sqrt(suffix_sums[start_idx]);
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

} // namespace faiss
