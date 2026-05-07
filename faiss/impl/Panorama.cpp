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

#include <faiss/impl/FaissAssert.h>

namespace faiss {

namespace {

/// Helper function to compute cumulative sums by iterating backwards through
/// levels. This is the core logic shared by compute_cumulative_sums and
/// compute_query_cum_sums.
template <typename OffsetFunc>
inline void compute_cum_sums_impl(
        const float* vector,
        float* output,
        size_t d,
        size_t n_levels,
        size_t level_width_floats,
        OffsetFunc&& get_offset) {
    // Iterate backwards through levels, accumulating sum as we go.
    // This avoids computing the suffix sum for each vector, which takes
    // extra memory.
    float sum = 0.0f;

    for (int level = n_levels - 1; level >= 0; level--) {
        size_t start_idx = level * level_width_floats;
        size_t end_idx = std::min(
                (level + 1) * level_width_floats, static_cast<size_t>(d));

        for (size_t j = start_idx; j < end_idx; j++) {
            sum += vector[j] * vector[j];
        }

        output[get_offset(level)] = std::sqrt(sum);
    }

    output[get_offset(n_levels)] = 0.0f;
}

} // namespace

/**************************************************************
 * Panorama structure implementation
 **************************************************************/

Panorama::Panorama(
        size_t code_size_in,
        size_t n_levels_in,
        size_t batch_size_in,
        bool inline_layout_in)
        : code_size(code_size_in),
          n_levels(n_levels_in),
          batch_size(batch_size_in),
          inline_layout(inline_layout_in) {
    set_derived_values();
}

void Panorama::set_derived_values() {
    FAISS_THROW_IF_NOT_MSG(n_levels > 0, "Panorama: n_levels must be > 0");
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
    // Inline mode prepends a per-batch cum-sums prefix; chunked mode
    // starts the feature region at the batch boundary. `batch_byte_offset`
    // and `feat_region_byte_offset` collapse the two cases.
    for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
        size_t current_pos = offset + entry_idx;

        size_t batch_no = current_pos / batch_size;
        size_t pos_in_batch = current_pos % batch_size;

        uint8_t* batch_feats = codes + batch_byte_offset(batch_no) +
                feat_region_byte_offset();

        for (size_t level = 0; level < n_levels; level++) {
            size_t level_offset = level * level_width * batch_size;
            size_t start_byte = level * level_width;
            size_t actual_level_width =
                    std::min(level_width, code_size - level * level_width);

            const uint8_t* src = code + entry_idx * code_size + start_byte;
            uint8_t* dest = batch_feats + level_offset +
                    pos_in_batch * actual_level_width;

            memcpy(dest, src, actual_level_width);
        }
    }
}

void Panorama::compute_cumulative_sums(
        float* cumsum_base,
        size_t offset,
        size_t n_entry,
        const float* vectors) const {
    // Per-batch cum-sums shape is identical for both layouts:
    //   [cs[0]_0..B-1 | cs[1]_0..B-1 | ... | cs[L]_0..B-1]
    // BUT the per-batch *stride* in the destination buffer differs:
    //   chunked: cum_sums is a separate vector packed tightly at
    //     stride cs_floats_per_batch() floats per batch.
    //   inline: the cum-sums prefix sits at the head of each batch in
    //     `codes`, but each batch in `codes` is `inline_batch_bytes()`
    //     long (cum-sums prefix + feature region), so the stride
    //     between batch starts is the larger inline batch width.
    const size_t batch_stride_floats = inline_layout
            ? (inline_batch_bytes() / sizeof(float))
            : cs_floats_per_batch();
    for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
        size_t current_pos = offset + entry_idx;
        size_t batch_no = current_pos / batch_size;
        size_t pos_in_batch = current_pos % batch_size;

        const float* vector = vectors + entry_idx * d;
        size_t cumsum_batch_offset = batch_no * batch_stride_floats;

        auto get_offset = [&](size_t level) {
            return cumsum_batch_offset + level * batch_size + pos_in_batch;
        };

        compute_cum_sums_impl(
                vector,
                cumsum_base,
                d,
                n_levels,
                level_width_floats,
                get_offset);
    }
}

void Panorama::compute_query_cum_sums(const float* query, float* query_cum_sums)
        const {
    auto get_offset = [](size_t level) { return level; };
    compute_cum_sums_impl(
            query, query_cum_sums, d, n_levels, level_width_floats, get_offset);
}

void Panorama::reconstruct(idx_t key, float* recons, const uint8_t* codes_base)
        const {
    uint8_t* recons_buffer = reinterpret_cast<uint8_t*>(recons);

    size_t batch_no = static_cast<size_t>(key) / batch_size;
    size_t pos_in_batch = static_cast<size_t>(key) % batch_size;
    const uint8_t* batch_feats = codes_base + batch_byte_offset(batch_no) +
            feat_region_byte_offset();

    for (size_t level = 0; level < n_levels; level++) {
        size_t level_offset = level * level_width * batch_size;
        const uint8_t* src =
                batch_feats + level_offset + pos_in_batch * level_width;
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

    // Per-batch byte/float offsets.
    //   feat region: chunked starts at the batch boundary; inline sits
    //     past the per-batch cum-sums prefix inside `*_codes`.
    //   cum_sums: chunked is packed tightly (stride cs_floats_per_batch
    //     floats per batch); inline sits at the head of each batch in
    //     `*_codes` so the float* base stride is inline_batch_bytes /
    //     sizeof(float).
    const uint8_t* src_feats = src_codes + batch_byte_offset(src_batch_no) +
            feat_region_byte_offset();
    uint8_t* dest_feats = dest_codes + batch_byte_offset(dest_batch_no) +
            feat_region_byte_offset();
    const size_t cs_batch_stride = inline_layout
            ? (inline_batch_bytes() / sizeof(float))
            : cs_floats_per_batch();
    const size_t src_cumsum_batch_offset = src_batch_no * cs_batch_stride;
    const size_t dest_cumsum_batch_offset = dest_batch_no * cs_batch_stride;

    for (size_t level = 0; level < n_levels; level++) {
        // Copy code
        size_t level_offset = level * level_width * batch_size;
        size_t actual_level_width =
                std::min(level_width, code_size - level * level_width);

        const uint8_t* src = src_feats + level_offset +
                src_pos_in_batch * actual_level_width;
        uint8_t* dest = dest_feats + level_offset +
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
