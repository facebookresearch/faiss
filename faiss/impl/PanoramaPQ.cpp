/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/PanoramaPQ.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

void PanoramaPQ::copy_codes_to_level_layout(
        uint8_t* codes,
        size_t offset,
        size_t n_entry,
        const uint8_t* code) {
    const size_t cs = chunk_size;
    const size_t bs = batch_size;

    for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
        size_t current_pos = offset + entry_idx;
        size_t batch_no = current_pos / bs;
        size_t pos_in_batch = current_pos % bs;
        size_t batch_offset = batch_no * bs * code_size;

        for (size_t level = 0; level < n_levels; level++) {
            size_t level_offset = level * cs * bs;
            size_t start_byte = level * cs;

            for (size_t ci = 0; ci < cs && (start_byte + ci) < code_size;
                 ci++) {
                codes[batch_offset + level_offset + ci * bs + pos_in_batch] =
                        code[entry_idx * code_size + start_byte + ci];
            }
        }
    }
}

void PanoramaPQ::reconstruct(
        idx_t key,
        float* recons,
        const uint8_t* codes_base) const {
    uint8_t* recons_buffer = reinterpret_cast<uint8_t*>(recons);
    const size_t cs = chunk_size;
    const size_t bs = batch_size;

    size_t batch_no = key / bs;
    size_t pos_in_batch = key % bs;
    size_t batch_offset = batch_no * bs * code_size;

    for (size_t level = 0; level < n_levels; level++) {
        size_t level_offset = level * cs * bs;
        size_t start_byte = level * cs;

        for (size_t ci = 0; ci < cs && (start_byte + ci) < code_size; ci++) {
            recons_buffer[start_byte + ci] =
                    codes_base[batch_offset + level_offset + ci * bs +
                               pos_in_batch];
        }
    }
}

PanoramaPQ::PanoramaPQ(
        size_t d,
        size_t code_size,
        size_t n_levels,
        size_t batch_size,
        const ProductQuantizer* pq,
        const Index* quantizer)
        : Panorama(d, code_size, n_levels, batch_size),
          pq(pq),
          quantizer(quantizer),
          chunk_size(code_size / n_levels),
          levels_size(d / n_levels) {
    FAISS_THROW_IF_NOT_MSG(
            code_size % n_levels == 0,
            "PanoramaPQ: code_size must be divisible by n_levels");
    FAISS_THROW_IF_NOT_MSG(pq != nullptr, "PanoramaPQ: pq must not be null");
}

void PanoramaPQ::compute_cumulative_sums(
        float* cumsum_base,
        size_t offset,
        size_t n_entry,
        const uint8_t* code) const {
    for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
        size_t current_pos = offset + entry_idx;
        size_t batch_no = current_pos / batch_size;
        size_t pos_in_batch = current_pos % batch_size;

        // Decode PQ code to float vector.
        std::vector<float> vec(d);
        pq->decode(code + entry_idx * code_size, vec.data());

        // Compute suffix sums of squared norms.
        std::vector<float> suffix(d + 1, 0.0f);
        for (int j = d - 1; j >= 0; j--) {
            suffix[j] = suffix[j + 1] + vec[j] * vec[j];
        }

        // Write into batch-oriented layout.
        size_t cumsum_batch_offset = batch_no * batch_size * (n_levels + 1);
        for (size_t level = 0; level < n_levels; level++) {
            size_t start_idx = level * levels_size;
            size_t out_offset = cumsum_batch_offset + level * batch_size +
                    pos_in_batch;
            cumsum_base[out_offset] = start_idx < d
                    ? std::sqrt(suffix[start_idx])
                    : 0.0f;
        }

        size_t last_offset = cumsum_batch_offset + n_levels * batch_size +
                pos_in_batch;
        cumsum_base[last_offset] = 0.0f;
    }
}

void PanoramaPQ::compute_init_distances(
        float* init_dists_base,
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const uint8_t* code) const {
    FAISS_THROW_IF_NOT_MSG(
            quantizer != nullptr,
            "PanoramaPQ: quantizer required for compute_init_distances");

    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());

    for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
        std::vector<float> vec(d);
        pq->decode(code + entry_idx * code_size, vec.data());

        float init_dist = 0.0f;
        for (size_t j = 0; j < d; j++) {
            init_dist += vec[j] * vec[j] + 2 * vec[j] * centroid[j];
        }

        size_t point_idx = offset + entry_idx;
        init_dists_base[point_idx] = init_dist;
    }
}

} // namespace faiss
