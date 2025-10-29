/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PANORAMA_H
#define FAISS_PANORAMA_H

#include <cstddef>
#include <cstdint>

namespace faiss {

struct Panorama {
    explicit Panorama(size_t code_size, size_t n_levels, size_t batch_size);

    void set_derived_values();

    /// Helper method to copy codes into level-oriented batch layout at a given
    /// offset in the list.
    void copy_codes_to_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code);

    /// Helper method to compute the cumulative sums of the codes.
    /// The cumsums also follow the level-oriented batch layout to minimize the
    /// number of random memory accesses.
    void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const uint8_t* code);

   private:
    size_t d;
    size_t code_size;
    size_t n_levels;
    size_t level_width;
    size_t batch_size;
};

} // namespace faiss

#endif
