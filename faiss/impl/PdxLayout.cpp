/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/PdxLayout.h>

#include <cstddef>
#include <cstring>

namespace faiss {
namespace detail {

void pdxify(
        const float* Y,
        int k,
        int d_trail,
        int pdx_block_size,
        float* Y_pdx) {
    const int n_full_blocks = d_trail / pdx_block_size;
    const int tail = d_trail % pdx_block_size;
    size_t offset = 0;
    for (int b = 0; b < n_full_blocks; ++b) {
        const size_t src_start = static_cast<size_t>(b) * pdx_block_size;
        for (int j = 0; j < k; ++j) {
            std::memcpy(
                    Y_pdx + offset,
                    Y + static_cast<size_t>(j) * d_trail + src_start,
                    pdx_block_size * sizeof(float));
            offset += pdx_block_size;
        }
    }
    if (tail > 0) {
        const size_t src_start =
                static_cast<size_t>(n_full_blocks) * pdx_block_size;
        for (int j = 0; j < k; ++j) {
            std::memcpy(
                    Y_pdx + offset,
                    Y + static_cast<size_t>(j) * d_trail + src_start,
                    tail * sizeof(float));
            offset += tail;
        }
    }
}

void de_pdxify(
        const float* Y_pdx,
        int k,
        int d_trail,
        int pdx_block_size,
        float* Y) {
    const int n_full_blocks = d_trail / pdx_block_size;
    const int tail = d_trail % pdx_block_size;
    size_t offset = 0;
    for (int b = 0; b < n_full_blocks; ++b) {
        const size_t dst_start = static_cast<size_t>(b) * pdx_block_size;
        for (int j = 0; j < k; ++j) {
            std::memcpy(
                    Y + static_cast<size_t>(j) * d_trail + dst_start,
                    Y_pdx + offset,
                    pdx_block_size * sizeof(float));
            offset += pdx_block_size;
        }
    }
    if (tail > 0) {
        const size_t dst_start =
                static_cast<size_t>(n_full_blocks) * pdx_block_size;
        for (int j = 0; j < k; ++j) {
            std::memcpy(
                    Y + static_cast<size_t>(j) * d_trail + dst_start,
                    Y_pdx + offset,
                    tail * sizeof(float));
            offset += tail;
        }
    }
}

void compute_partial_norms(const float* X, int n, int d, int p, float* norms) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        const float* row = X + static_cast<size_t>(i) * d;
        for (int m = 0; m < p; ++m) {
            s += row[m] * row[m];
        }
        norms[i] = s;
    }
}

} // namespace detail
} // namespace faiss
