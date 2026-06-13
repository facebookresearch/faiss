/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/IndexIVF.h>
#include <faiss/impl/EDENQuantizer.h>

namespace faiss {

struct IndexIVFEDEN : IndexIVF {
    EDENQuantizer eden;

    // Factory strings: IVF<nlist>,EDEN, IVF<nlist>,EDEN<n>, and the same
    // forms with the BIASED suffix for EDEN's MSE-minimizing scale.
    IndexIVFEDEN(
            Index* quantizer,
            const size_t d,
            const size_t nlist,
            MetricType metric = METRIC_L2,
            bool own_invlists = true,
            uint8_t nb_bits = 1,
            EDENScaleType scale_type = EDENScaleType_UNBIASED);

    IndexIVFEDEN();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    DistanceComputer* get_distance_computer() const override;
};

} // namespace faiss
