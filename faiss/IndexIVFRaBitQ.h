/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/Index.h>
#include <faiss/IndexIVF.h>

#include <faiss/impl/RaBitQuantizer.h>

namespace faiss {

struct IVFRaBitQSearchParameters : IVFSearchParameters {
    uint8_t qb = 0;
};

// * by_residual is true, just by design
struct IndexIVFRaBitQ : IndexIVF {
    RaBitQuantizer rabitq;

    // the default number of bits to quantize a query with.
    // use '0' to disable quantization and use raw fp32 values.
    uint8_t qb = 0;

    IndexIVFRaBitQ(
            Index* quantizer,
            const size_t d,
            const size_t nlist,
            MetricType metric = METRIC_L2);

    IndexIVFRaBitQ();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

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

    // unfortunately
    DistanceComputer* get_distance_computer() const override;
};

} // namespace faiss
