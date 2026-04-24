/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IndexIVFTurboQ: IVF index with TurboQuant fine quantizer.
//
// Key difference from RaBitQ: by_residual = false.
// TurboQ operates on the raw rotated vector, not the IVF residual,
// because its Lloyd-Max codebook is calibrated for unit-sphere coordinates,
// not for residual distributions which vary per centroid.

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/TurboQuantizer.h>

namespace faiss {

struct IVFTurboQSearchParameters : IVFSearchParameters {
    /// Query quantization bits for integer MSE pre-screening.
    /// 0 = float path (default), 1-8 = integer popcount path.
    uint8_t qb = 0;

    /// Also use integer popcount for QJL stage (requires qb > 0).
    bool int_qjl = false;
};

struct IndexIVFTurboQ : IndexIVF {
    TurboQuantizer turboq;

    IndexIVFTurboQ(
            Index* quantizer,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_L2,
            bool own_invlists = true,
            uint8_t nb_bits = 2,
            QJLProjectionType qjl_type = QJLProjectionType::FWHT,
            uint8_t nb_bits_lo = 0,
            size_t n_hi_dims = 0);

    IndexIVFTurboQ();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override {
        return 1;
    }

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
