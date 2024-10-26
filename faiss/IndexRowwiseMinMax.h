/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/// Index wrapper that performs rowwise normalization to [0,1], preserving
/// the coefficients. This is a vector codec index only.
///
/// Basically, this index performs a rowwise scaling to [0,1] of every row
/// in an input dataset before calling subindex::train() and
/// subindex::sa_encode(). sa_encode() call stores the scaling coefficients
///  (scaler and minv) in the very beginning of every output code. The format:
///     [scaler][minv][subindex::sa_encode() output]
/// The de-scaling in sa_decode() is done using:
///     output_rescaled = scaler * output + minv
///
/// An additional ::train_inplace() function is provided in order to do
/// an inplace scaling before calling subindex::train() and, thus, avoiding
/// the cloning of the input dataset, but modifying the input dataset because
/// of the scaling and the scaling back. It is up to user to call
/// this function instead of ::train()
///
/// Derived classes provide different data types for scaling coefficients.
/// Currently, versions with fp16 and fp32 scaling coefficients are available.
/// * fp16 version adds 4 extra bytes per encoded vector
/// * fp32 version adds 8 extra bytes per encoded vector

/// Provides base functions for rowwise normalizing indices.
struct IndexRowwiseMinMaxBase : Index {
    /// sub-index
    Index* index;

    /// whether the subindex needs to be freed in the destructor.
    bool own_fields;

    explicit IndexRowwiseMinMaxBase(Index* index);

    IndexRowwiseMinMaxBase();
    ~IndexRowwiseMinMaxBase() override;

    void add(idx_t n, const float* x) override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    virtual void train_inplace(idx_t n, float* x) = 0;
};

/// Stores scaling coefficients as fp16 values.
struct IndexRowwiseMinMaxFP16 : IndexRowwiseMinMaxBase {
    explicit IndexRowwiseMinMaxFP16(Index* index);

    IndexRowwiseMinMaxFP16();

    void train(idx_t n, const float* x) override;
    void train_inplace(idx_t n, float* x) override;

    size_t sa_code_size() const override;
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

/// Stores scaling coefficients as fp32 values.
struct IndexRowwiseMinMax : IndexRowwiseMinMaxBase {
    explicit IndexRowwiseMinMax(Index* index);

    IndexRowwiseMinMax();

    void train(idx_t n, const float* x) override;
    void train_inplace(idx_t n, float* x) override;

    size_t sa_code_size() const override;
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

/// block size for performing sa_encode and sa_decode
FAISS_API extern int rowwise_minmax_sa_encode_bs;
FAISS_API extern int rowwise_minmax_sa_decode_bs;

} // namespace faiss
