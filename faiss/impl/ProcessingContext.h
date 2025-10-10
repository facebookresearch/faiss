/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace faiss {

// Forward declarations
struct NormTableScaler;
struct QueryFactorsProcessor;

/**
 * Simple context object that holds processors for FastScan operations.
 * */
struct ProcessingContext {
    /// Norm scaling processor for Additive Quantizers (nullptr if not needed)
    const NormTableScaler* norm_scaler = nullptr;

    /// Query factors processor for RaBitQ (nullptr if not needed)
    QueryFactorsProcessor* query_factors = nullptr;

    /// Default constructor - no processing
    ProcessingContext() = default;

    /// Check if norm scaling is enabled
    bool has_norm_scaling() const {
        return norm_scaler != nullptr;
    }

    /// Check if query factors processing is enabled
    bool has_query_processing() const {
        return query_factors != nullptr;
    }
};

} // namespace faiss
