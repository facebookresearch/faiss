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

namespace rabitq_utils {
struct QueryFactorsData;
}

/**
 * Simple context object that holds processors for FastScan operations.
 * */
struct FastScanDistancePostProcessing {
    /// Norm scaling processor for Additive Quantizers (nullptr if not needed)
    const NormTableScaler* norm_scaler = nullptr;

    /// Query factors data pointer for RaBitQ (nullptr if not needed)
    /// This pointer should point to the beginning of the relevant
    /// QueryFactorsData subset for this context.
    rabitq_utils::QueryFactorsData* query_factors = nullptr;

    /// Default constructor - no processing
    FastScanDistancePostProcessing() = default;

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
