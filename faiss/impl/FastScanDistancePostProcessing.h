/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

namespace faiss {

/**
 * Norm table scaling for Additive Quantizer fast-scan operations.
 * This class holds the norm scale parameter used by factory functions
 * to select the appropriate SIMD scaler at runtime.
 *
 * For SIMD code paths, the templated DummyScaler<SL> and Scaler2x4bit<SL>
 * from LookupTableScaler.h are used internally by the handlers.
 */
struct NormTableScaler {
    /// Number of sub-quantizer indices used for norm scaling (typically 0 or 2)
    int nscale;

    explicit NormTableScaler(int nscale_in) : nscale(nscale_in) {}
    virtual ~NormTableScaler() = default;

    /// Scale a single distance value (for non-SIMD fallback paths)
    /// Default implementation applies the norm scale factor.
    virtual float scale_one(float x) const {
        // The scale factor is 2^(4*nscale) - 1, which is the maximum value
        // representable by nscale 4-bit indices.
        // For nscale=2: factor = 255 (0xFF)
        // For nscale=0: factor = 0 (no scaling, return x unchanged)
        if (nscale <= 0) {
            return x;
        }
        float factor = static_cast<float>((1 << (4 * nscale)) - 1);
        return x * factor;
    }
};

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

    /// The nprobe value used when allocating query_factors storage.
    /// This is needed because the allocation size (n * nprobe) may use a
    /// different nprobe than index->nprobe if search params override it.
    /// Set to 0 to use index->nprobe as fallback.
    size_t nprobe = 0;

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
