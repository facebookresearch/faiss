/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>
#include <cstddef>

namespace faiss {

/// Statistics for RaBitQ multi-bit two-stage search.
///
/// These stats are ONLY collected for multi-bit mode (nb_bits > 1).
/// In 1-bit mode, there is no two-stage filtering - all candidates are
/// evaluated with a single distance computation, so there is nothing
/// meaningful to track. For 1-bit mode, both counters remain 0.
///
/// Multi-bit mode uses a two-stage search:
///   Stage 1: Compute 1-bit lower bound distance for all candidates
///   Stage 2: Compute full multi-bit distance only for promising candidates
///
/// The skip_percentage() metric measures filtering effectiveness:
/// how many candidates were filtered out by the 1-bit lower bound
/// without needing the more expensive multi-bit distance computation.
///
/// WARNING: Statistics are not robust to internal threading nor to
/// concurrent RaBitQ searches. Use these values in a single-threaded
/// context to accurately gauge RaBitQ's filtering effectiveness.
/// Call reset() before search, then read stats after search completes.
struct RaBitQStats {
    /// Number of candidates evaluated using 1-bit (lower bound) distance.
    /// This is the first stage of two-stage search in multi-bit mode.
    /// Always 0 in 1-bit mode (stats not tracked).
    size_t n_1bit_evaluations = 0;

    /// Number of candidates that passed 1-bit filtering and required
    /// full multi-bit distance computation (second stage).
    /// Always 0 in 1-bit mode (stats not tracked).
    size_t n_multibit_evaluations = 0;

    void reset();

    /// Compute percentage of candidates skipped (filtered out by 1-bit stage).
    /// Returns 0 if no candidates were evaluated (including 1-bit mode).
    double skip_percentage() const;
};

/// Global stats for RaBitQ indexes
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
FAISS_API extern RaBitQStats rabitq_stats;

} // namespace faiss
