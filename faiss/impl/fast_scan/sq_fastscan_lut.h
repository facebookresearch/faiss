/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

/* Shared distance-LUT helpers for the scalar-quantizer FastScan indexes
 * (IndexSQFastScan and IndexIVFSQFastScan).  Both map SQ codes onto the PQ4
 * FastScan SIMD layout with M = d subquantizers and ksub = 16 levels, so the
 * per-dimension reconstruction table and the query LUT are built identically.
 */

namespace sq_fastscan {

constexpr size_t ksub = 16;

/// True for SQ types whose reconstruction range is a single (vmin, vdiff)
/// pair shared by all dimensions rather than a per-dimension range.
inline bool is_uniform_range(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed ||
            qtype == ScalarQuantizer::QT_4bit_uniform;
}

/// Resolve the scalar (vmin, vdiff) range for a uniform SQ type.
inline void get_uniform_range(
        const ScalarQuantizer& sq,
        float& vmin,
        float& vdiff) {
    if (sq.qtype == ScalarQuantizer::QT_8bit_direct) {
        vmin = 0;
        vdiff = 255;
    } else if (sq.qtype == ScalarQuantizer::QT_8bit_direct_signed) {
        vmin = -128;
        vdiff = 255;
    } else {
        vmin = sq.trained[0];
        vdiff = sq.trained[1];
    }
}

/// Build the dim x 16 reconstruction table: for every dimension, the 16
/// centroid values the 4-bit codes map back to.
inline void build_recon_table(
        const ScalarQuantizer& sq,
        size_t dim,
        std::vector<float>& recon_table) {
    recon_table.resize(dim * ksub);
    if (is_uniform_range(sq.qtype)) {
        float vmin, vdiff;
        get_uniform_range(sq, vmin, vdiff);
        for (size_t c = 0; c < ksub; c++) {
            float recon = vmin + ((c + 0.5f) / 15.0f) * vdiff;
            for (size_t m = 0; m < dim; m++) {
                recon_table[m * ksub + c] = recon;
            }
        }
    } else {
        // Per-dimension ranges (e.g. QT_4bit, QT_8bit).
        const float* vmin = sq.trained.data();
        const float* vdiff = sq.trained.data() + dim;
        for (size_t m = 0; m < dim; m++) {
            for (size_t c = 0; c < ksub; c++) {
                recon_table[m * ksub + c] =
                        vmin[m] + ((c + 0.5f) / 15.0f) * vdiff[m];
            }
        }
    }
}

/// Fill an n x (dim x 16) distance LUT from a reconstruction table.  For L2
/// each entry is the squared difference; for inner product it is the product.
/// by_residual must be false, so the LUT is computed directly from the raw
/// query and no biases are needed.
inline void fill_lut(
        const float* recon_table,
        const float* x,
        float* lut,
        size_t n,
        size_t dim,
        MetricType metric_type) {
    const size_t dim12 = dim * ksub;
    if (metric_type == METRIC_L2) {
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = lut + i * dim12;
            for (size_t m = 0; m < dim; m++) {
                float qi = xi[m];
                const float* recon_m = recon_table + m * ksub;
                float* lut_m = lut_i + m * ksub;
                for (size_t c = 0; c < ksub; c++) {
                    float diff = qi - recon_m[c];
                    lut_m[c] = diff * diff;
                }
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = lut + i * dim12;
            for (size_t m = 0; m < dim; m++) {
                float qi = xi[m];
                const float* recon_m = recon_table + m * ksub;
                float* lut_m = lut_i + m * ksub;
                for (size_t c = 0; c < ksub; c++) {
                    lut_m[c] = qi * recon_m[c];
                }
            }
        }
    }
}

} // namespace sq_fastscan

} // namespace faiss
