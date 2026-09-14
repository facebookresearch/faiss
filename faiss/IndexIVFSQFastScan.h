/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

/** Fast scan version of IndexIVFScalarQuantizer, for native 4-bit types only.
 *
 * Supports QT_4bit and QT_4bit_uniform: the 4-bit codes are packed directly
 * into the PQ4 FastScan SIMD block layout (vpshufb) and scanned within the
 * inverted lists.  This mirrors IndexIVFPQFastScan.
 *
 * For higher-precision scalar quantizers, wrap a 4-bit IndexIVFSQFastScan with
 * IndexRefine, or use IndexIVFScalarQuantizer directly.
 *
 * M = d subquantizers with 16 levels and uint16 SIMD accumulators.  Each LUT
 * entry is a uint8 in [0, 255], so the per-query accumulator d * 255 must fit
 * in a uint16; d is therefore limited to <= 257 (the constructors throw
 * otherwise).  by_residual is always false (the 2D LUT cannot handle per-probe
 * residuals, same rationale as IndexIVFPQFastScan).
 */
struct IndexIVFSQFastScan : IndexIVFFastScan {
    ScalarQuantizer sq;

    IndexIVFSQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexIVFSQFastScan();

    /// Build from an existing IndexIVFScalarQuantizer (must be QT_4bit*).
    explicit IndexIVFSQFastScan(
            const IndexIVFScalarQuantizer& orig,
            int bbs = 32);

    size_t fast_scan_code_size() const override;

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const override;

    bool lookup_table_is_3d() const override;

    void compute_LUT(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases,
            const FastScanDistancePostProcessing& context) const override;
};

} // namespace faiss
