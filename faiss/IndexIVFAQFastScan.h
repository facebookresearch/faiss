/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFFastScan.h>
#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

/** Fast scan version of IVFAQ. Works for 4-bit AQ for now.
 *
 * The codes in the inverted lists are not stored sequentially but
 * grouped in blocks of size bbs. This makes it possible to very quickly
 * compute distances with SIMD instructions.
 *
 * Implementations (implem):
 * 0: auto-select implementation (default)
 * 1: orig's search, re-implemented
 * 2: orig's search, re-ordered by invlist
 * 10: optimizer int16 search, collect results in heap, no qbs
 * 11: idem, collect results in reservoir
 * 12: optimizer int16 search, collect results in heap, uses qbs
 * 13: idem, collect results in reservoir
 */

struct IndexIVFAQFastScan : IndexIVFFastScan {
    using Search_type_t = AdditiveQuantizer::Search_type_t;

    AdditiveQuantizer* aq;

    // max number of training vectors
    size_t max_train_points;

    IndexIVFAQFastScan(
            Index* quantizer,
            AdditiveQuantizer* aq,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    void init(AdditiveQuantizer* aq, size_t nlist, MetricType metric, int bbs);

    IndexIVFAQFastScan();

    ~IndexIVFAQFastScan();

    // built from an IndexIVFAQ
    explicit IndexIVFAQFastScan(
            const IndexIVFAdditiveQuantizer& orig,
            int bbs = 32);

    void train_residual(idx_t n, const float* x) override;

    /// same as the regular IVFAQ encoder. The codes are not reorganized by
    /// blocks a that point
    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const override;

    // prepare look-up tables

    bool lookup_table_is_3d() const override;

    void compute_LUT(
            size_t n,
            const float* x,
            const idx_t* coarse_ids,
            const float* coarse_dis,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases) const;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

struct IndexIVFLSQFastScan : IndexIVFAQFastScan {
    LocalSearchQuantizer lsq;

    IndexIVFLSQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_norm_lsq2x4,
            int bbs = 32);

    IndexIVFLSQFastScan();

    ~IndexIVFLSQFastScan();
};

struct IndexIVFRQFastScan : IndexIVFAQFastScan {
    ResidualQuantizer rq;

    IndexIVFRQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_norm_lsq2x4,
            int bbs = 32);

    IndexIVFRQFastScan();

    ~IndexIVFRQFastScan();
};

} // namespace faiss