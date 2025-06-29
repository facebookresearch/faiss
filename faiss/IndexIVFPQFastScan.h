/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

/** Fast scan version of IVFPQ. Works for 4-bit PQ for now.
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

struct IndexIVFPQFastScan : IndexIVFFastScan {
    ProductQuantizer pq; ///< produces the codes

    /// precomputed tables management
    int use_precomputed_table = 0;
    /// if use_precompute_table size (nlist, pq.M, pq.ksub)
    AlignedTable<float> precomputed_table;

    IndexIVFPQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            int bbs = 32,
            bool own_invlists = true);

    IndexIVFPQFastScan();

    // built from an IndexIVFPQ
    explicit IndexIVFPQFastScan(const IndexIVFPQ& orig, int bbs = 32);

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    /// build precomputed table, possibly updating use_precomputed_table
    void precompute_table();

    /// same as the regular IVFPQ encoder. The codes are not reorganized by
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
            const CoarseQuantized& cq,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases) const override;
};

} // namespace faiss
