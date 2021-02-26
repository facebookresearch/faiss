/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

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

struct IndexIVFPQFastScan : IndexIVF {
    bool by_residual;    ///< Encode residual or plain vector?
    ProductQuantizer pq; ///< produces the codes

    // size of the kernel
    int bbs; // set at build time

    // M rounded up to a multiple of 2
    size_t M2;

    /// precomputed tables management
    int use_precomputed_table = 0;
    /// if use_precompute_table size (nlist, pq.M, pq.ksub)
    AlignedTable<float> precomputed_table;

    // search-time implementation
    int implem = 0;
    // skip some parts of the computation (for timing)
    int skip = 0;

    // batching factors at search time (0 = default)
    int qbs = 0;
    size_t qbs2 = 0;

    IndexIVFPQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexIVFPQFastScan();

    // built from an IndexIVFPQ
    explicit IndexIVFPQFastScan(const IndexIVFPQ& orig, int bbs = 32);

    /// orig's inverted lists (for debugging)
    InvertedLists* orig_invlists = nullptr;

    void train_residual(idx_t n, const float* x) override;

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

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    // prepare look-up tables

    void compute_LUT(
            size_t n,
            const float* x,
            const idx_t* coarse_ids,
            const float* coarse_dis,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases) const;

    void compute_LUT_uint8(
            size_t n,
            const float* x,
            const idx_t* coarse_ids,
            const float* coarse_dis,
            AlignedTable<uint8_t>& dis_tables,
            AlignedTable<uint16_t>& biases,
            float* normalizers) const;

    // internal search funcs

    template <bool is_max>
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    template <class C>
    void search_implem_1(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    template <class C>
    void search_implem_2(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    // implem 10 and 12 are not multithreaded internally, so
    // export search stats
    template <class C>
    void search_implem_10(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            size_t* ndis_out,
            size_t* nlist_out) const;

    template <class C>
    void search_implem_12(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            size_t* ndis_out,
            size_t* nlist_out) const;
};

struct IVFFastScanStats {
    uint64_t times[10];
    uint64_t t_compute_distance_tables, t_round;
    uint64_t t_copy_pack, t_scan, t_to_flat;
    uint64_t reservoir_times[4];

    double Mcy_at(int i) {
        return times[i] / (1000 * 1000.0);
    }

    double Mcy_reservoir_at(int i) {
        return reservoir_times[i] / (1000 * 1000.0);
    }
    IVFFastScanStats() {
        reset();
    }
    void reset() {
        memset(this, 0, sizeof(*this));
    }
};

FAISS_API extern IVFFastScanStats IVFFastScan_stats;

} // namespace faiss
