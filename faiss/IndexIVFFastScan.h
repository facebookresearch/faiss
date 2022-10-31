/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <faiss/IndexIVF.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

/** Fast scan version of IVFPQ and IVFAQ. Works for 4-bit PQ/AQ for now.
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

struct IndexIVFFastScan : IndexIVF {
    // size of the kernel
    int bbs; // set at build time

    size_t M;
    size_t nbits;
    size_t ksub;

    // M rounded up to a multiple of 2
    size_t M2;

    // search-time implementation
    int implem = 0;
    // skip some parts of the computation (for timing)
    int skip = 0;
    bool by_residual = false;

    // batching factors at search time (0 = default)
    int qbs = 0;
    size_t qbs2 = 0;

    IndexIVFFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t code_size,
            MetricType metric = METRIC_L2);

    IndexIVFFastScan();

    void init_fastscan(
            size_t M,
            size_t nbits,
            size_t nlist,
            MetricType metric,
            int bbs);

    ~IndexIVFFastScan() override;

    /// orig's inverted lists (for debugging)
    InvertedLists* orig_invlists = nullptr;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    // prepare look-up tables

    virtual bool lookup_table_is_3d() const = 0;

    virtual void compute_LUT(
            size_t n,
            const float* x,
            const idx_t* coarse_ids,
            const float* coarse_dis,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases) const = 0;

    void compute_LUT_uint8(
            size_t n,
            const float* x,
            const idx_t* coarse_ids,
            const float* coarse_dis,
            AlignedTable<uint8_t>& dis_tables,
            AlignedTable<uint16_t>& biases,
            float* normalizers) const;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// will just fail
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    // internal search funcs

    template <bool is_max, class Scaler>
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const Scaler& scaler) const;

    template <class C, class Scaler>
    void search_implem_1(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const Scaler& scaler) const;

    template <class C, class Scaler>
    void search_implem_2(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const Scaler& scaler) const;

    // implem 10 and 12 are not multithreaded internally, so
    // export search stats
    template <class C, class Scaler>
    void search_implem_10(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            size_t* ndis_out,
            size_t* nlist_out,
            const Scaler& scaler) const;

    template <class C, class Scaler>
    void search_implem_12(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            size_t* ndis_out,
            size_t* nlist_out,
            const Scaler& scaler) const;

    // implem 14 is mukltithreaded internally across nprobes and queries
    template <class C, class Scaler>
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            const Scaler& scaler) const;

    // reconstruct vectors from packed invlists
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    // reconstruct orig invlists (for debugging)
    void reconstruct_orig_invlists();
};

struct IVFFastScanStats {
    uint64_t times[10];
    uint64_t t_compute_distance_tables, t_round;
    uint64_t t_copy_pack, t_scan, t_to_flat;
    uint64_t reservoir_times[4];
    double t_aq_encode;
    double t_aq_norm_encode;

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
