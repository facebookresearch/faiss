/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <faiss/IndexIVF.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct NormTableScaler;
struct SIMDResultHandlerToFloat;
struct Quantizer;

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
 * 14: internally multithreaded implem over nq * nprobe
 * 15: same with reservoir
 *
 * For range search, only 10 and 12 are supported.
 * add 100 to the implem to force single-thread scanning (the coarse quantizer
 * may still use multiple threads).
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

    // batching factors at search time (0 = default)
    int qbs = 0;
    size_t qbs2 = 0;

    // quantizer used to pack the codes
    Quantizer* fine_quantizer = nullptr;

    IndexIVFFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t code_size,
            MetricType metric = METRIC_L2);

    IndexIVFFastScan();

    /// called by implementations
    void init_fastscan(
            Quantizer* fine_quantizer,
            size_t M,
            size_t nbits,
            size_t nlist,
            MetricType metric,
            int bbs);

    // initialize the CodePacker in the InvertedLists
    void init_code_packer();

    ~IndexIVFFastScan() override;

    /// orig's inverted lists (for debugging)
    InvertedLists* orig_invlists = nullptr;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    // prepare look-up tables

    virtual bool lookup_table_is_3d() const = 0;

    // compact way of conveying coarse quantization results
    struct CoarseQuantized {
        size_t nprobe;
        const float* dis = nullptr;
        const idx_t* ids = nullptr;
    };

    virtual void compute_LUT(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases) const = 0;

    void compute_LUT_uint8(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
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

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    // internal search funcs

    // dispatch to implementations and parallelize
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    void range_search_dispatch_implem(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult& rres,
            const CoarseQuantized& cq_in,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // impl 1 and 2 are just for verification
    template <class C>
    void search_implem_1(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    template <class C>
    void search_implem_2(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // implem 10 and 12 are not multithreaded internally, so
    // export search stats
    void search_implem_10(
            idx_t n,
            const float* x,
            SIMDResultHandlerToFloat& handler,
            const CoarseQuantized& cq,
            size_t* ndis_out,
            size_t* nlist_out,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    void search_implem_12(
            idx_t n,
            const float* x,
            SIMDResultHandlerToFloat& handler,
            const CoarseQuantized& cq,
            size_t* ndis_out,
            size_t* nlist_out,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // implem 14 is multithreaded internally across nprobes and queries
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            int impl,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // reconstruct vectors from packed invlists
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    CodePacker* get_CodePacker() const override;

    // reconstruct orig invlists (for debugging)
    void reconstruct_orig_invlists();

    /** Decode a set of vectors.
     *
     *  NOTE: The codes in the IndexFastScan object are non-contiguous.
     *        But this method requires a contiguous representation.
     *
     * @param n       number of vectors
     * @param bytes   input encoded vectors, size n * code_size
     * @param x       output vectors, size n * d
     */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
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
