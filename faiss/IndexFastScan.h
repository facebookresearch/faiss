/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct CodePacker;
struct NormTableScaler;

/** Fast scan version of IndexPQ and IndexAQ. Works for 4-bit PQ and AQ for now.
 *
 * The codes are not stored sequentially but grouped in blocks of size bbs.
 * This makes it possible to compute distances quickly with SIMD instructions.
 * The trailing codes (padding codes that are added to complete the last code)
 * are garbage.
 *
 * Implementations:
 * 12: blocked loop with internal loop on Q with qbs
 * 13: same with reservoir accumulator to store results
 * 14: no qbs with heap accumulator
 * 15: no qbs with reservoir accumulator
 */
struct IndexFastScan : Index {
    // implementation to select
    int implem = 0;
    // skip some parts of the computation (for timing)
    int skip = 0;

    // size of the kernel
    int bbs;     // set at build time
    int qbs = 0; // query block size 0 = use default

    // vector quantizer
    size_t M;
    size_t nbits;
    size_t ksub;
    size_t code_size;

    // packed version of the codes
    size_t ntotal2;
    size_t M2;

    AlignedTable<uint8_t> codes;

    // this is for testing purposes only
    // (set when initialized by IndexPQ or IndexAQ)
    const uint8_t* orig_codes = nullptr;

    void init_fastscan(
            int d,
            size_t M,
            size_t nbits,
            MetricType metric,
            int bbs);

    IndexFastScan();

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void add(idx_t n, const float* x) override;

    virtual void compute_codes(uint8_t* codes, idx_t n, const float* x)
            const = 0;

    virtual void compute_float_LUT(float* lut, idx_t n, const float* x)
            const = 0;

    // called by search function
    void compute_quantized_LUT(
            idx_t n,
            const float* x,
            uint8_t* lut,
            float* normalizers) const;

    template <bool is_max>
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const NormTableScaler* scaler) const;

    template <class Cfloat>
    void search_implem_234(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const NormTableScaler* scaler) const;

    template <class C>
    void search_implem_12(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            const NormTableScaler* scaler) const;

    template <class C>
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            const NormTableScaler* scaler) const;

    void reconstruct(idx_t key, float* recons) const override;
    size_t remove_ids(const IDSelector& sel) override;

    CodePacker* get_CodePacker() const;

    void merge_from(Index& otherIndex, idx_t add_id = 0) override;
    void check_compatible_for_merge(const Index& otherIndex) const override;

    /// standalone codes interface (but the codes are flattened)
    size_t sa_code_size() const override {
        return code_size;
    }

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override {
        compute_codes(bytes, n, x);
    }
};

struct FastScanStats {
    uint64_t t0, t1, t2, t3;
    FastScanStats() {
        reset();
    }
    void reset() {
        memset(this, 0, sizeof(*this));
    }
};

FAISS_API extern FastScanStats FastScan_stats;

} // namespace faiss
