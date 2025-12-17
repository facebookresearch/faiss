/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/FastScanDistancePostProcessing.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct CodePacker;
struct NormTableScaler;
struct IDSelector;
struct SIMDResultHandlerToFloat;

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

    /** Initialize the fast scan index
     *
     * @param d         dimensionality of vectors
     * @param M         number of subquantizers
     * @param nbits     number of bits per subquantizer
     * @param metric    distance metric to use
     * @param bbs       block size for SIMD processing
     */
    void init_fastscan(
            int d,
            size_t M,
            size_t nbits,
            MetricType metric,
            int bbs);

    IndexFastScan();

    void reset() override;

    /** Search for k nearest neighbors
     *
     * @param n          number of query vectors
     * @param x          query vectors (n * d)
     * @param k          number of nearest neighbors to find
     * @param distances  output distances (n * k)
     * @param labels     output labels/indices (n * k)
     * @param params     optional search parameters
     */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /** Add vectors to the index
     *
     * @param n  number of vectors to add
     * @param x  vectors to add (n * d)
     */
    void add(idx_t n, const float* x) override;

    /** Compute codes for vectors
     *
     * @param codes  output codes
     * @param n      number of vectors to encode
     * @param x      vectors to encode (n * d)
     */
    virtual void compute_codes(uint8_t* codes, idx_t n, const float* x)
            const = 0;

    /** Compute floating-point lookup table for distance computation
     *
     * @param lut          output lookup table
     * @param n            number of query vectors
     * @param x            query vectors (n * d)
     * @param context      processing context containing all processors
     */
    virtual void compute_float_LUT(
            float* lut,
            idx_t n,
            const float* x,
            const FastScanDistancePostProcessing& context) const = 0;

    /** Create a KNN handler for this index type
     *
     * This method can be overridden by derived classes to provide
     * specialized handlers (e.g., RaBitQHeapHandler for RaBitQ indexes).
     * Base implementation creates standard handlers based on k and impl.
     *
     * @param is_max       whether to use CMax comparator (true) or CMin (false)
     * @param impl         implementation number
     * @param n            number of queries
     * @param k            number of neighbors to find
     * @param ntotal       total number of vectors in database
     * @param distances    output distances array
     * @param labels       output labels array
     * @param sel          optional ID selector
     * @param context      processing context for distance post-processing
     * @return             pointer to created handler (never returns nullptr)
     */
    virtual SIMDResultHandlerToFloat* make_knn_handler(
            bool is_max,
            int impl,
            idx_t n,
            idx_t k,
            size_t ntotal,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context) const;

    // called by search function
    void compute_quantized_LUT(
            idx_t n,
            const float* x,
            uint8_t* lut,
            float* normalizers,
            const FastScanDistancePostProcessing& context) const;

    template <bool is_max>
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const FastScanDistancePostProcessing& context) const;

    template <class Cfloat>
    void search_implem_234(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const FastScanDistancePostProcessing& context) const;

    template <class C>
    void search_implem_12(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            const FastScanDistancePostProcessing& context) const;

    template <class C>
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            const FastScanDistancePostProcessing& context) const;

    /** Reconstruct a vector from its code
     *
     * @param key     index of vector to reconstruct
     * @param recons  output reconstructed vector
     */
    void reconstruct(idx_t key, float* recons) const override;

    /** Remove vectors by ID selector
     *
     * @param sel  selector defining which vectors to remove
     * @return     number of vectors removed
     */
    size_t remove_ids(const IDSelector& sel) override;

    /** Get the code packer for this index
     *
     * @return  pointer to the code packer
     */
    CodePacker* get_CodePacker() const;

    /** Merge another index into this one
     *
     * @param otherIndex  index to merge from
     * @param add_id      ID offset to add to merged vectors
     */
    void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    /** Check if another index is compatible for merging
     *
     * @param otherIndex  index to check compatibility with
     */
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
