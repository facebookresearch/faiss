/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/impl/FastScanDistancePostProcessing.h>
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

    /** Constructor for IndexIVFFastScan
     *
     * @param quantizer     coarse quantizer for IVF clustering
     * @param d             dimensionality of vectors
     * @param nlist         number of inverted lists
     * @param code_size     size of each code in bytes
     * @param metric        distance metric to use
     * @param own_invlists  whether to own the inverted lists
     */
    IndexIVFFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t code_size,
            MetricType metric = METRIC_L2,
            bool own_invlists = true);

    IndexIVFFastScan();

    /** Initialize the fast scan functionality (called by implementations)
     *
     * @param fine_quantizer  fine quantizer for encoding
     * @param M               number of subquantizers
     * @param nbits           number of bits per subquantizer
     * @param nlist           number of inverted lists
     * @param metric          distance metric to use
     * @param bbs             block size for SIMD processing
     * @param own_invlists    whether to own the inverted lists
     */
    void init_fastscan(
            Quantizer* fine_quantizer,
            size_t M,
            size_t nbits,
            size_t nlist,
            MetricType metric,
            int bbs,
            bool own_invlists);

    // initialize the CodePacker in the InvertedLists
    void init_code_packer();

    ~IndexIVFFastScan() override;

    /// orig's inverted lists (for debugging)
    InvertedLists* orig_invlists = nullptr;

    /** Add vectors with specific IDs to the index
     *
     * @param n     number of vectors to add
     * @param x     vectors to add (n * d)
     * @param xids  IDs for the vectors (n)
     */
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
    // prepare look-up tables

    virtual bool lookup_table_is_3d() const = 0;

    // compact way of conveying coarse quantization results
    struct CoarseQuantized {
        size_t nprobe = 0;
        const float* dis = nullptr;
        const idx_t* ids = nullptr;
    };

    /* Compute distance table for query set, given a list of coarse
     * quantizers.
     *
     * @param n             number of queries
     * @param x             query vectors (n, d)
     * @param cq            coarse quantization results
     * @param dis_tables    output distance tables
     * @param biases        output bias values
     * @param context       processing context containing query factors
    processor
     */
    virtual void compute_LUT(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases,
            const FastScanDistancePostProcessing& context) const = 0;

    /** Compute quantized lookup tables for distance computation
     *
     * @param n             number of query vectors
     * @param x             query vectors (n * d)
     * @param cq            coarse quantization results
     * @param dis_tables    output quantized distance tables
     * @param biases        output quantized bias values
     * @param normalizers   output normalization factors
     * @param context       processing context containing query factors
     * processor
     */
    void compute_LUT_uint8(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<uint8_t>& dis_tables,
            AlignedTable<uint16_t>& biases,
            float* normalizers,
            const FastScanDistancePostProcessing& context) const;

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

    /** Search with pre-assigned coarse quantization
     *
     * @param n             number of query vectors
     * @param x             query vectors (n * d)
     * @param k             number of nearest neighbors to find
     * @param assign        coarse cluster assignments (n * nprobe)
     * @param centroid_dis  distances to centroids (n * nprobe)
     * @param distances     output distances (n * k)
     * @param labels        output labels/indices (n * k)
     * @param store_pairs   whether to store cluster-relative pairs
     * @param params        optional IVF search parameters
     * @param stats         optional search statistics
     */
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

    /** Range search for all neighbors within radius
     *
     * @param n       number of query vectors
     * @param x       query vectors (n * d)
     * @param radius  search radius
     * @param result  output range search results
     * @param params  optional search parameters
     */
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    /** Create a KNN handler for this index type
     *
     * This method can be overridden by derived classes to provide
     * specialized handlers (e.g., IVFRaBitQHeapHandler for RaBitQ indexes).
     * Base implementation creates standard handlers based on k and impl.
     *
     * @param is_max        true for max-heap (inner product), false for
     *                      min-heap (L2 distance)
     * @param impl          implementation number:
     *                      - even (10, 12, 14): use heap for top-k
     *                      - odd (11, 13, 15): use reservoir sampling
     * @param n             number of queries
     * @param k             number of neighbors to find per query
     * @param distances     output array for distances (n * k), will be
     *                      populated by handler
     * @param labels        output array for result IDs (n * k), will be
     *                      populated by handler
     * @param sel           optional ID selector to filter results (nullptr =
     *                      no filtering)
     * @param context       processing context containing additional data
     * @param normalizers   optional array of size 2*n for converting quantized
     *                      uint16 distances to float.
     *
     * @return Allocated result handler (caller owns and must delete).
     *         Handler processes SIMD batches and populates distances/labels.
     *
     * @note The returned handler must be deleted by caller after use.
     *       Typical usage: handler->begin() → process batches → handler->end()
     */
    virtual SIMDResultHandlerToFloat* make_knn_handler(
            bool is_max,
            int impl,
            idx_t n,
            idx_t k,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context,
            const float* normalizers = nullptr) const;

    // dispatch to implementations and parallelize
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const FastScanDistancePostProcessing& context,
            const IVFSearchParameters* params = nullptr) const;

    void range_search_dispatch_implem(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult& rres,
            const CoarseQuantized& cq_in,
            const FastScanDistancePostProcessing& context,
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
            const FastScanDistancePostProcessing& context,
            const IVFSearchParameters* params = nullptr) const;

    template <class C>
    void search_implem_2(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const FastScanDistancePostProcessing& context,
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
            const FastScanDistancePostProcessing& context,
            const IVFSearchParameters* params = nullptr) const;

    void search_implem_12(
            idx_t n,
            const float* x,
            SIMDResultHandlerToFloat& handler,
            const CoarseQuantized& cq,
            size_t* ndis_out,
            size_t* nlist_out,
            const FastScanDistancePostProcessing& context,
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
            const FastScanDistancePostProcessing& context,
            const IVFSearchParameters* params = nullptr) const;

    // reconstruct vectors from packed invlists
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    CodePacker* get_CodePacker() const override;

    // reconstruct orig invlists (for debugging)
    void reconstruct_orig_invlists();

    /** Decode a set of vectors
     *
     * NOTE: The codes in the IndexFastScan object are non-contiguous.
     *       But this method requires a contiguous representation.
     *
     * @param n       number of vectors
     * @param bytes   input encoded vectors, size n * code_size
     * @param x       output vectors, size n * d
     */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

   protected:
    /** Preprocess metadata from encoded vectors before packing.
     *
     * Called during add_with_ids after encode_vectors but before codes
     * are packed into SIMD-friendly blocks. Subclasses can override to
     * extract and store metadata embedded in codes or perform other
     * pre-packing operations.
     *
     * Default implementation: no-op
     *
     * Example use case:
     * - IndexIVFRaBitQFastScan extracts factor data from codes for use
     *   during search-time distance corrections
     *
     * @param n                  number of vectors encoded
     * @param flat_codes         encoded vectors (n * code_size bytes)
     * @param start_global_idx   starting global index (ntotal before add)
     */
    virtual void preprocess_code_metadata(
            idx_t n,
            const uint8_t* flat_codes,
            idx_t start_global_idx);

    /** Get stride for interpreting codes during SIMD packing.
     *
     * The stride determines how to read codes when packing them into
     * SIMD-friendly block format. This is needed when codes contain
     * embedded metadata that should be skipped during packing.
     *
     * Default implementation: returns 0 (use standard M-byte stride)
     *
     * Example use case:
     * - IndexIVFRaBitQFastScan returns code_size because codes contain
     *   embedded factor data after the quantized bits
     *
     * @return stride in bytes:
     *         - 0: use default stride (M bytes, standard PQ/AQ codes)
     *         - >0: use custom stride (e.g., code_size for embedded metadata)
     */
    virtual size_t code_packing_stride() const;
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
