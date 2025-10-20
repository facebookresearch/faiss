/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/Heap.h>

namespace faiss {

// Forward declarations
struct FastScanDistancePostProcessing;

// Import shared utilities from RaBitQUtils
using rabitq_utils::FactorsData;
using rabitq_utils::QueryFactorsData;

struct IVFRaBitQFastScanSearchParameters : IVFSearchParameters {
    uint8_t qb = 0;
    bool centered = false;
};

/** Fast-scan version of IndexIVFRaBitQ that processes vectors in batches
 * using SIMD operations. Combines the inverted file structure of IVF
 * with RaBitQ's bit-level quantization and FastScan's batch processing.
 *
 * Key features:
 * - Inherits from IndexIVFFastScan for IVF structure and search algorithms
 * - Processes 32 database vectors at a time using SIMD
 * - Separates factors from quantized bits for efficient processing
 * - Supports both L2 and inner product metrics
 * - Maintains compatibility with existing IVF search parameters
 *
 * Implementation details:
 * - Batch size (bbs) is typically 32 for optimal SIMD performance
 * - Factors are stored separately from packed codes for cache efficiency
 * - Query factors are computed once per search and reused across lists
 * - Uses specialized result handlers for RaBitQ distance corrections
 */
struct IndexIVFRaBitQFastScan : IndexIVFFastScan {
    RaBitQuantizer rabitq;

    /// Default number of bits to quantize a query with
    uint8_t qb = 8;

    /// Use zero-centered scalar quantizer for queries
    bool centered = false;

    /// Extracted factors storage for batch processing
    /// Size: ntotal, stores factors separately from packed codes
    std::vector<FactorsData> factors_storage;

    // Constructors

    IndexIVFRaBitQFastScan();

    IndexIVFRaBitQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_L2,
            int bbs = 32,
            bool own_invlists = true);

    /// Build from an existing IndexIVFRaBitQ
    explicit IndexIVFRaBitQFastScan(const IndexIVFRaBitQ& orig, int bbs = 32);

    // Required overrides

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

   protected:
    /// Extract and store RaBitQ factors from encoded vectors
    void preprocess_code_metadata(
            idx_t n,
            const uint8_t* flat_codes,
            idx_t start_global_idx) override;

    /// Return code_size as stride to skip embedded factor data during packing
    size_t code_packing_stride() const override;

   public:
    /// Reconstruct a single vector from an inverted list
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /// Override sa_decode to handle RaBitQ reconstruction
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

   private:
    /// Encode a vector to FastScan format without computing factors
    void encode_vector_to_fastscan(
            const float* xi,
            const float* centroid,
            uint8_t* fastscan_code) const;

    /// Compute query factors and lookup table for a residual vector
    /// (similar to IndexRaBitQFastScan::compute_float_LUT)
    void compute_residual_LUT(
            const float* residual,
            QueryFactorsData& query_factors,
            float* lut_out,
            const float* original_query = nullptr) const;

    /// Decode FastScan code to RaBitQ residual vector
    void decode_fastscan_to_residual(
            const uint8_t* fastscan_code,
            float* residual) const;

   public:
    /// Implementation methods for IVFRaBitQFastScan specialization
    bool lookup_table_is_3d() const override;

    void compute_LUT(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases,
            const FastScanDistancePostProcessing& context) const override;

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

    /// Override to create RaBitQ-specific handlers
    SIMDResultHandlerToFloat* make_knn_handler(
            bool is_max,
            int /* impl */,
            idx_t n,
            idx_t k,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context,
            const float* normalizers = nullptr) const override;

    /** SIMD result handler for IndexIVFRaBitQFastScan that applies
     * RaBitQ-specific distance corrections during batch processing.
     *
     * This handler processes batches of 32 distance computations from SIMD
     * kernels, applies RaBitQ distance formula adjustments (factors and
     * normalizers), and immediately updates result heaps. This eliminates the
     * need for post-processing and provides significant performance benefits.
     *
     * Key optimizations:
     * - Direct heap integration with no intermediate result storage
     * - Batch-level computation of normalizers and query factors
     * - Specialized handling for both centered and non-centered quantization
     * modes
     * - Efficient inner product metric corrections
     *
     * @tparam C Comparator type (CMin/CMax) for heap operations
     */
    template <class C>
    struct IVFRaBitQHeapHandler
            : simd_result_handlers::ResultHandlerCompare<C, true> {
        const IndexIVFRaBitQFastScan* index;
        float* heap_distances; // [nq * k]
        int64_t* heap_labels;  // [nq * k]
        const size_t nq, k;
        size_t current_list_no = 0;
        std::vector<int>
                probe_indices; // probe index for each query in current batch
        const FastScanDistancePostProcessing*
                context; // Processing context with query factors

        // Use float-based comparator for heap operations
        using Cfloat = typename std::conditional<
                C::is_max,
                CMax<float, int64_t>,
                CMin<float, int64_t>>::type;

        IVFRaBitQHeapHandler(
                const IndexIVFRaBitQFastScan* idx,
                size_t nq_val,
                size_t k_val,
                float* distances,
                int64_t* labels,
                const FastScanDistancePostProcessing* ctx = nullptr);

        void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final;

        /// Override base class virtual method to receive context information
        void set_list_context(size_t list_no, const std::vector<int>& probe_map)
                override;

        void begin(const float* norms) override;

        void end() override;
    };
};

} // namespace faiss
