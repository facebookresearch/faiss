/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/RaBitQStats.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/fast_scan/rabitq_result_handler.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/Heap.h>

namespace faiss {

// Forward declarations
struct FastScanDistancePostProcessing;

// Import shared utilities from RaBitQUtils
using rabitq_utils::QueryFactorsData;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

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

    // Constructors

    IndexIVFRaBitQFastScan();

    IndexIVFRaBitQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_L2,
            int bbs = 32,
            bool own_invlists = true,
            uint8_t nb_bits = 1);

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
    /// Return code_size as stride to skip embedded factor data during packing
    size_t code_packing_stride() const override;

   public:
    /// Return CodePackerRaBitQ with enlarged block size
    CodePacker* get_CodePacker() const override;

    /// Write per-vector auxiliary data into block auxiliary region
    void postprocess_packed_codes(
            idx_t list_no,
            size_t list_offset,
            size_t n_added,
            const uint8_t* flat_codes) override;

    /// Reconstruct a single vector from an inverted list
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /// Override sa_decode to handle RaBitQ reconstruction
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// Compute per-vector auxiliary storage size based on nb_bits
    size_t compute_per_vector_storage_size() const;

   private:
    /// Compute query factors and lookup table for a residual vector
    /// (similar to IndexRaBitQFastScan::compute_float_LUT)
    void compute_residual_LUT(
            const float* residual,
            QueryFactorsData& query_factors,
            float* lut_out,
            const float* original_query = nullptr) const;

    /// Decode FastScan code to RaBitQ residual vector with explicit
    /// dp_multiplier
    void decode_fastscan_to_residual(
            const uint8_t* fastscan_code,
            float* residual,
            float dp_multiplier) const;

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

    /// Get an InvertedListScanner for single-query scanning.
    /// This provides compatibility with the standard IVF search interface
    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr,
            const IVFSearchParameters* params = nullptr) const override;

    /// RaBitQ-specific result handler (defined in impl/fast_scan/)
    template <class C>
    using IVFRaBitQHeapHandler = simd_result_handlers::IVFRaBitQHeapHandler<C>;
};

} // namespace faiss
