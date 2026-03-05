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
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/RaBitQStats.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/simd_result_handlers.h>
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

    /// RaBitQ uses custom handlers; scanner support pending.
    std::unique_ptr<PQ4CodeScanner> make_knn_scanner(
            bool is_max,
            idx_t n,
            idx_t k,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context) const override;

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
     * - Uses runtime boolean for multi-bit mode
     *
     * @tparam C Comparator type (CMin/CMax) for heap operations
     */
    template <class C, SIMDLevel SL = SINGLE_SIMD_LEVEL_256>
    struct IVFRaBitQHeapHandler
            : simd_result_handlers::ResultHandlerCompare<C, true, SL> {
        using SIMDResultHandler::handle;
        static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
        const IndexIVFRaBitQFastScan* index;
        float* heap_distances; // [nq * k]
        int64_t* heap_labels;  // [nq * k]
        const size_t nq, k;
        size_t current_list_no = 0;
        const uint8_t* list_codes_ptr = nullptr; // raw block data for list
        std::vector<int>
                probe_indices; // probe index for each query in current batch
        const FastScanDistancePostProcessing*
                context;        // Processing context with query factors
        const bool is_multibit; // Whether to use multi-bit two-stage search
        size_t nup = 0;         // Number of heap updates

        // Cached block-layout constants (invariant for handler lifetime)
        const size_t storage_size;
        const size_t packed_block_size;
        const size_t full_block_size;
        std::unique_ptr<CodePacker> packer; // cached for unpack in hot path

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
                const FastScanDistancePostProcessing* ctx = nullptr,
                bool multibit = false)
                : simd_result_handlers::ResultHandlerCompare<C, true, SL>(
                          nq_val,
                          0,
                          nullptr),
                  index(idx),
                  heap_distances(distances),
                  heap_labels(labels),
                  nq(nq_val),
                  k(k_val),
                  context(ctx),
                  is_multibit(multibit),
                  storage_size(idx->compute_per_vector_storage_size()),
                  packed_block_size(((idx->M2 + 1) / 2) * idx->bbs),
                  full_block_size(idx->get_block_stride()),
                  packer(idx->get_CodePacker()) {
            for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
                heap_heapify<Cfloat>(
                        k, heap_distances + q * k, heap_labels + q * k);
            }
        }

        void handle(
                size_t q,
                size_t b,
                simd16uint16<SL256> d0,
                simd16uint16<SL256> d1) {
            using namespace rabitq_utils;
            size_t local_q = q;
            this->adjust_with_origin(q, d0, d1);

            ALIGNED(32) uint16_t d32tab[32];
            d0.store(d32tab);
            d1.store(d32tab + 16);

            float* const heap_dis = heap_distances + q * k;
            int64_t* const heap_ids = heap_labels + q * k;

            FAISS_THROW_IF_NOT_FMT(
                    !probe_indices.empty() && local_q < probe_indices.size(),
                    "set_list_context() must be called before handle() - "
                    "probe_indices size: %zu, local_q: %zu, global_q: %zu",
                    probe_indices.size(),
                    local_q,
                    q);

            if (!context || !context->query_factors) {
                FAISS_THROW_MSG("Query factors not available");
            }

            size_t probe_rank = probe_indices[local_q];
            size_t nprobe_val =
                    context->nprobe > 0 ? context->nprobe : index->nprobe;
            size_t storage_idx = q * nprobe_val + probe_rank;
            const auto& query_factors = context->query_factors[storage_idx];

            const float one_a = this->normalizers
                    ? (1.0f / this->normalizers[2 * q])
                    : 1.0f;
            const float bias =
                    this->normalizers ? this->normalizers[2 * q + 1] : 0.0f;

            uint64_t idx_base = this->j0 + b * 32;
            if (idx_base >= this->ntotal)
                return;
            size_t max_positions =
                    std::min<size_t>(32, this->ntotal - idx_base);

            size_t local_1bit_evaluations = 0;
            size_t local_multibit_evaluations = 0;

            for (size_t j = 0; j < max_positions; j++) {
                const int64_t result_id = this->adjust_id(b, j);
                if (result_id < 0)
                    continue;

                const float normalized_distance = d32tab[j] * one_a + bias;
                const uint8_t* base_ptr = get_block_aux_ptr(
                        list_codes_ptr,
                        idx_base + j,
                        index->bbs,
                        packed_block_size,
                        full_block_size,
                        storage_size);

                if (is_multibit) {
                    local_1bit_evaluations++;
                    const SignBitFactorsWithError& full_factors =
                            *reinterpret_cast<const SignBitFactorsWithError*>(
                                    base_ptr);

                    float dist_1bit = compute_1bit_adjusted_distance(
                            normalized_distance,
                            full_factors,
                            query_factors,
                            index->centered,
                            index->qb,
                            index->d);

                    const bool is_similarity = index->metric_type ==
                            MetricType::METRIC_INNER_PRODUCT;
                    bool should_refine = should_refine_candidate(
                            dist_1bit,
                            full_factors.f_error,
                            query_factors.g_error,
                            heap_dis[0],
                            is_similarity);

                    if (should_refine) {
                        local_multibit_evaluations++;
                        size_t local_offset = this->j0 + b * 32 + j;
                        float dist_full = compute_full_multibit_distance(
                                result_id, local_q, q, local_offset);
                        if (Cfloat::cmp(heap_dis[0], dist_full)) {
                            heap_replace_top<Cfloat>(
                                    k,
                                    heap_dis,
                                    heap_ids,
                                    dist_full,
                                    result_id);
                            nup++;
                        }
                    }
                } else {
                    const auto& db_factors =
                            *reinterpret_cast<const SignBitFactors*>(base_ptr);
                    float adjusted_distance = compute_1bit_adjusted_distance(
                            normalized_distance,
                            db_factors,
                            query_factors,
                            index->centered,
                            index->qb,
                            index->d);
                    if (Cfloat::cmp(heap_dis[0], adjusted_distance)) {
                        heap_replace_top<Cfloat>(
                                k,
                                heap_dis,
                                heap_ids,
                                adjusted_distance,
                                result_id);
                        nup++;
                    }
                }
            }

#pragma omp atomic
            rabitq_stats.n_1bit_evaluations += local_1bit_evaluations;
#pragma omp atomic
            rabitq_stats.n_multibit_evaluations += local_multibit_evaluations;
        }

        void set_list_context(
                size_t list_no,
                const std::vector<int>& probe_map) {
            current_list_no = list_no;
            probe_indices = probe_map;
            list_codes_ptr = index->invlists->get_codes(list_no);
        }

        void begin(const float* norms) {
            this->normalizers = norms;
        }

        void end() {
#pragma omp parallel for
            for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
                heap_reorder<Cfloat>(
                        k, heap_distances + q * k, heap_labels + q * k);
            }
        }

        size_t num_updates() {
            return nup;
        }

       private:
        float compute_full_multibit_distance(
                size_t /*db_idx*/,
                size_t local_q,
                size_t global_q,
                size_t local_offset) const {
            using namespace rabitq_utils;
            const size_t ex_bits = index->rabitq.nb_bits - 1;
            const size_t dim = index->d;

            const uint8_t* base_ptr = get_block_aux_ptr(
                    list_codes_ptr,
                    local_offset,
                    index->bbs,
                    packed_block_size,
                    full_block_size,
                    storage_size);

            const size_t ex_code_size = (dim * ex_bits + 7) / 8;
            const uint8_t* ex_code = base_ptr + sizeof(SignBitFactorsWithError);
            const ExtraBitsFactors& ex_fac =
                    *reinterpret_cast<const ExtraBitsFactors*>(
                            base_ptr + sizeof(SignBitFactorsWithError) +
                            ex_code_size);

            size_t probe_rank = probe_indices[local_q];
            size_t nprobe_val =
                    context->nprobe > 0 ? context->nprobe : index->nprobe;
            size_t storage_idx = global_q * nprobe_val + probe_rank;
            const auto& qf = context->query_factors[storage_idx];

            InvertedLists::ScopedCodes list_codes(
                    index->invlists, current_list_no);
            std::vector<uint8_t> unpacked_code(index->code_size);
            packer->unpack_1(
                    list_codes.get(), local_offset, unpacked_code.data());

            return rabitq_utils::compute_full_multibit_distance(
                    unpacked_code.data(),
                    ex_code,
                    ex_fac,
                    qf.rotated_q.data(),
                    qf.qr_to_c_L2sqr,
                    qf.qr_norm_L2sqr,
                    dim,
                    ex_bits,
                    index->metric_type);
        }
    };
};

} // namespace faiss
