/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/fast_scan/FastScanDistancePostProcessing.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>

namespace faiss {

// Forward declaration — full definition in IndexIVFRaBitQFastScan.h
struct IndexIVFRaBitQFastScan;

namespace simd_result_handlers {

// Import shared utilities from RaBitQUtils
using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

/** SIMD result handler for IndexIVFRaBitQFastScan that applies
 * RaBitQ-specific distance corrections during batch processing.
 *
 * @tparam C Comparator type (CMin/CMax) for heap operations
 * @tparam SL SIMD level for dynamic dispatch
 */
template <class C, SIMDLevel SL = SINGLE_SIMD_LEVEL_256>
struct IVFRaBitQHeapHandler : ResultHandlerCompare<C, true, SL> {
    using RHC = ResultHandlerCompare<C, true, SL>;
    using typename RHC::simd16uint16;

    const IndexIVFRaBitQFastScan* index;
    float* heap_distances; // [nq * k]
    int64_t* heap_labels;  // [nq * k]
    const size_t nq, k;
    size_t current_list_no = 0;
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
    std::vector<uint8_t> unpack_buf; // sign bits scratch buffer

    // Cached per-list values (set in set_list_context, avoid recomputing in
    // handle)
    size_t cached_nprobe = 0;
    bool is_similarity = false; // metric == INNER_PRODUCT

    // Use float-based comparator for heap operations
    using Cfloat = typename std::conditional<
            C::is_max,
            CMax<float, int64_t>,
            CMin<float, int64_t>>::type;

    // Constructor and method bodies are defined at the bottom of
    // IndexIVFRaBitQFastScan.h (after the full struct definition is
    // available) to break the circular header dependency.
    IVFRaBitQHeapHandler(
            const IndexIVFRaBitQFastScan* idx,
            size_t nq_val,
            size_t k_val,
            float* distances,
            int64_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing* ctx = nullptr,
            bool multibit = false);

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) override;

    void set_list_context(size_t list_no, const std::vector<int>& probe_map)
            override;

    void begin(const float* norms) override;

    void end() override;

    size_t num_updates() override {
        return nup;
    }

   private:
    float compute_full_multibit_distance(
            size_t local_q,
            size_t global_q,
            size_t local_offset,
            const uint8_t* aux_ptr);
};

} // namespace simd_result_handlers

} // namespace faiss
