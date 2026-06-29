/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/utils/simd_levels.h>

#include <cstddef>
#include <cstdint>

namespace faiss {

// Forward declarations
struct DistanceComputer;
struct InvertedListScanner;
struct BinaryInvertedListScanner;
struct IndexBinaryFlat;
struct IndexBinaryHash;
struct IndexBinaryMultiHash;
struct IndexBinaryIVF;
struct IndexIVFSpectralHash;
struct IndexPQ;
struct SearchParametersIVF;
using IVFSearchParameters = SearchParametersIVF;
struct RangeQueryResult;
using idx_t = int64_t;

/** @name IndexBinaryHNSW dispatch
 * @{ */
template <SIMDLevel SL>
DistanceComputer* make_binary_hnsw_distance_computer_fixSL(
        int code_size,
        IndexBinaryFlat* flat_storage);
/** @} */

/** @name IndexBinaryIVF dispatch
 * @{ */
template <SIMDLevel SL>
BinaryInvertedListScanner* make_binary_ivf_scanner_fixSL(
        size_t code_size,
        bool store_pairs);

template <SIMDLevel SL>
void search_knn_hamming_per_invlist_fixSL(
        int code_size,
        const IndexBinaryIVF* ivf,
        size_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* keys_in,
        const int32_t* coarse_dis,
        int32_t* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params);

template <SIMDLevel SL>
void search_knn_hamming_count_fixSL(
        int code_size,
        bool store_pairs,
        const IndexBinaryIVF* ivf,
        size_t nx,
        const uint8_t* x,
        const idx_t* keys,
        int k,
        int32_t* distances,
        idx_t* labels,
        const IVFSearchParameters* params);
/** @} */

/** @name IndexBinaryHash dispatch
 * @{ */
template <SIMDLevel SL>
void binary_hash_knn_search_fixSL(
        const IndexBinaryHash& index,
        const uint8_t* q,
        idx_t k,
        int32_t* heap_sim,
        idx_t* heap_ids,
        size_t& n0,
        size_t& nlist,
        size_t& ndis);

template <SIMDLevel SL>
void binary_hash_range_search_fixSL(
        const IndexBinaryHash& index,
        const uint8_t* q,
        int radius,
        RangeQueryResult& qres,
        size_t& n0,
        size_t& nlist,
        size_t& ndis);

template <SIMDLevel SL>
void binary_multihash_knn_search_fixSL(
        const IndexBinaryMultiHash& index,
        const uint8_t* q,
        idx_t k,
        int32_t* heap_sim,
        idx_t* heap_ids,
        size_t& n0,
        size_t& nlist,
        size_t& ndis);

template <SIMDLevel SL>
void binary_multihash_range_search_fixSL(
        const IndexBinaryMultiHash& index,
        const uint8_t* q,
        int radius,
        RangeQueryResult& qres,
        size_t& n0,
        size_t& nlist,
        size_t& ndis);
/** @} */

/** @name IndexIVFSpectralHash dispatch
 * @{ */
template <SIMDLevel SL>
InvertedListScanner* make_spectral_hash_scanner_fixSL(
        int code_size,
        const IndexIVFSpectralHash* index,
        bool store_pairs);
/** @} */

/** @name IndexPQ polysemous dispatch
 * @{ */
template <SIMDLevel SL>
size_t polysemous_inner_loop_fixSL(
        int code_size,
        const IndexPQ* index,
        const float* dis_table_qi,
        const uint8_t* q_code,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        int ht);
/** @} */

} // namespace faiss
