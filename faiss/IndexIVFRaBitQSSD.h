/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/impl/RaBitQuantizer.h>

namespace faiss {

struct RaBitQSSDInvertedLists;

struct IVFRaBitQSSDSearchParameters : IVFSearchParameters {
    /// overrides IndexIVFRaBitQSSD::use_hacc
    bool use_hacc = true;
    /// overrides IndexIVFRaBitQSSD::drain_alpha
    float drain_alpha = 1.0f;
    /// overrides IndexIVFRaBitQSSD::qd_per_thread
    size_t qd_per_thread = 16;
};

/** IVF index whose RaBitQ codes are split between memory and an SSD
 * (RaBitQ-SSD).
 *
 * Reference: "RaBitQ-SSD: Split Codes and Pipelined I/O for SSD-Resident
 * Vector Search", Yuexuan Xu et al.
 *
 * Vectors are encoded with RaBitQuantizer relative to their list centroid,
 * with nb_bits bits per dimension: a sign bit and nb_bits - 1 extra bits.
 * The sign bits of the first mem_dim dimensions stay in memory; the other
 * sign bits, the extra bits and the vector ids are stored in a file, see
 * RaBitQSSDInvertedLists.
 *
 * Search. Each search thread takes one query at a time, runs the coarse
 * quantizer on it and scans the probed lists in batches of 32 vectors. For
 * each vector, FastScan over its in-memory bits gives a distance estimate and
 * a lower bound (the RaBitQ error bound). Vectors whose lower bound does not
 * exceed the current k-th distance are grouped by page, and the pages are
 * queued by their smallest lower bound. Between batches, the search re-ranks
 * the pages whose reads completed and submits the most promising queued
 * pages, keeping at most qd_per_thread reads in flight. After the last list,
 * the remaining reads are drained.
 *
 * The distance estimates assume randomly rotated vectors, as for all RaBitQ
 * indexes: use the index inside an IndexPreTransform with
 * HadamardRotation(d, seed, false).
 *
 * index_factory accepts "IVF<nlist>,RaBitQSSD<nb_bits>_<mem_dim>", usually
 * after the rotation: "HRk,IVF<nlist>,RaBitQSSD<nb_bits>_<mem_dim>". It
 * creates an index without inverted lists. Attach RaBitQSSDInvertedLists
 * with replace_invlists() before adding or searching vectors.
 *
 * Not supported: remove_ids, update_vectors, merge_from, range search,
 * reconstruction and standalone codes (sa_encode).
 */
struct IndexIVFRaBitQSSD : IndexIVF {
    /// most SSD reads a search thread can have in flight
    static constexpr size_t kMaxReadsInFlight = 200;

    /// encodes the vectors; its nb_bits is the total number of bits per
    /// dimension, in [2, 9]
    RaBitQuantizer rabitq;

    /// dimensions whose sign bits stay in memory: a multiple of 64 in [64, d]
    size_t mem_dim = 0;

    /// use 16-bit look-up tables for the in-memory scan (accumulate_hacc);
    /// if false, 8-bit tables
    bool use_hacc = true;

    /// after the last list, pages whose lower bound exceeds the
    /// drain_alpha-quantile of the current results are not read. 1 keeps
    /// the k-th distance as the threshold.
    float drain_alpha = 1.0f;

    /// SSD reads in flight per search thread, at most kMaxReadsInFlight
    size_t qd_per_thread = 16;

    /// squared norms of the centroids, needed to derive <q, c> and |q - c|
    /// from the coarse distances when the index or the quantizer uses the
    /// inner product. Set by add().
    std::vector<float> centroid_norms;

    /** @param quantizer  coarse quantizer, which must implement reconstruct
     * @param d          dimensionality, a multiple of 64
     * @param nlist      number of lists
     * @param mem_dim    dimensions whose sign bits stay in memory
     * @param nb_bits    bits per dimension, in [2, 9]
     * @param filename   file for the disk-resident part, created or
     *                   truncated; its file system must support O_DIRECT.
     *                   nullptr leaves invlists unset; training and
     *                   serialization are still supported
     * @param metric     METRIC_L2 or METRIC_INNER_PRODUCT
     */
    IndexIVFRaBitQSSD(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t mem_dim,
            uint8_t nb_bits,
            const char* filename = nullptr,
            MetricType metric = METRIC_L2);

    IndexIVFRaBitQSSD();

    IndexIVFRaBitQSSD(const IndexIVFRaBitQSSD&) = delete;
    IndexIVFRaBitQSSD& operator=(const IndexIVFRaBitQSSD&) = delete;

    /// the inverted lists, or an exception if they are not of the right type
    RaBitQSSDInvertedLists* get_ssd_invlists() const;

    /// empty the index, including when inverted lists have not been attached
    void reset() override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    /// not supported: the codes are not stored as flat codes
    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    /** Unlike IndexIVF::search, which runs the coarse quantizer on all the
     * queries first, each search thread runs it on its own query, so that it
     * overlaps the reads of the other threads. The queries are searched in
     * parallel whatever the parallel_mode.
     */
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

    /// not supported
    void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            const idx_t* keys,
            const float* coarse_dis,
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    /// not supported: the vectors are not stored, and their codes are
    /// split between memory and the file
    void reconstruct(idx_t key, float* recons) const override;

    /// not supported, see reconstruct
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /// not supported
    size_t remove_ids(const IDSelector& sel) override;

    /// not supported
    void update_vectors(int nv, const idx_t* idx, const float* v) override;

    /// not supported
    void merge_from(Index& otherIndex, idx_t add_id) override;

    /// recompute centroid_norms from the quantizer
    void update_centroid_norms();
};

/// Counters of IndexIVFRaBitQSSD searches, summed over all searches since
/// the last reset().
struct IndexIVFRaBitQSSDStats {
    size_t n_pages_read = 0; ///< pages read from the file
    size_t n_reranked = 0;   ///< vectors re-ranked with their full code

    void reset();
    void add(const IndexIVFRaBitQSSDStats& other);
};

FAISS_API extern IndexIVFRaBitQSSDStats indexIVFRaBitQSSD_stats;

} // namespace faiss
