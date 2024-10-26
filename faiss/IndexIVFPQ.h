/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVFPQ_H
#define FAISS_INDEX_IVFPQ_H

#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct IVFPQSearchParameters : IVFSearchParameters {
    size_t scan_table_threshold; ///< use table computation or on-the-fly?
    int polysemous_ht;           ///< Hamming thresh for polysemous filtering
    IVFPQSearchParameters() : scan_table_threshold(0), polysemous_ht(0) {}
    ~IVFPQSearchParameters() {}
};

FAISS_API extern size_t precomputed_table_max_bytes;

/** Inverted file with Product Quantizer encoding. Each residual
 * vector is encoded as a product quantizer code.
 */
struct IndexIVFPQ : IndexIVF {
    ProductQuantizer pq; ///< produces the codes

    bool do_polysemous_training; ///< reorder PQ centroids after training?
    PolysemousTraining* polysemous_training; ///< if NULL, use default

    // search-time parameters
    size_t scan_table_threshold; ///< use table computation or on-the-fly?
    int polysemous_ht;           ///< Hamming thresh for polysemous filtering

    /** Precompute table that speed up query preprocessing at some
     * memory cost (used only for by_residual with L2 metric)
     */
    int use_precomputed_table;

    /// if use_precompute_table
    /// size nlist * pq.M * pq.ksub
    AlignedTable<float> precomputed_table;

    IndexIVFPQ(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            MetricType metric = METRIC_L2);

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    /// same as add_core, also:
    /// - output 2nd level residuals if residuals_2 != NULL
    /// - accepts precomputed_idx = nullptr
    void add_core_o(
            idx_t n,
            const float* x,
            const idx_t* xids,
            float* residuals_2,
            const idx_t* precomputed_idx = nullptr,
            void* inverted_list_context = nullptr);

    /// trains the product quantizer
    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /** Find exact duplicates in the dataset.
     *
     * the duplicates are returned in pre-allocated arrays (see the
     * max sizes).
     *
     * @param lims   limits between groups of duplicates
     *                (max size ntotal / 2 + 1)
     * @param ids    ids[lims[i]] : ids[lims[i+1]-1] is a group of
     *                duplicates (max size ntotal)
     * @return n      number of groups found
     */
    size_t find_duplicates(idx_t* ids, size_t* lims) const;

    // map a vector to a binary code knowning the index
    void encode(idx_t key, const float* x, uint8_t* code) const;

    /** Encode multiple vectors
     *
     * @param n       nb vectors to encode
     * @param keys    posting list ids for those vectors (size n)
     * @param x       vectors (size n * d)
     * @param codes   output codes (size n * code_size)
     * @param compute_keys  if false, assume keys are precomputed,
     *                      otherwise compute them
     */
    void encode_multiple(
            size_t n,
            idx_t* keys,
            const float* x,
            uint8_t* codes,
            bool compute_keys = false) const;

    /// inverse of encode_multiple
    void decode_multiple(
            size_t n,
            const idx_t* keys,
            const uint8_t* xcodes,
            float* x) const;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    /// build precomputed table
    void precompute_table();

    IndexIVFPQ();
};

// block size used in IndexIVFPQ::add_core_o
FAISS_API extern int index_ivfpq_add_core_o_bs;

/** Pre-compute distance tables for IVFPQ with by-residual and METRIC_L2
 *
 * @param use_precomputed_table (I/O)
 *        =-1: force disable
 *        =0: decide heuristically (default: use tables only if they are
 *            < precomputed_tables_max_bytes), set use_precomputed_table on
 * output =1: tables that work for all quantizers (size 256 * nlist * M) =2:
 * specific version for MultiIndexQuantizer (much more compact)
 * @param precomputed_table precomputed table to initialize
 */

void initialize_IVFPQ_precomputed_table(
        int& use_precomputed_table,
        const Index* quantizer,
        const ProductQuantizer& pq,
        AlignedTable<float>& precomputed_table,
        bool by_residual,
        bool verbose);

/// statistics are robust to internal threading, but not if
/// IndexIVFPQ::search_preassigned is called by multiple threads
struct IndexIVFPQStats {
    size_t nrefine; ///< nb of refines (IVFPQR)

    size_t n_hamming_pass;
    ///< nb of passed Hamming distance tests (for polysemous)

    // timings measured with the CPU RTC on all threads
    size_t search_cycles;
    size_t refine_cycles; ///< only for IVFPQR

    IndexIVFPQStats() {
        reset();
    }
    void reset();
};

// global var that collects them all
FAISS_API extern IndexIVFPQStats indexIVFPQ_stats;

} // namespace faiss

#endif
