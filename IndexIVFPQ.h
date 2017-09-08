/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef FAISS_INDEX_IVFPQ_H
#define FAISS_INDEX_IVFPQ_H


#include <vector>

#include "IndexIVF.h"
#include "IndexPQ.h"


namespace faiss {



/** Inverted file with Product Quantizer encoding. Each residual
 * vector is encoded as a product quantizer code.
 */
struct IndexIVFPQ: IndexIVF {
    bool by_residual;              ///< Encode residual or plain vector?
    int use_precomputed_table;     ///< if by_residual, build precompute tables
    ProductQuantizer pq;           ///< produces the codes

    bool do_polysemous_training;   ///< reorder PQ centroids after training?
    PolysemousTraining *polysemous_training; ///< if NULL, use default

    // search-time parameters
    size_t scan_table_threshold;   ///< use table computation or on-the-fly?
    size_t max_codes;              ///< max nb of codes to visit to do a query
    int polysemous_ht;             ///< Hamming thresh for polysemous filtering


    /// if use_precompute_table
    /// size nlist * pq.M * pq.ksub
    std::vector <float> precomputed_table;

    IndexIVFPQ (
            Index * quantizer, size_t d, size_t nlist,
            size_t M, size_t nbits_per_idx);

    void add_with_ids(idx_t n, const float* x, const long* xids = nullptr)
        override;

    /// same as add_core, also:
    /// - output 2nd level residuals if residuals_2 != NULL
    /// - use precomputed list numbers if precomputed_idx != NULL
    void add_core_o (idx_t n, const float *x,
                     const long *xids, float *residuals_2,
                     const long *precomputed_idx = nullptr);

    /// trains the product quantizer
    void train_residual(idx_t n, const float* x) override;

    /// same as train_residual, also output 2nd level residuals
    void train_residual_o (idx_t n, const float *x, float *residuals_2);


    /** Reconstruct a subset of the indexed vectors
     *
     * @param i0     first vector to reconstruct
     * @param ni     nb of vectors to reconstruct
     * @param recons output array of reconstructed vectors, size ni * d
     */
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /** Find exact duplicates in the dataset.
     *
     * the duplicates are returned in pre-allocated arrays (see the
     * max sizes).
     *
     * @params lims   limits between groups of duplicates
     *                (max size ntotal / 2 + 1)
     * @params ids    ids[lims[i]] : ids[lims[i+1]-1] is a group of
     *                duplicates (max size ntotal)
     * @return n      number of groups found
     */
    size_t find_duplicates (idx_t *ids, size_t *lims) const;

    // map a vector to a binary code knowning the index
    void encode (long key, const float * x, uint8_t * code) const;

    /** Encode multiple vectors
     *
     * @param n       nb vectors to encode
     * @param keys    posting list ids for those vectors (size n)
     * @param x       vectors (size n * d)
     * @param codes   output codes (size n * code_size)
     * @param compute_keys  if false, assume keys are precomputed,
     *                      otherwise compute them
     */
    void encode_multiple (size_t n, long *keys,
                          const float * x, uint8_t * codes,
                          bool compute_keys = false) const;

    /// inverse of encode_multiple
    void decode_multiple (size_t n, const long *keys,
                          const uint8_t * xcodes, float * x) const;

    void search_preassigned (idx_t n, const float *x, idx_t k,
                             const idx_t *assign,
                             const float *centroid_dis,
                             float *distances, idx_t *labels,
                             bool store_pairs) const override;


    /// build precomputed table
    void precompute_table ();

    IndexIVFPQ ();

};


/// statistics are robust to internal threading, but not if
/// IndexIVFPQ::search_preassigned is called by multiple threads
struct IndexIVFPQStats {
    size_t nq;       // nb of queries run
    size_t nlist;    // nb of inverted lists scanned
    size_t ncode;    // nb of codes visited
    size_t nrefine;  // nb of refines (IVFPQR)

    size_t n_hamming_pass;
    // nb of passed Hamming distance tests (for polysemous)

    // timings measured with the CPU RTC
    // on all threads
    size_t assign_cycles;
    size_t search_cycles;
    size_t refine_cycles; // only for IVFPQR

    // single thread (double-counted with search_cycles)
    size_t init_query_cycles;
    size_t init_list_cycles;
    size_t scan_cycles;
    size_t heap_cycles;

    IndexIVFPQStats () {reset (); }
    void reset ();
};

// global var that collects them all
extern IndexIVFPQStats indexIVFPQ_stats;



/** Index with an additional level of PQ refinement */
struct IndexIVFPQR: IndexIVFPQ {
    ProductQuantizer refine_pq;           ///< 3rd level quantizer
    std::vector <uint8_t> refine_codes;   ///< corresponding codes

    /// factor between k requested in search and the k requested from the IVFPQ
    float k_factor;

    IndexIVFPQR (
            Index * quantizer, size_t d, size_t nlist,
            size_t M, size_t nbits_per_idx,
            size_t M_refine, size_t nbits_per_idx_refine);

    void reset() override;

    long remove_ids(const IDSelector& sel) override;

    /// trains the two product quantizers
    void train_residual(idx_t n, const float* x) override;

    void add_with_ids(idx_t n, const float* x, const long* xids) override;

    /// same as add_with_ids, but optionally use the precomputed list ids
    void add_core (idx_t n, const float *x, const long *xids,
                     const long *precomputed_idx = nullptr);

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void merge_from (IndexIVF &other, idx_t add_id) override;


    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    IndexIVFPQR();
};


/** Index with 32-bit ids and flat tables. Must be constructed from an
 *  exisiting IndexIVFPQ. Cannot be copy-constructed/assigned. The
 *  actual data is stored in the compact_* tables, the ids and codes
 *  tables are not used.  */
struct IndexIVFPQCompact: IndexIVFPQ {

    explicit IndexIVFPQCompact (const IndexIVFPQ &other);

    /// how were the compact tables allocated?
    enum Alloc_type_t {
        Alloc_type_none,     ///< alloc from outside
        Alloc_type_new,      ///< was allocated with new
        Alloc_type_mmap      ///< was mmapped
    };

    Alloc_type_t alloc_type;

    uint32_t *limits;        ///< size nlist + 1
    uint32_t *compact_ids;   ///< size ntotal
    uint8_t *compact_codes;  ///< size ntotal * code_size

    // file and buffer this was mmapped (will be unmapped when object
    // is deleted)
    char * mmap_buffer;
    long mmap_length;

    void search_preassigned (idx_t n, const float *x, idx_t k,
                             const idx_t *assign,
                             const float *centroid_dis,
                             float *distances, idx_t *labels,
                             bool store_pairs) const override;

    /// the three following functions will fail at runtime
    void add(idx_t, const float*) override;
    void reset() override;
    void train(idx_t, const float*) override;

    ~IndexIVFPQCompact() override;

    IndexIVFPQCompact ();

};



} // namespace faiss





#endif
