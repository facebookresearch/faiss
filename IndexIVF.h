
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_H
#define FAISS_INDEX_IVF_H


#include <vector>


#include "Index.h"
#include "Clustering.h"
#include "Heap.h"


namespace faiss {



/** Index based on a inverted file (IVF)
 *
 * In the inverted file, the quantizer (an Index instance) provides a
 * quantization index for each vector to be added. The quantization
 * index maps to a list (aka inverted list or posting list), where the
 * id of the vector is then stored.
 *
 * At search time, the vector to be searched is also quantized, and
 * only the list corresponding to the quantization index is
 * searched. This speeds up the search by making it
 * non-exhaustive. This can be relaxed using multi-probe search: a few
 * (nprobe) quantization indices are selected and several inverted
 * lists are visited.
 *
 * Sub-classes implement a post-filtering of the index that refines
 * the distance estimation from the query to databse vectors.
 */
struct IndexIVF: Index {
    size_t nlist;             ///< number of possible key values
    size_t nprobe;            ///< number of probes at query time

    Index * quantizer;        ///< quantizer that maps vectors to inverted lists
    bool quantizer_trains_alone;   ///< just pass over the trainset to quantizer
    bool own_fields;          ///< whether object owns the quantizer

    ClusteringParameters cp; ///< to override default clustering params

    std::vector < std::vector<long> > ids;  ///< Inverted lists for indexes

    /// map for direct access to the elements. Enables reconstruct().
    bool maintain_direct_map;
    std::vector <long> direct_map;

    /** The Inverted file takes a quantizer (an Index) on input,
     * which implements the function mapping a vector to a list
     * identifier. The pointer is borrowed: the quantizer should not
     * be deleted while the IndexIVF is in use.
     */
    IndexIVF (Index * quantizer, size_t d, size_t nlist,
              MetricType metric = METRIC_INNER_PRODUCT);

    virtual void reset () override;

    /// Trains the quantizer and calls train_residual to train sub-quantizers
    virtual void train (idx_t n, const float *x) override;

    /// Quantizes x and calls add_with_key
    virtual void add (idx_t n, const float *x) override;



    /// Sub-classes that encode the residuals can train their encoders here
    /// does nothing by default
    virtual void train_residual (idx_t n, const float *x);

    /** moves the entries from another dataset to self. On output,
     * other is empty. add_id is added to all moved ids (for
     * sequential ids, this would be this->ntotal */
    virtual void merge_from (IndexIVF &other, idx_t add_id);

    /** implemented by sub-classes */
    virtual void merge_from_residuals (IndexIVF &other) = 0;

    virtual ~IndexIVF();

    /// intialize a direct map
    void make_direct_map ();

    /// 1= perfectly balanced, >1: imbalanced
    double imbalance_factor () const;

    /// display some stats about the inverted lists
    void print_stats () const;

    IndexIVF ();
};




/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded.
 */
struct IndexIVFFlat: IndexIVF {
    /** Inverted list of original vectors. Each list is a nl * d
     * matrix, where nl is the nb of vectors stored in the list. */
    std::vector < std::vector<float> > vecs;

    IndexIVFFlat (
            Index * quantizer, size_t d, size_t nlist_,
            MetricType = METRIC_INNER_PRODUCT);

    /** Return the typeName of the index (which includes main parameters */
    virtual void set_typename () override;


    /// same as add_with_ids, with precomputed coarse quantizer
    void add_core (idx_t n, const float * x, const long *xids,
                   const long *precomputed_idx);


    /// implemented for all IndexIVF* classes
    virtual void add_with_ids (idx_t n, const float * x, const long *xids)
        override;

    virtual void search (
            idx_t n, const float *x, idx_t k,
            float *distances, idx_t *labels) const override;

    virtual void range_search (
            idx_t n, const float *x, float radius,
            RangeSearchResult *result) const override;

    /** copy a subset of the entries index to the other index
     *
     * if subset_type == 0: copies ids in [a1, a2)
     * if subset_type == 1: copies ids if id % a1 == a2
     */
    void copy_subset_to (IndexIVFFlat & other, int subset_type,
                         long a1, long a2) const;


    virtual void reset() override;

    virtual long remove_ids (const IDSelector & sel) override;

    /// Implementation of the search for the inner product metric
    void search_knn_inner_product (
            size_t nx, const float * x,
            const long * keys,
            float_minheap_array_t * res) const;

    /// Implementation of the search for the L2 metric
    void search_knn_L2sqr (
            size_t nx, const float * x,
            const long * keys,
            float_maxheap_array_t * res) const;

    virtual void reconstruct (idx_t key, float * recons) const override;

    virtual void merge_from_residuals (IndexIVF &other) override;


    IndexIVFFlat () {}
};






} // namespace faiss





#endif
