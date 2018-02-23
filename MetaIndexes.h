/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef META_INDEXES_H
#define META_INDEXES_H


#include <vector>
#include <unordered_map>


#include "Index.h"


namespace faiss {

/** Index that translates search results to ids */
struct IndexIDMap : Index {
    Index * index;            ///! the sub-index
    bool own_fields;          ///! whether pointers are deleted in destructo
    std::vector<long> id_map;

    explicit IndexIDMap (Index *index);

    /// Same as add_core, but stores xids instead of sequential ids
    /// @param xids if non-null, ids to store for the vectors (size n)
    void add_with_ids(idx_t n, const float* x, const long* xids) override;

    /// this will fail. Use add_with_ids
    void add(idx_t n, const float* x) override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    void train(idx_t n, const float* x) override;

    void reset() override;

    /// remove ids adapted to IndexFlat
    long remove_ids(const IDSelector& sel) override;

    void range_search (idx_t n, const float *x, float radius,
                       RangeSearchResult *result) const override;

    ~IndexIDMap() override;
    IndexIDMap () {own_fields=false; index=nullptr; }
};

/** same as IndexIDMap but also provides an efficient reconstruction
    implementation via a 2-way index */
struct IndexIDMap2 : IndexIDMap {

    std::unordered_map<idx_t, idx_t> rev_map;

    explicit IndexIDMap2 (Index *index);

    /// make the rev_map from scratch
    void construct_rev_map ();

    void add_with_ids(idx_t n, const float* x, const long* xids) override;

    long remove_ids(const IDSelector& sel) override;

    void reconstruct (idx_t key, float * recons) const override;

    ~IndexIDMap2() override {}
    IndexIDMap2 () {}
};


/** Index that concatenates the results from several sub-indexes
 *
 */
struct IndexShards : Index {

    std::vector<Index*> shard_indexes;
    bool own_fields;      /// should the sub-indexes be deleted along with this?
    bool threaded;
    bool successive_ids;

    /**
     * @param threaded     do we use one thread per sub_index or do
     *                     queries sequentially?
     * @param successive_ids should we shift the returned ids by
     *                     the size of each sub-index or return them
     *                     as they are?
     */
    explicit IndexShards (idx_t d, bool threaded = false,
                         bool successive_ids = true);

    void add_shard (Index *);

    // update metric_type and ntotal. Call if you changes something in
    // the shard indexes.
    void sync_with_shard_indexes ();

    Index *at(int i) {return shard_indexes[i]; }

    /// supported only for sub-indices that implement add_with_ids
    void add(idx_t n, const float* x) override;

    /**
     * Cases (successive_ids, xids):
     * - true, non-NULL       ERROR: it makes no sense to pass in ids and
     *                        request them to be shifted
     * - true, NULL           OK, but should be called only once (calls add()
     *                        on sub-indexes).
     * - false, non-NULL      OK: will call add_with_ids with passed in xids
     *                        distributed evenly over shards
     * - false, NULL          OK: will call add_with_ids on each sub-index,
     *                        starting at ntotal
     */
    void add_with_ids(idx_t n, const float* x, const long* xids) override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    void train(idx_t n, const float* x) override;

    void reset() override;

    ~IndexShards() override;
};

/** splits input vectors in segments and assigns each segment to a sub-index
 * used to distribute a MultiIndexQuantizer
 */

struct IndexSplitVectors: Index {
    bool own_fields;
    bool threaded;
    std::vector<Index*> sub_indexes;
    idx_t sum_d;  /// sum of dimensions seen so far

    explicit IndexSplitVectors (idx_t d, bool threaded = false);

    void add_sub_index (Index *);
    void sync_with_sub_indexes ();

    void add(idx_t n, const float* x) override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    void train(idx_t n, const float* x) override;

    void reset() override;

    ~IndexSplitVectors() override;
};



}


#endif
