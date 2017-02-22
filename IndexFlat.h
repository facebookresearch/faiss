
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#ifndef INDEX_FLAT_H
#define INDEX_FLAT_H

#include <vector>

#include "Index.h"


namespace faiss {

/** Index that stores the full vectors and performs exhaustive search */
struct IndexFlat: Index {
    /// database vectors, size ntotal * d
    std::vector<float> xb;

    explicit IndexFlat (idx_t d, MetricType metric = METRIC_INNER_PRODUCT);

    virtual void set_typename() override;

    virtual void add (idx_t n, const float *x) override;

    virtual void reset() override;

    virtual void search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const override;



    virtual void range_search (
            idx_t n,
            const float *x,
            float radius,
            RangeSearchResult *result) const override;

    virtual void reconstruct (idx_t key, float * recons)
        const override;


    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            const idx_t *labels) const;

    IndexFlat () {}
};



struct IndexFlatIP:IndexFlat {
    explicit IndexFlatIP (idx_t d): IndexFlat (d, METRIC_INNER_PRODUCT) {}
    IndexFlatIP () {}
};


struct IndexFlatL2:IndexFlat {
    explicit IndexFlatL2 (idx_t d): IndexFlat (d, METRIC_L2) {}
    IndexFlatL2 () {}
};


// same as an IndexFlatL2 but a value is subtracted from each distance
struct IndexFlatL2BaseShift: IndexFlatL2 {
    std::vector<float> shift;

    IndexFlatL2BaseShift (idx_t d, size_t nshift, const float *shift);

    virtual void search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const override;
};


/** Index that queries in a base_index (a fast one) and refines the
 *  results with an exact search, hopefully improving the results.
 */
struct IndexRefineFlat: Index {

    /// storage for full vectors
    IndexFlat refine_index;

    /// faster index to pre-select the vectors that should be filtered
    Index *base_index;
    bool own_fields;  ///< should the base index be deallocated?

    /// factor between k requested in search and the k requested from
    /// the base_index (should be >= 1)
    float k_factor;

    explicit IndexRefineFlat (Index *base_index);

    IndexRefineFlat ();

    virtual void train (idx_t n, const float *x) override;

    virtual void add (idx_t n, const float *x) override;

    virtual void reset() override;

    virtual void search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const override;

    virtual void set_typename () override;

    virtual ~IndexRefineFlat ();
};


/// optimized version for 1D "vectors"
struct IndexFlat1D:IndexFlatL2 {
    bool continuous_update; ///< is the permutation updated continuously?

    std::vector<idx_t> perm; ///< sorted database indices

    explicit IndexFlat1D (bool continuous_update=true);

    /// if not continuous_update, call this between the last add and
    /// the first search
    void update_permutation ();

    virtual void add (idx_t n, const float *x) override;

    virtual void reset() override;

    /// Warn: the distances returned are L1 not L2
    virtual void search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const override;


};


}

#endif
