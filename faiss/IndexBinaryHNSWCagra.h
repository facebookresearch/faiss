/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexBinaryHNSW.h>

namespace faiss {

/** Binary HNSW index designed for interoperability with GPU Binary CAGRA.
 *  
 *  This index maintains a fixed-degree graph structure compatible with
 *  Binary CAGRA's requirements, unlike the standard IndexBinaryHNSW which
 *  allows variable-degree graphs.
 *  
 *  INVARIANT: The base layer (level 0) always has exactly nb_neighbors(0)
 *  neighbors for every node. This is enforced by setting keep_max_size_level0
 *  to true and calling ensure_fixed_degree() after adding vectors.
 */
struct IndexBinaryHNSWCagra : IndexBinaryHNSW {
    IndexBinaryHNSWCagra();
    IndexBinaryHNSWCagra(int d, int M);

    /// When set to true, the index is immutable.
    /// This option is used to copy the knn graph from GpuIndexBinaryCagra
    /// to the base level of IndexBinaryHNSWCagra without adding upper levels.
    /// Doing so enables to search the HNSW index, but removes the
    /// ability to add vectors.
    bool base_level_only = false;

    /// When `base_level_only` is set to `True`, the search function
    /// searches only the base level knn graph of the HNSW index.
    /// This parameter selects the entry point by randomly selecting
    /// some points and using the best one.
    int num_base_level_search_entrypoints = 32;

    void add(idx_t n, const uint8_t* x) override;
    void add(idx_t n, const void* x, NumericType numeric_type) override;

    /// entry point for search
    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void search(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Specific methods for compatibility with fixed-degree graphs
    /// Ensures all nodes have exactly the expected number of neighbors
    void ensure_fixed_degree();
    
    /// Check if the graph has fixed degree (all nodes have same number of neighbors)
    /// Always returns true for IndexBinaryHNSWCagra as fixed degree is maintained by design
    bool has_fixed_degree() const;
    
    /// Get the fixed degree of the graph at level 0
    size_t get_fixed_degree() const;
};

} // namespace faiss 