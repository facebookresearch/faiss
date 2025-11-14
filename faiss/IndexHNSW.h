/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>
#include "faiss/Index.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

namespace faiss {

struct IndexHNSW;

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSW : Index {
    typedef HNSW::storage_idx_t storage_idx_t;

    // the link structure
    HNSW hnsw;

    // the sequential storage
    bool own_fields = false;
    Index* storage = nullptr;

    // When set to false, level 0 in the knn graph is not initialized.
    // This option is used by GpuIndexCagra::copyTo(IndexHNSWCagra*)
    // as level 0 knn graph is copied over from the index built by
    // GpuIndexCagra.
    bool init_level0 = true;

    // When set to true, all neighbors in level 0 are filled up
    // to the maximum size allowed (2 * M). This option is used by
    // IndexHNSWCagra to create a full base layer graph that is
    // used when GpuIndexCagra::copyFrom(IndexHNSWCagra*) is invoked.
    bool keep_max_size_level0 = false;

    explicit IndexHNSW(int d = 0, int M = 32, MetricType metric = METRIC_L2);
    explicit IndexHNSW(Index* storage, int M = 32);

    ~IndexHNSW() override;

    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1,
            const SearchParameters* params = nullptr) const;

    /// alternative graph building
    void init_level_0_from_knngraph(int k, const float* D, const idx_t* I);

    /// alternative graph building
    void init_level_0_from_entry_points(
            int npt,
            const storage_idx_t* points,
            const storage_idx_t* nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    void link_singletons();

    virtual void permute_entries(const idx_t* perm);

    DistanceComputer* get_distance_computer() const override;
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlat : IndexHNSW {
    IndexHNSWFlat();
    IndexHNSWFlat(int d, int M, MetricType metric = METRIC_L2);
};

/** Panorama implementation of IndexHNSWFlat following
 * https://www.arxiv.org/pdf/2510.00566.
 *
 * Unlike cluster-based Panorama, the vectors have to be higher dimensional
 * (i.e. typically d > 512) and/or be able to compress a lot of their energy in
 * the early dimensions to be effective. This is because HNSW accesses vectors
 * in a random order, which makes cache misses dominate the distance computation
 * time.
 *
 * The `num_panorama_levels` parameter controls the granularity of progressive
 * distance refinement, allowing candidates to be eliminated early using partial
 * distance computations rather than computing full distances.
 *
 * NOTE: This version of HNSW handles search slightly differently than the
 * vanilla HNSW, as it uses partial distance computations with progressive
 * refinement bounds. Instead of computing full distances immediately for all
 * candidates, Panorama maintains lower and upper bounds that are incrementally
 * tightened across refinement levels. Candidates are inserted into the search
 * beam using approximate distance estimates (LB+UB)/2 and are only fully
 * evaluated when they survive pruning and enter the result heap. This allows
 * the algorithm to prune unpromising candidates early using Cauchy-Schwarz
 * bounds on partial inner products. Hence, recall is not guaranteed to be the
 * same as vanilla HNSW due to the heterogeneous precision within the search
 * beam (exact vs. partial distance estimates affecting traversal order).
 */
struct IndexHNSWFlatPanorama : IndexHNSWFlat {
    IndexHNSWFlatPanorama();
    IndexHNSWFlatPanorama(
            int d,
            int M,
            int num_panorama_levels,
            MetricType metric = METRIC_L2);

    void add(idx_t n, const float* x) override;
    void reset() override;
    void permute_entries(const idx_t* perm) override;

    /// Inline for performance - called frequently in search hot path.
    const float* get_cum_sum(idx_t i) const {
        return cum_sums.data() + i * (num_panorama_levels + 1);
    }

    /// Compute cumulative sums for a vector (used both for database points and
    /// queries).
    static void compute_cum_sums(
            const float* x,
            float* dst_cum_sums,
            int d,
            int num_panorama_levels,
            int panorama_level_width);

    std::vector<float> cum_sums;
    const size_t panorama_level_width;
    const size_t num_panorama_levels;
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWPQ : IndexHNSW {
    IndexHNSWPQ();
    IndexHNSWPQ(
            int d,
            int pq_m,
            int M,
            int pq_nbits = 8,
            MetricType metric = METRIC_L2);
    void train(idx_t n, const float* x) override;
};

/** SQ index topped with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWSQ : IndexHNSW {
    IndexHNSWSQ();
    IndexHNSWSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric = METRIC_L2);
};

/** 2-level code structure with fast random access
 */
struct IndexHNSW2Level : IndexHNSW {
    IndexHNSW2Level();
    IndexHNSW2Level(Index* quantizer, size_t nlist, int m_pq, int M);

    void flip_to_ivf();

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

struct IndexHNSWCagra : IndexHNSW {
    IndexHNSWCagra();
    IndexHNSWCagra(
            int d,
            int M,
            MetricType metric = METRIC_L2,
            NumericType numeric_type = NumericType::Float32);

    /// When set to true, the index is immutable.
    /// This option is used to copy the knn graph from GpuIndexCagra
    /// to the base level of IndexHNSWCagra without adding upper levels.
    /// Doing so enables to search the HNSW index, but removes the
    /// ability to add vectors.
    bool base_level_only = false;

    /// When `base_level_only` is set to `True`, the search function
    /// searches only the base level knn graph of the HNSW index.
    /// This parameter selects the entry point by randomly selecting
    /// some points and using the best one.
    int num_base_level_search_entrypoints = 32;

    void add(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    faiss::NumericType get_numeric_type() const;
    void set_numeric_type(faiss::NumericType numeric_type);
    NumericType numeric_type_;
};

} // namespace faiss
