/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <omp.h>

#include "IndexFlat.h"
#include "IndexPQ.h"
#include "IndexScalarQuantizer.h"
#include "utils.h"


namespace faiss {

/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implmentation is heavily influenced by the NMSlib
 * implementation by Yury Malkov and Leonid Boystov
 * (https://github.com/searchivarius/nmslib)
 *
 * The HNSW object stores only the neighbor link structure, see
 * IndexHNSW below for the full index object.
 */

struct VisitedTable;

struct HNSW {

    /// internal storage of vectors (32 bits: this is expensive)
    typedef int storage_idx_t;

    /// Faiss results are 64-bit
    typedef faiss::Index::idx_t idx_t;

    /** The HNSW structure does not store vectors, it only accesses
     * them through this class.
     *
     * Functions are guaranteed to be be accessed only from 1 thread. */
    struct DistanceComputer {

        idx_t d;

        /// called before computing distances
        virtual void set_query (const float *x) = 0;

        /// compute distance of vector i to current query
        virtual float operator () (storage_idx_t i) = 0;

        /// compute distance between two stored vectors
        virtual float symmetric_dis(storage_idx_t i, storage_idx_t j) = 0;

        virtual ~DistanceComputer () {}

    };

    /// assignment probability to each layer (sum=1)
    std::vector<double> assign_probas;

    /// number of neighbors stored per layer (cumulative), should not
    /// be changed after first add
    std::vector<int> cum_nneighbor_per_level;

    /// level of each vector (base level = 1), size = ntotal
    std::vector<int> levels;

    /// offsets[i] is the offset in the neighbors array where vector i is stored
    /// size ntotal + 1
    std::vector<size_t> offsets;

    /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
    /// for all levels. this is where all storage goes.
    std::vector<storage_idx_t> neighbors;

    /// entry point in the search structure (one of the points with maximum level
    storage_idx_t entry_point;

    faiss::RandomGenerator rng;

    /// maximum level
    int max_level;

    /// expansion factor at construction time
    int efConstruction;

    /// expansion factor at search time
    int efSearch;

    /// during search: do we check whether the next best distance is good enough?
    bool check_relative_distance;

    /// number of entry points in levels > 0.
    int upper_beam;

    // methods that initialize the tree sizes

    /// initialize the assign_probas and cum_nneighbor_per_level to
    /// have 2*M links on level 0 and M links on levels > 0
    void set_default_probas(int M, float levelMult);

    /// set nb of neighbors for this level (before adding anything)
    void set_nb_neighbors(int level_no, int n);

    // methods that access the tree sizes

    /// nb of neighbors for this level
    int nb_neighbors(int layer_no) const;

    /// cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    /// range of entries in the neighbors table of vertex no at layer_no
    void neighbor_range(idx_t no, int layer_no,
                        size_t * begin, size_t * end) const;

    /// only mandatory parameter: nb of neighbors
    explicit HNSW(int M = 32);

    /// pick a random level for a new point
    int random_level();

    /// add n random levels to table (for debugging...)
    void fill_with_random_links(size_t n);

    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. */
    void add_with_locks(DistanceComputer & ptdis, int pt_level, int pt_id,
                        std::vector<omp_lock_t> & locks,
                        VisitedTable &vt);


    /// search interface
    void search(DistanceComputer & qdis, int k,
                idx_t *I, float * D,
                VisitedTable &vt) const;

    void reset();

    void clear_neighbor_tables(int level);
    void print_neighbor_stats(int level) const;
};


struct HNSWStats {
    size_t n1, n2, n3;
    size_t ndis;
    size_t nreorder;
    bool view;

    HNSWStats () {reset (); }
    void reset ();
};

// global var that collects them all
extern HNSWStats hnsw_stats;

class IndexHNSW;

struct ReconstructFromNeighbors {
    typedef Index::idx_t idx_t;
    typedef HNSW::storage_idx_t storage_idx_t;

    const IndexHNSW & index;
    size_t M; // number of neighbors
    size_t k; // number of codebook entries
    size_t nsq; // number of subvectors
    size_t code_size;
    int k_reorder; // nb to reorder. -1 = all

    std::vector<float> codebook; // size nsq * k * (M + 1)

    std::vector<uint8_t> codes; // size ntotal * code_size
    size_t ntotal;
    size_t d, dsub; // derived values

    ReconstructFromNeighbors(const IndexHNSW & index,
                             size_t k=256, size_t nsq=1);

    /// codes must be added in the correct order and the IndexHNSW
    /// must be populated and sorted
    void add_codes(size_t n, const float *x);

    size_t compute_distances(size_t n, const idx_t *shortlist,
                           const float *query, float *distances) const;

    /// called by add_codes
    void estimate_code(const float *x, storage_idx_t i, uint8_t *code) const;

    /// called by compute_distances
    void reconstruct(storage_idx_t i, float *x, float *tmp) const;

    void reconstruct_n(storage_idx_t n0, storage_idx_t ni, float *x) const;

    /// get the M+1 -by-d table for neighbor coordinates for vector i
    void get_neighbor_table(storage_idx_t i, float *out) const;

};


/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSW: Index {

    typedef HNSW::storage_idx_t storage_idx_t;

    // the link strcuture
    HNSW hnsw;

    // the sequential storage
    bool own_fields;
    Index * storage;

    ReconstructFromNeighbors *reconstruct_from_neighbors;

    explicit IndexHNSW (int d = 0, int M = 32);
    explicit IndexHNSW (Index * storage, int M = 32);

    ~IndexHNSW() override;

    // get a DistanceComputer object for this kind of storage
    virtual HNSW::DistanceComputer * get_distance_computer() const = 0;

    void add(idx_t n, const float *x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search (idx_t n, const float *x, idx_t k,
                 float *distances, idx_t *labels) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset () override;

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(idx_t n, const float *x, idx_t k,
                        const storage_idx_t *nearest, const float *nearest_d,
                        float *distances, idx_t *labels, int nprobe = 1,
                        int search_type = 1) const;

    /// alternative graph building
    void init_level_0_from_knngraph(
                        int k, const float *D, const idx_t *I);

    /// alternative graph building
    void init_level_0_from_entry_points(
                        int npt, const storage_idx_t *points,
                        const storage_idx_t *nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    void link_singletons();
};



/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlat: IndexHNSW {
    IndexHNSWFlat();
    IndexHNSWFlat(int d, int M);
    HNSW::DistanceComputer * get_distance_computer() const override;
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWPQ: IndexHNSW {
    IndexHNSWPQ();
    IndexHNSWPQ(int d, int pq_m, int M);
    void train(idx_t n, const float* x) override;
    HNSW::DistanceComputer * get_distance_computer() const override;
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWSQ: IndexHNSW {
    IndexHNSWSQ();
    IndexHNSWSQ(int d, ScalarQuantizer::QuantizerType qtype, int M);
    HNSW::DistanceComputer * get_distance_computer() const override;
};

/** 2-level code structure with fast random access
 */
struct IndexHNSW2Level: IndexHNSW {
    IndexHNSW2Level();
    IndexHNSW2Level(Index *quantizer, size_t nlist, int m_pq, int M);
    HNSW::DistanceComputer * get_distance_computer() const override;
    void flip_to_ivf();

    /// entry point for search
    void search (idx_t n, const float *x, idx_t k,
                 float *distances, idx_t *labels) const override;

};



};
