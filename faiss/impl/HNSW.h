/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <queue>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

namespace faiss {

/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implementation is heavily influenced by the NMSlib
 * implementation by Yury Malkov and Leonid Boystov
 * (https://github.com/searchivarius/nmslib)
 *
 * The HNSW object stores only the neighbor link structure, see
 * IndexHNSW.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer; // from AuxIndexStructures
struct HNSWStats;
template <class C>
struct ResultHandler;

struct SearchParametersHNSW : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    bool bounded_queue = true;

    ~SearchParametersHNSW() {}
};

struct HNSW {
    /// internal storage of vectors (32 bits: this is expensive)
    using storage_idx_t = int32_t;

    // for now we do only these distances
    using C = CMax<float, int64_t>;

    typedef std::pair<float, storage_idx_t> Node;

    /** Heap structure that allows fast
     */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(float* vmin_out = nullptr);

        int count_below(float thresh);
    };

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistFarther& obj1) const {
            return d > obj1.d;
        }
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

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point = -1;

    faiss::RandomGenerator rng;

    /// maximum level
    int max_level = -1;

    /// expansion factor at construction time
    int efConstruction = 40;

    /// expansion factor at search time
    int efSearch = 16;

    /// during search: do we check whether the next best distance is good
    /// enough?
    bool check_relative_distance = true;

    /// use bounded queue during exploration
    bool search_bounded_queue = true;

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
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
            const;

    /// only mandatory parameter: nb of neighbors
    explicit HNSW(int M = 32);

    /// pick a random level for a new point
    int random_level();

    /// add n random levels to table (for debugging...)
    void fill_with_random_links(size_t n);

    void add_links_starting_from(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t* locks,
            VisitedTable& vt,
            bool keep_max_size_level0 = false);

    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. */
    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt,
            bool keep_max_size_level0 = false);

    /// search interface for 1 point, single thread
    HNSWStats search(
            DistanceComputer& qdis,
            ResultHandler<C>& res,
            VisitedTable& vt,
            const SearchParametersHNSW* params = nullptr) const;

    /// search only in level 0 from a given vertex
    void search_level_0(
            DistanceComputer& qdis,
            ResultHandler<C>& res,
            idx_t nprobe,
            const storage_idx_t* nearest_i,
            const float* nearest_d,
            int search_type,
            HNSWStats& search_stats,
            VisitedTable& vt,
            const SearchParametersHNSW* params = nullptr) const;

    void reset();

    void clear_neighbor_tables(int level);
    void print_neighbor_stats(int level) const;

    int prepare_level_tab(size_t n, bool preset_levels = false);

    static void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistFarther>& input,
            std::vector<NodeDistFarther>& output,
            int max_size,
            bool keep_max_size_level0 = false);

    void permute_entries(const idx_t* map);
};

struct HNSWStats {
    size_t n1 = 0; /// number of vectors searched
    size_t n2 =
            0; /// number of queries for which the candidate list is exhausted
    size_t ndis = 0;  /// number of distances computed
    size_t nhops = 0; /// number of hops aka number of edges traversed

    void reset() {
        n1 = n2 = 0;
        ndis = 0;
        nhops = 0;
    }

    void combine(const HNSWStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        ndis += other.ndis;
        nhops += other.nhops;
    }
};

// global var that collects them all
FAISS_API extern HNSWStats hnsw_stats;

int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<HNSW::C>& res,
        HNSW::MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersHNSW* params = nullptr);

HNSWStats greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        HNSW::storage_idx_t& nearest,
        float& d_nearest);

std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
        const HNSW& hnsw,
        const HNSW::Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats);

} // namespace faiss
