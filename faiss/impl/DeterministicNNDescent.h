/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Index.h>

namespace faiss {

/// parameters of the local joins, shared by DeterministicNNDescent and
/// NNDescentShard
struct NNDescentParams {
    int K = 64; ///< neighbors per node
    int R = 0;  ///< reverse neighbors used per node; 0: 2 * K
    /// new neighbors of each kind (neighbors, reverse neighbors) that a local
    /// join uses; 0: all
    int sample = 0;

    int nreverse() const {
        return R != 0 ? R : 2 * K;
    }

    int nsample() const {
        return sample != 0 ? sample : K + nreverse();
    }
};

/** A K-nearest-neighbor graph built by NN-descent, with a result that does
 * not depend on the thread count.
 *
 * Dong, Charikar and Li, "Efficient K-Nearest Neighbor Graph Construction for
 * Generic Similarity Measures", WWW 2011. The batch structure follows
 * pyNNDescent in ParlayANN (Manohar et al., PPoPP 2024).
 *
 * - Initialization: random cluster trees. Each tree splits the points by
 *   their nearer of two random pivots, down to leaves of at most 512 points,
 *   and links each point to its K nearest in its leaf. With ntrees = 0, each
 *   node gets K random neighbors. Rows that stay short get random neighbors.
 * - Rounds: a round reads a fixed copy of the graph. For each node u, a
 *   local join computes the distances between the members of the
 *   neighborhood of u (its K neighbors and R nearest reverse neighbors), for
 *   the pairs where at least one member is new (added by the previous
 *   round), and offers each distance to both members. The new row of a node
 *   holds the K best of its old row and of its offers by (distance, id),
 *   with the smallest distance of each id. This set does not depend on the
 *   order of the offers, so the rows do not depend on the thread count.
 *
 * The vectors are read in place for IndexFlat, decoded for
 * IndexScalarQuantizer, and reconstructed for other storage types. The
 * distances of a leaf and of a local join come from the inner products of
 * the float vectors: from the nndescent::gram_upper_rows kernel on AVX2,
 * AVX-512 and ARM NEON or SVE, else from BLAS. The values of a level do not
 * depend on the thread count, but differ between levels. Distances are
 * smaller-is-better: for METRIC_INNER_PRODUCT they are negated inner products.
 * A distance that overflows to NaN becomes +inf. */
struct DeterministicNNDescent : NNDescentParams {
    int ntrees = 8;      ///< random cluster trees for init(); 0: random init
    int max_rounds = 20; ///< maximum rounds of build()
    float delta = 0.05f; ///< build() stops when fewer nodes change
    int64_t seed = 1234;
    bool verbose = false;

    // the storage of init()
    idx_t n = 0;
    size_t d = 0;
    MetricType metric_type = METRIC_L2;
    /// n * K neighbors per node, sorted by (distance, id), -1 padded
    std::vector<int32_t> ids;
    std::vector<float> dis;

    /// distance computations of the last init() or round()
    size_t ndis = 0;

    /// Initializes the graph over the ntotal vectors of storage.
    void init(const Index& storage);

    /// One round; returns the number of nodes whose row changed.
    idx_t round(const Index& storage);

    /// init(storage), then rounds until fewer than delta * n nodes change
    void build(const Index& storage);

    // internal state
    /// n * K, nonzero for an entry added by the last init() or round()
    std::vector<uint8_t> is_new;
    /// order in which round() visits the nodes: the leaf order of the first
    /// cluster tree, so that consecutive nodes share neighbors in the cache
    std::vector<int32_t> order;

    void check_params() const;
};

/** The rows of one process in an NN-descent round over a graph that is split
 * between processes (for example a distributed build, after the
 * per-partition builds).
 *
 * Graph ids are int64. Process p owns the ids owner_begin[p] ..
 * owner_begin[p + 1] - 1 and holds their rows and vectors in that order. A
 * round has four steps. Between the steps, the caller sends the lists to the
 * other processes; each list is grouped by destination process, with one
 * count per process.
 *  1. reverse_edges(): the edges u -> v of the home rows, for the owner of v.
 *  2. set_reverse_edges(): the received edges. Each home row keeps its R
 *     nearest reverse edges by (distance, source). The remote ids are the
 *     sorted ids that other processes own, of the neighborhoods that have a
 *     new member: a local join of a neighborhood without one computes no
 *     distances.
 *  3. join(): with the vectors and the K-th distances of the remote ids, the
 *     local joins of the home rows, as in DeterministicNNDescent::round().
 *     Offers to home rows are merged. Offers to remote rows are returned, the
 *     K best for each row. With select_rows(), several join() calls can each
 *     cover a range of the home rows, with the remote ids of that range only.
 *  4. add_offers(): the offers that the other processes sent, in any order,
 *     then end_round(), which returns the number of home rows that changed.
 * With the same rows, a round gives the same rows as
 * DeterministicNNDescent::round() on the whole graph, except for the order of
 * equal distances. The result does not depend on the thread count or on the
 * order in which the lists arrive.
 *
 * The offers to a remote row can include ids that are already in the row:
 * the owner drops them in add_offers(). A pair can get two distances that
 * differ in the last bits: init() computes some distances directly and the
 * local joins from the squared norms, and coded remote vectors differ from
 * the home vectors. The cases below that give "the same rows" assume one
 * value per pair; otherwise the rows can differ slightly, but are still the
 * same in every run with the same input. */
struct NNDescentShard : NNDescentParams {
    /// nhome() x K rows: graph ids sorted by (distance, id), -1 padded
    std::vector<int64_t> ids;
    std::vector<float> dis;
    /// ids that other processes own, of the neighborhoods of this round that
    /// have a new member
    std::vector<int64_t> remote_ids;

    /// distance computations of the join() calls of the round
    size_t ndis = 0;

    /// rows of the home ids; every entry is new. Every row id is -1 or in
    /// [owner_begin[0], owner_begin[world]).
    void set_rows(
            int K,
            MetricType metric_type,
            size_t d,
            int rank,
            int world,
            const int64_t* owner_begin,
            const float* x,
            const int64_t* ids,
            const float* dis);

    idx_t nhome() const;

    /// edges u -> v grouped by the owner of v
    void reverse_edges(
            std::vector<int64_t>& counts,
            std::vector<int64_t>& target,
            std::vector<int64_t>& source,
            std::vector<float>& edge_dis,
            std::vector<uint8_t>& edge_new) const;

    /// the received edges, whose targets are home ids and whose sources are
    /// graph ids; computes remote_ids
    void set_reverse_edges(
            size_t n,
            const int64_t* target,
            const int64_t* source,
            const float* edge_dis,
            const uint8_t* edge_new);

    /** Selects the home rows [begin, end) for the next join() and sets
     * remote_ids to the remote ids of their neighborhoods that have a new
     * member. set_reverse_edges() selects all home rows. join() calls on
     * disjoint ranges that cover the home rows need only the remote ids of
     * their range. With one distance per pair, they give the same rows as one
     * join() on all rows. When a pair can get two distances (see above), a
     * range can send an offer to a remote row that one join() would cut, so
     * the rows can differ slightly. */
    void select_rows(idx_t begin, idx_t end);

    /// number of remote ids that each process owns
    void remote_counts(std::vector<int64_t>& counts) const;

    /** Local joins of the selected home rows. For remote id i: its vector
     * remote_x[i * d] and the K-th distance of its row remote_kth[i] (+inf if
     * the row is short). Throws if a neighborhood id of the selected rows is
     * not in remote_ids. Returns the offers to remote rows grouped by owner:
     * (target, candidate, distance), sorted by target, then by (distance,
     * candidate). */
    void join(
            const float* remote_x,
            const float* remote_kth,
            std::vector<int64_t>& counts,
            std::vector<int64_t>& target,
            std::vector<int64_t>& candidate,
            std::vector<float>& offer_dis);

    /// join() with float16 remote vectors, which take half the memory
    void join_fp16(
            const uint16_t* remote_x,
            const float* remote_kth,
            std::vector<int64_t>& counts,
            std::vector<int64_t>& target,
            std::vector<int64_t>& candidate,
            std::vector<float>& offer_dis);

    /** join() with 8-bit remote codes, a quarter of the float memory.
     * Component j of a remote vector is vmin[j] + (code + 0.5) / 255 *
     * vdiff[j], as in the 8-bit ScalarQuantizer. */
    void join_sq8(
            const uint8_t* remote_codes,
            const float* vmin,
            const float* vdiff,
            const float* remote_kth,
            std::vector<int64_t>& counts,
            std::vector<int64_t>& target,
            std::vector<int64_t>& candidate,
            std::vector<float>& offer_dis);

    /// sets remote_ids to the sorted ids of the home rows that other
    /// processes own
    void set_remote_ids_of_rows();

    /** The rows with the original ids: home_original[i] for home id
     * owner_begin[rank] + i, remote_original[j] for remote_ids[j]. out:
     * nhome() x K, -1 padded. */
    void original_ids(
            const int64_t* home_original,
            const int64_t* remote_original,
            int64_t* out) const;

    /** The offers to home rows that other processes sent: (target,
     * candidate, distance), with home targets and candidates that are graph
     * ids. The caller can add the offers of each join() as they arrive, in
     * any order, then call end_round(). */
    void add_offers(
            size_t n,
            const int64_t* target,
            const int64_t* candidate,
            const float* offer_dis);

    /// ends the round; returns the number of home rows that changed
    idx_t end_round();

    // internal state, set by set_rows()
    MetricType metric_type = METRIC_L2;
    size_t d = 0;
    int rank = 0;
    /// world + 1 boundaries of the graph ids that each process owns
    std::vector<int64_t> owner_begin;
    /// nhome() x d home vectors; the caller keeps them alive
    const float* x = nullptr;
    /// nhome() x K flags of the rows
    std::vector<uint8_t> is_new;

    // state between the steps of a round
    std::vector<int64_t> reverse_ids; ///< nhome() x nreverse(), -1 padded
    std::vector<uint8_t> reverse_new; ///< flag of each reverse edge
    std::vector<int64_t> next_ids;    ///< rows after the merges
    std::vector<float> next_dis;
    std::vector<uint8_t> next_new;
    idx_t selected_begin = 0; ///< home rows of the next join()
    idx_t selected_end = 0;

    /// remote vectors of join_impl(): exactly one of x, x16 and codes
    struct RemoteVectors {
        const float* x = nullptr;
        const uint16_t* x16 = nullptr;
        const uint8_t* codes = nullptr;
        const float* vmin = nullptr;
        const float* vdiff = nullptr;
    };

    void join_impl(
            const RemoteVectors& rv,
            const float* remote_kth,
            std::vector<int64_t>& counts,
            std::vector<int64_t>& target,
            std::vector<int64_t>& candidate,
            std::vector<float>& offer_dis);

    /// whether the neighborhood of home row t has a new member, so that its
    /// local join computes distances
    bool has_new_member(idx_t t) const;

    /** The remote ids of the home rows [begin, end), sorted. With the
     * reverse lists: the remote ids of the neighborhoods that have a new
     * member, which are the only ones that a local join reads. */
    void collect_remote_ids(bool with_reverse, idx_t begin, idx_t end);
};

} // namespace faiss
