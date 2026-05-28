/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSW.h>

#include <cinttypes>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include <faiss/IndexHNSW.h>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>

namespace faiss {

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

int HNSW::nb_neighbors(int layer_no) const {
    FAISS_THROW_IF_NOT(
            static_cast<size_t>(layer_no + 1) < cum_nneighbor_per_level.size());
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (size_t i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const {
    FAISS_CHECK_RANGE_DEBUG(layer_no, 0, (int)cum_nneighbor_per_level.size());
    return cum_nneighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    FAISS_CHECK_RANGE_DEBUG(no, 0, (idx_t)offsets.size());
    FAISS_CHECK_RANGE_DEBUG(
            layer_no, 0, (int)cum_nneighbor_per_level.size() - 1);
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

HNSW::HNSW(int M) : rng(12345) {
    set_default_probas(M, 1.0 / log(M));
    offsets.push_back(0);
}

int HNSW::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bisection
    for (size_t level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9) {
            break;
        }
        assign_probas.push_back(proba);
        nn += level == 0 ? M * 2 : M;
        cum_nneighbor_per_level.push_back(nn);
    }
}

void HNSW::clear_neighbor_tables(int level) {
    for (size_t i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = -1;
        }
    }
}

void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

void HNSW::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(
            static_cast<size_t>(level) < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
        reduction(+ : tot_reciprocal) reduction(+ : n_node)
    for (idx_t i = 0; i < static_cast<idx_t>(levels.size()); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) {
                    break;
                }
                neighset.insert(neighbors[j]);
            }
            size_t n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0) {
                    break;
                }
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0) {
                        break;
                    }
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

void HNSW::fill_with_random_links(size_t n) {
    int max_level_2 = prepare_level_tab(n);
    RandomGenerator rng2(456);

    for (int level = max_level_2 - 1; level >= 0; --level) {
        std::vector<int> elts;
        for (size_t i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }
        printf("linking %zd elements in level %d\n", elts.size(), level);

        if (elts.size() == 1) {
            continue;
        }

        for (size_t ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while (other == i);

                neighbors[j] = other;
            }
        }
    }
}

int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (size_t i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_level_2 = 0;
    for (size_t i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level_2) {
            max_level_2 = pt_level;
        }
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    }
    neighbors.resize(offsets.back(), -1);

    return max_level_2;
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
template <class C>
void HNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFartherT<C>>& input,
        std::vector<NodeDistFartherT<C>>& output,
        size_t max_size,
        bool keep_max_size_level0) {
    // This prevents number of neighbors at
    // level 0 from being shrunk to less than 2 * M.
    // This is essential in making sure
    // `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
    std::vector<NodeDistFartherT<C>> outsiders;

    while (input.size() > 0) {
        NodeDistFartherT<C> v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFartherT<C> v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            // "v1 is bad" if some previously-kept neighbor v2 is closer
            // (more similar, under CMin) to v1 than the query is. Encoded
            // generically as: v1v2 is "better than" v1q under C.
            if (C::cmp(dist_v1_q, dist_v1_v2)) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= static_cast<size_t>(max_size)) {
                return;
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(v1);
        }
    }
    size_t idx = 0;
    while (keep_max_size_level0 &&
           (output.size() < static_cast<size_t>(max_size)) &&
           (idx < outsiders.size())) {
        output.push_back(outsiders[idx++]);
    }
}

// Explicit instantiations for the two supported comparators.
template void HNSW::shrink_neighbor_list<HNSW::C_distance>(
        DistanceComputer&,
        std::priority_queue<HNSW::NodeDistFartherT<HNSW::C_distance>>&,
        std::vector<HNSW::NodeDistFartherT<HNSW::C_distance>>&,
        size_t,
        bool);
template void HNSW::shrink_neighbor_list<HNSW::C_similarity>(
        DistanceComputer&,
        std::priority_queue<HNSW::NodeDistFartherT<HNSW::C_similarity>>&,
        std::vector<HNSW::NodeDistFartherT<HNSW::C_similarity>>&,
        size_t,
        bool);

namespace {

using storage_idx_t = HNSW::storage_idx_t;

// Map a (high-level) HNSW comparator C — which uses int64_t IDs — to the
// (low-level) MinimaxHeap comparator HC, which uses int32_t IDs.
template <class C>
using HC_for = std::
        conditional_t<C::is_max, CMax<float, int32_t>, CMin<float, int32_t>>;

// Priority queue types used by the unbounded search variant. For CMax
// (distance) "top_candidates" is a max-heap of the kept-so-far results
// (top is the farthest) and "candidates" is a min-heap of the next nodes
// to explore (top is the closest). For CMin (similarity) the orderings are
// swapped: top_candidates is a min-heap (top is the least similar) and
// candidates is a max-heap (top is the most similar).
template <class C>
using TopCandidatesQueue = std::conditional_t<
        C::is_max,
        std::priority_queue<HNSW::Node>,
        std::priority_queue<
                HNSW::Node,
                std::vector<HNSW::Node>,
                std::greater<HNSW::Node>>>;

template <class C>
using CandidatesQueue = std::conditional_t<
        C::is_max,
        std::priority_queue<
                HNSW::Node,
                std::vector<HNSW::Node>,
                std::greater<HNSW::Node>>,
        std::priority_queue<HNSW::Node>>;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
template <class C>
void shrink_neighbor_list_inner(
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloserT<C>>& resultSet1,
        size_t max_size,
        bool keep_max_size_level0 = false) {
    if (resultSet1.size() < static_cast<size_t>(max_size)) {
        return;
    }
    std::priority_queue<HNSW::NodeDistFartherT<C>> resultSet;
    std::vector<HNSW::NodeDistFartherT<C>> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list<C>(
            qdis, resultSet, returnlist, max_size, keep_max_size_level0);

    for (HNSW::NodeDistFartherT<C> curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
template <class C>
void add_link_tpl(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) {
                break;
            }
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<HNSW::NodeDistCloserT<C>> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) {
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    size_t max_size = end - begin;
    max_size -= max_size * std::clamp(hnsw.prune_headroom, 0.0f, 0.5f);
    shrink_neighbor_list_inner<C>(
            qdis, resultSet, max_size, keep_max_size_level0);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

/** Templated body of `search_neighbors_to_add` — instantiated once per final
 * VisitedTable subclass × comparator so that `vt.set/advance` are inlined
 * and the cost of virtual dispatch is paid only once at the top of the call.
 */
template <typename VTType, class C>
static void search_neighbors_to_add_fixVT(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloserT<C>>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VTType& vt,
        bool reference_version) {
    // top is nearest candidate
    std::priority_queue<HNSW::NodeDistFartherT<C>> candidates;

    HNSW::NodeDistFartherT<C> ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const HNSW::NodeDistFartherT<C>& currEv = candidates.top();

        if (C::cmp(currEv.d, results.top().d)) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        // The reference version is not used, but kept here because:
        // 1. It is easier to switch back if the optimized version has a problem
        // 2. It serves as a starting point for new optimizations
        // 3. It helps understand the code
        // 4. It ensures the reference version is still compilable if the
        // optimized version changes
        // The reference and the optimized versions' results are compared in
        // test_hnsw.cpp
        if (reference_version) {
            // a reference version
            for (size_t i = begin; i < end; i++) {
                HNSW::storage_idx_t nodeId = hnsw.neighbors[i];
                if (nodeId < 0) {
                    break;
                }
                if (!vt.set(nodeId)) {
                    continue;
                }

                float dis = qdis(nodeId);
                HNSW::NodeDistFartherT<C> evE1(dis, nodeId);

                if (results.size() < static_cast<size_t>(hnsw.efConstruction) ||
                    C::cmp(results.top().d, dis)) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() >
                        static_cast<size_t>(hnsw.efConstruction)) {
                        results.pop();
                    }
                }
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const HNSW::storage_idx_t idx,
                                             const float dis) {
                if (results.size() < static_cast<size_t>(hnsw.efConstruction) ||
                    C::cmp(results.top().d, dis)) {
                    results.emplace(dis, idx);
                    candidates.emplace(dis, idx);
                    if (results.size() >
                        static_cast<size_t>(hnsw.efConstruction)) {
                        results.pop();
                    }
                }
            };

            int n_buffered = 0;
            HNSW::storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++) {
                HNSW::storage_idx_t nodeId = hnsw.neighbors[j];
                if (nodeId < 0) {
                    break;
                }
                if (!vt.set(nodeId)) {
                    continue;
                }

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                if (n_buffered == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            buffered_ids[0],
                            buffered_ids[1],
                            buffered_ids[2],
                            buffered_ids[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(buffered_ids[id4], dis[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (int icnt = 0; icnt < n_buffered; icnt++) {
                float dis = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], dis);
            }
        }
    }

    vt.advance();
}

/// Dispatches the VisitedTable concrete type for a given C, then calls
/// the templated `search_neighbors_to_add_fixVT<VTType, C>`.
template <class C>
void search_neighbors_to_add_dispatch(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloserT<C>>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version) {
    auto call = [&]<typename VTType>(VTType& vt_concrete) {
        search_neighbors_to_add_fixVT<VTType, C>(
                hnsw,
                qdis,
                results,
                entry_point,
                d_entry_point,
                level,
                vt_concrete,
                reference_version);
    };
    if (VisitedTableVector* vtv = dynamic_cast<VisitedTableVector*>(&vt)) {
        call(*vtv);
        return;
    }
    VisitedTableSet& vts = dynamic_cast<VisitedTableSet&>(vt);
    call(vts);
}

/// Templated implementation of `HNSW::add_links_starting_from`.
template <class C>
void add_links_starting_from_impl(
        HNSW& hnsw,
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    std::priority_queue<HNSW::NodeDistCloserT<C>> link_targets;

    search_neighbors_to_add_dispatch<C>(
            hnsw, ptdis, link_targets, nearest, d_nearest, level, vt, false);

    // but we can afford only this many neighbors
    int M = hnsw.nb_neighbors(level);

    shrink_neighbor_list_inner<C>(ptdis, link_targets, M, keep_max_size_level0);

    std::vector<storage_idx_t> neighbors_to_add;
    neighbors_to_add.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link_tpl<C>(
                hnsw, ptdis, pt_id, other_id, level, keep_max_size_level0);
        neighbors_to_add.push_back(other_id);
        link_targets.pop();
    }

    locks.unlock(pt_id);
    for (storage_idx_t other_id : neighbors_to_add) {
        locks.lock(other_id);
        add_link_tpl<C>(
                hnsw, ptdis, other_id, pt_id, level, keep_max_size_level0);
        locks.unlock(other_id);
    }
    locks.lock(pt_id);
}

} // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    if (is_similarity) {
        add_links_starting_from_impl<C_similarity>(
                *this,
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks,
                vt,
                keep_max_size_level0);
    } else {
        add_links_starting_from_impl<C_distance>(
                *this,
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks,
                vt,
                keep_max_size_level0);
    }
}

/// search neighbors on a single level, starting from an entry point.
/// Public dispatcher: always operates in distance (CMax) mode because its
/// `priority_queue<HNSW::NodeDistCloser>` signature is the back-compat
/// distance flavor. Internal callers that need similarity mode reach the
/// templated implementation directly via `search_neighbors_to_add_dispatch`.
void hnsw_detail::search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version) {
    search_neighbors_to_add_dispatch<HNSW::C_distance>(
            hnsw,
            qdis,
            results,
            entry_point,
            d_entry_point,
            level,
            vt,
            reference_version);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

namespace {

/// Greedy update of the nearest entry point at a given level.
template <class C>
HNSWStats greedy_update_nearest_impl(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    HNSWStats stats;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        size_t ndis = 0;

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        auto update_with_candidate = [&](const storage_idx_t idx,
                                         const float dis) {
            if (C::cmp(d_nearest, dis)) {
                nearest = idx;
                d_nearest = dis;
            }
        };

        int n_buffered = 0;
        storage_idx_t buffered_ids[4];

        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) {
                break;
            }
            ndis += 1;

            buffered_ids[n_buffered] = v;
            n_buffered += 1;

            if (n_buffered == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        buffered_ids[0],
                        buffered_ids[1],
                        buffered_ids[2],
                        buffered_ids[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    update_with_candidate(buffered_ids[id4], dis[id4]);
                }

                n_buffered = 0;
            }
        }

        // process leftovers
        for (int icnt = 0; icnt < n_buffered; icnt++) {
            float dis = qdis(buffered_ids[icnt]);
            update_with_candidate(buffered_ids[icnt], dis);
        }

        // update stats
        stats.ndis += ndis;
        stats.nhops += 1;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
}

} // namespace

/// greedily update a nearest vector at a given level
HNSWStats hnsw_detail::greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    if (hnsw.is_similarity) {
        return greedy_update_nearest_impl<HNSW::C_similarity>(
                hnsw, qdis, level, nearest, d_nearest);
    }
    return greedy_update_nearest_impl<HNSW::C_distance>(
            hnsw, qdis, level, nearest, d_nearest);
}

namespace {

template <class C>
void add_with_locks_impl(
        HNSW& hnsw,
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    storage_idx_t nearest = hnsw.entry_point;
    if (nearest == -1) { // avoid locking after the first point.
#pragma omp critical
        if (hnsw.entry_point == -1) { // double-check under lock.
            hnsw.max_level = pt_level;
            hnsw.entry_point = pt_id;
            // leave nearest = -1 to trigger early exit after critical block.
        } else {
            // else: Another thread set the entry point.
            nearest = hnsw.entry_point;
        }
    }

    if (nearest < 0) {
        return;
    }

    locks.lock(pt_id);

    int level = hnsw.max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    //  greedy search on upper levels
    for (; level > pt_level; level--) {
        greedy_update_nearest_impl<C>(hnsw, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
        add_links_starting_from_impl<C>(
                hnsw,
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks,
                vt,
                keep_max_size_level0);
    }

    locks.unlock(pt_id);

#pragma omp critical
    {
        if (pt_level > hnsw.max_level) {
            hnsw.max_level = pt_level;
            hnsw.entry_point = pt_id;
        }
    }
}

} // namespace

void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    if (is_similarity) {
        add_with_locks_impl<C_similarity>(
                *this, ptdis, pt_level, pt_id, locks, vt, keep_max_size_level0);
    } else {
        add_with_locks_impl<C_distance>(
                *this, ptdis, pt_level, pt_id, locks, vt, keep_max_size_level0);
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {

/** Helper to extract search parameters from HNSW and SearchParameters */
inline void extract_search_params(
        const HNSW& hnsw,
        const SearchParameters* params,
        bool& do_dis_check,
        int& efSearch,
        const IDSelector*& sel) {
    // can be overridden by search params
    do_dis_check = hnsw.check_relative_distance;
    efSearch = hnsw.efSearch;
    sel = nullptr;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            do_dis_check = hnsw_params->check_relative_distance;
            efSearch = hnsw_params->efSearch;
        }
        sel = params->sel;
    }
}

/** Templated body of `search_from_candidates` — instantiated once per
 * VisitedTable subclass × comparator.
 */
template <typename VTType, class C>
int search_from_candidates_fixVT(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeapT<HC_for<C>>& candidates,
        VTType& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    typename C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (C::cmp(threshold, d)) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            vt.prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        threshold = res.threshold;

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (!sel || sel->is_member(idx)) {
                if (C::cmp(threshold, dis)) {
                    if (res.add_result(dis, idx)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            saved_j[counter] = v1;
            counter += vt.set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;

                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);

            ndis += 1;
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}

/// Dispatches the VisitedTable concrete type for a given C, then calls
/// the templated `search_from_candidates_fixVT<VTType, C>`.
template <class C>
int search_from_candidates_dispatch(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeapT<HC_for<C>>& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    auto call = [&]<typename VTType>(VTType& vt_concrete) -> int {
        return search_from_candidates_fixVT<VTType, C>(
                hnsw,
                qdis,
                res,
                candidates,
                vt_concrete,
                stats,
                level,
                nres_in,
                params);
    };
    if (VisitedTableVector* vtv = dynamic_cast<VisitedTableVector*>(&vt)) {
        return call(*vtv);
    }
    VisitedTableSet& vts = dynamic_cast<VisitedTableSet&>(vt);
    return call(vts);
}

} // namespace

/** Do a BFS on the candidates list. Public dispatcher: only handles the
 *  distance (CMax) flavor because its `MinimaxHeap` parameter is the
 *  CMax instantiation. */
int hnsw_detail::search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    return search_from_candidates_dispatch<HNSW::C_distance>(
            hnsw, qdis, res, candidates, vt, stats, level, nres_in, params);
}

int hnsw_detail::search_from_candidates_panorama(
        const HNSW& hnsw,
        const IndexHNSW* index,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    // Panorama's progressive-bound math is L2-specific: refuse to run in
    // similarity mode.
    FAISS_THROW_IF_NOT_MSG(
            !hnsw.is_similarity,
            "search_from_candidates_panorama does not support is_similarity=true");

    using C = HNSW::C_distance;

    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (C::cmp(threshold, d)) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    // Validate the index type so we can access cumulative sums, n_levels, and
    // get the ability to compute partial dot products.
    const auto* panorama_index =
            dynamic_cast<const IndexHNSWFlatPanorama*>(index);
    FAISS_THROW_IF_NOT_MSG(
            panorama_index, "Index must be a IndexHNSWFlatPanorama");
    auto* flat_codes_qdis = dynamic_cast<FlatCodesDistanceComputer*>(&qdis);
    FAISS_THROW_IF_NOT_MSG(
            flat_codes_qdis,
            "DistanceComputer must be a FlatCodesDistanceComputer");

    // Allocate space for the index array and exact distances.
    size_t M = hnsw.nb_neighbors(0);
    std::vector<idx_t> index_array(M);
    std::vector<float> exact_distances(M);

    const float* query = flat_codes_qdis->q;
    std::vector<float> query_cum_sums(panorama_index->pano.n_levels + 1);
    panorama_index->pano.compute_query_cum_sums(query, query_cum_sums.data());
    float query_norm_sq = query_cum_sums[0] * query_cum_sums[0];

    int nstep = 0;
    const size_t d = static_cast<size_t>(panorama_index->d);

    PanoramaStats local_pano_stats;
    local_pano_stats.reset();

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more than ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // Unlike the vanilla HNSW, we already remove (and compact) the visited
        // nodes from the candidates list at this stage. We also remove nodes
        // that are not selected.
        size_t initial_size = 0;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            const float* cum_sums_v1 = panorama_index->get_cum_sum(v1);
            index_array[initial_size] = v1;
            exact_distances[initial_size] =
                    query_norm_sq + cum_sums_v1[0] * cum_sums_v1[0];

            bool is_selected = !sel || sel->is_member(v1);
            initial_size += is_selected && vt.set(v1) ? 1 : 0;
        }

        local_pano_stats.total_dims += initial_size * d;
        size_t batch_size = initial_size;
        size_t curr_panorama_level = 0;
        const size_t num_panorama_levels = panorama_index->pano.n_levels;
        while (curr_panorama_level < num_panorama_levels && batch_size > 0) {
            float query_cum_norm = query_cum_sums[curr_panorama_level + 1];

            size_t start_dim = curr_panorama_level *
                    panorama_index->pano.level_width_floats;
            size_t end_dim = (curr_panorama_level + 1) *
                    panorama_index->pano.level_width_floats;
            end_dim = std::min(end_dim, static_cast<size_t>(panorama_index->d));

            size_t i = 0;
            size_t next_batch_size = 0;
            for (; i + 3 < batch_size; i += 4) {
                idx_t idx_0 = index_array[i];
                idx_t idx_1 = index_array[i + 1];
                idx_t idx_2 = index_array[i + 2];
                idx_t idx_3 = index_array[i + 3];

                float dp[4];
                flat_codes_qdis->partial_dot_product_batch_4(
                        idx_0,
                        idx_1,
                        idx_2,
                        idx_3,
                        dp[0],
                        dp[1],
                        dp[2],
                        dp[3],
                        start_dim,
                        end_dim - start_dim);
                ndis += 4;

                float new_exact_0 = exact_distances[i + 0] - 2 * dp[0];
                float new_exact_1 = exact_distances[i + 1] - 2 * dp[1];
                float new_exact_2 = exact_distances[i + 2] - 2 * dp[2];
                float new_exact_3 = exact_distances[i + 3] - 2 * dp[3];

                float cum_sum_0 = panorama_index->get_cum_sum(
                        idx_0)[curr_panorama_level + 1];
                float cum_sum_1 = panorama_index->get_cum_sum(
                        idx_1)[curr_panorama_level + 1];
                float cum_sum_2 = panorama_index->get_cum_sum(
                        idx_2)[curr_panorama_level + 1];
                float cum_sum_3 = panorama_index->get_cum_sum(
                        idx_3)[curr_panorama_level + 1];

                float cs_bound_0 = 2.0f * cum_sum_0 * query_cum_norm;
                float cs_bound_1 = 2.0f * cum_sum_1 * query_cum_norm;
                float cs_bound_2 = 2.0f * cum_sum_2 * query_cum_norm;
                float cs_bound_3 = 2.0f * cum_sum_3 * query_cum_norm;

                float lower_bound_0 = new_exact_0 - cs_bound_0;
                float lower_bound_1 = new_exact_1 - cs_bound_1;
                float lower_bound_2 = new_exact_2 - cs_bound_2;
                float lower_bound_3 = new_exact_3 - cs_bound_3;

                // The following code is not the most branch friendly (due to
                // the maintenance of the candidate heap), but micro-benchmarks
                // have shown that it is not worth it to write horrible code to
                // squeeze out those cycles.
                if (!C::cmp(lower_bound_0, threshold)) {
                    exact_distances[next_batch_size] = new_exact_0;
                    index_array[next_batch_size] = idx_0;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_0, new_exact_0);
                }
                if (!C::cmp(lower_bound_1, threshold)) {
                    exact_distances[next_batch_size] = new_exact_1;
                    index_array[next_batch_size] = idx_1;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_1, new_exact_1);
                }
                if (!C::cmp(lower_bound_2, threshold)) {
                    exact_distances[next_batch_size] = new_exact_2;
                    index_array[next_batch_size] = idx_2;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_2, new_exact_2);
                }
                if (!C::cmp(lower_bound_3, threshold)) {
                    exact_distances[next_batch_size] = new_exact_3;
                    index_array[next_batch_size] = idx_3;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_3, new_exact_3);
                }
            }

            // Process the remaining candidates.
            for (; i < batch_size; i++) {
                idx_t idx = index_array[i];

                float dp = flat_codes_qdis->partial_dot_product(
                        idx, start_dim, end_dim - start_dim);
                ndis += 1;
                float new_exact = exact_distances[i] - 2.0f * dp;

                float cum_sum = panorama_index->get_cum_sum(
                        idx)[curr_panorama_level + 1];
                float cs_bound = 2.0f * cum_sum * query_cum_norm;
                float lower_bound = new_exact - cs_bound;

                if (!C::cmp(lower_bound, threshold)) {
                    exact_distances[next_batch_size] = new_exact;
                    index_array[next_batch_size] = idx;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx, new_exact);
                }
            }

            local_pano_stats.total_dims_scanned +=
                    batch_size * (end_dim - start_dim);
            batch_size = next_batch_size;
            curr_panorama_level++;
        }

        // Add surviving candidates to the result handler.
        for (size_t i = 0; i < batch_size; i++) {
            idx_t idx = index_array[i];
            if (res.add_result(exact_distances[i], idx)) {
                threshold = res.threshold;
                nres += 1;
            }
            candidates.push(idx, exact_distances[i]);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    indexPanorama_stats.add(local_pano_stats);
    return nres;
}

namespace {

template <typename T, typename Container, typename Compare>
void reservePriorityQueue(
        std::priority_queue<T, Container, Compare>& q,
        std::size_t size) {
    struct Access : std::priority_queue<T, Container, Compare> {
        using std::priority_queue<T, Container, Compare>::c;
    };
    Access access{std::move(q)};
    access.c.reserve(size);
    q = std::move(access);
}

/// Templated body of `search_from_candidate_unbounded`. The choice of
/// max-heap vs min-heap for both `top_candidates` and `candidates` is
/// derived from C via `TopCandidatesQueue` / `CandidatesQueue`.
template <class C>
TopCandidatesQueue<C> search_from_candidate_unbounded_impl(
        const HNSW& hnsw,
        const HNSW::Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats) {
    int ndis = 0;
    TopCandidatesQueue<C> top_candidates;
    reservePriorityQueue(top_candidates, ef);

    CandidatesQueue<C> candidates;
    reservePriorityQueue(candidates, ef);

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (C::cmp(d0, top_candidates.top().first)) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            vt->prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (C::cmp(top_candidates.top().first, dis) ||
                top_candidates.size() < static_cast<size_t>(ef)) {
                candidates.emplace(dis, idx);
                top_candidates.emplace(dis, idx);

                if (top_candidates.size() > static_cast<size_t>(ef)) {
                    top_candidates.pop();
                }
            }
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            saved_j[counter] = v1;
            counter += vt->set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;

                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);

            ndis += 1;
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

} // namespace

/// Public dispatcher: only the distance (CMax) flavor is exposed because
/// its return type — `std::priority_queue<HNSW::Node>` — is the CMax
/// max-heap. Internal callers that need similarity mode use
/// `search_from_candidate_unbounded_impl<C_similarity>` directly.
std::priority_queue<HNSW::Node> hnsw_detail::search_from_candidate_unbounded(
        const HNSW& hnsw,
        const HNSW::Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats) {
    return search_from_candidate_unbounded_impl<HNSW::C_distance>(
            hnsw, node, qdis, ef, vt, stats);
}

namespace {

// just used as a lower bound for the minmaxheap, but it is set for heap search
template <class C>
int extract_k_from_ResultHandler(ResultHandler& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<typename RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}

template <class C>
HNSWStats search_impl(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler& res,
        VisitedTable& vt,
        const SearchParameters* params) {
    HNSWStats stats;
    if (hnsw.entry_point == -1) {
        return stats;
    }
    int k = extract_k_from_ResultHandler<C>(res);

    bool bounded_queue = hnsw.search_bounded_queue;
    int cur_efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            bounded_queue = hnsw_params->bounded_queue;
            cur_efSearch = hnsw_params->efSearch;
        }
    }

    //  greedy search on upper levels
    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);

    for (int level = hnsw.max_level; level >= 1; level--) {
        HNSWStats local_stats = greedy_update_nearest_impl<C>(
                hnsw, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    int ef = std::max(cur_efSearch, k);
    if (bounded_queue) { // this is the most common branch, for now we only
                         // support Panorama search in this branch
        MinimaxHeapT<HC_for<C>> candidates(ef);

        candidates.push(nearest, d_nearest);

        if (!hnsw.is_panorama) {
            search_from_candidates_dispatch<C>(
                    hnsw, qdis, res, candidates, vt, stats, 0, 0, params);
        } else {
            // Panorama is L2-specific and is only valid for C_distance.
            // The public dispatch ensures we never reach this code path
            // with C != C_distance, but assert in debug builds.
            if constexpr (std::is_same_v<C, HNSW::C_distance>) {
                hnsw_detail::search_from_candidates_panorama(
                        hnsw,
                        index,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        0,
                        params);
            } else {
                FAISS_THROW_MSG(
                        "Panorama search is not supported with is_similarity=true");
            }
        }
    } else {
        auto top_candidates = search_from_candidate_unbounded_impl<C>(
                hnsw, HNSW::Node(d_nearest, nearest), qdis, ef, &vt, stats);

        while (top_candidates.size() > static_cast<size_t>(k)) {
            top_candidates.pop();
        }

        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            res.add_result(d, label);
            top_candidates.pop();
        }
    }

    vt.advance();

    return stats;
}

template <class C>
void search_level_0_impl(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt,
        const SearchParameters* params) {
    auto cur_efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            cur_efSearch = hnsw_params->efSearch;
        }
    }

    int k = extract_k_from_ResultHandler<C>(res);

    if (search_type == 1) {
        int nres = 0;

        for (idx_t j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0) {
                break;
            }

            if (vt.get(cj)) {
                continue;
            }

            int candidates_size = std::max(cur_efSearch, k);
            MinimaxHeapT<HC_for<C>> candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates_dispatch<C>(
                    hnsw,
                    qdis,
                    res,
                    candidates,
                    vt,
                    search_stats,
                    0,
                    nres,
                    params);
            nres = std::min(nres, candidates_size);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(cur_efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeapT<HC_for<C>> candidates(candidates_size);
        for (idx_t j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0) {
                break;
            }
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates_dispatch<C>(
                hnsw, qdis, res, candidates, vt, search_stats, 0, 0, params);
    }
}

} // namespace

HNSWStats HNSW::search(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler& res,
        VisitedTable& vt,
        const SearchParameters* params) const {
    if (is_similarity) {
        return search_impl<C_similarity>(*this, qdis, index, res, vt, params);
    }
    return search_impl<C_distance>(*this, qdis, index, res, vt, params);
}

void HNSW::search_level_0(
        DistanceComputer& qdis,
        ResultHandler& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt,
        const SearchParameters* params) const {
    if (is_similarity) {
        search_level_0_impl<C_similarity>(
                *this,
                qdis,
                res,
                nprobe,
                nearest_i,
                nearest_d,
                search_type,
                search_stats,
                vt,
                params);
    } else {
        search_level_0_impl<C_distance>(
                *this,
                qdis,
                res,
                nprobe,
                nearest_i,
                nearest_d,
                search_type,
                search_stats,
                vt,
                params);
    }
}

void HNSW::permute_entries(const idx_t* map) {
    // remap levels
    storage_idx_t ntotal = levels.size();
    std::vector<storage_idx_t> imap(ntotal); // inverse mapping
    // map: new index -> old index
    // imap: old index -> new index
    for (int i = 0; i < ntotal; i++) {
        assert(map[i] >= 0 && map[i] < ntotal);
        imap[map[i]] = i;
    }
    if (entry_point != -1) {
        entry_point = imap[entry_point];
    }
    std::vector<int> new_levels(ntotal);
    std::vector<size_t> new_offsets(ntotal + 1);
    std::vector<storage_idx_t> new_neighbors(neighbors.size());
    size_t no = 0;
    for (int i = 0; i < ntotal; i++) {
        storage_idx_t o = map[i]; // corresponding "old" index
        new_levels[i] = levels[o];
        for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
            storage_idx_t neigh = neighbors[j];
            new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
        }
        new_offsets[i + 1] = no;
    }
    assert(new_offsets[ntotal] == offsets[ntotal]);
    // swap everyone
    std::swap(levels, new_levels);
    std::swap(offsets, new_offsets);
    neighbors = std::move(new_neighbors);
}

} // namespace faiss
