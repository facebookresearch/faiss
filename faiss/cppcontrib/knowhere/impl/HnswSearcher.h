// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

// standard headers
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>

// Faiss-specific headers
#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/ordered_key_value.h>

// Knowhere-specific headers
#include <faiss/cppcontrib/knowhere/impl/Neighbor.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

namespace {

// whether to track statistics
constexpr bool track_hnsw_stats = true;

} // namespace

// Accomodates all the search logic and variables.
/// * DistanceComputerT is responsible for computing distances
/// * GraphVisitorT records visited edges
/// * VisitedT is responsible for tracking visited nodes
/// * FilterT is resposible for filtering unneeded nodes
/// Interfaces of all templates are tweaked to accept standard Faiss structures
///   with dynamic dispatching. Custom Knowhere structures are also accepted.
template <
        typename DistanceComputerT,
        typename GraphVisitorT,
        typename VisitedT,
        typename FilterT>
struct v2_hnsw_searcher {
    using storage_idx_t = faiss::HNSW::storage_idx_t;
    using idx_t = faiss::idx_t;

    // hnsw structure.
    // the reference is not owned.
    const faiss::HNSW& hnsw;

    // computes distances. it already knows the query vector.
    // the reference is not owned.
    DistanceComputerT& qdis;

    // records visited edges.
    // the reference is not owned.
    GraphVisitorT& graph_visitor;

    // tracks the nodes that have been visited already.
    // the reference is not owned.
    VisitedT& visited_nodes;

    // a filter for disabled nodes.
    // the reference is not owned.
    const FilterT& filter;

    // parameter for the filtering
    const float kAlpha;

    // custom parameters of HNSW search.
    // the pointer is not owned.
    const faiss::SearchParametersHNSW* params;

    //
    v2_hnsw_searcher(
            const faiss::HNSW& hnsw_,
            DistanceComputerT& qdis_,
            GraphVisitorT& graph_visitor_,
            VisitedT& visited_nodes_,
            const FilterT& filter_,
            const float kAlpha_,
            const faiss::SearchParametersHNSW* params_)
            : hnsw{hnsw_},
              qdis{qdis_},
              graph_visitor{graph_visitor_},
              visited_nodes{visited_nodes_},
              filter{filter_},
              kAlpha{kAlpha_},
              params{params_} {}

    v2_hnsw_searcher(const v2_hnsw_searcher&) = delete;
    v2_hnsw_searcher(v2_hnsw_searcher&&) = delete;
    v2_hnsw_searcher& operator=(const v2_hnsw_searcher&) = delete;
    v2_hnsw_searcher& operator=(v2_hnsw_searcher&&) = delete;

    // greedily update a nearest vector at a given level.
    // * the update starts from the value in 'nearest'.
    faiss::HNSWStats greedy_update_nearest(
            const int level,
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::HNSWStats stats;

        for (;;) {
            storage_idx_t prev_nearest = nearest;

            size_t begin = 0;
            size_t end = 0;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const float dis) {
                graph_visitor.visit_edge(level, prev_nearest, idx, dis);

                if (dis < d_nearest) {
                    nearest = idx;
                    d_nearest = dis;
                }
            };

            size_t counter = 0;
            storage_idx_t saved_indices[4];

            // visit neighbors
            size_t count = 0;
            for (size_t i = begin; i < end; i++) {
                storage_idx_t v = hnsw.neighbors[i];
                if (v < 0) {
                    // no more neighbors
                    break;
                }

                count += 1;

                saved_indices[counter] = v;
                counter += 1;

                if (counter == 4) {
                    // evaluate 4x distances at once
                    float dis[4] = {0, 0, 0, 0};
                    qdis.distances_batch_4(
                            saved_indices[0],
                            saved_indices[1],
                            saved_indices[2],
                            saved_indices[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(saved_indices[id4], dis[id4]);
                    }

                    counter = 0;
                }
            }

            // process leftovers
            for (size_t id4 = 0; id4 < counter; id4++) {
                // evaluate a single distance
                const float dis = qdis(saved_indices[id4]);

                update_with_candidate(saved_indices[id4], dis);
            }

            // update stats
            if (track_hnsw_stats) {
                stats.ndis += count;
                stats.nhops += 1;
            }

            // we're done if there we no changes
            if (nearest == prev_nearest) {
                return stats;
            }
        }
    }

    // no loops, just check neighbors of a single node.
    template <typename FuncAddCandidate>
    faiss::HNSWStats evaluate_single_node(
            const idx_t node_id,
            const int level,
            float& accumulated_alpha,
            FuncAddCandidate func_add_candidate) {
        // // unused
        // bool do_dis_check = params ? params->check_relative_distance
        //                            : hnsw.check_relative_distance;

        faiss::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(node_id, level, &begin, &end);

        // todo: add prefetch
        size_t counter = 0;
        storage_idx_t saved_indices[4];
        int saved_statuses[4];

        size_t ndis = 0;
        for (size_t j = begin; j < end; j++) {
            const storage_idx_t v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                // no more neighbors
                break;
            }

            // already visited?
            if (visited_nodes.get(v1)) {
                // yes, visited.
                graph_visitor.visit_edge(level, node_id, v1, -1);
                continue;
            }

            // not visited. mark as visited.
            visited_nodes.set(v1);

            // is the node disabled?
            int status = knowhere::Neighbor::kValid;
            if (!filter.is_member(v1)) {
                // yes, disabled
                status = knowhere::Neighbor::kInvalid;

                // sometimes, disabled nodes are allowed to be used
                accumulated_alpha += kAlpha;
                if (accumulated_alpha < 1.0f) {
                    continue;
                }

                accumulated_alpha -= 1.0f;
            }

            saved_indices[counter] = v1;
            saved_statuses[counter] = status;
            counter += 1;

            ndis += 1;

            if (counter == 4) {
                // evaluate 4x distances at once
                float dis[4] = {0, 0, 0, 0};
                qdis.distances_batch_4(
                        saved_indices[0],
                        saved_indices[1],
                        saved_indices[2],
                        saved_indices[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    // record a traversed edge
                    graph_visitor.visit_edge(
                            level, node_id, saved_indices[id4], dis[id4]);

                    // add a record of visited nodes
                    knowhere::Neighbor nn(
                            saved_indices[id4], dis[id4], saved_statuses[id4]);
                    if (func_add_candidate(nn)) {
#if defined(USE_PREFETCH)
                        // TODO
                        // _mm_prefetch(get_linklist0(v), _MM_HINT_T0);
#endif
                    }
                }

                counter = 0;
            }
        }

        // process leftovers
        for (size_t id4 = 0; id4 < counter; id4++) {
            // evaluate a single distance
            const float dis = qdis(saved_indices[id4]);

            // record a traversed edge
            graph_visitor.visit_edge(level, node_id, saved_indices[id4], dis);

            // add a record of visited
            knowhere::Neighbor nn(saved_indices[id4], dis, saved_statuses[id4]);
            if (func_add_candidate(nn)) {
#if defined(USE_PREFETCH)
                // TODO
                // _mm_prefetch(get_linklist0(v), _MM_HINT_T0);
#endif
            }
        }

        // update stats
        if (track_hnsw_stats) {
            stats.ndis = ndis;
            stats.nhops = 1;
        }

        // done
        return stats;
    }

    // perform the search on a given level.
    // it is assumed that retset is initialized and contains the initial nodes.
    faiss::HNSWStats search_on_a_level(
            knowhere::NeighborSetDoublePopList& retset,
            const int level,
            knowhere::IteratorMinHeap* const __restrict disqualified = nullptr,
            const float initial_accumulated_alpha = 1.0f) {
        faiss::HNSWStats stats;

        //
        float accumulated_alpha = initial_accumulated_alpha;

        // what to do with a accepted candidate
        auto add_search_candidate = [&](const knowhere::Neighbor n) {
            return retset.insert(n, disqualified);
        };

        // iterate while possible
        while (retset.has_next()) {
            // get a node to be processed
            const knowhere::Neighbor neighbor = retset.pop();

            // analyze its neighbors
            faiss::HNSWStats local_stats = evaluate_single_node(
                    neighbor.id,
                    level,
                    accumulated_alpha,
                    add_search_candidate);

            // update stats
            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        // done
        return stats;
    }

    // traverse down to the level 0
    faiss::HNSWStats greedy_search_top_levels(
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::HNSWStats stats;

        // iterate through upper levels
        for (int level = hnsw.max_level; level >= 1; level--) {
            // update the visitor
            graph_visitor.visit_level(level);

            // alter the value of 'nearest'
            faiss::HNSWStats local_stats =
                    greedy_update_nearest(level, nearest, d_nearest);

            // update stats
            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        return stats;
    }

    // perform the search.
    faiss::HNSWStats search(
            const idx_t k,
            float* __restrict distances,
            idx_t* __restrict labels) {
        faiss::HNSWStats stats;

        // is the graph empty?
        if (hnsw.entry_point == -1) {
            return stats;
        }

        // grab some needed parameters
        const int efSearch = params ? params->efSearch : hnsw.efSearch;

        // yes.
        // greedy search on upper levels.

        // initialize the starting point.
        storage_idx_t nearest = hnsw.entry_point;
        float d_nearest = qdis(nearest);

        // iterate through upper levels
        auto bottom_levels_stats = greedy_search_top_levels(nearest, d_nearest);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(bottom_levels_stats);
        }

        // level 0 search

        // update the visitor
        graph_visitor.visit_level(0);

        // initialize the container for candidates
        const idx_t n_candidates = std::max((idx_t)efSearch, k);
        knowhere::NeighborSetDoublePopList retset(n_candidates);

        // initialize retset with a single 'nearest' point
        {
            if (!filter.is_member(nearest)) {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kInvalid));
            } else {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kValid));
            }

            visited_nodes[nearest] = true;
        }

        // perform the search of the level 0.
        faiss::HNSWStats local_stats = search_on_a_level(retset, 0);

        // todo: switch to brute-force in case of (retset.size() < k)

        // populate the result
        const idx_t len = std::min((idx_t)retset.size(), k);
        for (idx_t i = 0; i < len; i++) {
            distances[i] = retset[i].distance;
            labels[i] = (idx_t)retset[i].id;
        }

        // update stats
        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        // done
        return stats;
    }

    faiss::HNSWStats range_search(
            const float radius,
            typename faiss::RangeSearchBlockResultHandler<
                    faiss::CMax<float, int64_t>>::
                    SingleResultHandler* const __restrict rres) {
        faiss::HNSWStats stats;

        // is the graph empty?
        if (hnsw.entry_point == -1) {
            return stats;
        }

        // grab some needed parameters
        const int efSearch = params ? params->efSearch : hnsw.efSearch;

        // yes.
        // greedy search on upper levels.

        // initialize the starting point.
        storage_idx_t nearest = hnsw.entry_point;
        float d_nearest = qdis(nearest);

        // iterate through upper levels
        auto bottom_levels_stats = greedy_search_top_levels(nearest, d_nearest);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(bottom_levels_stats);
        }

        // level 0 search

        // update the visitor
        graph_visitor.visit_level(0);

        // initialize the container for candidates
        const idx_t n_candidates = efSearch;
        knowhere::NeighborSetDoublePopList retset(n_candidates);

        // initialize retset with a single 'nearest' point
        {
            if (!filter.is_member(nearest)) {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kInvalid));
            } else {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kValid));
            }

            visited_nodes[nearest] = true;
        }

        // perform the search of the level 0.
        faiss::HNSWStats local_stats = search_on_a_level(retset, 0);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        // select candidates that match our criteria
        faiss::HNSWStats pick_stats;

        visited_nodes.clear();

        std::queue<std::pair<float, int64_t>> radius_queue;
        for (size_t i = retset.size(); (i--) > 0;) {
            const auto candidate = retset[i];
            if (candidate.distance < radius) {
                radius_queue.push({candidate.distance, candidate.id});
                rres->add_result(candidate.distance, candidate.id);

                visited_nodes[candidate.id] = true;
            }
        }

        while (!radius_queue.empty()) {
            auto current = radius_queue.front();
            radius_queue.pop();

            size_t id_begin = 0;
            size_t id_end = 0;
            hnsw.neighbor_range(current.second, 0, &id_begin, &id_end);

            for (size_t id = id_begin; id < id_end; id++) {
                const auto ngb = hnsw.neighbors[id];
                if (ngb == -1) {
                    break;
                }

                if (visited_nodes[ngb]) {
                    continue;
                }

                visited_nodes[ngb] = true;

                if (filter.is_member(ngb)) {
                    const float dis = qdis(ngb);
                    if (dis < radius) {
                        radius_queue.push({dis, ngb});
                        rres->add_result(dis, ngb);
                    }

                    if (track_hnsw_stats) {
                        pick_stats.ndis += 1;
                    }
                }
            }
        }

        // update stats
        if (track_hnsw_stats) {
            stats.combine(pick_stats);
        }

        return stats;
    }
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
