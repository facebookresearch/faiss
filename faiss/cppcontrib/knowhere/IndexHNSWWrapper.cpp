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

#include <faiss/cppcontrib/knowhere/IndexHNSWWrapper.h>

#include <algorithm>
#include <memory>

#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>

#include <faiss/cppcontrib/knowhere/impl/HnswSearcher.h>
#include <faiss/cppcontrib/knowhere/utils/Bitset.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// a visitor that does nothing
struct DummyVisitor {
    using storage_idx_t = HNSW::storage_idx_t;

    void visit_level(const int level) {
        // does nothing
    }

    void visit_edge(
            const int level,
            const storage_idx_t node_from,
            const storage_idx_t node_to,
            const float distance) {
        // does nothing
    }
};

/**************************************************************
 * Utilities
 **************************************************************/

namespace {

// cloned from IndexHNSW.cpp
DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

} // namespace

/**************************************************************
 * IndexHNSWWrapper implementation
 **************************************************************/

IndexHNSWWrapper::IndexHNSWWrapper(IndexHNSW* underlying_index)
        : IndexWrapper(underlying_index) {}

void IndexHNSWWrapper::search(
        idx_t n,
        const float* __restrict x,
        idx_t k,
        float* __restrict distances,
        idx_t* __restrict labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    const IndexHNSW* index_hnsw = dynamic_cast<const IndexHNSW*>(index);
    FAISS_THROW_IF_NOT(index_hnsw);

    FAISS_THROW_IF_NOT_MSG(index_hnsw->storage, "No storage index");

    // set up
    using C = HNSW::C;

    // check if the graph is empty
    if (index_hnsw->hnsw.entry_point == -1) {
        for (idx_t i = 0; i < k * n; i++) {
            distances[i] = C::neutral();
            labels[i] = -1;
        }

        return;
    }

    // check parameters
    const SearchParametersHNSWWrapper* params = nullptr;
    const HNSW& hnsw = index_hnsw->hnsw;

    float kAlpha = 0.0f;
    int efSearch = hnsw.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSWWrapper*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
        kAlpha = params->kAlpha;
    }

    // set up hnsw_stats
    HNSWStats* __restrict const hnsw_stats =
            (params == nullptr) ? nullptr : params->hnsw_stats;

    //
    size_t n1 = 0;
    size_t n2 = 0;
    size_t ndis = 0;
    size_t nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            Bitset bitset_visited_nodes =
                    Bitset::create_uninitialized(index->ntotal);

            // create a distance computer
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index_hnsw->storage));

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                // prepare the query
                dis->set_query(x + i * index->d);

                // prepare the table of visited elements
                bitset_visited_nodes.clear();

                // a visitor
                DummyVisitor graph_visitor;

                // future results
                HNSWStats local_stats;

                // set up a filter
                IDSelector* sel = (params == nullptr) ? nullptr : params->sel;
                if (sel == nullptr) {
                    // no filter.
                    // It it expected that a compile will be able to
                    //   de-virtualize the class.
                    IDSelectorAll sel_all;

                    using searcher_type = v2_hnsw_searcher<
                            DistanceComputer,
                            DummyVisitor,
                            Bitset,
                            IDSelectorAll>;

                    searcher_type searcher{
                            hnsw,
                            *(dis.get()),
                            graph_visitor,
                            bitset_visited_nodes,
                            sel_all,
                            kAlpha,
                            params};

                    local_stats = searcher.search(
                            k, distances + i * k, labels + i * k);
                } else {
                    // there is a filter
                    using searcher_type = v2_hnsw_searcher<
                            DistanceComputer,
                            DummyVisitor,
                            Bitset,
                            IDSelector>;

                    searcher_type searcher{
                            hnsw,
                            *(dis.get()),
                            graph_visitor,
                            bitset_visited_nodes,
                            *sel,
                            kAlpha,
                            params};

                    local_stats = searcher.search(
                            k, distances + i * k, labels + i * k);
                }

                // update stats if possible
                if (hnsw_stats != nullptr) {
                    n1 += local_stats.n1;
                    n2 += local_stats.n2;
                    ndis += local_stats.ndis;
                    nhops += local_stats.nhops;
                }
            }
        }

        InterruptCallback::check();
    }

    // update stats if possible
    if (hnsw_stats != nullptr) {
        hnsw_stats->combine({n1, n2, ndis, nhops});
    }

    // done, update the results, if needed
    if (is_similarity_metric(index->metric_type)) {
        // we need to revert the negated distances
        for (idx_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSWWrapper::range_search(
        idx_t n,
        const float* x,
        float radius_in,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    const IndexHNSW* index_hnsw = dynamic_cast<const IndexHNSW*>(index);
    FAISS_THROW_IF_NOT(index_hnsw);

    FAISS_THROW_IF_NOT_MSG(index_hnsw->storage, "No storage index");

    // check if the graph is empty
    if (index_hnsw->hnsw.entry_point == -1) {
        return;
    }

    // check parameters
    const SearchParametersHNSWWrapper* params = nullptr;
    const HNSW& hnsw = index_hnsw->hnsw;

    float kAlpha = 0.0f;
    int efSearch = hnsw.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSWWrapper*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");

        kAlpha = params->kAlpha;
        efSearch = params->efSearch;
    }

    // set up hnsw_stats
    HNSWStats* __restrict const hnsw_stats =
            (params == nullptr) ? nullptr : params->hnsw_stats;

    //
    size_t n1 = 0;
    size_t n2 = 0;
    size_t ndis = 0;
    size_t nhops = 0;

    // radius
    float radius = radius_in;
    if (is_similarity_metric(this->metric_type)) {
        radius *= (-1);
    }

    // initialize a ResultHandler
    using RH_min = RangeSearchBlockResultHandler<CMax<float, int64_t>>;
    RH_min bres_min(result, radius);

    // no parallelism by design
    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            //
            Bitset bitset_visited_nodes =
                    Bitset::create_uninitialized(index->ntotal);

            // create a distance computer
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index_hnsw->storage));

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                typename RH_min::SingleResultHandler res_min(bres_min);
                res_min.begin(i);

                // prepare the query
                dis->set_query(x + i * index->d);

                // prepare the table of visited elements
                bitset_visited_nodes.clear();

                // future results
                HNSWStats local_stats;

                // set up a filter
                IDSelector* sel = (params == nullptr) ? nullptr : params->sel;

                if (sel == nullptr) {
                    IDSelectorAll sel_all;
                    DummyVisitor graph_visitor;

                    using searcher_type = v2_hnsw_searcher<
                            DistanceComputer,
                            DummyVisitor,
                            Bitset,
                            IDSelectorAll>;

                    searcher_type searcher(
                            hnsw,
                            *(dis.get()),
                            graph_visitor,
                            bitset_visited_nodes,
                            sel_all,
                            kAlpha,
                            params);

                    local_stats = searcher.range_search(radius, &res_min);
                } else {
                    DummyVisitor graph_visitor;

                    using searcher_type = v2_hnsw_searcher<
                            DistanceComputer,
                            DummyVisitor,
                            Bitset,
                            IDSelector>;

                    searcher_type searcher{
                            hnsw,
                            *(dis.get()),
                            graph_visitor,
                            bitset_visited_nodes,
                            *sel,
                            kAlpha,
                            params};

                    local_stats = searcher.range_search(radius, &res_min);
                }

                // update stats if possible
                if (hnsw_stats != nullptr) {
                    n1 += local_stats.n1;
                    n2 += local_stats.n2;
                    ndis += local_stats.ndis;
                    nhops += local_stats.nhops;
                }

                //
                res_min.end();
            }
        }
    }

    // update stats if possible
    if (hnsw_stats != nullptr) {
        hnsw_stats->combine({n1, n2, ndis, nhops});
    }

    // done, update the results, if needed
    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
