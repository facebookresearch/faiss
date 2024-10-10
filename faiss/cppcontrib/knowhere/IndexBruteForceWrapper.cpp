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

#include <faiss/cppcontrib/knowhere/IndexBruteForceWrapper.h>

#include <algorithm>
#include <memory>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>

#include <faiss/cppcontrib/knowhere/impl/Bruteforce.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

IndexBruteForceWrapper::IndexBruteForceWrapper(Index* underlying_index)
        : IndexWrapper{underlying_index} {}

void IndexBruteForceWrapper::search(
        idx_t n,
        const float* __restrict x,
        idx_t k,
        float* __restrict distances,
        idx_t* __restrict labels,
        const SearchParameters* __restrict params) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t check_period =
            InterruptCallback::get_period_hint(index->d * index->ntotal);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<DistanceComputer> dis(
                    index->get_distance_computer());

#pragma omp for schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                // prepare the query
                dis->set_query(x + i * index->d);

                // allocate heap
                idx_t* const __restrict local_ids = labels + i * index->d;
                float* const __restrict local_distances =
                        distances + i * index->d;

                // set up a filter
                IDSelector* __restrict sel =
                        (params == nullptr) ? nullptr : params->sel;

                if (is_similarity_metric(index->metric_type)) {
                    using C = CMin<float, idx_t>;

                    if (sel == nullptr) {
                        // Compiler is expected to de-virtualize virtual method
                        // calls
                        IDSelectorAll sel_all;
                        brute_force_search_impl<
                                C,
                                DistanceComputer,
                                IDSelectorAll>(
                                index->ntotal,
                                *dis,
                                sel_all,
                                k,
                                local_distances,
                                local_ids);
                    } else {
                        brute_force_search_impl<
                                C,
                                DistanceComputer,
                                IDSelector>(
                                index->ntotal,
                                *dis,
                                *sel,
                                k,
                                local_distances,
                                local_ids);
                    }
                } else {
                    using C = CMax<float, idx_t>;

                    if (sel == nullptr) {
                        // Compiler is expected to de-virtualize virtual method
                        // calls
                        IDSelectorAll sel_all;
                        brute_force_search_impl<
                                C,
                                DistanceComputer,
                                IDSelectorAll>(
                                index->ntotal,
                                *dis,
                                sel_all,
                                k,
                                local_distances,
                                local_ids);
                    } else {
                        brute_force_search_impl<
                                C,
                                DistanceComputer,
                                IDSelector>(
                                index->ntotal,
                                *dis,
                                *sel,
                                k,
                                local_distances,
                                local_ids);
                    }
                }
            }
        }

        InterruptCallback::check();
    }
}

void IndexBruteForceWrapper::range_search(
        idx_t n,
        const float* __restrict x,
        float radius,
        RangeSearchResult* __restrict result,
        const SearchParameters* __restrict params) const {
    using RH_min = RangeSearchBlockResultHandler<CMax<float, int64_t>>;
    using RH_max = RangeSearchBlockResultHandler<CMin<float, int64_t>>;
    RH_min bres_min(result, radius);
    RH_max bres_max(result, radius);

    idx_t check_period =
            InterruptCallback::get_period_hint(index->d * index->ntotal);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<DistanceComputer> dis(
                    index->get_distance_computer());

            typename RH_min::SingleResultHandler res_min(bres_min);
            typename RH_max::SingleResultHandler res_max(bres_max);

#pragma omp for schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                // prepare the query
                dis->set_query(x + i * index->d);

                // set up a filter
                IDSelector* __restrict sel =
                        (params == nullptr) ? nullptr : params->sel;

                if (is_similarity_metric(index->metric_type)) {
                    res_max.begin(i);

                    if (sel == nullptr) {
                        // Compiler is expected to de-virtualize virtual method
                        // calls
                        IDSelectorAll sel_all;

                        brute_force_range_search_impl<
                                typename RH_max::SingleResultHandler,
                                DistanceComputer,
                                IDSelectorAll>(
                                index->ntotal, *dis, sel_all, res_max);
                    } else {
                        brute_force_range_search_impl<
                                typename RH_max::SingleResultHandler,
                                DistanceComputer,
                                IDSelector>(index->ntotal, *dis, *sel, res_max);
                    }

                    res_max.end();
                } else {
                    res_min.begin(i);

                    if (sel == nullptr) {
                        // Compiler is expected to de-virtualize virtual method
                        // calls
                        IDSelectorAll sel_all;

                        brute_force_range_search_impl<
                                typename RH_min::SingleResultHandler,
                                DistanceComputer,
                                IDSelectorAll>(
                                index->ntotal, *dis, sel_all, res_min);
                    } else {
                        brute_force_range_search_impl<
                                typename RH_min::SingleResultHandler,
                                DistanceComputer,
                                IDSelector>(index->ntotal, *dis, *sel, res_min);
                    }

                    res_min.end();
                }
            }
        }

        InterruptCallback::check();
    }
}

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
