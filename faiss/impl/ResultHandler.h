/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Structures that collect search results from distance computations
 */

#pragma once

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/partitioning.h>

#include <algorithm>
#include <iostream>

namespace faiss {

/*****************************************************************
 * The classes below are intended to be used as template arguments
 * they handle results for batches of queries (size nq).
 * They can be called in two ways:
 * - by instanciating a SingleResultHandler that tracks results for a single
 *   query
 * - with begin_multiple/add_results/end_multiple calls where a whole block of
 *   results is submitted
 * All classes are templated on C which to define wheter the min or the max of
 * results is to be kept, and on sel, so that the codepaths for with / without
 * selector can be separated at compile time.
 *****************************************************************/

template <class C, bool use_sel = false>
struct BlockResultHandler {
    size_t nq; // number of queries for which we search
    const IDSelector* sel;

    explicit BlockResultHandler(size_t nq, const IDSelector* sel = nullptr)
            : nq(nq), sel(sel) {
        assert(!use_sel || sel);
    }

    // currently handled query range
    size_t i0 = 0, i1 = 0;

    // start collecting results for queries [i0, i1)
    virtual void begin_multiple(size_t i0_2, size_t i1_2) {
        this->i0 = i0_2;
        this->i1 = i1_2;
    }

    // add results for queries [i0, i1) and database [j0, j1)
    virtual void add_results(size_t, size_t, const typename C::T*) {}

    // series of results for queries i0..i1 is done
    virtual void end_multiple() {}

    virtual ~BlockResultHandler() {}

    bool is_in_selection(idx_t i) const {
        return !use_sel || sel->is_member(i);
    }
};

// handler for a single query
template <class C>
struct ResultHandler {
    // if not better than threshold, then not necessary to call add_result
    typename C::T threshold = C::neutral();

    // return whether threshold was updated
    virtual bool add_result(typename C::T dis, typename C::TI idx) = 0;

    virtual ~ResultHandler() {}
};

/*****************************************************************
 * Single best result handler.
 * Tracks the only best result, thus avoiding storing
 * some temporary data in memory.
 *****************************************************************/

template <class C, bool use_sel = false>
struct Top1BlockResultHandler : BlockResultHandler<C, use_sel> {
    using T = typename C::T;
    using TI = typename C::TI;
    using BlockResultHandler<C, use_sel>::i0;
    using BlockResultHandler<C, use_sel>::i1;

    // contains exactly nq elements
    T* dis_tab;
    // contains exactly nq elements
    TI* ids_tab;

    Top1BlockResultHandler(
            size_t nq,
            T* dis_tab,
            TI* ids_tab,
            const IDSelector* sel = nullptr)
            : BlockResultHandler<C, use_sel>(nq, sel),
              dis_tab(dis_tab),
              ids_tab(ids_tab) {}

    struct SingleResultHandler : ResultHandler<C> {
        Top1BlockResultHandler& hr;
        using ResultHandler<C>::threshold;

        TI min_idx;
        size_t current_idx = 0;

        explicit SingleResultHandler(Top1BlockResultHandler& hr) : hr(hr) {}

        /// begin results for query # i
        void begin(const size_t current_idx_2) {
            this->current_idx = current_idx_2;
            threshold = C::neutral();
            min_idx = -1;
        }

        /// add one result for query i
        bool add_result(T dis, TI idx) final {
            if (C::cmp(this->threshold, dis)) {
                threshold = dis;
                min_idx = idx;
                return true;
            }
            return false;
        }

        /// series of results for query i is done
        void end() {
            hr.dis_tab[current_idx] = threshold;
            hr.ids_tab[current_idx] = min_idx;
        }
    };

    /// begin
    void begin_multiple(size_t i0, size_t i1) final {
        this->i0 = i0;
        this->i1 = i1;

        for (size_t i = i0; i < i1; i++) {
            this->dis_tab[i] = C::neutral();
        }
    }

    /// add results for query i0..i1 and j0..j1
    void add_results(size_t j0, size_t j1, const T* dis_tab_2) final {
        for (int64_t i = i0; i < i1; i++) {
            const T* dis_tab_i = dis_tab_2 + (j1 - j0) * (i - i0) - j0;

            auto& min_distance = this->dis_tab[i];
            auto& min_index = this->ids_tab[i];

            for (size_t j = j0; j < j1; j++) {
                const T distance = dis_tab_i[j];

                if (C::cmp(min_distance, distance)) {
                    min_distance = distance;
                    min_index = j;
                }
            }
        }
    }

    void add_result(const size_t i, const T dis, const TI idx) {
        auto& min_distance = this->dis_tab[i];
        auto& min_index = this->ids_tab[i];

        if (C::cmp(min_distance, dis)) {
            min_distance = dis;
            min_index = idx;
        }
    }
};

/*****************************************************************
 * Heap based result handler
 *****************************************************************/

template <class C, bool use_sel = false>
struct HeapBlockResultHandler : BlockResultHandler<C, use_sel> {
    using T = typename C::T;
    using TI = typename C::TI;
    using BlockResultHandler<C, use_sel>::i0;
    using BlockResultHandler<C, use_sel>::i1;

    T* heap_dis_tab;
    TI* heap_ids_tab;

    int64_t k; // number of results to keep

    HeapBlockResultHandler(
            size_t nq,
            T* heap_dis_tab,
            TI* heap_ids_tab,
            size_t k,
            const IDSelector* sel = nullptr)
            : BlockResultHandler<C, use_sel>(nq, sel),
              heap_dis_tab(heap_dis_tab),
              heap_ids_tab(heap_ids_tab),
              k(k) {}

    /******************************************************
     * API for 1 result at a time (each SingleResultHandler is
     * called from 1 thread)
     */

    struct SingleResultHandler : ResultHandler<C> {
        HeapBlockResultHandler& hr;
        using ResultHandler<C>::threshold;
        size_t k;

        T* heap_dis;
        TI* heap_ids;

        explicit SingleResultHandler(HeapBlockResultHandler& hr)
                : hr(hr), k(hr.k) {}

        /// begin results for query # i
        void begin(size_t i) {
            heap_dis = hr.heap_dis_tab + i * k;
            heap_ids = hr.heap_ids_tab + i * k;
            heap_heapify<C>(k, heap_dis, heap_ids);
            threshold = heap_dis[0];
        }

        /// add one result for query i
        bool add_result(T dis, TI idx) final {
            if (C::cmp(threshold, dis)) {
                heap_replace_top<C>(k, heap_dis, heap_ids, dis, idx);
                threshold = heap_dis[0];
                return true;
            }
            return false;
        }

        /// series of results for query i is done
        void end() {
            heap_reorder<C>(k, heap_dis, heap_ids);
        }
    };

    /******************************************************
     * API for multiple results (called from 1 thread)
     */

    /// begin
    void begin_multiple(size_t i0_2, size_t i1_2) final {
        this->i0 = i0_2;
        this->i1 = i1_2;
        for (size_t i = i0; i < i1; i++) {
            heap_heapify<C>(k, heap_dis_tab + i * k, heap_ids_tab + i * k);
        }
    }

    /// add results for query i0..i1 and j0..j1
    void add_results(size_t j0, size_t j1, const T* dis_tab) final {
#pragma omp parallel for
        for (int64_t i = i0; i < i1; i++) {
            T* heap_dis = heap_dis_tab + i * k;
            TI* heap_ids = heap_ids_tab + i * k;
            const T* dis_tab_i = dis_tab + (j1 - j0) * (i - i0) - j0;
            T thresh = heap_dis[0];
            for (size_t j = j0; j < j1; j++) {
                T dis = dis_tab_i[j];
                if (C::cmp(thresh, dis)) {
                    heap_replace_top<C>(k, heap_dis, heap_ids, dis, j);
                    thresh = heap_dis[0];
                }
            }
        }
    }

    /// series of results for queries i0..i1 is done
    void end_multiple() final {
        // maybe parallel for
        for (size_t i = i0; i < i1; i++) {
            heap_reorder<C>(k, heap_dis_tab + i * k, heap_ids_tab + i * k);
        }
    }
};

/*****************************************************************
 * Reservoir result handler
 *
 * A reservoir is a result array of size capacity > n (number of requested
 * results) all results below a threshold are stored in an arbitrary order. When
 * the capacity is reached, a new threshold is chosen by partitionning the
 * distance array.
 *****************************************************************/

/// Reservoir for a single query
template <class C>
struct ReservoirTopN : ResultHandler<C> {
    using T = typename C::T;
    using TI = typename C::TI;
    using ResultHandler<C>::threshold;

    T* vals;
    TI* ids;

    size_t i;        // number of stored elements
    size_t n;        // number of requested elements
    size_t capacity; // size of storage

    ReservoirTopN() {}

    ReservoirTopN(size_t n, size_t capacity, T* vals, TI* ids)
            : vals(vals), ids(ids), i(0), n(n), capacity(capacity) {
        assert(n < capacity);
        threshold = C::neutral();
    }

    bool add_result(T val, TI id) final {
        bool updated_threshold = false;
        if (C::cmp(threshold, val)) {
            if (i == capacity) {
                shrink_fuzzy();
                updated_threshold = true;
            }
            vals[i] = val;
            ids[i] = id;
            i++;
        }
        return updated_threshold;
    }

    void add(T val, TI id) {
        add_result(val, id);
    }

    // reduce storage from capacity to anything
    // between n and (capacity + n) / 2
    void shrink_fuzzy() {
        assert(i == capacity);

        threshold = partition_fuzzy<C>(
                vals, ids, capacity, n, (capacity + n) / 2, &i);
    }

    void shrink() {
        threshold = partition<C>(vals, ids, i, n);
        i = n;
    }

    void to_result(T* heap_dis, TI* heap_ids) const {
        for (int j = 0; j < std::min(i, n); j++) {
            heap_push<C>(j + 1, heap_dis, heap_ids, vals[j], ids[j]);
        }

        if (i < n) {
            heap_reorder<C>(i, heap_dis, heap_ids);
            // add empty results
            heap_heapify<C>(n - i, heap_dis + i, heap_ids + i);
        } else {
            // add remaining elements
            heap_addn<C>(n, heap_dis, heap_ids, vals + n, ids + n, i - n);
            heap_reorder<C>(n, heap_dis, heap_ids);
        }
    }
};

template <class C, bool use_sel = false>
struct ReservoirBlockResultHandler : BlockResultHandler<C, use_sel> {
    using T = typename C::T;
    using TI = typename C::TI;
    using BlockResultHandler<C, use_sel>::i0;
    using BlockResultHandler<C, use_sel>::i1;

    T* heap_dis_tab;
    TI* heap_ids_tab;

    int64_t k;       // number of results to keep
    size_t capacity; // capacity of the reservoirs

    ReservoirBlockResultHandler(
            size_t nq,
            T* heap_dis_tab,
            TI* heap_ids_tab,
            size_t k,
            const IDSelector* sel = nullptr)
            : BlockResultHandler<C, use_sel>(nq, sel),
              heap_dis_tab(heap_dis_tab),
              heap_ids_tab(heap_ids_tab),
              k(k) {
        // double then round up to multiple of 16 (for SIMD alignment)
        capacity = (2 * k + 15) & ~15;
    }

    /******************************************************
     * API for 1 result at a time (each SingleResultHandler is
     * called from 1 thread)
     */

    struct SingleResultHandler : ReservoirTopN<C> {
        ReservoirBlockResultHandler& hr;

        std::vector<T> reservoir_dis;
        std::vector<TI> reservoir_ids;

        explicit SingleResultHandler(ReservoirBlockResultHandler& hr)
                : ReservoirTopN<C>(hr.k, hr.capacity, nullptr, nullptr),
                  hr(hr) {}

        size_t qno;

        /// begin results for query # i
        void begin(size_t qno_2) {
            reservoir_dis.resize(hr.capacity);
            reservoir_ids.resize(hr.capacity);
            this->vals = reservoir_dis.data();
            this->ids = reservoir_ids.data();
            this->i = 0; // size of reservoir
            this->threshold = C::neutral();
            this->qno = qno_2;
        }

        /// series of results for query qno is done
        void end() {
            T* heap_dis = hr.heap_dis_tab + qno * hr.k;
            TI* heap_ids = hr.heap_ids_tab + qno * hr.k;
            this->to_result(heap_dis, heap_ids);
        }
    };

    /******************************************************
     * API for multiple results (called from 1 thread)
     */

    std::vector<T> reservoir_dis;
    std::vector<TI> reservoir_ids;
    std::vector<ReservoirTopN<C>> reservoirs;

    /// begin
    void begin_multiple(size_t i0_2, size_t i1_2) {
        this->i0 = i0_2;
        this->i1 = i1_2;
        reservoir_dis.resize((i1 - i0) * capacity);
        reservoir_ids.resize((i1 - i0) * capacity);
        reservoirs.clear();
        for (size_t i = i0_2; i < i1_2; i++) {
            reservoirs.emplace_back(
                    k,
                    capacity,
                    reservoir_dis.data() + (i - i0_2) * capacity,
                    reservoir_ids.data() + (i - i0_2) * capacity);
        }
    }

    /// add results for query i0..i1 and j0..j1
    void add_results(size_t j0, size_t j1, const T* dis_tab) {
#pragma omp parallel for
        for (int64_t i = i0; i < i1; i++) {
            ReservoirTopN<C>& reservoir = reservoirs[i - i0];
            const T* dis_tab_i = dis_tab + (j1 - j0) * (i - i0) - j0;
            for (size_t j = j0; j < j1; j++) {
                T dis = dis_tab_i[j];
                reservoir.add_result(dis, j);
            }
        }
    }

    /// series of results for queries i0..i1 is done
    void end_multiple() final {
        // maybe parallel for
        for (size_t i = i0; i < i1; i++) {
            reservoirs[i - i0].to_result(
                    heap_dis_tab + i * k, heap_ids_tab + i * k);
        }
    }
};

/*****************************************************************
 * Result handler for range searches
 *****************************************************************/

template <class C, bool use_sel = false>
struct RangeSearchBlockResultHandler : BlockResultHandler<C, use_sel> {
    using T = typename C::T;
    using TI = typename C::TI;
    using BlockResultHandler<C, use_sel>::i0;
    using BlockResultHandler<C, use_sel>::i1;

    RangeSearchResult* res;
    T radius;

    RangeSearchBlockResultHandler(
            RangeSearchResult* res,
            float radius,
            const IDSelector* sel = nullptr)
            : BlockResultHandler<C, use_sel>(res->nq, sel),
              res(res),
              radius(radius) {}

    /******************************************************
     * API for 1 result at a time (each SingleResultHandler is
     * called from 1 thread)
     ******************************************************/

    struct SingleResultHandler : ResultHandler<C> {
        // almost the same interface as RangeSearchResultHandler
        using ResultHandler<C>::threshold;
        RangeSearchPartialResult pres;
        RangeQueryResult* qr = nullptr;

        explicit SingleResultHandler(RangeSearchBlockResultHandler& rh)
                : pres(rh.res) {
            threshold = rh.radius;
        }

        /// begin results for query # i
        void begin(size_t i) {
            qr = &pres.new_result(i);
        }

        /// add one result for query i
        bool add_result(T dis, TI idx) final {
            if (C::cmp(threshold, dis)) {
                qr->add(dis, idx);
            }
            return false;
        }

        /// series of results for query i is done
        void end() {}

        ~SingleResultHandler() {
            try {
                // finalize the partial result
                pres.finalize();
            } catch ([[maybe_unused]] const faiss::FaissException& e) {
                // Do nothing if allocation fails in finalizing partial results.
#ifndef NDEBUG
                std::cerr << e.what() << std::endl;
#endif
            }
        }
    };

    /******************************************************
     * API for multiple results (called from 1 thread)
     ******************************************************/

    std::vector<RangeSearchPartialResult*> partial_results;
    std::vector<size_t> j0s;
    int pr = 0;

    /// begin
    void begin_multiple(size_t i0_2, size_t i1_2) {
        this->i0 = i0_2;
        this->i1 = i1_2;
    }

    /// add results for query i0..i1 and j0..j1

    void add_results(size_t j0, size_t j1, const T* dis_tab) {
        RangeSearchPartialResult* pres;
        // there is one RangeSearchPartialResult structure per j0
        // (= block of columns of the large distance matrix)
        // it is a bit tricky to find the poper PartialResult structure
        // because the inner loop is on db not on queries.

        if (pr < j0s.size() && j0 == j0s[pr]) {
            pres = partial_results[pr];
            pr++;
        } else if (j0 == 0 && j0s.size() > 0) {
            pr = 0;
            pres = partial_results[pr];
            pr++;
        } else { // did not find this j0
            pres = new RangeSearchPartialResult(res);
            partial_results.push_back(pres);
            j0s.push_back(j0);
            pr = partial_results.size();
        }

        for (size_t i = i0; i < i1; i++) {
            const float* ip_line = dis_tab + (i - i0) * (j1 - j0);
            RangeQueryResult& qres = pres->new_result(i);

            for (size_t j = j0; j < j1; j++) {
                float dis = *ip_line++;
                if (C::cmp(radius, dis)) {
                    qres.add(dis, j);
                }
            }
        }
    }

    ~RangeSearchBlockResultHandler() {
        try {
            if (partial_results.size() > 0) {
                RangeSearchPartialResult::merge(partial_results);
            }
        } catch ([[maybe_unused]] const faiss::FaissException& e) {
            // Do nothing if allocation fails in merge.
#ifndef NDEBUG
            std::cerr << e.what() << std::endl;
#endif
        }
    }
};

/*****************************************************************
 * Dispatcher function to choose the right knn result handler depending on k
 *****************************************************************/

// declared in distances.cpp
FAISS_API extern int distance_compute_min_k_reservoir;

template <class Consumer, class... Types>
typename Consumer::T dispatch_knn_ResultHandler(
        size_t nx,
        float* vals,
        int64_t* ids,
        size_t k,
        MetricType metric,
        const IDSelector* sel,
        Consumer& consumer,
        Types... args) {
#define DISPATCH_C_SEL(C, use_sel)                                          \
    if (k == 1) {                                                           \
        Top1BlockResultHandler<C, use_sel> res(nx, vals, ids, sel);         \
        return consumer.template f<>(res, args...);                         \
    } else if (k < distance_compute_min_k_reservoir) {                      \
        HeapBlockResultHandler<C, use_sel> res(nx, vals, ids, k, sel);      \
        return consumer.template f<>(res, args...);                         \
    } else {                                                                \
        ReservoirBlockResultHandler<C, use_sel> res(nx, vals, ids, k, sel); \
        return consumer.template f<>(res, args...);                         \
    }

    if (is_similarity_metric(metric)) {
        using C = CMin<float, int64_t>;
        if (sel) {
            DISPATCH_C_SEL(C, true);
        } else {
            DISPATCH_C_SEL(C, false);
        }
    } else {
        using C = CMax<float, int64_t>;
        if (sel) {
            DISPATCH_C_SEL(C, true);
        } else {
            DISPATCH_C_SEL(C, false);
        }
    }
#undef DISPATCH_C_SEL
}

template <class Consumer, class... Types>
typename Consumer::T dispatch_range_ResultHandler(
        RangeSearchResult* res,
        float radius,
        MetricType metric,
        const IDSelector* sel,
        Consumer& consumer,
        Types... args) {
#define DISPATCH_C_SEL(C, use_sel)                                    \
    RangeSearchBlockResultHandler<C, use_sel> resb(res, radius, sel); \
    return consumer.template f<>(resb, args...);

    if (is_similarity_metric(metric)) {
        using C = CMin<float, int64_t>;
        if (sel) {
            DISPATCH_C_SEL(C, true);
        } else {
            DISPATCH_C_SEL(C, false);
        }
    } else {
        using C = CMax<float, int64_t>;
        if (sel) {
            DISPATCH_C_SEL(C, true);
        } else {
            DISPATCH_C_SEL(C, false);
        }
    }
#undef DISPATCH_C_SEL
}

} // namespace faiss
