/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


/*
 * Structures that collect search results from distance computations
 */

#pragma once


#include <faiss/utils/Heap.h>
#include <faiss/impl/AuxIndexStructures.h>


namespace faiss {



template<class C>
struct HeapResultHandler {

    using T = typename C::T;
    using TI = typename C::TI;

    int nq;
    T *heap_dis_tab;
    TI *heap_ids_tab;

    int64_t k;  // number of results to keep

    HeapResultHandler(
        size_t nq,
        T * heap_dis_tab, TI * heap_ids_tab,
        size_t k):
        nq(nq),
        heap_dis_tab(heap_dis_tab), heap_ids_tab(heap_ids_tab), k(k)
    {

    }

    /******************************************************
     * API for 1 result at a time (each SingleResultHandler is
     * called from 1 thread)
     ******************************************************/

    struct SingleResultHandler {
        HeapResultHandler & hr;
        size_t k;

        T *heap_dis;
        TI *heap_ids;
        T thresh;

        SingleResultHandler(HeapResultHandler &hr): hr(hr), k(hr.k) {}

        /// begin results for query # i
        void begin(size_t i) {
            heap_dis = hr.heap_dis_tab + i * k;
            heap_ids = hr.heap_ids_tab + i * k;
            heap_heapify<C> (k, heap_dis, heap_ids);
            thresh = heap_dis[0];
        }

        /// add one result for query i
        void add_result(T dis, TI idx) {
            if (C::cmp(heap_dis[0], dis)) {
                heap_pop<C>(k, heap_dis, heap_ids);
                heap_push<C>(k, heap_dis, heap_ids, dis, idx);
                thresh = heap_dis[0];
            }
        }

        /// series of results for query i is done
        void end() {
            heap_reorder<C> (k, heap_dis, heap_ids);
        }
    };


    /******************************************************
     * API for multiple results (called from 1 thread)
     ******************************************************/

    size_t i0, i1;

    /// begin
    void begin_multiple(size_t i0, size_t i1) {
        this->i0 = i0;
        this->i1 = i1;
        for(size_t i = i0; i < i1; i++) {
            heap_heapify<C> (k, heap_dis_tab + i * k, heap_ids_tab + i * k);
        }
    }

    /// add results for query i0..i1 and j0..j1
    void add_results(size_t j0, size_t j1, const T *dis_tab) {
        // maybe parallel for
        for (size_t i = i0; i < i1; i++) {
            T * heap_dis = heap_dis_tab + i * k;
            TI * heap_ids = heap_ids_tab + i * k;
            T thresh = heap_dis[0];
            for (size_t j = j0; j < j1; j++) {
                T dis = *dis_tab++;
                if (C::cmp(thresh, dis)) {
                    heap_pop<C>(k, heap_dis, heap_ids);
                    heap_push<C>(k, heap_dis, heap_ids, dis, j);
                    thresh = heap_dis[0];
                }
            }
        }
    }

    /// series of results for queries i0..i1 is done
    void end_multiple() {
        // maybe parallel for
        for(size_t i = i0; i < i1; i++) {
            heap_reorder<C> (k, heap_dis_tab + i * k, heap_ids_tab + i * k);
        }
    }

};


template<class C>
struct RangeSearchResultHandler {
    using T = typename C::T;
    using TI = typename C::TI;

    RangeSearchResult *res;
    float radius;

    RangeSearchResultHandler(RangeSearchResult *res, float radius):
        res(res), radius(radius)
    {}

    /******************************************************
     * API for 1 result at a time (each SingleResultHandler is
     * called from 1 thread)
     ******************************************************/

    struct SingleResultHandler {
        // almost the same interface as RangeSearchResultHandler
        RangeSearchPartialResult pres;
        float radius;
        RangeQueryResult *qr = nullptr;

        SingleResultHandler(RangeSearchResultHandler &rh):
            pres(rh.res), radius(rh.radius)
        {}

        /// begin results for query # i
        void begin(size_t i) {
            qr = &pres.new_result(i);
        }

        /// add one result for query i
        void add_result(T dis, TI idx) {

            if (C::cmp(radius, dis)) {
                qr->add(dis, idx);
            }
        }

        /// series of results for query i is done
        void end() {
        }

        ~SingleResultHandler() {
            pres.finalize();
        }
    };

    /******************************************************
     * API for multiple results (called from 1 thread)
     ******************************************************/

    size_t i0, i1;

    std::vector <RangeSearchPartialResult *> partial_results;
    std::vector <size_t> j0s;
    int pr = 0;

    /// begin
    void begin_multiple(size_t i0, size_t i1) {
        this->i0 = i0;
        this->i1 = i1;
    }

    /// add results for query i0..i1 and j0..j1

    void add_results(size_t j0, size_t j1, const T *dis_tab) {
        RangeSearchPartialResult *pres;
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
            pres = new RangeSearchPartialResult (res);
            partial_results.push_back(pres);
            j0s.push_back(j0);
            pr = partial_results.size();
        }

        for (size_t i = i0; i < i1; i++) {
            const float *ip_line = dis_tab + (i - i0) * (j1 - j0);
            RangeQueryResult & qres = pres->new_result (i);

            for (size_t j = j0; j < j1; j++) {
                float dis = *ip_line++;
                if (C::cmp(radius, dis)) {
                    qres.add (dis, j);
                }
            }
        }
    }

    void end_multiple() {

    }

    ~RangeSearchResultHandler() {
        if (partial_results.size() > 0) {
            RangeSearchPartialResult::merge (partial_results);
        }
    }

};




}  // namespace faiss

