/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>

#include <faiss/utils/Heap.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/partitioning.h>

/** This file contains callbacks for kernels that compute distances.
 */

namespace faiss {

struct SIMDResultHandler {
    // used to dispatch templates
    bool is_CMax = false;
    uint8_t sizeof_ids = 0;
    bool with_fields = false;

    /**  called when 32 distances are computed and provided in two
     *   simd16uint16. (q, b) indicate which entry it is in the block. */
    virtual void handle(
            size_t q,
            size_t b,
            simd16uint16 d0,
            simd16uint16 d1) = 0;

    /// set the sub-matrix that is being computed
    virtual void set_block_origin(size_t i0, size_t j0) = 0;

    virtual ~SIMDResultHandler() {}
};

/* Result handler that will return float resutls eventually */
struct SIMDResultHandlerToFloat : SIMDResultHandler {
    size_t nq;     // number of queries
    size_t ntotal; // ignore excess elements after ntotal

    /// these fields are used mainly for the IVF variants (with_id_map=true)
    const idx_t* id_map = nullptr; // map offset in invlist to vector id
    const int* q_map = nullptr;    // map q to global query
    const uint16_t* dbias =
            nullptr; // table of biases to add to each query (for IVF L2 search)
    const float* normalizers = nullptr; // size 2 * nq, to convert

    SIMDResultHandlerToFloat(size_t nq, size_t ntotal)
            : nq(nq), ntotal(ntotal) {}

    virtual void begin(const float* norms) {
        normalizers = norms;
    }

    // called at end of search to convert int16 distances to float, before
    // normalizers are deallocated
    virtual void end() {
        normalizers = nullptr;
    }
};

FAISS_API extern bool simd_result_handlers_accept_virtual;

namespace simd_result_handlers {

/** Dummy structure that just computes a chqecksum on results
 * (to avoid the computation to be optimized away) */
struct DummyResultHandler : SIMDResultHandler {
    size_t cs = 0;

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
        cs += q * 123 + b * 789 + d0.get_scalar_0() + d1.get_scalar_0();
    }

    void set_block_origin(size_t, size_t) final {}

    ~DummyResultHandler() {}
};

/** memorize results in a nq-by-nb matrix.
 *
 * j0 is the current upper-left block of the matrix
 */
struct StoreResultHandler : SIMDResultHandler {
    uint16_t* data;
    size_t ld; // total number of columns
    size_t i0 = 0;
    size_t j0 = 0;

    StoreResultHandler(uint16_t* data, size_t ld) : data(data), ld(ld) {}

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
        size_t ofs = (q + i0) * ld + j0 + b * 32;
        d0.store(data + ofs);
        d1.store(data + ofs + 16);
    }

    void set_block_origin(size_t i0_in, size_t j0_in) final {
        this->i0 = i0_in;
        this->j0 = j0_in;
    }
};

/** stores results in fixed-size matrix. */
template <int NQ, int BB>
struct FixedStorageHandler : SIMDResultHandler {
    simd16uint16 dis[NQ][BB];
    int i0 = 0;

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
        dis[q + i0][2 * b] = d0;
        dis[q + i0][2 * b + 1] = d1;
    }

    void set_block_origin(size_t i0_in, size_t j0_in) final {
        this->i0 = i0_in;
        assert(j0_in == 0);
    }

    template <class OtherResultHandler>
    void to_other_handler(OtherResultHandler& other) const {
        for (int q = 0; q < NQ; q++) {
            for (int b = 0; b < BB; b += 2) {
                other.handle(q, b / 2, dis[q][b], dis[q][b + 1]);
            }
        }
    }

    virtual ~FixedStorageHandler() {}
};

/** Result handler that compares distances to check if they need to be kept */
template <class C, bool with_id_map>
struct ResultHandlerCompare : SIMDResultHandlerToFloat {
    using TI = typename C::TI;

    bool disable = false;

    int64_t i0 = 0; // query origin
    int64_t j0 = 0; // db origin

    const IDSelector* sel;

    ResultHandlerCompare(size_t nq, size_t ntotal, const IDSelector* sel_in)
            : SIMDResultHandlerToFloat(nq, ntotal), sel{sel_in} {
        this->is_CMax = C::is_max;
        this->sizeof_ids = sizeof(typename C::TI);
        this->with_fields = with_id_map;
    }

    void set_block_origin(size_t i0_in, size_t j0_in) final {
        this->i0 = i0_in;
        this->j0 = j0_in;
    }

    // adjust handler data for IVF.
    void adjust_with_origin(size_t& q, simd16uint16& d0, simd16uint16& d1) {
        q += i0;

        if (dbias) {
            simd16uint16 dbias16(dbias[q]);
            d0 += dbias16;
            d1 += dbias16;
        }

        if (with_id_map) { // FIXME test on q_map instead
            q = q_map[q];
        }
    }

    // compute and adjust idx
    int64_t adjust_id(size_t b, size_t j) {
        int64_t idx = j0 + 32 * b + j;
        if (with_id_map) {
            idx = id_map[idx];
        }
        return idx;
    }

    /// return binary mask of elements below thr in (d0, d1)
    /// inverse_test returns elements above
    uint32_t get_lt_mask(
            uint16_t thr,
            size_t b,
            simd16uint16 d0,
            simd16uint16 d1) {
        simd16uint16 thr16(thr);
        uint32_t lt_mask;

        constexpr bool keep_min = C::is_max;
        if (keep_min) {
            lt_mask = ~cmp_ge32(d0, d1, thr16);
        } else {
            lt_mask = ~cmp_le32(d0, d1, thr16);
        }

        if (lt_mask == 0) {
            return 0;
        }
        uint64_t idx = j0 + b * 32;
        if (idx + 32 > ntotal) {
            if (idx >= ntotal) {
                return 0;
            }
            int nbit = (ntotal - idx);
            lt_mask &= (uint32_t(1) << nbit) - 1;
        }
        return lt_mask;
    }

    virtual ~ResultHandlerCompare() {}
};

/** Special version for k=1 */
template <class C, bool with_id_map = false>
struct SingleResultHandler : ResultHandlerCompare<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;
    using RHC = ResultHandlerCompare<C, with_id_map>;
    using RHC::normalizers;

    std::vector<int16_t> idis;
    float* dis;
    int64_t* ids;

    SingleResultHandler(
            size_t nq,
            size_t ntotal,
            float* dis,
            int64_t* ids,
            const IDSelector* sel_in)
            : RHC(nq, ntotal, sel_in), idis(nq), dis(dis), ids(ids) {
        for (size_t i = 0; i < nq; i++) {
            ids[i] = -1;
            idis[i] = C::neutral();
        }
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
        if (this->disable) {
            return;
        }

        this->adjust_with_origin(q, d0, d1);

        uint32_t lt_mask = this->get_lt_mask(idis[q], b, d0, d1);
        if (!lt_mask) {
            return;
        }

        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        if (this->sel != nullptr) {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                auto real_idx = this->adjust_id(b, j);
                lt_mask -= 1 << j;
                if (this->sel->is_member(real_idx)) {
                    T d = d32tab[j];
                    if (C::cmp(idis[q], d)) {
                        idis[q] = d;
                        ids[q] = real_idx;
                    }
                }
            }
        } else {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                lt_mask -= 1 << j;
                T d = d32tab[j];
                if (C::cmp(idis[q], d)) {
                    idis[q] = d;
                    ids[q] = this->adjust_id(b, j);
                }
            }
        }
    }

    void end() {
        for (size_t q = 0; q < this->nq; q++) {
            if (!normalizers) {
                dis[q] = idis[q];
            } else {
                float one_a = 1 / normalizers[2 * q];
                float b = normalizers[2 * q + 1];
                dis[q] = b + idis[q] * one_a;
            }
        }
    }
};

/** Structure that collects results in a min- or max-heap */
template <class C, bool with_id_map = false>
struct HeapHandler : ResultHandlerCompare<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;
    using RHC = ResultHandlerCompare<C, with_id_map>;
    using RHC::normalizers;

    std::vector<uint16_t> idis;
    std::vector<TI> iids;
    float* dis;
    int64_t* ids;

    int64_t k; // number of results to keep

    HeapHandler(
            size_t nq,
            size_t ntotal,
            int64_t k,
            float* dis,
            int64_t* ids,
            const IDSelector* sel_in)
            : RHC(nq, ntotal, sel_in),
              idis(nq * k),
              iids(nq * k),
              dis(dis),
              ids(ids),
              k(k) {
        heap_heapify<C>(k * nq, idis.data(), iids.data());
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
        if (this->disable) {
            return;
        }

        this->adjust_with_origin(q, d0, d1);

        T* heap_dis = idis.data() + q * k;
        TI* heap_ids = iids.data() + q * k;

        uint16_t cur_thresh =
                heap_dis[0] < 65536 ? (uint16_t)(heap_dis[0]) : 0xffff;

        // here we handle the reverse comparison case as well
        uint32_t lt_mask = this->get_lt_mask(cur_thresh, b, d0, d1);

        if (!lt_mask) {
            return;
        }

        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        if (this->sel != nullptr) {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                auto real_idx = this->adjust_id(b, j);
                lt_mask -= 1 << j;
                if (this->sel->is_member(real_idx)) {
                    T dis_2 = d32tab[j];
                    if (C::cmp(heap_dis[0], dis_2)) {
                        heap_replace_top<C>(
                                k, heap_dis, heap_ids, dis_2, real_idx);
                    }
                }
            }
        } else {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                lt_mask -= 1 << j;
                T dis_2 = d32tab[j];
                if (C::cmp(heap_dis[0], dis_2)) {
                    int64_t idx = this->adjust_id(b, j);
                    heap_replace_top<C>(k, heap_dis, heap_ids, dis_2, idx);
                }
            }
        }
    }

    void end() override {
        for (size_t q = 0; q < this->nq; q++) {
            T* heap_dis_in = idis.data() + q * k;
            TI* heap_ids_in = iids.data() + q * k;
            heap_reorder<C>(k, heap_dis_in, heap_ids_in);
            float* heap_dis = dis + q * k;
            int64_t* heap_ids = ids + q * k;

            float one_a = 1.0, b = 0.0;
            if (normalizers) {
                one_a = 1 / normalizers[2 * q];
                b = normalizers[2 * q + 1];
            }
            for (int j = 0; j < k; j++) {
                heap_dis[j] = heap_dis_in[j] * one_a + b;
                heap_ids[j] = heap_ids_in[j];
            }
        }
    }
};

/** Simple top-N implementation using a reservoir.
 *
 * Results are stored when they are below the threshold until the capacity is
 * reached. Then a partition sort is used to update the threshold. */

/** Handler built from several ReservoirTopN (one per query) */
template <class C, bool with_id_map = false>
struct ReservoirHandler : ResultHandlerCompare<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;
    using RHC = ResultHandlerCompare<C, with_id_map>;
    using RHC::normalizers;

    size_t capacity; // rounded up to multiple of 16

    // where the final results will be written
    float* dis;
    int64_t* ids;

    std::vector<TI> all_ids;
    AlignedTable<T> all_vals;
    std::vector<ReservoirTopN<C>> reservoirs;

    ReservoirHandler(
            size_t nq,
            size_t ntotal,
            size_t k,
            size_t cap,
            float* dis,
            int64_t* ids,
            const IDSelector* sel_in)
            : RHC(nq, ntotal, sel_in),
              capacity((cap + 15) & ~15),
              dis(dis),
              ids(ids) {
        assert(capacity % 16 == 0);
        all_ids.resize(nq * capacity);
        all_vals.resize(nq * capacity);
        for (size_t q = 0; q < nq; q++) {
            reservoirs.emplace_back(
                    k,
                    capacity,
                    all_vals.get() + q * capacity,
                    all_ids.data() + q * capacity);
        }
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
        if (this->disable) {
            return;
        }
        this->adjust_with_origin(q, d0, d1);

        ReservoirTopN<C>& res = reservoirs[q];
        uint32_t lt_mask = this->get_lt_mask(res.threshold, b, d0, d1);

        if (!lt_mask) {
            return;
        }
        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        if (this->sel != nullptr) {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                auto real_idx = this->adjust_id(b, j);
                lt_mask -= 1 << j;
                if (this->sel->is_member(real_idx)) {
                    T dis_2 = d32tab[j];
                    res.add(dis_2, real_idx);
                }
            }
        } else {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                lt_mask -= 1 << j;
                T dis_2 = d32tab[j];
                res.add(dis_2, this->adjust_id(b, j));
            }
        }
    }

    void end() override {
        using Cf = typename std::conditional<
                C::is_max,
                CMax<float, int64_t>,
                CMin<float, int64_t>>::type;

        std::vector<int> perm(reservoirs[0].n);
        for (size_t q = 0; q < reservoirs.size(); q++) {
            ReservoirTopN<C>& res = reservoirs[q];
            size_t n = res.n;

            if (res.i > res.n) {
                res.shrink();
            }
            int64_t* heap_ids = ids + q * n;
            float* heap_dis = dis + q * n;

            float one_a = 1.0, b = 0.0;
            if (normalizers) {
                one_a = 1 / normalizers[2 * q];
                b = normalizers[2 * q + 1];
            }
            for (size_t i = 0; i < res.i; i++) {
                perm[i] = i;
            }
            // indirect sort of result arrays
            std::sort(perm.begin(), perm.begin() + res.i, [&res](int i, int j) {
                return C::cmp(res.vals[j], res.vals[i]);
            });
            for (size_t i = 0; i < res.i; i++) {
                heap_dis[i] = res.vals[perm[i]] * one_a + b;
                heap_ids[i] = res.ids[perm[i]];
            }

            // possibly add empty results
            heap_heapify<Cf>(n - res.i, heap_dis + res.i, heap_ids + res.i);
        }
    }
};

/** Result handler for range search. The difficulty is that the range distances
 * have to be scaled using the scaler.
 */

template <class C, bool with_id_map = false>
struct RangeHandler : ResultHandlerCompare<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;
    using RHC = ResultHandlerCompare<C, with_id_map>;
    using RHC::normalizers;
    using RHC::nq;

    RangeSearchResult& rres;
    float radius;
    std::vector<uint16_t> thresholds;
    std::vector<size_t> n_per_query;
    size_t q0 = 0;

    // we cannot use the RangeSearchPartialResult interface because queries can
    // be performed by batches
    struct Triplet {
        idx_t q;
        idx_t b;
        uint16_t dis;
    };
    std::vector<Triplet> triplets;

    RangeHandler(
            RangeSearchResult& rres,
            float radius,
            size_t ntotal,
            const IDSelector* sel_in)
            : RHC(rres.nq, ntotal, sel_in), rres(rres), radius(radius) {
        thresholds.resize(nq);
        n_per_query.resize(nq + 1);
    }

    virtual void begin(const float* norms) override {
        normalizers = norms;
        for (int q = 0; q < nq; ++q) {
            thresholds[q] =
                    int(normalizers[2 * q] * (radius - normalizers[2 * q + 1]));
        }
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
        if (this->disable) {
            return;
        }
        this->adjust_with_origin(q, d0, d1);

        uint32_t lt_mask = this->get_lt_mask(thresholds[q], b, d0, d1);

        if (!lt_mask) {
            return;
        }
        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        if (this->sel != nullptr) {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                lt_mask -= 1 << j;

                auto real_idx = this->adjust_id(b, j);
                if (this->sel->is_member(real_idx)) {
                    T dis = d32tab[j];
                    n_per_query[q]++;
                    triplets.push_back({idx_t(q + q0), real_idx, dis});
                }
            }
        } else {
            while (lt_mask) {
                // find first non-zero
                int j = __builtin_ctz(lt_mask);
                lt_mask -= 1 << j;
                T dis = d32tab[j];
                n_per_query[q]++;
                triplets.push_back({idx_t(q + q0), this->adjust_id(b, j), dis});
            }
        }
    }

    void end() override {
        memcpy(rres.lims, n_per_query.data(), sizeof(n_per_query[0]) * nq);
        rres.do_allocation();
        for (auto it = triplets.begin(); it != triplets.end(); ++it) {
            size_t& l = rres.lims[it->q];
            rres.distances[l] = it->dis;
            rres.labels[l] = it->b;
            l++;
        }
        memmove(rres.lims + 1, rres.lims, sizeof(*rres.lims) * rres.nq);
        rres.lims[0] = 0;

        for (int q = 0; q < nq; q++) {
            float one_a = 1 / normalizers[2 * q];
            float b = normalizers[2 * q + 1];
            for (size_t i = rres.lims[q]; i < rres.lims[q + 1]; i++) {
                rres.distances[i] = rres.distances[i] * one_a + b;
            }
        }
    }
};

#ifndef SWIG

// handler for a subset of queries
template <class C, bool with_id_map = false>
struct PartialRangeHandler : RangeHandler<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;
    using RHC = RangeHandler<C, with_id_map>;
    using RHC::normalizers;
    using RHC::nq, RHC::q0, RHC::triplets, RHC::n_per_query;

    RangeSearchPartialResult& pres;

    PartialRangeHandler(
            RangeSearchPartialResult& pres,
            float radius,
            size_t ntotal,
            size_t q0,
            size_t q1,
            const IDSelector* sel_in)
            : RangeHandler<C, with_id_map>(*pres.res, radius, ntotal, sel_in),
              pres(pres) {
        nq = q1 - q0;
        this->q0 = q0;
    }

    // shift left n_per_query
    void shift_n_per_query() {
        memmove(n_per_query.data() + 1,
                n_per_query.data(),
                nq * sizeof(n_per_query[0]));
        n_per_query[0] = 0;
    }

    // commit to partial result instead of full RangeResult
    void end() override {
        std::vector<typename RHC::Triplet> sorted_triplets(triplets.size());
        for (int q = 0; q < nq; q++) {
            n_per_query[q + 1] += n_per_query[q];
        }
        shift_n_per_query();

        for (size_t i = 0; i < triplets.size(); i++) {
            sorted_triplets[n_per_query[triplets[i].q - q0]++] = triplets[i];
        }
        shift_n_per_query();

        size_t* lims = n_per_query.data();

        for (int q = 0; q < nq; q++) {
            float one_a = 1 / normalizers[2 * q];
            float b = normalizers[2 * q + 1];
            RangeQueryResult& qres = pres.new_result(q + q0);
            for (size_t i = lims[q]; i < lims[q + 1]; i++) {
                qres.add(
                        sorted_triplets[i].dis * one_a + b,
                        sorted_triplets[i].b);
            }
        }
    }
};

#endif

/********************************************************************************
 * Dynamic dispatching function. The consumer should have a templatized method f
 * that will be replaced with the actual SIMDResultHandler that is determined
 * dynamically.
 */

template <class C, bool W, class Consumer, class... Types>
void dispatch_SIMDResultHandler_fixedCW(
        SIMDResultHandler& res,
        Consumer& consumer,
        Types... args) {
    if (auto resh = dynamic_cast<SingleResultHandler<C, W>*>(&res)) {
        consumer.template f<SingleResultHandler<C, W>>(*resh, args...);
    } else if (auto resh_2 = dynamic_cast<HeapHandler<C, W>*>(&res)) {
        consumer.template f<HeapHandler<C, W>>(*resh_2, args...);
    } else if (auto resh_2 = dynamic_cast<ReservoirHandler<C, W>*>(&res)) {
        consumer.template f<ReservoirHandler<C, W>>(*resh_2, args...);
    } else { // generic handler -- will not be inlined
        FAISS_THROW_IF_NOT_FMT(
                simd_result_handlers_accept_virtual,
                "Running vitrual handler for %s",
                typeid(res).name());
        consumer.template f<SIMDResultHandler>(res, args...);
    }
}

template <class C, class Consumer, class... Types>
void dispatch_SIMDResultHandler_fixedC(
        SIMDResultHandler& res,
        Consumer& consumer,
        Types... args) {
    if (res.with_fields) {
        dispatch_SIMDResultHandler_fixedCW<C, true>(res, consumer, args...);
    } else {
        dispatch_SIMDResultHandler_fixedCW<C, false>(res, consumer, args...);
    }
}

template <class Consumer, class... Types>
void dispatch_SIMDResultHandler(
        SIMDResultHandler& res,
        Consumer& consumer,
        Types... args) {
    if (res.sizeof_ids == 0) {
        if (auto resh = dynamic_cast<StoreResultHandler*>(&res)) {
            consumer.template f<StoreResultHandler>(*resh, args...);
        } else if (auto resh_2 = dynamic_cast<DummyResultHandler*>(&res)) {
            consumer.template f<DummyResultHandler>(*resh_2, args...);
        } else { // generic path
            FAISS_THROW_IF_NOT_FMT(
                    simd_result_handlers_accept_virtual,
                    "Running vitrual handler for %s",
                    typeid(res).name());
            consumer.template f<SIMDResultHandler>(res, args...);
        }
    } else if (res.sizeof_ids == sizeof(int)) {
        if (res.is_CMax) {
            dispatch_SIMDResultHandler_fixedC<CMax<uint16_t, int>>(
                    res, consumer, args...);
        } else {
            dispatch_SIMDResultHandler_fixedC<CMin<uint16_t, int>>(
                    res, consumer, args...);
        }
    } else if (res.sizeof_ids == sizeof(int64_t)) {
        if (res.is_CMax) {
            dispatch_SIMDResultHandler_fixedC<CMax<uint16_t, int64_t>>(
                    res, consumer, args...);
        } else {
            dispatch_SIMDResultHandler_fixedC<CMin<uint16_t, int64_t>>(
                    res, consumer, args...);
        }
    } else {
        FAISS_THROW_FMT("Unknown id size %d", res.sizeof_ids);
    }
}

} // namespace simd_result_handlers

} // namespace faiss
