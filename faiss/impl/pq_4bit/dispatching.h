/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** This header contains functions that dispatch the runtime parameters to
 * compile-time template parameters */

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/pq_4bit/pq4_fast_scan.h>
#include <faiss/impl/pq_4bit/simd_result_handlers.h>

#include <faiss/impl/pq_4bit/kernels_simd256.h>

namespace faiss {

/** Mix-in class that manages both an SIMD result hander and offers the actual
 * scanning routines. */
template <class ResultHandler, class Scaler>
struct ScannerMixIn : ResultHandler {
    Scaler scaler;

    // args are forwarded to the ResutlHandler constructor
    template <class... ConstructorTypes>
    ScannerMixIn(int norm_scale, ConstructorTypes... args)
            : ResultHandler(args...), scaler(norm_scale) {}

    void accumulate_loop(
            int nq,
            size_t nb,
            int bbs,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) override {
        constexpr SIMDLevel SL = ResultHandler::SL;
        pq4_accumulate_loop_fixed_scaler<SL, ResultHandler>(
                nq, nb, bbs, nsq, codes, LUT, *this, scaler);
    }

    void accumulate_loop_qbs(
            int qbs,
            size_t nb,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) override {
        pq4_accumulate_loop_qbs_fixed_scaler<ResultHandler, Scaler>(
                qbs, nb, nsq, codes, LUT, *this, scaler);
    }
};

// instantiate the ResultHandler part of the PQ4Scanner. The type of handler is
// determined by the function parameters (so make_handler_2 is overloaded
// several times)

template <SIMDLevel SL, bool with_id_map, class C, class Scaler>
PQ4CodeScanner* make_handler_2(
        int norm_scale,
        bool use_reservoir,
        size_t nq,
        size_t ntotal,
        size_t k,
        float* dis,
        int64_t* ids,
        const IDSelector* sel) {
    if (k == 1) {
        return new ScannerMixIn<
                SingleResultHandler<C, with_id_map, SL>,
                Scaler>(norm_scale, nq, ntotal, dis, ids, sel);
    } else if (use_reservoir) {
        return new ScannerMixIn<ReservoirHandler<C, with_id_map, SL>, Scaler>(
                norm_scale, nq, ntotal, k, 2 * k, dis, ids, sel);
    } else {
        return new ScannerMixIn<HeapHandler<C, with_id_map, SL>, Scaler>(
                norm_scale, nq, ntotal, k, dis, ids, sel);
    }
}

template <SIMDLevel SL, bool with_id_map, class C, class Scaler>
PQ4CodeScanner* make_handler_2(
        int norm_scale,
        RangeSearchResult* rres,
        float radius,
        size_t ntotal,
        const IDSelector* sel) {
    return new ScannerMixIn<RangeHandler<C, with_id_map, SL>, Scaler>(
            norm_scale, rres, radius, ntotal, sel);
}

template <SIMDLevel SL, bool with_id_map, class C, class Scaler>
PQ4CodeScanner* make_handler_2(
        int norm_scale,
        RangeSearchPartialResult* pres,
        float radius,
        size_t ntotal,
        size_t q0,
        size_t q1,
        const IDSelector* sel) {
    return new ScannerMixIn<PartialRangeHandler<C, with_id_map, SL>, Scaler>(
            norm_scale, pres, radius, ntotal, q0, q1, sel);
}

// this function dispatches runtime -> template parameters. It is generic for
//  the different instances of make_handler_2. Be careful not to pass
// structs by references here becasue they will be copied by value not by ref
// (better use pointers).

template <SIMDLevel SL, bool with_id_map, class... Types>
PQ4CodeScanner* make_pq4_scanner_1(bool is_max, int norm_scale, Types... args) {
    if (is_max) {
        using C = CMax<uint16_t, int64_t>;
        if (norm_scale == -1) {
            return make_handler_2<SL, with_id_map, C, DummyScaler<SL>>(
                    norm_scale, args...);
        } else {
            return make_handler_2<SL, with_id_map, C, Scaler2x4bit<SL>>(
                    norm_scale, args...);
        }
    } else {
        using C = CMin<uint16_t, int64_t>;
        if (norm_scale == -1) {
            return make_handler_2<SL, with_id_map, C, DummyScaler<SL>>(
                    norm_scale, args...);
        } else {
            return make_handler_2<SL, with_id_map, C, Scaler2x4bit<SL>>(
                    norm_scale, args...);
        }
    }
}

// make_pq4_scanner should not be instantiated automatically (even if the
// function is defined just above), because here is where the different SIMD
// versions become separate.

// Because it is tedious to repleat the parameters all the time, define a few
// macros. this does not pollute the namespace too much because this is an
// internal header.
#define KNN_ARGS_LIST                                                        \
    bool is_max, int norm_scale, bool use_reservoir, idx_t nq, idx_t ntotal, \
            idx_t k, float *dis, idx_t *ids, const IDSelector *sel
#define KNN_ARGS_LIST_2 \
    is_max, norm_scale, use_reservoir, nq, ntotal, k, dis, ids, sel

template <SIMDLevel SL, bool with_id_map>
PQ4CodeScanner* make_pq4_scanner(KNN_ARGS_LIST);

#define RRES_ARGS_LIST                                                  \
    bool is_max, int norm_scale, RangeSearchResult *rres, float radius, \
            idx_t ntotal, const IDSelector *sel
#define RRES_ARGS_LIST_2 is_max, norm_scale, rres, radius, ntotal, sel

template <SIMDLevel SL, bool with_id_map>
PQ4CodeScanner* make_pq4_scanner(RRES_ARGS_LIST);

#define PRES_ARGS_LIST                                                         \
    bool is_max, int norm_scale, RangeSearchPartialResult *pres, float radius, \
            idx_t ntotal, idx_t i0, idx_t i1, const IDSelector *sel
#define PRES_ARGS_LIST_2 is_max, norm_scale, pres, radius, ntotal, i0, i1, sel

template <SIMDLevel SL, bool with_id_map>
PQ4CodeScanner* make_pq4_scanner(PRES_ARGS_LIST);

} // namespace faiss
