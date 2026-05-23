/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Shared implementation header for Hamming distance dynamic dispatch.
// Included by per-ISA TUs (hamming_avx2.cpp, hamming_avx512.cpp,
// hamming_neon.cpp) and by hamming.cpp (for the NONE fallback TU).
//
// THE_SIMD_LEVEL must be defined before including this header.
// Callers must also include the appropriate hamming_computer-*.h
// (generic, avx2, avx512, or neon) before this header so that the
// HammingComputer struct specializations are available.
//
// ODR CONTRACT: each TU that includes this header MUST define a UNIQUE
// THE_SIMD_LEVEL value across the linked binary. The template
// specializations at the bottom of this file (hammings_knn_hc_fixSL<SL>,
// etc.) have external linkage with SL-keyed mangled names; two TUs sharing
// THE_SIMD_LEVEL would emit conflicting definitions of the same symbol.
// In DD mode, the per-ISA TUs use distinct values (NONE/AVX2/AVX512/NEON);
// in static mode, only hamming.cpp includes this header so uniqueness is
// trivially satisfied.

#pragma once

#ifndef THE_SIMD_LEVEL
#error "Define THE_SIMD_LEVEL before including hamming_impl.h"
#endif

// Forward declarations and dispatch function.
#include <faiss/utils/hamming_distance/hamming_computer.h>

#include <faiss/utils/hamming.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/approx_topk_hamming/approx_topk_hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

// All HammingComputer-dependent implementation templates live in an anonymous
// namespace for ODR safety. Different TUs compile this header with different
// HammingComputer struct layouts (generic vs AVX2 vs NEON). The anonymous
// namespace ensures each TU gets its own copy with internal linkage, preventing
// the linker from merging incompatible instantiations.
namespace {

/******************************************************************
 * HammingComputer-based search templates
 ******************************************************************/

template <class HammingComputer>
void hammings_knn_hc_impl(
        int bytes_per_code,
        int_maxheap_array_t* __restrict ha,
        const uint8_t* __restrict bs1,
        const uint8_t* __restrict bs2,
        size_t n2,
        bool order = true,
        bool init_heap = true,
        ApproxTopK_mode_t approx_topk_mode = ApproxTopK_mode_t::EXACT_TOPK,
        const faiss::IDSelector* sel = nullptr) {
    size_t k = ha->k;
    if (init_heap) {
        ha->heapify();
    }

    const size_t block_size = hamming_batch_size;
    for (size_t j0 = 0; j0 < n2; j0 += block_size) {
        const size_t j1 = std::min(j0 + block_size, n2);
#pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(ha->nh); i++) {
            HammingComputer hc(bs1 + i * bytes_per_code, bytes_per_code);

            const uint8_t* __restrict bs2_ = bs2 + j0 * bytes_per_code;
            hamdis_t dis;
            hamdis_t* __restrict bh_val_ = ha->val + i * k;
            int64_t* __restrict bh_ids_ = ha->ids + i * k;

#define HANDLE_APPROX(NB, BD)                                                \
    case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B##NB##_D##BD:               \
        FAISS_THROW_IF_NOT_FMT(                                              \
                k <= NB * BD,                                                \
                "The chosen mode (%d) of approximate top-k supports "        \
                "up to %d values, but %zd is requested.",                    \
                (int)(ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B##NB##_D##BD), \
                NB * BD,                                                     \
                k);                                                          \
        HeapWithBucketsForHamming32<                                         \
                CMax<hamdis_t, int64_t>,                                     \
                NB,                                                          \
                BD,                                                          \
                HammingComputer>::                                           \
                addn(j1 - j0, hc, bs2_, k, bh_val_, bh_ids_, sel);           \
        break;

            switch (approx_topk_mode) {
                HANDLE_APPROX(8, 3)
                HANDLE_APPROX(8, 2)
                HANDLE_APPROX(16, 2)
                HANDLE_APPROX(32, 2)
                default: {
                    for (size_t j = j0; j < j1; j++, bs2_ += bytes_per_code) {
                        if (sel && !sel->is_member(j)) {
                            continue;
                        }
                        dis = hc.hamming(bs2_);
                        if (dis < bh_val_[0]) {
                            faiss::maxheap_replace_top<hamdis_t>(
                                    k, bh_val_, bh_ids_, dis, j);
                        }
                    }
                } break;
            }
        }
    }
    if (order) {
        ha->reorder();
    }
}

#undef HANDLE_APPROX

template <class HammingComputer>
void hammings_knn_mc_impl(
        int bytes_per_code,
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t na,
        size_t nb,
        size_t k,
        int32_t* __restrict distances,
        int64_t* __restrict labels,
        const faiss::IDSelector* sel) {
    const int nBuckets = bytes_per_code * 8 + 1;
    std::vector<int> all_counters(na * nBuckets, 0);
    std::unique_ptr<int64_t[]> all_ids_per_dis(new int64_t[na * nBuckets * k]);

    std::vector<HCounterState<HammingComputer>> cs;
    for (size_t i = 0; i < na; ++i) {
        cs.push_back(
                HCounterState<HammingComputer>(
                        all_counters.data() + i * nBuckets,
                        all_ids_per_dis.get() + i * nBuckets * k,
                        a + i * bytes_per_code,
                        8 * bytes_per_code,
                        k));
    }

    const size_t block_size = hamming_batch_size;
    for (size_t j0 = 0; j0 < nb; j0 += block_size) {
        const size_t j1 = std::min(j0 + block_size, nb);
#pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(na); ++i) {
            for (size_t j = j0; j < j1; ++j) {
                if (!sel || sel->is_member(j)) {
                    cs[i].update_counter(b + j * bytes_per_code, j);
                }
            }
        }
    }

    for (size_t i = 0; i < na; ++i) {
        HCounterState<HammingComputer>& csi = cs[i];

        size_t nres = 0;
        for (int b_2 = 0; b_2 < nBuckets && nres < k; b_2++) {
            for (int l = 0; l < csi.counters[b_2] && nres < k; l++) {
                labels[i * k + nres] = csi.ids_per_dis[b_2 * k + l];
                distances[i * k + nres] = b_2;
                nres++;
            }
        }
        while (nres < k) {
            labels[i * k + nres] = -1;
            distances[i * k + nres] = std::numeric_limits<int32_t>::max();
            ++nres;
        }
    }
}

template <class HammingComputer>
void hamming_range_search_impl(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        int radius,
        size_t code_size,
        RangeSearchResult* res,
        const faiss::IDSelector* sel) {
#pragma omp parallel
    {
        RangeSearchPartialResult pres(res);

#pragma omp for
        for (int64_t i = 0; i < static_cast<int64_t>(na); i++) {
            HammingComputer hc(a + i * code_size, code_size);
            const uint8_t* yi = b;
            RangeQueryResult& qres = pres.new_result(i);

            for (size_t j = 0; j < nb; j++) {
                if (!sel || sel->is_member(j)) {
                    int dis = hc.hamming(yi);
                    if (dis < radius) {
                        qres.add(dis, j);
                    }
                }
                yi += code_size;
            }
        }
        pres.finalize();
    }
}

/******************************************************************
 * Generalized Hamming distances
 ******************************************************************/

template <class HammingComputer>
void hamming_dis_inner_loop(
        const uint8_t* __restrict ca,
        const uint8_t* __restrict cb,
        size_t nb,
        size_t code_size,
        int k,
        hamdis_t* __restrict bh_val_,
        int64_t* __restrict bh_ids_) {
    HammingComputer hc(ca, code_size);

    for (size_t j = 0; j < nb; j++) {
        int ndiff = hc.hamming(cb);
        cb += code_size;
        if (ndiff < bh_val_[0]) {
            maxheap_replace_top<hamdis_t>(k, bh_val_, bh_ids_, ndiff, j);
        }
    }
}

void generalized_hammings_knn_hc_impl(
        int_maxheap_array_t* __restrict ha,
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t nb,
        size_t code_size,
        int ordered) {
    int na = ha->nh;
    int k = ha->k;

    if (ordered) {
        ha->heapify();
    }

#pragma omp parallel for
    for (int i = 0; i < na; i++) {
        const uint8_t* __restrict ca = a + i * code_size;
        const uint8_t* __restrict cb = b;

        hamdis_t* __restrict bh_val_ = ha->val + i * k;
        int64_t* __restrict bh_ids_ = ha->ids + i * k;

        switch (code_size) {
            case 8:
                hamming_dis_inner_loop<GenHammingComputer8_tpl<THE_SIMD_LEVEL>>(
                        ca, cb, nb, 8, k, bh_val_, bh_ids_);
                break;
            case 16:
                hamming_dis_inner_loop<
                        GenHammingComputer16_tpl<THE_SIMD_LEVEL>>(
                        ca, cb, nb, 16, k, bh_val_, bh_ids_);
                break;
            case 32:
                hamming_dis_inner_loop<
                        GenHammingComputer32_tpl<THE_SIMD_LEVEL>>(
                        ca, cb, nb, 32, k, bh_val_, bh_ids_);
                break;
            default:
                hamming_dis_inner_loop<
                        GenHammingComputerM8_tpl<THE_SIMD_LEVEL>>(
                        ca, cb, nb, code_size, k, bh_val_, bh_ids_);
                break;
        }
    }

    if (ordered) {
        ha->reorder();
    }
}

} // anonymous namespace

/******************************************************************
 * Entry point template specializations at THE_SIMD_LEVEL
 ******************************************************************/

#define C64(x) ((uint64_t*)x)

template <>
void hammings_knn_hc_fixSL<THE_SIMD_LEVEL>(
        int_maxheap_array_t* ha,
        const uint8_t* a,
        const uint8_t* b,
        size_t nb,
        size_t ncodes,
        int ordered,
        ApproxTopK_mode_t approx_topk_mode,
        const IDSelector* sel) {
    with_HammingComputer<THE_SIMD_LEVEL>(ncodes, [&]<class HammingComputer>() {
        hammings_knn_hc_impl<HammingComputer>(
                ncodes, ha, a, b, nb, ordered, true, approx_topk_mode, sel);
    });
}

template <>
void hammings_knn_mc_fixSL<THE_SIMD_LEVEL>(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t k,
        size_t ncodes,
        int32_t* distances,
        int64_t* labels,
        const IDSelector* sel) {
    with_HammingComputer<THE_SIMD_LEVEL>(ncodes, [&]<class HammingComputer>() {
        hammings_knn_mc_impl<HammingComputer>(
                ncodes, a, b, na, nb, k, distances, labels, sel);
    });
}

template <>
void hamming_range_search_fixSL<THE_SIMD_LEVEL>(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        int radius,
        size_t code_size,
        RangeSearchResult* result,
        const IDSelector* sel) {
    with_HammingComputer<THE_SIMD_LEVEL>(
            code_size, [&]<class HammingComputer>() {
                hamming_range_search_impl<HammingComputer>(
                        a, b, na, nb, radius, code_size, result, sel);
            });
}

template <>
void hammings_fixSL<THE_SIMD_LEVEL>(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t ncodes,
        hamdis_t* dis) {
    FAISS_THROW_IF_NOT(ncodes % 8 == 0);
    switch (ncodes) {
        case 8:
            hammings_impl<64>(C64(a), C64(b), na, nb, dis);
            return;
        case 16:
            hammings_impl<128>(C64(a), C64(b), na, nb, dis);
            return;
        case 32:
            hammings_impl<256>(C64(a), C64(b), na, nb, dis);
            return;
        case 64:
            hammings_impl<512>(C64(a), C64(b), na, nb, dis);
            return;
        default:
            hammings_impl_runtime(C64(a), C64(b), na, nb, ncodes * 8, dis);
            return;
    }
}

template <>
void generalized_hammings_knn_hc_fixSL<THE_SIMD_LEVEL>(
        int_maxheap_array_t* ha,
        const uint8_t* a,
        const uint8_t* b,
        size_t nb,
        size_t code_size,
        int ordered) {
    generalized_hammings_knn_hc_impl(ha, a, b, nb, code_size, ordered);
}

template <>
void hamming_count_thres_fixSL<THE_SIMD_LEVEL>(
        const uint8_t* bs1,
        const uint8_t* bs2,
        size_t n1,
        size_t n2,
        hamdis_t ht,
        size_t ncodes,
        size_t* nptr) {
    switch (ncodes) {
        case 8:
            hamming_count_thres_impl<64>(C64(bs1), C64(bs2), n1, n2, ht, nptr);
            return;
        case 16:
            hamming_count_thres_impl<128>(C64(bs1), C64(bs2), n1, n2, ht, nptr);
            return;
        case 32:
            hamming_count_thres_impl<256>(C64(bs1), C64(bs2), n1, n2, ht, nptr);
            return;
        case 64:
            hamming_count_thres_impl<512>(C64(bs1), C64(bs2), n1, n2, ht, nptr);
            return;
        default:
            FAISS_THROW_FMT("not implemented for %zu bits", ncodes);
    }
}

template <>
void crosshamming_count_thres_fixSL<THE_SIMD_LEVEL>(
        const uint8_t* dbs,
        size_t n,
        hamdis_t ht,
        size_t ncodes,
        size_t* nptr) {
    switch (ncodes) {
        case 8:
            crosshamming_count_thres_impl<64>(C64(dbs), n, ht, nptr);
            return;
        case 16:
            crosshamming_count_thres_impl<128>(C64(dbs), n, ht, nptr);
            return;
        case 32:
            crosshamming_count_thres_impl<256>(C64(dbs), n, ht, nptr);
            return;
        case 64:
            crosshamming_count_thres_impl<512>(C64(dbs), n, ht, nptr);
            return;
        default:
            FAISS_THROW_FMT("not implemented for %zu bits", ncodes);
    }
}

template <>
size_t match_hamming_thres_fixSL<THE_SIMD_LEVEL>(
        const uint8_t* bs1,
        const uint8_t* bs2,
        size_t n1,
        size_t n2,
        hamdis_t ht,
        size_t ncodes,
        int64_t* idx,
        hamdis_t* dis) {
    switch (ncodes) {
        case 8:
            return match_hamming_thres_impl<64>(
                    C64(bs1), C64(bs2), n1, n2, ht, idx, dis);
        case 16:
            return match_hamming_thres_impl<128>(
                    C64(bs1), C64(bs2), n1, n2, ht, idx, dis);
        case 32:
            return match_hamming_thres_impl<256>(
                    C64(bs1), C64(bs2), n1, n2, ht, idx, dis);
        case 64:
            return match_hamming_thres_impl<512>(
                    C64(bs1), C64(bs2), n1, n2, ht, idx, dis);
        default:
            FAISS_THROW_FMT("not implemented for %zu bits", ncodes);
            return 0;
    }
}

#undef C64

} // namespace faiss
