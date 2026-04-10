/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Shared implementation header for Hamming distance dynamic dispatch.
// Included by per-ISA TUs (hamming_avx2.cpp, hamming_neon.cpp) and by
// hamming.cpp (for the NONE fallback).
//
// THE_SIMD_LEVEL must be defined before including this header.
//
// INCLUDE ORDERING IS LOAD-BEARING: hamdis-inl.h must be included before
// hamming.h so that the ISA-specific struct definitions (selected by
// __AVX2__ / __aarch64__ / etc.) are established before hamming.h's own
// #include of hamdis-inl.h becomes a no-op via include guards.

#pragma once

#ifndef THE_SIMD_LEVEL
#error "Define THE_SIMD_LEVEL before including hamming_impl.h"
#endif

// ISA-specific struct definitions — MUST come first (see comment above).
#include <faiss/utils/hamming_distance/hamdis-inl.h>

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

// Alias for readability in the template specializations below.
constexpr SIMDLevel hamming_impl_SL = THE_SIMD_LEVEL;

// All implementation templates live in an anonymous namespace for ODR safety.
// Different TUs compile this header with different HammingComputer struct
// layouts (generic vs AVX2 vs NEON). The anonymous namespace ensures each
// TU gets its own copy with internal linkage, preventing the linker from
// merging incompatible instantiations.
namespace {

/******************************************************************
 * Bit-level Hamming distance templates
 ******************************************************************/

template <size_t nbits>
void hammings_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        hamdis_t* __restrict dis) {
    size_t i, j;
    const size_t nwords = nbits / 64;
    for (i = 0; i < n1; i++) {
        const uint64_t* __restrict bs1_ = bs1 + i * nwords;
        hamdis_t* __restrict dis_ = dis + i * n2;
        for (j = 0; j < n2; j++) {
            dis_[j] = hamming<nbits>(bs1_, bs2 + j * nwords);
        }
    }
}

void hammings_impl_runtime(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        size_t nbits,
        hamdis_t* __restrict dis) {
    size_t i, j;
    const size_t nwords = nbits / 64;
    for (i = 0; i < n1; i++) {
        const uint64_t* __restrict bs1_ = bs1 + i * nwords;
        hamdis_t* __restrict dis_ = dis + i * n2;
        for (j = 0; j < n2; j++) {
            dis_[j] = hamming(bs1_, bs2 + j * nwords, nwords);
        }
    }
}

template <size_t nbits>
void hamming_count_thres_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        hamdis_t ht,
        size_t* __restrict nptr) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    const uint64_t* bs2_ = bs2;

    for (i = 0; i < n1; i++) {
        bs2 = bs2_;
        for (j = 0; j < n2; j++) {
            if (hamming<nbits>(bs1, bs2) <= ht) {
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    *nptr = posm;
}

template <size_t nbits>
void crosshamming_count_thres_impl(
        const uint64_t* __restrict dbs,
        size_t n,
        int ht,
        size_t* __restrict nptr) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    const uint64_t* bs1 = dbs;
    for (i = 0; i < n; i++) {
        const uint64_t* bs2 = bs1 + 2;
        for (j = i + 1; j < n; j++) {
            if (hamming<nbits>(bs1, bs2) <= ht) {
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    *nptr = posm;
}

template <size_t nbits>
size_t match_hamming_thres_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        int ht,
        int64_t* __restrict idx,
        hamdis_t* __restrict hams) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    hamdis_t h;
    const uint64_t* bs2_ = bs2;
    for (i = 0; i < n1; i++) {
        bs2 = bs2_;
        for (j = 0; j < n2; j++) {
            h = hamming<nbits>(bs1, bs2);
            if (h <= ht) {
                *idx = i;
                idx++;
                *idx = j;
                idx++;
                *hams = h;
                hams++;
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    return posm;
}

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

// Consumer structs for dispatch_HammingComputer.
// These MUST be in the anonymous namespace so that each TU's
// dispatch_HammingComputer instantiation uses unique consumer types,
// preventing ODR violations across TUs with different struct layouts.

struct Run_hammings_knn_hc {
    using T = void;
    template <class HammingComputer, class... Types>
    void f(Types... args) {
        hammings_knn_hc_impl<HammingComputer>(args...);
    }
};

struct Run_hammings_knn_mc {
    using T = void;
    template <class HammingComputer, class... Types>
    void f(Types... args) {
        hammings_knn_mc_impl<HammingComputer>(args...);
    }
};

struct Run_hamming_range_search {
    using T = void;
    template <class HammingComputer, class... Types>
    void f(Types... args) {
        hamming_range_search_impl<HammingComputer>(args...);
    }
};

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
                hamming_dis_inner_loop<GenHammingComputer8>(
                        ca, cb, nb, 8, k, bh_val_, bh_ids_);
                break;
            case 16:
                hamming_dis_inner_loop<GenHammingComputer16>(
                        ca, cb, nb, 16, k, bh_val_, bh_ids_);
                break;
            case 32:
                hamming_dis_inner_loop<GenHammingComputer32>(
                        ca, cb, nb, 32, k, bh_val_, bh_ids_);
                break;
            default:
                hamming_dis_inner_loop<GenHammingComputerM8>(
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
void hammings_knn_hc_dispatch<hamming_impl_SL>(
        int_maxheap_array_t* ha,
        const uint8_t* a,
        const uint8_t* b,
        size_t nb,
        size_t ncodes,
        int ordered,
        ApproxTopK_mode_t approx_topk_mode,
        const IDSelector* sel) {
    Run_hammings_knn_hc r;
    dispatch_HammingComputer(
            ncodes,
            r,
            ncodes,
            ha,
            a,
            b,
            nb,
            ordered,
            true,
            approx_topk_mode,
            sel);
}

template <>
void hammings_knn_mc_dispatch<hamming_impl_SL>(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t k,
        size_t ncodes,
        int32_t* distances,
        int64_t* labels,
        const IDSelector* sel) {
    Run_hammings_knn_mc r;
    dispatch_HammingComputer(
            ncodes, r, ncodes, a, b, na, nb, k, distances, labels, sel);
}

template <>
void hamming_range_search_dispatch<hamming_impl_SL>(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        int radius,
        size_t code_size,
        RangeSearchResult* result,
        const IDSelector* sel) {
    Run_hamming_range_search r;
    dispatch_HammingComputer(
            code_size, r, a, b, na, nb, radius, code_size, result, sel);
}

template <>
void hammings_dispatch<hamming_impl_SL>(
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
void generalized_hammings_knn_hc_dispatch<hamming_impl_SL>(
        int_maxheap_array_t* ha,
        const uint8_t* a,
        const uint8_t* b,
        size_t nb,
        size_t code_size,
        int ordered) {
    generalized_hammings_knn_hc_impl(ha, a, b, nb, code_size, ordered);
}

template <>
void hamming_count_thres_dispatch<hamming_impl_SL>(
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
void crosshamming_count_thres_dispatch<hamming_impl_SL>(
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
size_t match_hamming_thres_dispatch<hamming_impl_SL>(
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
