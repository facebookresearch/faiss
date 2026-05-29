/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Per-ISA implementation of Hamming distance computation for
 * IndexBinaryIVF. Included once per SIMD TU with THE_SIMD_LEVEL
 * set to the desired SIMDLevel.
 *
 * Contains: IVFBinaryScannerL2, search_knn_hamming_count,
 * BlockSearch, BlockSearchVariableK, search_knn_hamming_per_invlist.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including this file"
#endif

#include <faiss/utils/hamming_distance/hamming_computer.h>

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <memory>

#include <faiss/IndexBinaryIVF.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/binary_hamming/dispatch.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/sorting.h>

namespace faiss {

namespace {

template <class HammingComputer>
struct IVFBinaryScannerL2 : BinaryInvertedListScanner {
    HammingComputer hc;
    size_t code_size;
    bool store_pairs;

    IVFBinaryScannerL2(size_t code_size_, bool store_pairs_)
            : code_size(code_size_), store_pairs(store_pairs_) {}

    void set_query(const uint8_t* query_vector) override {
        hc.set(query_vector, code_size);
    }

    idx_t list_no = 0;
    void set_list(idx_t list_no_2, uint8_t /* coarse_dis */) override {
        this->list_no = list_no_2;
    }

    uint32_t distance_to_code(const uint8_t* code) const override {
        return hc.hamming(code);
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* __restrict codes,
            const idx_t* __restrict ids,
            int32_t* __restrict simi,
            idx_t* __restrict idxi,
            size_t k) const override {
        using C = CMax<int32_t, idx_t>;

        size_t nup = 0;
        for (size_t j = 0; j < n; j++) {
            uint32_t dis = hc.hamming(codes);
            if (dis < static_cast<uint32_t>(simi[0])) {
                idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t n,
            const uint8_t* __restrict codes,
            const idx_t* __restrict ids,
            int radius,
            RangeQueryResult& result) const override {
        for (size_t j = 0; j < n; j++) {
            uint32_t dis = hc.hamming(codes);
            if (dis < static_cast<uint32_t>(radius)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                result.add(dis, id);
            }
            codes += code_size;
        }
    }
};

template <class HammingComputer, bool store_pairs>
void search_knn_hamming_count(
        const IndexBinaryIVF* ivf,
        size_t nx,
        const uint8_t* __restrict x,
        const idx_t* __restrict keys,
        int k,
        int32_t* __restrict distances,
        idx_t* __restrict labels,
        const IVFSearchParameters* params) {
    const int nBuckets = ivf->d + 1;
    std::vector<int> all_counters(nx * nBuckets, 0);
    std::unique_ptr<idx_t[]> all_ids_per_dis(new idx_t[nx * nBuckets * k]);

    idx_t nprobe = params ? params->nprobe : ivf->nprobe;
    nprobe = std::min((idx_t)ivf->nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf->max_codes;

    std::vector<HCounterState<HammingComputer>> cs;
    cs.reserve(nx);
    for (size_t i = 0; i < nx; ++i) {
        cs.push_back(
                HCounterState<HammingComputer>(
                        all_counters.data() + i * nBuckets,
                        all_ids_per_dis.get() + i * nBuckets * k,
                        x + i * ivf->code_size,
                        ivf->d,
                        k));
    }

    size_t nlistv = 0, ndis = 0;

#pragma omp parallel for reduction(+ : nlistv, ndis)
    for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
        const idx_t* keysi = keys + i * nprobe;
        HCounterState<HammingComputer>& csi = cs[i];

        size_t nscan = 0;

        for (idx_t ik = 0; ik < nprobe; ik++) {
            idx_t key = keysi[ik]; /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)ivf->nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    static_cast<size_t>(ik),
                    ivf->nlist);

            nlistv++;
            size_t list_size = ivf->invlists->list_size(key);
            size_t list_size_max = static_cast<size_t>(max_codes) - nscan;
            if (list_size > list_size_max) {
                list_size = list_size_max;
            }
            InvertedLists::ScopedCodes scodes(ivf->invlists, key);
            const uint8_t* list_vecs = scodes.get();
            const idx_t* ids =
                    store_pairs ? nullptr : ivf->invlists->get_ids(key);

            for (size_t j = 0; j < list_size; j++) {
                const uint8_t* yj = list_vecs + ivf->code_size * j;

                idx_t id = store_pairs ? (key << 32 | j) : ids[j];
                csi.update_counter(yj, id);
            }
            if (ids) {
                ivf->invlists->release_ids(key, ids);
            }

            nscan += list_size;
            if (nscan >= static_cast<size_t>(max_codes)) {
                break;
            }
        }
        ndis += nscan;

        int nres = 0;
        for (int b = 0; b < nBuckets && nres < k; b++) {
            for (int l = 0; l < csi.counters[b] && nres < k; l++) {
                labels[i * k + nres] = csi.ids_per_dis[b * k + l];
                distances[i * k + nres] = b;
                nres++;
            }
        }
        while (nres < k) {
            labels[i * k + nres] = -1;
            distances[i * k + nres] = std::numeric_limits<int32_t>::max();
            ++nres;
        }
    }

    indexIVF_stats.nq += nx;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}

/* Manages NQ queries at a time, stores results */
template <class HammingComputer, int NQ, int K>
struct BlockSearch {
    HammingComputer hcs[NQ];
    // heaps to update for each query
    int32_t* distances[NQ] = {};
    idx_t* labels[NQ] = {};
    // curent top of heap
    int32_t heap_tops[NQ] = {};

    BlockSearch(
            size_t code_size,
            const uint8_t* __restrict x,
            const int32_t* __restrict keys,
            int32_t* __restrict all_distances,
            idx_t* __restrict all_labels) {
        for (idx_t q = 0; q < NQ; q++) {
            idx_t qno = keys[q];
            hcs[q] = HammingComputer(x + qno * code_size, code_size);
            distances[q] = all_distances + qno * K;
            labels[q] = all_labels + qno * K;
            heap_tops[q] = distances[q][0];
        }
    }

    void add_bcode(const uint8_t* bcode, idx_t id) {
        using C = CMax<int32_t, idx_t>;
        for (int q = 0; q < NQ; q++) {
            int dis = hcs[q].hamming(bcode);
            if (dis < heap_tops[q]) {
                heap_replace_top<C>(K, distances[q], labels[q], dis, id);
                heap_tops[q] = distances[q][0];
            }
        }
    }
};

template <class HammingComputer, int NQ>
struct BlockSearchVariableK {
    int k;
    HammingComputer hcs[NQ];
    // heaps to update for each query
    int32_t* distances[NQ] = {};
    idx_t* labels[NQ] = {};
    // curent top of heap
    int32_t heap_tops[NQ] = {};

    BlockSearchVariableK(
            size_t code_size,
            int k_,
            const uint8_t* __restrict x,
            const int32_t* __restrict keys,
            int32_t* __restrict all_distances,
            idx_t* __restrict all_labels)
            : k(k_) {
        for (idx_t q = 0; q < NQ; q++) {
            idx_t qno = keys[q];
            hcs[q] = HammingComputer(x + qno * code_size, code_size);
            distances[q] = all_distances + qno * k;
            labels[q] = all_labels + qno * k;
            heap_tops[q] = distances[q][0];
        }
    }

    void add_bcode(const uint8_t* bcode, idx_t id) {
        using C = CMax<int32_t, idx_t>;
        for (int q = 0; q < NQ; q++) {
            int dis = hcs[q].hamming(bcode);
            if (dis < heap_tops[q]) {
                heap_replace_top<C>(k, distances[q], labels[q], dis, id);
                heap_tops[q] = distances[q][0];
            }
        }
    }
};

template <class HammingComputer>
void search_knn_hamming_per_invlist(
        const IndexBinaryIVF* ivf,
        size_t n,
        const uint8_t* __restrict x,
        idx_t k,
        const idx_t* __restrict keys_in,
        const int32_t* __restrict /* coarse_dis */,
        int32_t* __restrict distances,
        idx_t* __restrict labels,
        bool store_pairs,
        const IVFSearchParameters* params) {
    idx_t nprobe = params ? params->nprobe : ivf->nprobe;
    nprobe = std::min((idx_t)ivf->nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf->max_codes;
    FAISS_THROW_IF_NOT(max_codes == 0);
    FAISS_THROW_IF_NOT(!store_pairs);

    // reorder buckets
    std::vector<int64_t> lims(n + 1);
    int32_t* keys = new int32_t[n * nprobe];
    std::unique_ptr<int32_t[]> delete_keys(keys);
    for (size_t i = 0; i < n * static_cast<size_t>(nprobe); i++) {
        keys[i] = static_cast<int32_t>(keys_in[i]);
    }
    matrix_bucket_sort_inplace(
            n, nprobe, keys, static_cast<int32_t>(ivf->nlist), lims.data(), 0);

    using C = CMax<int32_t, idx_t>;
    heap_heapify<C>(n * k, distances, labels);
    const size_t code_size = ivf->code_size;

    for (size_t l = 0; l < ivf->nlist; l++) {
        idx_t l0 = lims[l], nq = lims[l + 1] - l0;

        InvertedLists::ScopedCodes scodes(ivf->invlists, l);
        InvertedLists::ScopedIds sidx(ivf->invlists, l);
        idx_t nb = ivf->invlists->list_size(l);
        const uint8_t* bcodes = scodes.get();
        const idx_t* ids = sidx.get();

        idx_t i = 0;

        // process as much as possible by blocks
        constexpr int BS = 4;

        if (k == 1) {
            for (; i + BS <= nq; i += BS) {
                BlockSearch<HammingComputer, BS, 1> bc(
                        code_size, x, keys + l0 + i, distances, labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        } else if (k == 2) {
            for (; i + BS <= nq; i += BS) {
                BlockSearch<HammingComputer, BS, 2> bc(
                        code_size, x, keys + l0 + i, distances, labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        } else if (k == 4) {
            for (; i + BS <= nq; i += BS) {
                BlockSearch<HammingComputer, BS, 4> bc(
                        code_size, x, keys + l0 + i, distances, labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        } else {
            for (; i + BS <= nq; i += BS) {
                BlockSearchVariableK<HammingComputer, BS> bc(
                        code_size,
                        static_cast<int>(k),
                        x,
                        keys + l0 + i,
                        distances,
                        labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        }

        // leftovers
        for (; i < nq; i++) {
            idx_t qno = keys[l0 + i];
            HammingComputer hc(
                    x + qno * code_size, static_cast<int>(code_size));
            idx_t* __restrict idxi = labels + qno * k;
            int32_t* __restrict simi = distances + qno * k;
            int32_t simi0 = simi[0];
            for (idx_t j = 0; j < nb; j++) {
                int dis = hc.hamming(bcodes + j * code_size);

                if (dis < simi0) {
                    idx_t id = store_pairs ? lo_build(l, j) : ids[j];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    simi0 = simi[0];
                }
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        heap_reorder<C>(k, distances + i * k, labels + i * k);
    }
}

} // anonymous namespace

// --- Entry points ---

template <>
BinaryInvertedListScanner* make_binary_ivf_scanner_fixSL<THE_SIMD_LEVEL>(
        size_t code_size,
        bool store_pairs) {
    return with_HammingComputer<THE_SIMD_LEVEL>(
            code_size,
            [&]<class HammingComputer>() -> BinaryInvertedListScanner* {
                return new IVFBinaryScannerL2<HammingComputer>(
                        code_size, store_pairs);
            });
}

template <>
void search_knn_hamming_per_invlist_fixSL<THE_SIMD_LEVEL>(
        int code_size,
        const IndexBinaryIVF* ivf,
        size_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* keys_in,
        const int32_t* coarse_dis,
        int32_t* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params) {
    with_HammingComputer<THE_SIMD_LEVEL>(
            code_size, [&]<class HammingComputer>() {
                search_knn_hamming_per_invlist<HammingComputer>(
                        ivf,
                        n,
                        x,
                        k,
                        keys_in,
                        coarse_dis,
                        distances,
                        labels,
                        store_pairs,
                        params);
            });
}

template <>
void search_knn_hamming_count_fixSL<THE_SIMD_LEVEL>(
        int code_size,
        bool store_pairs,
        const IndexBinaryIVF* ivf,
        size_t nx,
        const uint8_t* x,
        const idx_t* keys,
        int k,
        int32_t* distances,
        idx_t* labels,
        const IVFSearchParameters* params) {
    if (store_pairs) {
        with_HammingComputer<THE_SIMD_LEVEL>(
                code_size, [&]<class HammingComputer>() {
                    search_knn_hamming_count<HammingComputer, true>(
                            ivf, nx, x, keys, k, distances, labels, params);
                });
    } else {
        with_HammingComputer<THE_SIMD_LEVEL>(
                code_size, [&]<class HammingComputer>() {
                    search_knn_hamming_count<HammingComputer, false>(
                            ivf, nx, x, keys, k, distances, labels, params);
                });
    }
}

} // namespace faiss
