/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Per-ISA implementation of Hamming distance computation for
 * IndexBinaryHash and IndexBinaryMultiHash. Included once per SIMD TU
 * with THE_SIMD_LEVEL set to the desired SIMDLevel.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including this file"
#endif

#include <faiss/utils/hamming_distance/hamming_computer.h>

#include <unordered_set>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/binary_hamming/dispatch.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>

namespace faiss {

namespace {

/** Enumerate all bit vectors of size nbit with up to maxflip 1s */
struct FlipEnumerator {
    int nbit, nflip, maxflip;
    uint64_t mask, x;

    FlipEnumerator(int nbit_, int maxflip_) : nbit(nbit_), maxflip(maxflip_) {
        nflip = 0;
        mask = 0;
        x = 0;
    }

    bool next() {
        if (x == mask) {
            if (nflip == maxflip) {
                return false;
            }
            // increase Hamming radius
            nflip++;
            mask = (((uint64_t)1 << nflip) - 1);
            x = mask << (nbit - nflip);
            return true;
        }

        int i = __builtin_ctzll(x);

        if (i > 0) {
            x ^= (uint64_t)3 << (i - 1);
        } else {
            // nb of LSB 1s
            int n1 = __builtin_ctzll(~x);
            // clear them
            x &= ((uint64_t)(-1) << n1);
            int n2 = __builtin_ctzll(x);
            x ^= (((uint64_t)1 << (n1 + 2)) - 1) << (n2 - n1 - 1);
        }
        return true;
    }
};

struct RangeSearchResults {
    int radius;
    RangeQueryResult& qres;

    inline void add(float dis, idx_t id) {
        if (dis < radius) {
            qres.add(dis, id);
        }
    }
};

struct KnnSearchResults {
    // heap params
    idx_t k;
    int32_t* heap_sim;
    idx_t* heap_ids;

    using C = CMax<int, idx_t>;

    inline void add(float dis, idx_t id) {
        if (dis < heap_sim[0]) {
            heap_replace_top<C>(k, heap_sim, heap_ids, dis, id);
        }
    }
};

template <class HammingComputer, class SearchResults>
void search_single_query_template(
        const IndexBinaryHash& index,
        const uint8_t* q,
        SearchResults& res,
        size_t& n0,
        size_t& nlist,
        size_t& ndis) {
    size_t code_size = index.code_size;
    BitstringReader br(q, code_size);
    uint64_t qhash = br.read(index.b);
    HammingComputer hc(q, code_size);
    FlipEnumerator fe(index.b, index.nflip);

    // loop over neighbors that are at most at nflip bits
    do {
        uint64_t hash = qhash ^ fe.x;
        auto it = index.invlists.find(hash);

        if (it == index.invlists.end()) {
            continue;
        }

        const IndexBinaryHash::InvertedList& il = it->second;

        size_t nv = il.ids.size();

        if (nv == 0) {
            n0++;
        } else {
            const uint8_t* codes = il.vecs.data();
            for (size_t i = 0; i < nv; i++) {
                int dis = hc.hamming(codes);
                res.add(dis, il.ids[i]);
                codes += code_size;
            }
            ndis += nv;
            nlist++;
        }
    } while (fe.next());
}

template <class HammingComputer, class SearchResults>
static void verify_shortlist(
        const IndexBinaryFlat* index,
        const uint8_t* q,
        const std::unordered_set<idx_t>& shortlist,
        SearchResults& res) {
    size_t code_size = index->code_size;

    HammingComputer hc(q, code_size);
    const uint8_t* codes = index->xb.data();

    for (auto i : shortlist) {
        int dis = hc.hamming(codes + i * code_size);
        res.add(dis, i);
    }
}

template <class SearchResults>
void search_1_query_multihash(
        const IndexBinaryMultiHash& index,
        const uint8_t* xi,
        SearchResults& res,
        size_t& n0,
        size_t& nlist,
        size_t& ndis) {
    std::unordered_set<idx_t> shortlist;
    int b = index.b;

    BitstringReader br(xi, index.code_size);
    for (int h = 0; h < index.nhash; h++) {
        uint64_t qhash = br.read(b);
        const IndexBinaryMultiHash::Map& map = index.maps[h];

        FlipEnumerator fe(index.b, index.nflip);
        // loop over neighbors that are at most at nflip bits
        do {
            uint64_t hash = qhash ^ fe.x;
            auto it = map.find(hash);

            if (it != map.end()) {
                const std::vector<idx_t>& v = it->second;
                for (auto i : v) {
                    shortlist.insert(i);
                }
                nlist++;
            } else {
                n0++;
            }
        } while (fe.next());
    }
    ndis += shortlist.size();

    // verify shortlist
    with_HammingComputer<THE_SIMD_LEVEL>(
            index.code_size, [&]<class HammingComputer>() {
                verify_shortlist<HammingComputer>(
                        index.storage, xi, shortlist, res);
            });
}

} // anonymous namespace

// --- IndexBinaryHash entry points ---

template <>
void binary_hash_knn_search_fixSL<THE_SIMD_LEVEL>(
        const IndexBinaryHash& index,
        const uint8_t* q,
        idx_t k,
        int32_t* heap_sim,
        idx_t* heap_ids,
        size_t& n0,
        size_t& nlist,
        size_t& ndis) {
    KnnSearchResults res = {k, heap_sim, heap_ids};
    with_HammingComputer<THE_SIMD_LEVEL>(
            index.code_size, [&]<class HammingComputer>() {
                search_single_query_template<HammingComputer>(
                        index, q, res, n0, nlist, ndis);
            });
}

template <>
void binary_hash_range_search_fixSL<THE_SIMD_LEVEL>(
        const IndexBinaryHash& index,
        const uint8_t* q,
        int radius,
        RangeQueryResult& qres,
        size_t& n0,
        size_t& nlist,
        size_t& ndis) {
    RangeSearchResults res = {radius, qres};
    with_HammingComputer<THE_SIMD_LEVEL>(
            index.code_size, [&]<class HammingComputer>() {
                search_single_query_template<HammingComputer>(
                        index, q, res, n0, nlist, ndis);
            });
}

// --- IndexBinaryMultiHash entry points ---

template <>
void binary_multihash_knn_search_fixSL<THE_SIMD_LEVEL>(
        const IndexBinaryMultiHash& index,
        const uint8_t* q,
        idx_t k,
        int32_t* heap_sim,
        idx_t* heap_ids,
        size_t& n0,
        size_t& nlist,
        size_t& ndis) {
    KnnSearchResults res = {k, heap_sim, heap_ids};
    search_1_query_multihash(index, q, res, n0, nlist, ndis);
}

template <>
void binary_multihash_range_search_fixSL<THE_SIMD_LEVEL>(
        const IndexBinaryMultiHash& index,
        const uint8_t* q,
        int radius,
        RangeQueryResult& qres,
        size_t& n0,
        size_t& nlist,
        size_t& ndis) {
    RangeSearchResults res = {radius, qres};
    search_1_query_multihash(index, q, res, n0, nlist, ndis);
}

} // namespace faiss
