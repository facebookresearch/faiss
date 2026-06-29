/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexBinaryHash.h>

#include <cinttypes>
#include <cstdio>
#include <memory>
#include <unordered_set>

#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>

#include <faiss/impl/simd_dispatch.h>

// Scalar (NONE) fallback for dynamic dispatch
#define THE_SIMD_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/binary_hamming/IndexBinaryHash_impl.h>
#include <faiss/utils/hamming_distance/hamming_computer-generic.h>
#undef THE_SIMD_LEVEL

namespace faiss {

void IndexBinaryHash::InvertedList::add(
        idx_t id,
        size_t code_size,
        const uint8_t* code) {
    ids.push_back(id);
    vecs.insert(vecs.end(), code, code + code_size);
}

IndexBinaryHash::IndexBinaryHash(int d_, int b_)
        : IndexBinary(d_), b(b_), nflip(0) {
    is_trained = true;
}

IndexBinaryHash::IndexBinaryHash() : b(0), nflip(0) {
    is_trained = true;
}

void IndexBinaryHash::reset() {
    invlists.clear();
    ntotal = 0;
}

void IndexBinaryHash::add(idx_t n, const uint8_t* x) {
    add_with_ids(n, x, nullptr);
}

void IndexBinaryHash::add_with_ids(
        idx_t n,
        const uint8_t* x,
        const idx_t* xids) {
    // simplistic add function. Cannot really be parallelized.

    for (idx_t i = 0; i < n; i++) {
        idx_t id = xids ? xids[i] : ntotal + i;
        const uint8_t* xi = x + i * code_size;
        BitstringReader br(xi, code_size);
        idx_t hash = br.read(b);
        invlists[hash].add(id, code_size, xi);
    }
    ntotal += n;
}

// search_single_query_template and helpers are now in
// impl/binary_hamming/IndexBinaryHash_impl.h (compiled per-ISA)

void IndexBinaryHash::range_search(
        idx_t n,
        const uint8_t* x,
        int radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    size_t nlist = 0, ndis = 0, n0 = 0;

#pragma omp parallel if (n > 100) reduction(+ : ndis, n0, nlist)
    {
        RangeSearchPartialResult pres(result);

#pragma omp for
        for (idx_t i = 0; i < n; i++) { // loop queries
            RangeQueryResult& qres = pres.new_result(i);
            const uint8_t* q = x + i * code_size;

            with_simd_level([&]<SIMDLevel SL>() {
                binary_hash_range_search_fixSL<SL>(
                        *this, q, radius, qres, n0, nlist, ndis);
            });
        }
        pres.finalize();
    }
    indexBinaryHash_stats.nq += n;
    indexBinaryHash_stats.n0 += n0;
    indexBinaryHash_stats.nlist += nlist;
    indexBinaryHash_stats.ndis += ndis;
}

void IndexBinaryHash::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);

    using HeapForL2 = CMax<int32_t, idx_t>;
    size_t nlist = 0, ndis = 0, n0 = 0;

#pragma omp parallel for if (n > 100) reduction(+ : nlist, ndis, n0)
    for (idx_t i = 0; i < n; i++) {
        int32_t* simi = distances + k * i;
        idx_t* idxi = labels + k * i;

        heap_heapify<HeapForL2>(k, simi, idxi);
        const uint8_t* q = x + i * code_size;

        with_simd_level([&]<SIMDLevel SL>() {
            binary_hash_knn_search_fixSL<SL>(
                    *this, q, k, simi, idxi, n0, nlist, ndis);
        });

        heap_reorder<HeapForL2>(k, simi, idxi);
    }
    indexBinaryHash_stats.nq += n;
    indexBinaryHash_stats.n0 += n0;
    indexBinaryHash_stats.nlist += nlist;
    indexBinaryHash_stats.ndis += ndis;
}

size_t IndexBinaryHash::hashtable_size() const {
    return invlists.size();
}

void IndexBinaryHash::display() const {
    for (auto it = invlists.begin(); it != invlists.end(); ++it) {
        printf("%" PRId64 ": [", it->first);
        const std::vector<idx_t>& v = it->second.ids;
        for (auto x : v) {
            printf("%" PRId64 " ", x);
        }
        printf("]\n");
    }
}

void IndexBinaryHashStats::reset() {
    memset((void*)this, 0, sizeof(*this));
}

IndexBinaryHashStats indexBinaryHash_stats;

/*******************************************************
 * IndexBinaryMultiHash implementation
 ******************************************************/

IndexBinaryMultiHash::IndexBinaryMultiHash(int d_, int nhash_, int b_)
        : IndexBinary(d_), maps(nhash_), nhash(nhash_), b(b_), nflip(0) {
    FAISS_THROW_IF_NOT(nhash_ * b_ <= d_);
    storage = std::make_unique<IndexBinaryFlat>(d_).release();
    own_fields = true;
}

IndexBinaryMultiHash::IndexBinaryMultiHash()
        : storage(nullptr), nhash(0), b(0), nflip(0) {}

IndexBinaryMultiHash::~IndexBinaryMultiHash() {
    if (own_fields) {
        delete storage;
    }
}

void IndexBinaryMultiHash::reset() {
    storage->reset();
    ntotal = 0;
    for (auto& map : maps) {
        map.clear();
    }
}

void IndexBinaryMultiHash::add(idx_t n, const uint8_t* x) {
    storage->add(n, x);
    // populate maps
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* xi = x + i * code_size;
        BitstringReader br(xi, code_size);
        for (int h = 0; h < nhash; h++) {
            uint64_t hash = br.read(b);
            maps[h][hash].push_back(i + ntotal);
        }
    }
    ntotal += n;
}

// verify_shortlist and search_1_query_multihash are now in
// impl/binary_hamming/IndexBinaryHash_impl.h (compiled per-ISA)

void IndexBinaryMultiHash::range_search(
        idx_t n,
        const uint8_t* x,
        int radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    size_t nlist = 0, ndis = 0, n0 = 0;

#pragma omp parallel if (n > 100) reduction(+ : ndis, n0, nlist)
    {
        RangeSearchPartialResult pres(result);

#pragma omp for
        for (idx_t i = 0; i < n; i++) { // loop queries
            RangeQueryResult& qres = pres.new_result(i);
            const uint8_t* q = x + i * code_size;

            with_simd_level([&]<SIMDLevel SL>() {
                binary_multihash_range_search_fixSL<SL>(
                        *this, q, radius, qres, n0, nlist, ndis);
            });
        }
        pres.finalize();
    }
    indexBinaryHash_stats.nq += n;
    indexBinaryHash_stats.n0 += n0;
    indexBinaryHash_stats.nlist += nlist;
    indexBinaryHash_stats.ndis += ndis;
}

void IndexBinaryMultiHash::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);

    using HeapForL2 = CMax<int32_t, idx_t>;
    size_t nlist = 0, ndis = 0, n0 = 0;

#pragma omp parallel for if (n > 100) reduction(+ : nlist, ndis, n0)
    for (idx_t i = 0; i < n; i++) {
        int32_t* simi = distances + k * i;
        idx_t* idxi = labels + k * i;

        heap_heapify<HeapForL2>(k, simi, idxi);
        const uint8_t* q = x + i * code_size;

        with_simd_level([&]<SIMDLevel SL>() {
            binary_multihash_knn_search_fixSL<SL>(
                    *this, q, k, simi, idxi, n0, nlist, ndis);
        });

        heap_reorder<HeapForL2>(k, simi, idxi);
    }
    indexBinaryHash_stats.nq += n;
    indexBinaryHash_stats.n0 += n0;
    indexBinaryHash_stats.nlist += nlist;
    indexBinaryHash_stats.ndis += ndis;
}

size_t IndexBinaryMultiHash::hashtable_size() const {
    size_t tot = 0;
    for (const auto& map : maps) {
        tot += map.size();
    }

    return tot;
}

} // namespace faiss
