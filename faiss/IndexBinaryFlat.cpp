/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexBinaryFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>
#include <cstring>

namespace faiss {

IndexBinaryFlat::IndexBinaryFlat(idx_t d) : IndexBinary(d) {}

void IndexBinaryFlat::add(idx_t n, const uint8_t* x) {
    xb.insert(xb.end(), x, x + n * code_size);
    ntotal += n;
}

void IndexBinaryFlat::add(idx_t n, const void* x, NumericType numeric_type) {
    IndexBinary::add(n, x, numeric_type);
}

void IndexBinaryFlat::reset() {
    xb.clear();
    ntotal = 0;
}

void IndexBinaryFlat::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // Extract IDSelector from params if present
    const IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);

    const idx_t block_size = query_batch_size;
    for (idx_t s = 0; s < n; s += block_size) {
        idx_t nn = block_size;
        if (s + block_size > n) {
            nn = n - s;
        }

        if (use_heap) {
            // We see the distances and labels as heaps.
            int_maxheap_array_t res = {
                    size_t(nn), size_t(k), labels + s * k, distances + s * k};

            hammings_knn_hc(
                    &res,
                    x + s * code_size,
                    xb.data(),
                    ntotal,
                    code_size,
                    /* ordered = */ true,
                    approx_topk_mode,
                    sel);
        } else {
            hammings_knn_mc(
                    x + s * code_size,
                    xb.data(),
                    nn,
                    ntotal,
                    k,
                    code_size,
                    distances + s * k,
                    labels + s * k,
                    sel);
        }
    }
}

void IndexBinaryFlat::search(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IndexBinary::search(n, x, numeric_type, k, distances, labels, params);
}

size_t IndexBinaryFlat::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove(&xb[code_size * j],
                        &xb[code_size * i],
                        sizeof(xb[0]) * code_size);
            }
            j++;
        }
    }
    long nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        xb.resize(ntotal * code_size);
    }
    return nremove;
}

void IndexBinaryFlat::reconstruct(idx_t key, uint8_t* recons) const {
    memcpy(recons, &(xb[code_size * key]), sizeof(*recons) * code_size);
}

void IndexBinaryFlat::range_search(
        idx_t n,
        const uint8_t* x,
        int radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    const IDSelector* sel = params ? params->sel : nullptr;
    hamming_range_search(
            x, xb.data(), n, ntotal, radius, code_size, result, sel);
}

} // namespace faiss
