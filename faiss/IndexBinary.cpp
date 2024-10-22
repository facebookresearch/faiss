/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexBinary.h>
#include <faiss/impl/FaissAssert.h>

#include <cinttypes>
#include <cstring>

namespace faiss {

IndexBinary::IndexBinary(idx_t d, MetricType metric)
        : d(d), code_size(d / 8), metric_type(metric) {
    FAISS_THROW_IF_NOT(d % 8 == 0);
}

IndexBinary::~IndexBinary() = default;

void IndexBinary::train(idx_t, const uint8_t*) {
    // Does nothing by default.
}

void IndexBinary::range_search(
        idx_t,
        const uint8_t*,
        int,
        RangeSearchResult*,
        const SearchParameters*) const {
    FAISS_THROW_MSG("range search not implemented");
}

void IndexBinary::assign(idx_t n, const uint8_t* x, idx_t* labels, idx_t k)
        const {
    std::vector<int> distances(n * k);
    search(n, x, k, distances.data(), labels);
}

void IndexBinary::add_with_ids(idx_t, const uint8_t*, const idx_t*) {
    FAISS_THROW_MSG("add_with_ids not implemented for this type of index");
}

size_t IndexBinary::remove_ids(const IDSelector&) {
    FAISS_THROW_MSG("remove_ids not implemented for this type of index");
    return 0;
}

void IndexBinary::reconstruct(idx_t, uint8_t*) const {
    FAISS_THROW_MSG("reconstruct not implemented for this type of index");
}

void IndexBinary::reconstruct_n(idx_t i0, idx_t ni, uint8_t* recons) const {
    for (idx_t i = 0; i < ni; i++) {
        reconstruct(i0 + i, recons + i * code_size);
    }
}

void IndexBinary::search_and_reconstruct(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        uint8_t* recons,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    search(n, x, k, distances, labels, params);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            uint8_t* reconstructed = recons + ij * code_size;
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, code_size);
            } else {
                reconstruct(key, reconstructed);
            }
        }
    }
}

void IndexBinary::display() const {
    printf("Index: %s  -> %" PRId64 " elements\n",
           typeid(*this).name(),
           ntotal);
}

void IndexBinary::merge_from(
        IndexBinary& /* otherIndex */,
        idx_t /* add_id */) {
    FAISS_THROW_MSG("merge_from() not implemented");
}

void IndexBinary::check_compatible_for_merge(
        const IndexBinary& /* otherIndex */) const {
    FAISS_THROW_MSG("check_compatible_for_merge() not implemented");
}

} // namespace faiss
