/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/Index.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <cstring>

namespace faiss {

Index::~Index() = default;

void Index::train(idx_t /*n*/, const float* /*x*/) {
    // does nothing by default
}

void Index::range_search(
        idx_t,
        const float*,
        float,
        RangeSearchResult*,
        const SearchParameters* params) const {
    FAISS_THROW_MSG("range search not implemented");
}

void Index::assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
    std::vector<float> distances(n * k);
    search(n, x, k, distances.data(), labels);
}

void Index::add_with_ids(
        idx_t /*n*/,
        const float* /*x*/,
        const idx_t* /*xids*/) {
    FAISS_THROW_MSG("add_with_ids not implemented for this type of index");
}

size_t Index::remove_ids(const IDSelector& /*sel*/) {
    FAISS_THROW_MSG("remove_ids not implemented for this type of index");
    return -1;
}

void Index::reconstruct(idx_t, float*) const {
    FAISS_THROW_MSG("reconstruct not implemented for this type of index");
}

void Index::reconstruct_batch(idx_t n, const idx_t* keys, float* recons) const {
    std::mutex exception_mutex;
    std::string exception_string;
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        try {
            reconstruct(keys[i], &recons[i * d]);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(exception_mutex);
            exception_string = e.what();
        }
    }
    if (!exception_string.empty()) {
        FAISS_THROW_MSG(exception_string.c_str());
    }
}

void Index::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
#pragma omp parallel for if (ni > 1000)
    for (idx_t i = 0; i < ni; i++) {
        reconstruct(i0 + i, recons + i * d);
    }
}

void Index::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    search(n, x, k, distances, labels, params);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            float* reconstructed = recons + ij * d;
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, sizeof(*reconstructed) * d);
            } else {
                reconstruct(key, reconstructed);
            }
        }
    }
}

void Index::compute_residual(const float* x, float* residual, idx_t key) const {
    reconstruct(key, residual);
    for (size_t i = 0; i < d; i++) {
        residual[i] = x[i] - residual[i];
    }
}

void Index::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
#pragma omp parallel for
    for (idx_t i = 0; i < n; ++i) {
        compute_residual(&xs[i * d], &residuals[i * d], keys[i]);
    }
}

size_t Index::sa_code_size() const {
    FAISS_THROW_MSG("standalone codec not implemented for this type of index");
}

void Index::sa_encode(idx_t, const float*, uint8_t*) const {
    FAISS_THROW_MSG("standalone codec not implemented for this type of index");
}

void Index::sa_decode(idx_t, const uint8_t*, float*) const {
    FAISS_THROW_MSG("standalone codec not implemented for this type of index");
}

void Index::add_sa_codes(idx_t, const uint8_t*, const idx_t*) {
    FAISS_THROW_MSG("add_sa_codes not implemented for this type of index");
}

namespace {

// storage that explicitly reconstructs vectors before computing distances
struct GenericDistanceComputer : DistanceComputer {
    size_t d;
    const Index& storage;
    std::vector<float> buf;
    const float* q;

    explicit GenericDistanceComputer(const Index& storage) : storage(storage) {
        d = storage.d;
        buf.resize(d * 2);
    }

    float operator()(idx_t i) override {
        storage.reconstruct(i, buf.data());
        return fvec_L2sqr(q, buf.data(), d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        storage.reconstruct(i, buf.data());
        storage.reconstruct(j, buf.data() + d);
        return fvec_L2sqr(buf.data() + d, buf.data(), d);
    }

    void set_query(const float* x) override {
        q = x;
    }
};

} // namespace

DistanceComputer* Index::get_distance_computer() const {
    if (metric_type == METRIC_L2) {
        return new GenericDistanceComputer(*this);
    } else {
        FAISS_THROW_MSG("get_distance_computer() not implemented");
    }
}

void Index::merge_from(Index& /* otherIndex */, idx_t /* add_id */) {
    FAISS_THROW_MSG("merge_from() not implemented");
}

void Index::check_compatible_for_merge(const Index& /* otherIndex */) const {
    FAISS_THROW_MSG("check_compatible_for_merge() not implemented");
}

} // namespace faiss
