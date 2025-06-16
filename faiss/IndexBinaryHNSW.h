/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexBinaryFlat.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

namespace faiss {

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexBinaryHNSW : IndexBinary {
    typedef HNSW::storage_idx_t storage_idx_t;

    // the link structure
    HNSW hnsw;

    // the sequential storage
    bool own_fields;
    IndexBinary* storage;

    // When set to false, level 0 in the knn graph is not initialized.
    // This option is used by GpuIndexBinaryCagra::copyTo(IndexBinaryHNSW*)
    // as level 0 knn graph is copied over from the index built by
    // GpuIndexBinaryCagra.
    bool init_level0 = true;

    // When set to true, all neighbors in level 0 are filled up
    // to the maximum size allowed (2 * M). This option is used by
    // IndexBinaryHHNSW to create a full base layer graph that is
    // used when GpuIndexBinaryCagra::copyFrom(IndexBinaryHNSW*) is called.
    bool keep_max_size_level0 = false;

    explicit IndexBinaryHNSW();
    explicit IndexBinaryHNSW(int d, int M = 32);
    explicit IndexBinaryHNSW(IndexBinary* storage, int M = 32);

    ~IndexBinaryHNSW() override;

    DistanceComputer* get_distance_computer() const;

    void add(idx_t n, const uint8_t* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const uint8_t* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, uint8_t* recons) const override;

    void reset() override;
};

} // namespace faiss
