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
    using storage_idx_t = HNSW::storage_idx_t;

    // the link structure
    HNSW hnsw;

    // the sequential storage
    bool own_fields = false;
    IndexBinary* storage = nullptr;

    // When set to false, level 0 in the knn graph is not initialized.
    // This option is used by GpuIndexBinaryCagra::copyTo(IndexBinaryHNSW*)
    // as level 0 knn graph is copied over from the index built by
    // GpuIndexBinaryCagra.
    bool init_level0 = true;

    // When set to true, all neighbors in level 0 are filled up
    // to the maximum size allowed (2 * M). This option is used by
    // IndexBinaryHNSW to create a full base layer graph that is
    // used when GpuIndexBinaryCagra::copyFrom(IndexBinaryHNSW*) is called.
    bool keep_max_size_level0 = false;

    // When this index is the coarse quantizer of an IndexBinaryIVF,
    // IndexBinaryIVF::train() calls add_level_0_links_from_sample() after it
    // adds the centroids. The sample is the first
    // min(n, level_0_link_samples_per_node * ntotal) training vectors. 0
    // disables the call. Training does not call it when the quantizer is
    // already trained. The index format does not store this value.
    int level_0_link_samples_per_node = 16;

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

    /** Adds level-0 links so that the search no longer stops at "sinks":
     * nodes where it stops for many vectors whose nearest node is elsewhere.
     * See IndexHNSW::add_level_0_links_from_sample(). x holds n binary
     * vectors. Call this function after the last add().
     */
    HNSWLevel0LinkStats add_level_0_links_from_sample(
            idx_t n,
            const uint8_t* x,
            const HNSWLevel0LinkParameters& params =
                    HNSWLevel0LinkParameters());
};

struct IndexBinaryHNSWCagra : IndexBinaryHNSW {
    IndexBinaryHNSWCagra();
    IndexBinaryHNSWCagra(int d, int M);

    /// When set to true, the index is immutable.
    /// This option is used to copy the knn graph from GpuIndexBinaryCagra
    /// to the base level of IndexBinaryHNSWCagra without adding upper levels.
    /// Doing so enables to search the HNSW index, but removes the
    /// ability to add vectors.
    bool base_level_only = false;

    /// When `base_level_only` is set to `True`, the search function
    /// searches only the base level knn graph of the HNSW index.
    /// This parameter selects the entry point by randomly selecting
    /// some points and using the best one.
    int num_base_level_search_entrypoints = 32;

    void add(idx_t n, const uint8_t* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
