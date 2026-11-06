/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#include <faiss/Index.h>
#include <faiss/impl/PiPNN.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

struct SearchParametersPiPNN : SearchParameters {
    /// beam width L; if 0, pipnn.search_L is used
    int search_L = 0;
    SearchParameters* clone() const override {
        return new SearchParametersPiPNN(*this);
    }
};

/** The PiPNN index is a normal random-access index with a PiPNN graph built
 * on top of a flat storage (IndexFlat or IndexScalarQuantizer). */
struct IndexPiPNN : Index {
    PiPNN pipnn;
    Index* storage = nullptr;
    bool own_fields = false;
    /// if true, add() only appends to storage; call build() afterwards
    bool defer_build = false;
    /// See impl/VisitedTable.h.
    std::optional<bool> use_visited_hashset;

    IndexPiPNN();
    IndexPiPNN(Index* storage, int R = 64);

    ~IndexPiPNN() override;

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    /// Builds the graph over all vectors in storage.
    void build();

    bool is_built() const {
        return pipnn.ntotal > 0;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /// Empties storage (whatever own_fields says) and drops the graph.
    void reset() override;

    /// Throws unless IndexPiPNN supports the type and metric of `storage`.
    static void check_storage_supported(const Index& storage);
};

/// IndexPiPNN over an owned IndexFlat.
struct IndexPiPNNFlat : IndexPiPNN {
    IndexPiPNNFlat();
    IndexPiPNNFlat(int d, int R, MetricType metric = METRIC_L2);
};

/// IndexPiPNN over an owned IndexScalarQuantizer.
struct IndexPiPNNSQ : IndexPiPNN {
    IndexPiPNNSQ();
    IndexPiPNNSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int R,
            MetricType metric = METRIC_L2);
};

} // namespace faiss
