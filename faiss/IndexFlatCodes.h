/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <vector>

namespace faiss {

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatCodes : Index {
    size_t code_size;

    /// encoded dataset, size ntotal * code_size
    std::vector<uint8_t> codes;

    IndexFlatCodes();

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);

    /// default add uses sa_encode
    void add(idx_t n, const float* x) override;

    void reset() override;

    /// reconstruction using the codec interface
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    size_t sa_code_size() const override;

    /** remove some ids. NB that Because of the structure of the
     * indexing structure, the semantics of this operation are
     * different from the usual ones: the new ids are shifted */
    size_t remove_ids(const IDSelector& sel) override;

    /** a FlatCodesDistanceComputer offers a distance_to_code method */
    virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }
};

} // namespace faiss
