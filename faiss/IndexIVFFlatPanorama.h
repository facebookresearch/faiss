/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_FLAT_PANORAMA_H
#define FAISS_INDEX_IVF_FLAT_PANORAMA_H

#include <stdint.h>
#include "faiss/IndexIVFFlat.h"

namespace faiss {

// We inherit from IndexIVFFlat instead of IndexIVF so we can keep the same
// insertion logic. The code responsible for level-oriented storage is in
// `ArrayInvertedListsPanorama`, which is a struct member of `IndexIVF`.
struct IndexIVFFlatPanorama : IndexIVFFlat {
    explicit IndexIVFFlatPanorama(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2,
            bool own_invlists = true,
            int n_levels = 1);

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

   private:
    const size_t n_levels;
    std::vector<MaybeOwnedVector<float>> cum_sums;
};

} // namespace faiss

#endif
