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

/// Panorama adaptation of IndexIVFFlat following
/// https://www.arxiv.org/pdf/2510.00566.
///
/// IDEA:
/// Panorama adapts the storage layout within each cluster and uses
/// pruning with bounds to improve the search performance.
/// Combined with orthogonal transforms upstream that concentrate the energy
/// in the early dimensions (like PCA, Cayley, etc.), Panorama can prune up
/// to 95% of the vectors in the cluster.
///
/// OVERHEAD:
/// To be more efficient, we compute the residual energies at insertion time
/// and store them along the vectors, which comes with an additional storage
/// overhead of exactly (nlevels + 1) floats per vector. Add time is also
/// slightly higher due to the overhead of transposing the vectors.
///
/// NOTE:
/// We inherit from IndexIVFFlat instead of IndexIVF so we can keep the same
/// insertion logic. The code responsible for level-oriented storage is in
/// `ArrayInvertedListsPanorama`, which is a struct member of `IndexIVF`.
struct IndexIVFFlatPanorama : IndexIVFFlat {
    size_t n_levels;

    std::vector<MaybeOwnedVector<float>> cum_sums;

    explicit IndexIVFFlatPanorama(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            int n_levels,
            MetricType = METRIC_L2,
            bool own_invlists = true);

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    IndexIVFFlatPanorama();
};

} // namespace faiss

#endif
