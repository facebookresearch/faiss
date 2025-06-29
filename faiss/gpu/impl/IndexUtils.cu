/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <limits>

namespace faiss {
namespace gpu {

/// A collection of various utility functions for index implementation

/// Returns the maximum k-selection value supported based on the CUDA SDK that
/// we were compiled with. .cu files can use DeviceDefs.cuh, but this is for
/// non-CUDA files
int getMaxKSelection(bool use_cuvs) {
    if (use_cuvs) {
        return 16384;
    }
    return GPU_MAX_SELECTION_K;
}

void validateKSelect(int k, bool use_cuvs) {
    FAISS_THROW_IF_NOT_FMT(
            k > 0 && k <= getMaxKSelection(use_cuvs),
            "GPU index only supports min/max-K selection up to %d (requested %d)",
            getMaxKSelection(use_cuvs),
            k);
}

void validateNProbe(size_t nprobe, bool use_cuvs) {
    FAISS_THROW_IF_NOT_FMT(
            nprobe > 0 && nprobe <= (size_t)getMaxKSelection(use_cuvs),
            "GPU IVF index only supports nprobe selection up to %d (requested %zu)",
            getMaxKSelection(use_cuvs),
            nprobe);
}

} // namespace gpu
} // namespace faiss
