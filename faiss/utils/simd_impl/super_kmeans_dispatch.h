/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Private dispatch wrapper for SuperKMeans's block_l2. Routes to the
// highest available SIMD specialization at runtime (DD mode) or the
// compiled-in level (static mode).
//
// The BASE_WITH_SVE level mask is required: plain with_simd_level() omits the
// ARM_SVE bit, so on an SVE host the SVE specialization would never be
// instantiated. ARM_NEON has no specialization and falls through to the scalar
// primary template.

#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/simd_impl/super_kmeans_kernels.h>

namespace faiss {
namespace detail {

inline float block_l2_dispatch(const float* x, const float* y, int n) {
    return with_simd_level_with_sve(
            [&]<SIMDLevel SL>() { return block_l2<SL>(x, y, n); });
}

} // namespace detail
} // namespace faiss
