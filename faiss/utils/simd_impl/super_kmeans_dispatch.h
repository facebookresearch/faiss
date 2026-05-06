/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Private dispatch wrapper for SuperKMeans's block_l2. Routes to the
// highest available SIMD specialization at runtime (DD mode) or the
// compiled-in level (static mode). aarch64 currently falls through to the
// scalar primary template; adding NEON/SVE means just adding a new
// specialization file alongside the AVX ones.
//
// Known perf gap: aarch64 (NEON/SVE) specializations are not implemented in v1.
// aarch64 falls through to the scalar primary template. Validating SVE requires
// a Graviton-class host; deferred to a focused follow-up.

#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/simd_impl/super_kmeans_kernels.h>

namespace faiss {
namespace detail {

inline float block_l2_dispatch(const float* x, const float* y, int n) {
    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_A0>(
            [&]<SIMDLevel SL>() { return block_l2<SL>(x, y, n); });
}

} // namespace detail
} // namespace faiss
