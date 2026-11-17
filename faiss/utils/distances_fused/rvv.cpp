/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances_fused/distances_fused.h>

namespace faiss {

// Scalar fallback: no fused kernel available.
template <>
bool exhaustive_L2sqr_fused_cmax<SIMDLevel::RISCV_RVV>(
        const float*,
        const float*,
        size_t,
        size_t,
        size_t,
        Top1BlockResultHandler<CMax<float, int64_t>>&,
        const float*) {
    return false;
}

} // namespace faiss
