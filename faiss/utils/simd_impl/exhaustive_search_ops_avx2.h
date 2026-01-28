/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/ordered_key_value.h>
#include <cstddef>
#include <cstdint>

namespace faiss {

void exhaustive_L2sqr_blas_cmax_avx2(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms = nullptr);

void exhaustive_L2sqr_fused_blas_cmax_avx2(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms = nullptr);

} // namespace faiss
