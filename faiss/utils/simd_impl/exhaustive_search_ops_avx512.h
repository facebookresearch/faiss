/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/simd_impl/exhaustive_search_ops_avx2.h>

namespace faiss {

template <SIMDLevel>
void exhaustive_L2sqr_blas_simd_avx512_with_avx2_fallback(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms);

} // namespace faiss
