/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/utils/distances.h>

#include <faiss/impl/platform_macros.h>

namespace faiss {

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_norm_L2sqr<AUTOVEC_LEVEL>(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_L2sqr<AUTOVEC_LEVEL>(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_inner_product<AUTOVEC_LEVEL>(
        const float* x,
        const float* y,
        size_t d) {
    float res = 0.F;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * y[i];
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

} // namespace faiss
