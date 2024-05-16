/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "FaissHook.h"

namespace faiss {

extern float fvec_L2sqr_default(const float* x, const float* y, size_t d);

extern float fvec_inner_product_default(
        const float* x,
        const float* y,
        size_t d);

FVEC_L2SQR_HOOK fvec_L2sqr_hook = fvec_L2sqr_default;
FVEC_INNER_PRODUCT_HOOK fvec_inner_product_hook = fvec_inner_product_default;

void set_fvec_L2sqr_hook(FVEC_L2SQR_HOOK_C hook) {
    if (nullptr != hook)
        fvec_L2sqr_hook = hook;
}
FVEC_L2SQR_HOOK_C get_fvec_L2sqr_hook() {
    return fvec_L2sqr_hook;
}

void set_fvec_inner_product_hook(FVEC_INNER_PRODUCT_HOOK_C hook) {
    if (nullptr != hook)
        fvec_inner_product_hook = hook;
}
FVEC_INNER_PRODUCT_HOOK_C get_fvec_inner_product_hook() {
    return fvec_inner_product_hook;
}

} // namespace faiss
