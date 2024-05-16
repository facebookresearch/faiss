/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <cstddef>
#include "faiss/impl/platform_macros.h"

namespace faiss {

using FVEC_L2SQR_HOOK = float (*)(const float*, const float*, size_t);

using FVEC_INNER_PRODUCT_HOOK = float (*)(const float*, const float*, size_t);

extern FVEC_L2SQR_HOOK fvec_L2sqr_hook;
extern FVEC_INNER_PRODUCT_HOOK fvec_inner_product_hook;

#ifdef __cplusplus
extern "C" {
#endif

typedef float (*FVEC_L2SQR_HOOK_C)(const float*, const float*, size_t);
typedef float (*FVEC_INNER_PRODUCT_HOOK_C)(const float*, const float*, size_t);

FAISS_API void set_fvec_L2sqr_hook(FVEC_L2SQR_HOOK_C hook);
FAISS_API FVEC_L2SQR_HOOK_C get_fvec_L2sqr_hook();

FAISS_API void set_fvec_inner_product_hook(FVEC_INNER_PRODUCT_HOOK_C hook);
FAISS_API FVEC_INNER_PRODUCT_HOOK_C get_fvec_inner_product_hook();

#ifdef __cplusplus
}
#endif

} // namespace faiss
