/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "WarpSelectImpl.cuh"

namespace faiss { namespace gpu {

#ifdef FAISS_USE_FLOAT16
WARP_SELECT_IMPL(half, true, 256, 4);
WARP_SELECT_IMPL(half, false, 256, 4);
#endif

} } // namespace
