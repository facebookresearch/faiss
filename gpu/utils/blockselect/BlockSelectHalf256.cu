/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "BlockSelectImpl.cuh"

namespace faiss { namespace gpu {

#ifdef FAISS_USE_FLOAT16
BLOCK_SELECT_IMPL(half, true, 256, 4);
BLOCK_SELECT_IMPL(half, false, 256, 4);
#endif

} } // namespace
