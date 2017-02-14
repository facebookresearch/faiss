
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
#include "WarpSelectImpl.cuh"

namespace faiss { namespace gpu {

#ifdef FAISS_USE_FLOAT16
WARP_SELECT_IMPL(half, true, 64, 3);
WARP_SELECT_IMPL(half, false, 64, 3);
#endif

} } // namespace
