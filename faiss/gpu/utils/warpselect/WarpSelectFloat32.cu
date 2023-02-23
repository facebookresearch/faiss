/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>

namespace faiss {
namespace gpu {

#if defined(USE_ROCM)
#if __AMDGCN_WAVEFRONT_SIZE == 32u
WARP_SELECT_IMPL(float, true, 32, 2);
WARP_SELECT_IMPL(float, false, 32, 2);
#else
WARP_SELECT_IMPL_DUMMY(float, true, 32, 2);
WARP_SELECT_IMPL_DUMMY(float, false, 32, 2);
#endif
#else
WARP_SELECT_IMPL(float, true, 32, 2);
WARP_SELECT_IMPL(float, false, 32, 2);
#endif

} // namespace gpu
} // namespace faiss
