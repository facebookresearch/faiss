/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>

namespace faiss {
namespace gpu {

BLOCK_SELECT_IMPL(float, true, 1, 1);
BLOCK_SELECT_IMPL(float, false, 1, 1);
BLOCK_SELECT_IMPL_INDEX(float, true, 1, 1, ushort);
BLOCK_SELECT_IMPL_INDEX(float, false, 1, 1, ushort);

} // namespace gpu
} // namespace faiss
