/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/scan/IVFInterleavedImpl.cuh>

namespace faiss {
namespace gpu {

#if GPU_MAX_SELECTION_K >= 2048
IVF_INTERLEAVED_IMPL(64, 2048, 8)
#endif

} // namespace gpu
} // namespace faiss
