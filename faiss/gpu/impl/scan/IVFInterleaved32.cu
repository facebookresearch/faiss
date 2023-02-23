/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/scan/IVFInterleavedImpl.cuh>

namespace faiss {
namespace gpu {

#if defined(USE_ROCM)
#if __AMDGCN_WAVEFRONT_SIZE == 32u
IVF_INTERLEAVED_IMPL(128, KWARPSIZE, 2)
#else
IVF_INTERLEAVED_IMPL_DUMMY(128, KWARPSIZE, 2)
#endif
#else
IVF_INTERLEAVED_IMPL(128, 32, 2)
#endif

}
} // namespace faiss
