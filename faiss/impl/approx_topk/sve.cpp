/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Explicit template instantiation of HeapWithBucketsCMaxFloat and
// accum_and_*_tab for SIMDLevel::ARM_SVE.

#ifdef COMPILE_SIMD_ARM_SVE

#define THE_SIMD_LEVEL SIMDLevel::ARM_SVE

#include <faiss/impl/approx_topk/rq_beam_search_tab-inl.h>
#include <faiss/impl/approx_topk/simdlib256-inl.h>
#include <faiss/impl/simdlib/simdlib_neon.h>

namespace faiss {

template struct HeapWithBucketsCMaxFloat<8, 3, SIMDLevel::ARM_SVE>;
template struct HeapWithBucketsCMaxFloat<8, 2, SIMDLevel::ARM_SVE>;
template struct HeapWithBucketsCMaxFloat<16, 2, SIMDLevel::ARM_SVE>;
template struct HeapWithBucketsCMaxFloat<16, 1, SIMDLevel::ARM_SVE>;
template struct HeapWithBucketsCMaxFloat<32, 2, SIMDLevel::ARM_SVE>;

#define INSTANCIATE_ACCUM_TAB(M)                                    \
    template void accum_and_store_tab<M, 4, SIMDLevel::ARM_SVE>(    \
            size_t,                                                 \
            const float* __restrict,                                \
            const uint64_t* __restrict,                             \
            const int32_t* __restrict,                              \
            size_t,                                                 \
            size_t,                                                 \
            size_t,                                                 \
            float* __restrict);                                     \
    template void accum_and_add_tab<M, 4, SIMDLevel::ARM_SVE>(      \
            size_t,                                                 \
            const float* __restrict,                                \
            const uint64_t* __restrict,                             \
            const int32_t* __restrict,                              \
            size_t,                                                 \
            size_t,                                                 \
            size_t,                                                 \
            float* __restrict);                                     \
    template void accum_and_finalize_tab<M, 4, SIMDLevel::ARM_SVE>( \
            const float* __restrict,                                \
            const uint64_t* __restrict,                             \
            const int32_t* __restrict,                              \
            size_t,                                                 \
            size_t,                                                 \
            size_t,                                                 \
            const float* __restrict,                                \
            const float* __restrict,                                \
            float* __restrict);

INSTANCIATE_ACCUM_TAB(1)
INSTANCIATE_ACCUM_TAB(2)
INSTANCIATE_ACCUM_TAB(3)
INSTANCIATE_ACCUM_TAB(4)
INSTANCIATE_ACCUM_TAB(5)
INSTANCIATE_ACCUM_TAB(6)
INSTANCIATE_ACCUM_TAB(7)
INSTANCIATE_ACCUM_TAB(8)

#undef INSTANCIATE_ACCUM_TAB

} // namespace faiss

#endif // COMPILE_SIMD_ARM_SVE
