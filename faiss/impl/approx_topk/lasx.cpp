/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Explicit template instantiations of accum_and_*_tab for
// SIMDLevel::LOONGARCH_LASX.

#include <faiss/impl/approx_topk/rq_beam_search_tab-inl.h>
#include <faiss/impl/simdlib/simdlib_lasx.h>

namespace faiss {

#define INSTANTIATE_ACCUM_TAB(M)                                           \
    template void accum_and_store_tab<M, 4, SIMDLevel::LOONGARCH_LASX>(    \
            size_t,                                                        \
            const float* __restrict,                                       \
            const uint64_t* __restrict,                                    \
            const int32_t* __restrict,                                     \
            size_t,                                                        \
            size_t,                                                        \
            size_t,                                                        \
            float* __restrict);                                            \
    template void accum_and_add_tab<M, 4, SIMDLevel::LOONGARCH_LASX>(      \
            size_t,                                                        \
            const float* __restrict,                                       \
            const uint64_t* __restrict,                                    \
            const int32_t* __restrict,                                     \
            size_t,                                                        \
            size_t,                                                        \
            size_t,                                                        \
            float* __restrict);                                            \
    template void accum_and_finalize_tab<M, 4, SIMDLevel::LOONGARCH_LASX>( \
            const float* __restrict,                                       \
            const uint64_t* __restrict,                                    \
            const int32_t* __restrict,                                     \
            size_t,                                                        \
            size_t,                                                        \
            size_t,                                                        \
            const float* __restrict,                                       \
            const float* __restrict,                                       \
            float* __restrict);

INSTANTIATE_ACCUM_TAB(1)
INSTANTIATE_ACCUM_TAB(2)
INSTANTIATE_ACCUM_TAB(3)
INSTANTIATE_ACCUM_TAB(4)
INSTANTIATE_ACCUM_TAB(5)
INSTANTIATE_ACCUM_TAB(6)
INSTANTIATE_ACCUM_TAB(7)
INSTANTIATE_ACCUM_TAB(8)

#undef INSTANTIATE_ACCUM_TAB

} // namespace faiss
