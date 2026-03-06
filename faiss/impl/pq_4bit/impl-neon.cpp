/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_NEON

#define THE_LEVEL_TO_DISPATCH SIMDLevel::ARM_NEON
#include <faiss/impl/pq_4bit/dispatching.h> // IWYU pragma: keep

// ARM_SVE forwards to ARM_NEON for PQ4 scanners
namespace faiss {
template <>
std::unique_ptr<PQ4CodeScanner> pq4_make_knn_scanner_impl<SIMDLevel::ARM_SVE>(
        bool is_max,
        size_t nq,
        size_t ntotal,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        bool with_id_map) {
    return pq4_make_knn_scanner_impl<SIMDLevel::ARM_NEON>(
            is_max, nq, ntotal, k, distances, ids, sel, with_id_map);
}
} // namespace faiss

#endif // COMPILE_SIMD_ARM_NEON
