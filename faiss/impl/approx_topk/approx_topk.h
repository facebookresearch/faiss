/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Approximate top-k search using bucketed heaps with SIMD acceleration.
//
// HeapWithBuckets<C, NBUCKETS, N> splits n elements into NBUCKETS buckets,
// tracks the top-N per bucket using SIMD, then merges into a regular heap.
// This trades a small accuracy loss for significant speedup.
//
// The core idea is the following.
// Say we need to find beam_size indices with the minimal distance
// values. It is done via heap (priority_queue) using the following
// pseudocode:
//
//   def baseline():
//     distances = np.empty([beam_size * n], dtype=float)
//     indices = np.empty([beam_size * n], dtype=int)
//
//     heap = Heap(max_heap_size=beam_size)
//
//     for i in range(0, beam_size * n):
//         heap.push(distances[i], indices[i])
//
// Basically, this is what heap_addn() function from utils/Heap.h does.
//
// The following scheme can be used for approximate beam search.
// Say, we need to find elements with min distance.
// Basically, we split n elements of every beam into NBUCKETS buckets
// and track the index with the minimal distance for every bucket.
// This can be effectively SIMD-ed and significantly lowers the number
// of operations, but yields approximate results for beam_size >= 2.
//
//  def approximate_v1():
//    distances = np.empty([beam_size * n], dtype=float)
//    indices = np.empty([beam_size * n], dtype=int)
//
//    heap = Heap(max_heap_size=beam_size)
//
//    for beam in range(0, beam_size):
//      # The value of 32 is just an example.
//      # The value may be varied: the larger the value is,
//      #  the slower and the more precise vs baseline beam search is
//      NBUCKETS = 32
//
//     local_min_distances = [HUGE_VALF] * NBUCKETS
//     local_min_indices = [0] * NBUCKETS
//
//      for i in range(0, n / NBUCKETS):
//        for j in range(0, NBUCKETS):
//          idx = beam * n + i * NBUCKETS + j
//          if distances[idx] < local_min_distances[j]:
//            local_min_distances[j] = distances[idx]
//            local_min_indices[j] = indices[idx]
//
//    for j in range(0, NBUCKETS):
//      heap.push(local_min_distances[j], local_min_indices[j])
//
// The accuracy can be improved by tracking min-2 elements for every
// bucket. Such a min-2 implementation with NBUCKETS buckets provides
// better accuracy than top-1 implementation with 2 * NBUCKETS buckets.
// Min-3 is also doable. One can use min-N approach, but I'm not sure
// whether min-4 and above are practical, because of the lack of SIMD
// registers (unless AVX-512 version is used).
//
// C++ template for top-N implementation is provided. The code
// assumes that indices[idx] == idx. One can write a code that lifts
// such an assumption easily.
//
// Currently, the code that tracks elements with min distances is implemented
//    (Max Heap). Min Heap option can be added easily.
//
// Dispatch:
//   AVX2 / ARM_NEON  →  HeapWithBucketsCMaxFloat (simdlib256-inl.h)
//   NONE (scalar)    →  HeapWithBucketsGenericCMaxFloat (generic.h)
//
// The SIMD definitions live in simdlib256-inl.h (only included by per-ISA
// .cpp files). Common TUs see only declarations here, so no extern template
// suppression is needed.

#pragma once

#include <cstdint>

#include <faiss/impl/approx_topk/generic.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/simd_levels.h>

// -----------------------------------------------------------------------
// ApproxTopK_mode_t
// -----------------------------------------------------------------------

/// Represents the mode of use of approximate top-k computations
/// that allows to trade accuracy vs speed. So, every options
/// besides EXACT_TOPK increases the speed.
///
/// B represents the number of buckets.
/// D is the number of min-k elements to track within every bucket.
///
/// Default option is EXACT_TOPK.
/// APPROX_TOPK_BUCKETS_B16_D2 is worth starting from, if you'd like
/// to experiment a bit.
///
/// It seems that only the limited number of combinations are
/// meaningful, because of the limited supply of SIMD registers.
/// Also, certain combinations, such as B32_D1 and B16_D1, were concluded
/// to be not very precise in benchmarks, so they were not introduced.

enum ApproxTopK_mode_t : int {
    EXACT_TOPK = 0,
    APPROX_TOPK_BUCKETS_B32_D2 = 1,
    APPROX_TOPK_BUCKETS_B8_D3 = 2,
    APPROX_TOPK_BUCKETS_B16_D2 = 3,
    APPROX_TOPK_BUCKETS_B8_D2 = 4,
};

// -----------------------------------------------------------------------
// HeapWithBuckets dispatch
// -----------------------------------------------------------------------

namespace faiss {

// Primary template: declared, bs_addn NOT defined here.
// The out-of-line definition lives in simdlib256-inl.h, which is only
// included by the per-ISA .cpp files (avx2.cpp, neon.cpp).
template <uint32_t NBUCKETS, uint32_t N, SIMDLevel SL>
struct HeapWithBucketsCMaxFloat {
    static_assert(
            (NBUCKETS) > 0 && ((NBUCKETS % 8) == 0),
            "Number of buckets needs to be 8, 16, 24, ...");

    static void addn(
            const uint32_t n,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        bs_addn(1, n, distances, k, bh_val, bh_ids);
    }

    // Declared but not defined — resolved at link time from avx2.cpp/neon.cpp.
    static void bs_addn(
            const uint32_t beam_size,
            const uint32_t n_per_beam,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids);
};

// NONE specialization: delegates to the scalar generic implementation.
template <uint32_t NBUCKETS, uint32_t N>
struct HeapWithBucketsCMaxFloat<NBUCKETS, N, SIMDLevel::NONE> {
    static void addn(
            const uint32_t n,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        bs_addn(1, n, distances, k, bh_val, bh_ids);
    }

    static void bs_addn(
            const uint32_t beam_size,
            const uint32_t n_per_beam,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        HeapWithBucketsGenericCMaxFloat<NBUCKETS, N>::bs_addn(
                beam_size, n_per_beam, distances, k, bh_val, bh_ids);
    }
};

// Primary template — not implemented for arbitrary comparators.
template <typename C, uint32_t NBUCKETS, uint32_t N>
struct HeapWithBuckets {
    static_assert(
            sizeof(C) == 0,
            "HeapWithBuckets: unsupported comparator type");
};

// Partial specialization for CMax<float, int> that dispatches
// via with_simd_level_256bit.
template <uint32_t NBUCKETS, uint32_t N>
struct HeapWithBuckets<CMax<float, int>, NBUCKETS, N> {
    static void addn(
            const uint32_t n,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        bs_addn(1, n, distances, k, bh_val, bh_ids);
    }

    static void bs_addn(
            const uint32_t beam_size,
            const uint32_t n_per_beam,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        with_simd_level_256bit([&]<SIMDLevel SL>() {
            HeapWithBucketsCMaxFloat<NBUCKETS, N, SL>::bs_addn(
                    beam_size, n_per_beam, distances, k, bh_val, bh_ids);
        });
    }
};

} // namespace faiss
