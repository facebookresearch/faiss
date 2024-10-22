/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file contains an implementation of approximate top-k search
// using heap. It was initially created for a beam search.
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
//            local_min_distances[i] = distances[idx]
//            local_min_indices[i] = indices[idx]
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

#pragma once

#include <faiss/impl/platform_macros.h>

// the list of available modes is in the following file
#include <faiss/utils/approx_topk/mode.h>

#ifdef __AVX2__
#include <faiss/utils/approx_topk/avx2-inl.h>
#else
#include <faiss/utils/approx_topk/generic.h>
#endif
