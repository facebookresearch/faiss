/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

/**
 * AVX-512 SIMD argpartition implementation
 * Inspired by NumPy's x86-simd-sort / introselect
 *
 * Partitions indices such that the k largest elements' indices are at positions
 * [n-k, n) Uses introselect (quickselect with heap fallback) with
 * SIMD-optimized partition
 *
 * @param n     Number of elements (must be multiple of 16)
 * @param val   Array of values (modified in place)
 * @param idx   Array of indices (modified in place)
 * @param k     Number of largest elements to partition
 *
 * After calling, the k largest elements will be at positions [n-k, n) in both
 * arrays. The relative order within each partition is undefined.
 */
void argpartition(size_t n, float* val, int32_t* idx, size_t k);

/**
 * AVX-512 optimized argsort implementation
 * Uses quicksort with SIMD-optimized partition for large arrays (>16 elements)
 * and bitonic sort for small arrays (<=16 elements)
 *
 * @param n     Number of elements
 * @param val   Array of values (modified in place, sorted in ascending order)
 * @param idx   Array of indices (modified in place, reordered with values)
 */
void argsort(size_t n, float* val, int32_t* idx);
