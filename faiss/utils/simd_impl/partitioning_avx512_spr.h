/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace partitioning_avx512_spr {

/// AVX-512 VBMI2 count_lt_and_eq for CMax (is_max = true).
void count_lt_and_eq_max_avx512(
        const uint16_t* vals,
        int n,
        uint16_t thresh,
        size_t& n_lt,
        size_t& n_eq);

/// AVX-512 VBMI2 count_lt_and_eq for CMin (is_max = false).
void count_lt_and_eq_min_avx512(
        const uint16_t* vals,
        int n,
        uint16_t thresh,
        size_t& n_lt,
        size_t& n_eq);

/// AVX-512 VBMI2 simd_compress_array with int ids, CMax.
int simd_compress_array_max_int_avx512(
        uint16_t* vals,
        int* ids,
        size_t n,
        uint16_t thresh,
        int n_eq);

/// AVX-512 VBMI2 simd_compress_array with int ids, CMin.
int simd_compress_array_min_int_avx512(
        uint16_t* vals,
        int* ids,
        size_t n,
        uint16_t thresh,
        int n_eq);

/// AVX-512 VBMI2 simd_compress_array with int64_t ids, CMax.
int simd_compress_array_max_int64_avx512(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        uint16_t thresh,
        int n_eq);

/// AVX-512 VBMI2 simd_compress_array with int64_t ids, CMin.
int simd_compress_array_min_int64_avx512(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        uint16_t thresh,
        int n_eq);

} // namespace partitioning_avx512_spr
} // namespace faiss
