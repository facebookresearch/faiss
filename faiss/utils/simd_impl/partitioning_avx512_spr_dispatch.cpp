/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Per-ISA partitioning TU for AVX-512 Sapphire Rapids.
// Including partitioning_simdlib256.h with THE_SIMD_LEVEL = AVX512_SPR
// emits explicit instantiations of partition_fuzzy_simd<SIMDLevel::AVX512_SPR,
// ...> and simd_histogram_{8,16}<SIMDLevel::AVX512_SPR>. The VBMI2 fast paths
// for count_lt_and_eq and simd_compress_array are hooked inside the header
// and forward to implementations in partitioning_avx512_spr.cpp.

#ifdef COMPILE_SIMD_AVX512_SPR

#define THE_SIMD_LEVEL SIMDLevel::AVX512_SPR
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/partitioning_simdlib256.h>

#endif // COMPILE_SIMD_AVX512_SPR
