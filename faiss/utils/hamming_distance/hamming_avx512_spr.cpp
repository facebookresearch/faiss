/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512_SPR

#define THE_SIMD_LEVEL SIMDLevel::AVX512_SPR
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-avx512_spr.h>
#include <faiss/utils/hamming_distance/hamming_impl.h>

// IndexBinaryIVF's scanner and search paths are instantiated here rather than
// in impl/binary_hamming/, because this is the one translation unit compiled
// with VPOPCNTDQ, which is what HammingComputerDefault_tpl<AVX512_SPR> needs.
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/binary_hamming/IndexBinaryIVF_impl.h>

#endif // COMPILE_SIMD_AVX512_SPR
