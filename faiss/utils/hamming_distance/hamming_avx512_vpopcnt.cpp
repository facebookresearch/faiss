/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512_VPOPCNT

#define THE_SIMD_LEVEL SIMDLevel::AVX512_VPOPCNT
#include <faiss/utils/hamming_distance/hamming_computer-avx512_vpopcnt.h>
#include <faiss/utils/hamming_distance/hamming_impl.h>

// Must follow the computer specializations above.
// clang-format off
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/binary_hamming/IndexBinaryIVF_impl.h>
// clang-format on

#endif // COMPILE_SIMD_AVX512_VPOPCNT
