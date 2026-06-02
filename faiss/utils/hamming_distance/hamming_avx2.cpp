/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX2

#define THE_SIMD_LEVEL SIMDLevel::AVX2
// NOLINTBEGIN(facebook-hte-InlineHeader,facebook-unused-include-check)
#include <faiss/utils/hamming_distance/hamming_computer-avx2.h>
#include <faiss/utils/hamming_distance/hamming_impl.h>
// NOLINTEND(facebook-hte-InlineHeader,facebook-unused-include-check)

#endif // COMPILE_SIMD_AVX2
