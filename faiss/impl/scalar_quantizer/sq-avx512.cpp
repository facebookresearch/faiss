/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/simdlib/simdlib_avx512.h>

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#include <faiss/impl/scalar_quantizer/sq-avx512-impl.h>

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_AVX512
