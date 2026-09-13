/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

namespace faiss::rabitq_integer_adc {

FAISS_API int64_t
dot_product_scalar(const int8_t* query, const int8_t* levels, size_t d);

FAISS_API void dot_product_batch_4_scalar(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3);

FAISS_API void dot_product_batch_8_scalar(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t dots[8]);

FAISS_API void dot_product_batch_16_scalar(
        const int8_t* query,
        const int8_t* const levels[16],
        size_t d,
        int64_t dots[16]);

FAISS_API void dot_product_batch_tail_scalar(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots);

#ifdef COMPILE_SIMD_ARM_NEON
FAISS_API bool arm_dotprod_supported();

FAISS_API int64_t
dot_product_arm(const int8_t* query, const int8_t* levels, size_t d);

FAISS_API void dot_product_batch_4_arm(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3);

FAISS_API void dot_product_batch_8_arm(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t dots[8]);

FAISS_API void dot_product_batch_16_arm(
        const int8_t* query,
        const int8_t* const levels[16],
        size_t d,
        int64_t dots[16]);

FAISS_API void dot_product_batch_tail_arm(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots);
#endif

#ifdef COMPILE_SIMD_AVX512_SPR
FAISS_API int64_t dot_product_avx512_vnni(
        const int8_t* query,
        const int8_t* levels,
        size_t d,
        int64_t query_correction);

FAISS_API void dot_product_batch_4_avx512_vnni(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t query_correction,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3);

FAISS_API void dot_product_batch_8_avx512_vnni(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t query_correction,
        int64_t dots[8]);
#endif

} // namespace faiss::rabitq_integer_adc
