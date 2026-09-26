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

/** Dot products against the native 2-bit RaBitQ layout. The database levels
 * are decoded 16 at a time from [sign bits, ..., packed 1-bit lows] and fed
 * directly to SDOT; no d-byte expanded row is materialized.
 */
FAISS_API int64_t dot_product_packed_2bit_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t extra_offset);

FAISS_API void dot_product_packed_2bit_batch_4_arm(
        const int8_t* query,
        const uint8_t* const codes[4],
        size_t d,
        size_t extra_offset,
        int64_t dots[4]);

FAISS_API void dot_product_packed_2bit_batch_8_arm(
        const int8_t* query,
        const uint8_t* const codes[8],
        size_t d,
        size_t extra_offset,
        int64_t dots[8]);

FAISS_API void dot_product_packed_2bit_batch_16_arm(
        const int8_t* query,
        const uint8_t* const codes[16],
        size_t d,
        size_t extra_offset,
        int64_t dots[16]);

FAISS_API void dot_product_packed_2bit_batch_tail_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        int64_t* dots);

/** Dot products against the native 4-bit RaBitQ layout. Three packed
 * magnitude bits and the separate sign plane are combined in NEON registers
 * and consumed immediately by SDOT, without a d-byte scratch row.
 */
FAISS_API int64_t dot_product_packed_4bit_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t extra_offset);

FAISS_API void dot_product_packed_4bit_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        int64_t* dots);

FAISS_API void dot_product_split_4bit_batch_arm(
        const int8_t* query,
        const uint8_t* const* signs,
        const uint8_t* const* tails,
        int count,
        size_t d,
        int64_t* dots);

/** Dot products against a progressive scorer row:
 * [2-bit prefix per dimension][factors][full-score magnitudes].
 * Prefix scoring reads only the first d/4 bytes. Full scoring combines the
 * sign from the prefix with a 5- or 6-bit tail in registers before SDOT.
 */
FAISS_API int64_t dot_product_progressive_prefix_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d);

FAISS_API int64_t dot_product_progressive_full6_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset);

FAISS_API int64_t dot_product_progressive_full7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset);

FAISS_API void dot_product_progressive_prefix_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        int64_t* dots);

FAISS_API void dot_product_progressive_full6_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots);

FAISS_API void dot_product_progressive_full7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots);

/** Seven-bit nested LUT scorer. The five-bit local magnitude code is split
 * into a nibble plane and a one-bit plane; the RQ2 coarse cell selects one of
 * two 32-byte tables before SDOT consumes the reconstructed signed levels.
 */
FAISS_API int64_t dot_product_nested_lut7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut);

FAISS_API void dot_product_nested_lut7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots);

/** Four-bit nested LUT scorer. Staged rows contain a packed 2-bit prefix and
 * packed 2-bit local plane. Nibble rows contain the equivalent sign/coarse/
 * local symbol directly (two coordinates per byte). Both perform a 16-entry
 * table lookup before SDOT.
 */
FAISS_API int64_t dot_product_nested_lut4_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout);

FAISS_API void dot_product_nested_lut4_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t* dots);

/** Exact compact RQ7 scorer. The coarse bit is the high magnitude bit, so
 * reconstruction is bit assembly and needs no table lookup.
 */
FAISS_API int64_t dot_product_nested_exact7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t low4_offset,
        size_t high1_offset);

FAISS_API void dot_product_nested_exact7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        int64_t* dots);

/** Three-bit nested navigation scorer. It reads the two-bit prefix and the
 * high local-code plane, grouping each 32-entry conditional table into two
 * ordered halves. No additional per-vector storage is required.
 */
FAISS_API int64_t dot_product_nested_lut3_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t high1_offset,
        const uint8_t* lut);

FAISS_API void dot_product_nested_lut3_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots);

/** Two-dot refinement for the residual progressive code. The prefix and tail
 * use independent per-vector scales, so their integer sums stay separate.
 */
FAISS_API void dot_product_progressive_refine6_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset,
        int64_t& prefix_dot,
        int64_t& tail_dot);

FAISS_API void dot_product_progressive_refine6_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* prefix_dots,
        int64_t* tail_dots);
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

/** Direct VNNI scoring over the packed two-bit navigation prefix. */
FAISS_API void dot_product_progressive_prefix_batch_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        int64_t query_correction,
        int64_t* dots);

/** Direct VNNI scoring over compact Nested-LUT rows. These kernels decode
 * one 64-coordinate block at a time and avoid materializing d-byte levels.
 */
FAISS_API int64_t dot_product_nested_lut4_avx512_vnni(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t query_correction);

FAISS_API void dot_product_nested_lut4_batch_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t query_correction,
        int64_t* dots);

FAISS_API int64_t dot_product_nested_lut7_avx512_vnni(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t query_correction);

FAISS_API void dot_product_nested_lut7_batch_avx512_vnni(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t query_correction,
        int64_t* dots);
#endif

} // namespace faiss::rabitq_integer_adc
