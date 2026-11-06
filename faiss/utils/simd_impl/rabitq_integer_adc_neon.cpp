/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>

#include <faiss/impl/FaissAssert.h>

#include <arm_neon.h>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

#define FAISS_RABITQ_NEON_NAMESPACE neon_dotprod
#define FAISS_RABITQ_NEON_DECLARATIONS_ONLY
#include "rabitq_integer_adc_neon-inl.h"
#undef FAISS_RABITQ_NEON_DECLARATIONS_ONLY
#undef FAISS_RABITQ_NEON_NAMESPACE

#define FAISS_RABITQ_NEON_NAMESPACE neon_fallback
#include "rabitq_integer_adc_neon-inl.h"
#undef FAISS_RABITQ_NEON_NAMESPACE

namespace faiss::rabitq_integer_adc {

int64_t dot_product_arm(const int8_t* query, const int8_t* levels, size_t d) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_arm(query, levels, d);
    }
    return neon_fallback::dot_product_arm(query, levels, d);
}

void dot_product_batch_4_arm(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_batch_4_arm(
                query,
                levels0,
                levels1,
                levels2,
                levels3,
                d,
                dot0,
                dot1,
                dot2,
                dot3);
    }
    return neon_fallback::dot_product_batch_4_arm(
            query,
            levels0,
            levels1,
            levels2,
            levels3,
            d,
            dot0,
            dot1,
            dot2,
            dot3);
}

void dot_product_batch_8_arm(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t dots[8]) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_batch_8_arm(query, levels, d, dots);
    }
    return neon_fallback::dot_product_batch_8_arm(query, levels, d, dots);
}

void dot_product_batch_16_arm(
        const int8_t* query,
        const int8_t* const levels[16],
        size_t d,
        int64_t dots[16]) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_batch_16_arm(query, levels, d, dots);
    }
    return neon_fallback::dot_product_batch_16_arm(query, levels, d, dots);
}

void dot_product_batch_tail_arm(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_batch_tail_arm(
                query, levels, count, d, dots);
    }
    return neon_fallback::dot_product_batch_tail_arm(
            query, levels, count, d, dots);
}

int64_t dot_product_packed_2bit_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t extra_offset) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_packed_2bit_arm(
                query, code, d, extra_offset);
    }
    return neon_fallback::dot_product_packed_2bit_arm(
            query, code, d, extra_offset);
}

void dot_product_packed_2bit_batch_4_arm(
        const int8_t* query,
        const uint8_t* const codes[4],
        size_t d,
        size_t extra_offset,
        int64_t dots[4]) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_packed_2bit_batch_4_arm(
                query, codes, d, extra_offset, dots);
    }
    return neon_fallback::dot_product_packed_2bit_batch_4_arm(
            query, codes, d, extra_offset, dots);
}

void dot_product_packed_2bit_batch_8_arm(
        const int8_t* query,
        const uint8_t* const codes[8],
        size_t d,
        size_t extra_offset,
        int64_t dots[8]) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_packed_2bit_batch_8_arm(
                query, codes, d, extra_offset, dots);
    }
    return neon_fallback::dot_product_packed_2bit_batch_8_arm(
            query, codes, d, extra_offset, dots);
}

void dot_product_packed_2bit_batch_16_arm(
        const int8_t* query,
        const uint8_t* const codes[16],
        size_t d,
        size_t extra_offset,
        int64_t dots[16]) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_packed_2bit_batch_16_arm(
                query, codes, d, extra_offset, dots);
    }
    return neon_fallback::dot_product_packed_2bit_batch_16_arm(
            query, codes, d, extra_offset, dots);
}

void dot_product_packed_2bit_batch_tail_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_packed_2bit_batch_tail_arm(
                query, codes, count, d, extra_offset, dots);
    }
    return neon_fallback::dot_product_packed_2bit_batch_tail_arm(
            query, codes, count, d, extra_offset, dots);
}

int64_t dot_product_packed_4bit_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t extra_offset) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_packed_4bit_arm(
                query, code, d, extra_offset);
    }
    return neon_fallback::dot_product_packed_4bit_arm(
            query, code, d, extra_offset);
}

void dot_product_packed_4bit_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t extra_offset,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_packed_4bit_batch_arm(
                query, codes, count, d, extra_offset, dots);
    }
    return neon_fallback::dot_product_packed_4bit_batch_arm(
            query, codes, count, d, extra_offset, dots);
}

int64_t dot_product_progressive_prefix_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_progressive_prefix_arm(query, code, d);
    }
    return neon_fallback::dot_product_progressive_prefix_arm(query, code, d);
}

int64_t dot_product_progressive_full6_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_progressive_full6_arm(
                query, code, d, tail_offset);
    }
    return neon_fallback::dot_product_progressive_full6_arm(
            query, code, d, tail_offset);
}

int64_t dot_product_progressive_full7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t tail_offset) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_progressive_full7_arm(
                query, code, d, tail_offset);
    }
    return neon_fallback::dot_product_progressive_full7_arm(
            query, code, d, tail_offset);
}

void dot_product_progressive_prefix_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_progressive_prefix_batch_arm(
                query, codes, count, d, dots);
    }
    return neon_fallback::dot_product_progressive_prefix_batch_arm(
            query, codes, count, d, dots);
}

void dot_product_progressive_full6_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_progressive_full6_batch_arm(
                query, codes, count, d, tail_offset, dots);
    }
    return neon_fallback::dot_product_progressive_full6_batch_arm(
            query, codes, count, d, tail_offset, dots);
}

void dot_product_progressive_full7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t tail_offset,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_progressive_full7_batch_arm(
                query, codes, count, d, tail_offset, dots);
    }
    return neon_fallback::dot_product_progressive_full7_batch_arm(
            query, codes, count, d, tail_offset, dots);
}

int64_t dot_product_nested_lut7_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_nested_lut7_arm(
                query, code, d, low4_offset, high1_offset, lut);
    }
    return neon_fallback::dot_product_nested_lut7_arm(
            query, code, d, low4_offset, high1_offset, lut);
}

void dot_product_nested_lut7_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t low4_offset,
        size_t high1_offset,
        const uint8_t* lut,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_nested_lut7_batch_arm(
                query, codes, count, d, low4_offset, high1_offset, lut, dots);
    }
    return neon_fallback::dot_product_nested_lut7_batch_arm(
            query, codes, count, d, low4_offset, high1_offset, lut, dots);
}

int64_t dot_product_nested_lut4_arm(
        const int8_t* query,
        const uint8_t* code,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_nested_lut4_arm(
                query, code, d, local_offset, lut, nibble_layout);
    }
    return neon_fallback::dot_product_nested_lut4_arm(
            query, code, d, local_offset, lut, nibble_layout);
}

void dot_product_nested_lut4_batch_arm(
        const int8_t* query,
        const uint8_t* const* codes,
        int count,
        size_t d,
        size_t local_offset,
        const uint8_t* lut,
        bool nibble_layout,
        int64_t* dots) {
    if (arm_dotprod_supported()) {
        return neon_dotprod::dot_product_nested_lut4_batch_arm(
                query, codes, count, d, local_offset, lut, nibble_layout, dots);
    }
    return neon_fallback::dot_product_nested_lut4_batch_arm(
            query, codes, count, d, local_offset, lut, nibble_layout, dots);
}

} // namespace faiss::rabitq_integer_adc
