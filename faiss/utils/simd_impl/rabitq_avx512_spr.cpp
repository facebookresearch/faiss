/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file rabitq_avx512_spr.cpp
 *
 * RaBitQ SIMD kernels specialized for SIMDLevel::AVX512_SPR.
 *
 * Sapphire Rapids (SPR) and later Intel microarchitectures expose
 * AVX-512 VPOPCNTDQ (vpopcntq), which performs a per-lane 64-bit
 * popcount in a single instruction. This is used here to replace the
 * multi-step shuffle/pshufb-based popcount used by the generic AVX-512
 * specialization in rabitq_avx512.cpp. The popcount-heavy kernels
 * (bitwise_and_dot_product, bitwise_xor_dot_product, popcount) become
 * substantially shorter and faster on SPR+ as a result.
 *
 * Build / dispatch behavior:
 *   - faiss_avx512 (AVX-512 only, no SPR features): NOT compiled.
 *     The existing AVX512 specialization in rabitq_avx512.cpp is used.
 *   - faiss_avx512_spr (statically built for SPR+): compiled. The
 *     SINGLE_SIMD_LEVEL is AVX512_SPR, so this specialization is
 *     selected by static dispatch.
 *   - faiss with FAISS_OPT_LEVEL=dd (dynamic dispatch): compiled with
 *     -mavx512vpopcntdq as a per-file flag. Selected at runtime when
 *     SIMDConfig::level == SIMDLevel::AVX512_SPR.
 *
 * The floating-point multi-bit inner-product kernel does not benefit
 * from VPOPCNTDQ, so this TU forwards compute_inner_product<SPR> to
 * the AVX512 implementation to avoid duplicating that code path.
 */

#ifdef COMPILE_SIMD_AVX512_SPR

#include <faiss/utils/popcount.h>
#include <faiss/utils/rabitq_simd.h>
#include <immintrin.h>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace faiss::rabitq {

// Forward declarations for the AVX512 specializations defined in
// rabitq_avx512.cpp. They live in the same TU group on SPR builds, so
// we can reuse them as a tail handler / fallback. Declaring rather
// than redefining avoids ODR risk and keeps a single source of truth
// for the floating-point kernel.
template <>
uint64_t bitwise_and_dot_product<SIMDLevel::AVX512>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb);
template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::AVX512>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb);
template <>
uint64_t popcount<SIMDLevel::AVX512>(const uint8_t* data, size_t size);

namespace {

// 512-bit popcount using AVX-512 VPOPCNTDQ (vpopcntq).
// Single-instruction per-lane popcount on 8x uint64 lanes.
inline __m512i popcount_512_vpopcntdq(__m512i v) {
    return _mm512_popcnt_epi64(v);
}

// 256-bit popcount using AVX-512VL VPOPCNTDQ.
// AVX512VL is part of the SPR feature set, so vpopcntq is available
// on 256-bit registers via _mm256_popcnt_epi64.
inline __m256i popcount_256_vpopcntdq(__m256i v) {
    return _mm256_popcnt_epi64(v);
}

// 128-bit popcount using AVX-512VL VPOPCNTDQ.
inline __m128i popcount_128_vpopcntdq(__m128i v) {
    return _mm_popcnt_epi64(v);
}

inline uint64_t reduce_add_256(__m256i v) {
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

inline uint64_t reduce_add_128(__m128i v) {
    alignas(16) uint64_t lanes[2];
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), v);
    return lanes[0] + lanes[1];
}

} // namespace

template <>
uint64_t bitwise_and_dot_product<SIMDLevel::AVX512_SPR>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;

    // 512-bit main loop: vpopcntq replaces the shuffle-based popcount,
    // halving the instruction count per iteration relative to AVX512.
    if (size_t step = 512 / 8; offset + step <= size) {
        __m512i sum_512 = _mm512_setzero_si512();
        for (; offset + step <= size; offset += step) {
            __m512i v_x = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(data + offset));
            for (size_t j = 0; j < qb; j++) {
                __m512i v_q = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(
                                query + j * size + offset));
                __m512i v_and = _mm512_and_si512(v_q, v_x);
                __m512i v_popcnt = popcount_512_vpopcntdq(v_and);
                __m512i v_shifted = _mm512_slli_epi64(v_popcnt, j);
                sum_512 = _mm512_add_epi64(sum_512, v_shifted);
            }
        }
        sum += _mm512_reduce_add_epi64(sum_512);
    }

    // 256-bit tail.
    if (size_t step = 256 / 8; offset + step <= size) {
        __m256i sum_256 = _mm256_setzero_si256();
        for (; offset + step <= size; offset += step) {
            __m256i v_x = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(data + offset));
            for (size_t j = 0; j < qb; j++) {
                __m256i v_q = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                                query + j * size + offset));
                __m256i v_and = _mm256_and_si256(v_q, v_x);
                __m256i v_popcnt = popcount_256_vpopcntdq(v_and);
                __m256i v_shifted = _mm256_slli_epi64(v_popcnt, j);
                sum_256 = _mm256_add_epi64(sum_256, v_shifted);
            }
        }
        sum += reduce_add_256(sum_256);
    }

    // 128-bit tail.
    __m128i sum_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(data + offset));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(
                            query + j * size + offset));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = popcount_128_vpopcntdq(v_and);
            __m128i v_shifted = _mm_slli_epi64(v_popcnt, j);
            sum_128 = _mm_add_epi64(sum_128, v_shifted);
        }
    }
    sum += reduce_add_128(sum_128);

    // 64-bit scalar tail.
    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *reinterpret_cast<const uint64_t*>(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const auto qv = *reinterpret_cast<const uint64_t*>(
                    query + j * size + offset);
            sum += static_cast<uint64_t>(popcount64(qv & yv)) << j;
        }
    }
    // Byte tail.
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const auto qv = *(query + j * size + offset);
            sum += static_cast<uint64_t>(popcount32(qv & yv)) << j;
        }
    }
    return sum;
}

template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::AVX512_SPR>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0;
    size_t offset = 0;

    if (size_t step = 512 / 8; offset + step <= size) {
        __m512i sum_512 = _mm512_setzero_si512();
        for (; offset + step <= size; offset += step) {
            __m512i v_x = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(data + offset));
            for (size_t j = 0; j < qb; j++) {
                __m512i v_q = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(
                                query + j * size + offset));
                __m512i v_xor = _mm512_xor_si512(v_q, v_x);
                __m512i v_popcnt = popcount_512_vpopcntdq(v_xor);
                __m512i v_shifted = _mm512_slli_epi64(v_popcnt, j);
                sum_512 = _mm512_add_epi64(sum_512, v_shifted);
            }
        }
        sum += _mm512_reduce_add_epi64(sum_512);
    }

    if (size_t step = 256 / 8; offset + step <= size) {
        __m256i sum_256 = _mm256_setzero_si256();
        for (; offset + step <= size; offset += step) {
            __m256i v_x = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(data + offset));
            for (size_t j = 0; j < qb; j++) {
                __m256i v_q = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                                query + j * size + offset));
                __m256i v_xor = _mm256_xor_si256(v_q, v_x);
                __m256i v_popcnt = popcount_256_vpopcntdq(v_xor);
                __m256i v_shifted = _mm256_slli_epi64(v_popcnt, j);
                sum_256 = _mm256_add_epi64(sum_256, v_shifted);
            }
        }
        sum += reduce_add_256(sum_256);
    }

    __m128i sum_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(data + offset));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(
                            query + j * size + offset));
            __m128i v_xor = _mm_xor_si128(v_q, v_x);
            __m128i v_popcnt = popcount_128_vpopcntdq(v_xor);
            __m128i v_shifted = _mm_slli_epi64(v_popcnt, j);
            sum_128 = _mm_add_epi64(sum_128, v_shifted);
        }
    }
    sum += reduce_add_128(sum_128);

    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *reinterpret_cast<const uint64_t*>(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const auto qv = *reinterpret_cast<const uint64_t*>(
                    query + j * size + offset);
            sum += static_cast<uint64_t>(popcount64(qv ^ yv)) << j;
        }
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const auto qv = *(query + j * size + offset);
            sum += static_cast<uint64_t>(popcount32(qv ^ yv)) << j;
        }
    }
    return sum;
}

template <>
uint64_t popcount<SIMDLevel::AVX512_SPR>(const uint8_t* data, size_t size) {
    uint64_t sum = 0;
    size_t offset = 0;

    if (offset + 512 / 8 <= size) {
        __m512i sum_512 = _mm512_setzero_si512();
        for (size_t end; (end = offset + 512 / 8) <= size; offset = end) {
            __m512i v_x = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(data + offset));
            __m512i v_popcnt = popcount_512_vpopcntdq(v_x);
            sum_512 = _mm512_add_epi64(sum_512, v_popcnt);
        }
        sum += _mm512_reduce_add_epi64(sum_512);
    }

    if (offset + 256 / 8 <= size) {
        __m256i sum_256 = _mm256_setzero_si256();
        for (size_t end; (end = offset + 256 / 8) <= size; offset = end) {
            __m256i v_x = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(data + offset));
            __m256i v_popcnt = popcount_256_vpopcntdq(v_x);
            sum_256 = _mm256_add_epi64(sum_256, v_popcnt);
        }
        sum += reduce_add_256(sum_256);
    }

    __m128i sum_128 = _mm_setzero_si128();
    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(data + offset));
        sum_128 = _mm_add_epi64(sum_128, popcount_128_vpopcntdq(v_x));
    }
    sum += reduce_add_128(sum_128);

    for (size_t step = 64 / 8; offset + step <= size; offset += step) {
        const auto yv = *reinterpret_cast<const uint64_t*>(data + offset);
        sum += popcount64(yv);
    }
    for (; offset < size; ++offset) {
        const auto yv = *(data + offset);
        sum += popcount32(yv);
    }
    return sum;
}

} // namespace faiss::rabitq

namespace faiss::rabitq::multibit {

// Forward-declare the AVX512 floating-point inner-product kernel.
// VPOPCNTDQ does not help this kernel (it operates on FP32), so we
// reuse the AVX512 implementation rather than duplicate it.
template <>
float compute_inner_product<SIMDLevel::AVX512>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb);

template <>
float compute_inner_product<SIMDLevel::AVX512_SPR>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    return compute_inner_product<SIMDLevel::AVX512>(
            sign_bits, ex_code, rotated_q, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_AVX512_SPR
