/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AVX512 hardware execution tests - this file is compiled with AVX512 flags.
// Verifies that AVX512 instructions execute correctly when SIMDConfig reports
// AVX512 support.

#if defined(__x86_64__) || defined(_M_X64)

#include <gtest/gtest.h>
#include <immintrin.h>
#include <setjmp.h>
#include <signal.h>
#include <vector>

#include <faiss/utils/simd_levels.h>

namespace {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
static jmp_buf jmpbuf;

[[noreturn]] static void sigill_handler(int /* sig */) {
    longjmp(jmpbuf, 1);
}

std::pair<bool, std::vector<int>> try_execute_avx512(
        std::vector<int> (*func)()) {
    signal(SIGILL, sigill_handler);
    if (setjmp(jmpbuf) == 0) {
        auto result = func();
        signal(SIGILL, SIG_DFL);
        return std::make_pair(true, result);
    } else {
        signal(SIGILL, SIG_DFL);
        return std::make_pair(false, std::vector<int>());
    }
}

std::vector<int> run_avx512f_computation() {
    alignas(64) long long result[8];
    alignas(64) long long input1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) long long input2[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    __m512i vec1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(input1));
    __m512i vec2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(input2));
    __m512i vec_result = _mm512_add_epi64(vec1, vec2);
    _mm512_store_si512(reinterpret_cast<__m512i*>(result), vec_result);

    return {result, result + 8};
}

std::vector<int> run_avx512cd_computation() {
    run_avx512f_computation();

    __m512i indices = _mm512_set_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i conflict_mask = _mm512_conflict_epi32(indices);

    alignas(64) int mask_array[16];
    _mm512_store_epi32(mask_array, conflict_mask);

    return std::vector<int>();
}

std::vector<int> run_avx512vl_computation() {
    run_avx512f_computation();

    __m256i vec1 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i vec2 = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i result = _mm256_add_epi32(vec1, vec2);
    alignas(32) int result_array[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(result_array), result);

    return std::vector<int>(result_array, result_array + 8);
}

std::vector<int> run_avx512dq_computation() {
    run_avx512f_computation();

    __m512i vec1 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    __m512i vec2 = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i result = _mm512_add_epi64(vec1, vec2);

    alignas(64) long long result_array[8];
    _mm512_store_si512(result_array, result);

    return std::vector<int>(result_array, result_array + 8);
}

std::vector<int> run_avx512bw_computation() {
    run_avx512f_computation();

    std::vector<int8_t> input1(64, 0);
    __m512i vec1 =
            _mm512_loadu_si512(reinterpret_cast<const void*>(input1.data()));
    std::vector<int8_t> input2(64, 7);
    __m512i vec2 =
            _mm512_loadu_si512(reinterpret_cast<const void*>(input2.data()));
    __m512i result = _mm512_add_epi8(vec1, vec2);

    alignas(64) int8_t result_array[64];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_array), result);

    return std::vector<int>(result_array, result_array + 64);
}

} // namespace

TEST(SIMDConfig, successful_avx512f_execution_on_x86arch) {
    if (faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual_result = try_execute_avx512(run_avx512f_computation);
        EXPECT_TRUE(actual_result.first);
        auto expected_result_vector = std::vector<int>(8, 9);
        EXPECT_EQ(actual_result.second, expected_result_vector);
    }
}

TEST(SIMDConfig, successful_avx512cd_execution_on_x86arch) {
    if (faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual = try_execute_avx512(run_avx512cd_computation);
        EXPECT_TRUE(actual.first);
    }
}

TEST(SIMDConfig, successful_avx512vl_execution_on_x86arch) {
    if (faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual = try_execute_avx512(run_avx512vl_computation);
        EXPECT_TRUE(actual.first);
        EXPECT_EQ(actual.second, std::vector<int>(8, 7));
    }
}

TEST(SIMDConfig, successful_avx512dq_execution_on_x86arch) {
    if (faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual = try_execute_avx512(run_avx512dq_computation);
        EXPECT_TRUE(actual.first);
        EXPECT_EQ(actual.second, std::vector<int>(8, 7));
    }
}

TEST(SIMDConfig, successful_avx512bw_execution_on_x86arch) {
    if (faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual = try_execute_avx512(run_avx512bw_computation);
        EXPECT_TRUE(actual.first);
        EXPECT_EQ(actual.second, std::vector<int>(64, 7));
    }
}

#endif // defined(__x86_64__) || defined(_M_X64)
