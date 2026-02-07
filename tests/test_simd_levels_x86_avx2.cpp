/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AVX2 hardware execution tests - this file is compiled with AVX2 flags.
// Verifies that AVX2 instructions execute correctly when SIMDConfig reports
// AVX2 support.

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

std::pair<bool, std::vector<int>> try_execute_avx2(std::vector<int> (*func)()) {
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

std::vector<int> run_avx2_computation() {
    alignas(32) int result[8];
    alignas(32) int input1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) int input2[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    __m256i vec1 = _mm256_load_si256(reinterpret_cast<__m256i*>(input1));
    __m256i vec2 = _mm256_load_si256(reinterpret_cast<__m256i*>(input2));
    __m256i vec_result = _mm256_add_epi32(vec1, vec2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(result), vec_result);

    return {result, result + 8};
}

} // namespace

TEST(SIMDConfig, successful_avx2_execution_on_x86arch) {
    if (faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        auto actual_result = try_execute_avx2(run_avx2_computation);
        EXPECT_TRUE(actual_result.first);
        auto expected_result_vector = std::vector<int>(8, 9);
        EXPECT_EQ(actual_result.second, expected_result_vector);
    }
}

TEST(SIMDConfig, on_avx512f_supported_we_should_have_avx2_support_as_well) {
    if (faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        EXPECT_TRUE(
                faiss::SIMDConfig::is_simd_level_available(
                        faiss::SIMDLevel::AVX2));
    }
}

#endif // defined(__x86_64__) || defined(_M_X64)
