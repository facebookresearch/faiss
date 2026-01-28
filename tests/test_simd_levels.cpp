/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <setjmp.h>
#include <vector>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <faiss/utils/simd_levels.h>

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
static jmp_buf jmpbuf;
[[noreturn]] static void sigill_handler(int /* sig */) {
    longjmp(jmpbuf, 1);
}

bool try_execute(void (*func)()) {
    signal(SIGILL, sigill_handler);
    if (setjmp(jmpbuf) == 0) {
        func();
        signal(SIGILL, SIG_DFL);
        return true;
    } else {
        signal(SIGILL, SIG_DFL);
        return false;
    }
}

#ifdef __x86_64__
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
#endif // __x86_64__

std::pair<bool, std::vector<int>> try_execute(std::vector<int> (*func)()) {
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

TEST(SIMDConfig, simd_level_auto_detect_architecture_only) {
    faiss::SIMDLevel detected_level =
            faiss::SIMDConfig::auto_detect_simd_level();

#if defined(__x86_64__) &&                                  \
        (defined(__AVX2__) ||                               \
         (defined(__AVX512F__) && defined(__AVX512CD__) &&  \
          defined(__AVX512VL__) && defined(__AVX512BW__) && \
          defined(__AVX512DQ__)))
    EXPECT_TRUE(
            detected_level == faiss::SIMDLevel::AVX2 ||
            detected_level == faiss::SIMDLevel::AVX512);
#elif defined(__aarch64__) && defined(__ARM_NEON) && \
        defined(COMPILE_SIMD_ARM_NEON)
    EXPECT_TRUE(detected_level == faiss::SIMDLevel::ARM_NEON);
#else
    EXPECT_EQ(detected_level, faiss::SIMDLevel::NONE);
#endif
}

#ifdef __x86_64__
TEST(SIMDConfig, successful_avx2_execution_on_x86arch) {
    faiss::SIMDConfig simd_config(nullptr);

    if (simd_config.is_simd_level_available(faiss::SIMDLevel::AVX2)) {
        auto actual_result = try_execute(run_avx2_computation);
        EXPECT_TRUE(actual_result.first);
        auto expected_result_vector = std::vector<int>(8, 9);
        EXPECT_EQ(actual_result.second, expected_result_vector);
    }
}

TEST(SIMDConfig, on_avx512f_supported_we_should_avx2_support_as_well) {
    faiss::SIMDConfig simd_config(nullptr);

    if (simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        EXPECT_TRUE(
                simd_config.is_simd_level_available(faiss::SIMDLevel::AVX2));
    }
}

TEST(SIMDConfig, successful_avx512f_execution_on_x86arch) {
    faiss::SIMDConfig simd_config(nullptr);

    if (simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual_result = try_execute(run_avx512f_computation);
        EXPECT_TRUE(actual_result.first);
        auto expected_result_vector = std::vector<int>(8, 9);
        EXPECT_EQ(actual_result.second, expected_result_vector);
    }
}

TEST(SIMDConfig, successful_avx512cd_execution_on_x86arch) {
    faiss::SIMDConfig simd_config(nullptr);

    if (simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual = try_execute(run_avx512cd_computation);
        EXPECT_TRUE(actual.first);
    }
}

TEST(SIMDConfig, successful_avx512vl_execution_on_x86arch) {
    faiss::SIMDConfig simd_config(nullptr);

    if (simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        auto actual = try_execute(run_avx512vl_computation);
        EXPECT_TRUE(actual.first);
        EXPECT_EQ(actual.second, std::vector<int>(8, 7));
    }
}

TEST(SIMDConfig, successful_avx512dq_execution_on_x86arch) {
    faiss::SIMDConfig simd_config(nullptr);

    if (simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        EXPECT_TRUE(
                simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512));
        auto actual = try_execute(run_avx512dq_computation);
        EXPECT_TRUE(actual.first);
        EXPECT_EQ(actual.second, std::vector<int>(8, 7));
    }
}

TEST(SIMDConfig, successful_avx512bw_execution_on_x86arch) {
    faiss::SIMDConfig simd_config(nullptr);

    if (simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512)) {
        EXPECT_TRUE(
                simd_config.is_simd_level_available(faiss::SIMDLevel::AVX512));
        auto actual = try_execute(run_avx512bw_computation);
        EXPECT_TRUE(actual.first);
        EXPECT_EQ(actual.second, std::vector<int>(64, 7));
    }
}
#endif // __x86_64__

TEST(SIMDConfig, override_simd_level) {
    const char* faiss_env_var_neon = "ARM_NEON";
    faiss::SIMDConfig simd_neon_config(&faiss_env_var_neon);
    EXPECT_EQ(simd_neon_config.level, faiss::SIMDLevel::ARM_NEON);

    EXPECT_EQ(simd_neon_config.supported_simd_levels().size(), 2);
    EXPECT_TRUE(simd_neon_config.is_simd_level_available(
            faiss::SIMDLevel::ARM_NEON));

    const char* faiss_env_var_avx512 = "AVX512";
    faiss::SIMDConfig simd_avx512_config(&faiss_env_var_avx512);
    EXPECT_EQ(simd_avx512_config.level, faiss::SIMDLevel::AVX512);
    EXPECT_EQ(simd_avx512_config.supported_simd_levels().size(), 2);
    EXPECT_TRUE(simd_avx512_config.is_simd_level_available(
            faiss::SIMDLevel::AVX512));
}

TEST(SIMDConfig, simd_config_get_level_name) {
    const char* faiss_env_var_neon = "ARM_NEON";
    faiss::SIMDConfig simd_neon_config(&faiss_env_var_neon);
    EXPECT_EQ(simd_neon_config.level, faiss::SIMDLevel::ARM_NEON);
    EXPECT_TRUE(simd_neon_config.is_simd_level_available(
            faiss::SIMDLevel::ARM_NEON));
    EXPECT_EQ(faiss_env_var_neon, simd_neon_config.get_level_name());

    const char* faiss_env_var_avx512 = "AVX512";
    faiss::SIMDConfig simd_avx512_config(&faiss_env_var_avx512);
    EXPECT_EQ(simd_avx512_config.level, faiss::SIMDLevel::AVX512);
    EXPECT_TRUE(simd_avx512_config.is_simd_level_available(
            faiss::SIMDLevel::AVX512));
    EXPECT_EQ(faiss_env_var_avx512, simd_avx512_config.get_level_name());
}

TEST(SIMDLevel, get_level_name_from_enum) {
    EXPECT_EQ("NONE", to_string(faiss::SIMDLevel::NONE).value_or(""));
    EXPECT_EQ("AVX2", to_string(faiss::SIMDLevel::AVX2).value_or(""));
    EXPECT_EQ("AVX512", to_string(faiss::SIMDLevel::AVX512).value_or(""));
    EXPECT_EQ("ARM_NEON", to_string(faiss::SIMDLevel::ARM_NEON).value_or(""));

    int actual_num_simd_levels = static_cast<int>(faiss::SIMDLevel::COUNT);
    EXPECT_EQ(5, actual_num_simd_levels);
    // Check that all SIMD levels have a name (except for COUNT which is not a
    // real SIMD level)
    for (int i = 0; i < actual_num_simd_levels - 1; ++i) {
        faiss::SIMDLevel simd_level = static_cast<faiss::SIMDLevel>(i);
        EXPECT_TRUE(faiss::to_string(simd_level).has_value());
    }
}

TEST(SIMDLevel, to_simd_level_from_string) {
    EXPECT_EQ(faiss::SIMDLevel::NONE, faiss::to_simd_level("NONE"));
    EXPECT_EQ(faiss::SIMDLevel::AVX2, faiss::to_simd_level("AVX2"));
    EXPECT_EQ(faiss::SIMDLevel::AVX512, faiss::to_simd_level("AVX512"));
    EXPECT_EQ(faiss::SIMDLevel::ARM_NEON, faiss::to_simd_level("ARM_NEON"));
    EXPECT_FALSE(faiss::to_simd_level("INVALID").has_value());
}
