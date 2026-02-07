/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Core SIMDConfig API tests - works in both static and DD modes.
// Hardware execution tests (DD-only) are in separate files:
// - test_simd_levels_x86_avx2.cpp (compiled with AVX2 flags)
// - test_simd_levels_x86_avx512.cpp (compiled with AVX512 flags)

#include <gtest/gtest.h>

#include <faiss/impl/FaissException.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/utils.h>

// Helper to check if we're in DD mode
static bool is_dd_mode() {
    return faiss::get_compile_options().find("DD") != std::string::npos;
}

TEST(SIMDConfig, get_level_returns_valid_level) {
    // Works in both static and DD modes
    faiss::SIMDLevel level = faiss::SIMDConfig::get_level();
    EXPECT_NE(level, faiss::SIMDLevel::COUNT);
    EXPECT_GE(static_cast<int>(level), 0);
    EXPECT_LT(
            static_cast<int>(level), static_cast<int>(faiss::SIMDLevel::COUNT));
}

TEST(SIMDConfig, supported_simd_levels_not_empty) {
    // Works in both static and DD modes
    // Current level should always be in supported levels
    EXPECT_TRUE(
            faiss::SIMDConfig::is_simd_level_available(
                    faiss::SIMDConfig::get_level()));
}

TEST(SIMDConfig, set_level_to_supported_level_succeeds) {
    // Works in both static and DD modes
    faiss::SIMDLevel original_level = faiss::SIMDConfig::get_level();

    // Setting to any supported level should succeed
    for (int i = 0; i < static_cast<int>(faiss::SIMDLevel::COUNT); ++i) {
        auto level = static_cast<faiss::SIMDLevel>(i);
        if (faiss::SIMDConfig::is_simd_level_available(level)) {
            EXPECT_NO_THROW(faiss::SIMDConfig::set_level(level))
                    << "set_level(" << faiss::to_string(level)
                    << ") should succeed";
            EXPECT_EQ(faiss::SIMDConfig::get_level(), level);
        }
    }

    // Restore original level
    faiss::SIMDConfig::set_level(original_level);
}

TEST(SIMDConfig, set_level_to_unsupported_level_throws) {
    // Works in both static and DD modes
    // Find a level that's NOT supported
    faiss::SIMDLevel unsupported = faiss::SIMDLevel::COUNT;
    for (int i = 0; i < static_cast<int>(faiss::SIMDLevel::COUNT); ++i) {
        auto level = static_cast<faiss::SIMDLevel>(i);
        if (!faiss::SIMDConfig::is_simd_level_available(level)) {
            unsupported = level;
            break;
        }
    }

    if (unsupported != faiss::SIMDLevel::COUNT) {
        EXPECT_THROW(
                faiss::SIMDConfig::set_level(unsupported),
                faiss::FaissException)
                << "set_level(" << faiss::to_string(unsupported)
                << ") should throw";
    }
}

TEST(SIMDConfig, static_mode_has_single_level) {
    // Static mode should have exactly 1 level: the compiled-in level
    if (is_dd_mode()) {
        GTEST_SKIP() << "DD build - has multiple levels";
    }

    int count = 0;
    for (int i = 0; i < static_cast<int>(faiss::SIMDLevel::COUNT); ++i) {
        if (faiss::SIMDConfig::is_simd_level_available(
                    static_cast<faiss::SIMDLevel>(i))) {
            ++count;
        }
    }
    EXPECT_EQ(count, 1)
            << "Static mode should have exactly 1 level (compiled-in)";
}

TEST(SIMDConfig, get_level_name_matches_level) {
    // Works in both static and DD modes
    faiss::SIMDLevel level = faiss::SIMDConfig::get_level();
    std::string name = faiss::SIMDConfig::get_level_name();
    std::string expected = faiss::to_string(level);
    EXPECT_EQ(name, expected);
}

TEST(SIMDConfig, get_dispatched_level_matches_get_level) {
    // Works in both static and DD modes
    // Verifies that dispatch mechanism returns the same level as get_level()
    faiss::SIMDLevel level = faiss::SIMDConfig::get_level();
    faiss::SIMDLevel dispatched = faiss::SIMDConfig::get_dispatched_level();
    EXPECT_EQ(level, dispatched)
            << "get_level() returned " << faiss::to_string(level)
            << " but get_dispatched_level() returned "
            << faiss::to_string(dispatched);
}

TEST(SIMDLevel, to_string_all_levels) {
    EXPECT_EQ("NONE", faiss::to_string(faiss::SIMDLevel::NONE));
    EXPECT_EQ("AVX2", faiss::to_string(faiss::SIMDLevel::AVX2));
    EXPECT_EQ("AVX512", faiss::to_string(faiss::SIMDLevel::AVX512));
    EXPECT_EQ("AVX512_SPR", faiss::to_string(faiss::SIMDLevel::AVX512_SPR));
    EXPECT_EQ("ARM_NEON", faiss::to_string(faiss::SIMDLevel::ARM_NEON));
    EXPECT_EQ("ARM_SVE", faiss::to_string(faiss::SIMDLevel::ARM_SVE));

    // COUNT should throw
    EXPECT_THROW(
            faiss::to_string(faiss::SIMDLevel::COUNT), faiss::FaissException);
}

TEST(SIMDLevel, to_simd_level_all_strings) {
    EXPECT_EQ(faiss::SIMDLevel::NONE, faiss::to_simd_level("NONE"));
    EXPECT_EQ(faiss::SIMDLevel::AVX2, faiss::to_simd_level("AVX2"));
    EXPECT_EQ(faiss::SIMDLevel::AVX512, faiss::to_simd_level("AVX512"));
    EXPECT_EQ(faiss::SIMDLevel::AVX512_SPR, faiss::to_simd_level("AVX512_SPR"));
    EXPECT_EQ(faiss::SIMDLevel::ARM_NEON, faiss::to_simd_level("ARM_NEON"));
    EXPECT_EQ(faiss::SIMDLevel::ARM_SVE, faiss::to_simd_level("ARM_SVE"));

    // Invalid strings should throw
    EXPECT_THROW(faiss::to_simd_level("INVALID"), faiss::FaissException);
    EXPECT_THROW(faiss::to_simd_level(""), faiss::FaissException);
}

TEST(SIMDConfig, modern_hardware_has_simd_support) {
    // In DD mode, verify modern hardware detects SIMD support
    if (!is_dd_mode()) {
        GTEST_SKIP() << "Static build - level is fixed at compile time";
    }

    faiss::SIMDLevel detected = faiss::SIMDConfig::auto_detect_simd_level();

#if defined(__x86_64__) || defined(_M_X64)
    // All modern x86_64 machines (Haswell 2013+) support at least AVX2
    EXPECT_NE(detected, faiss::SIMDLevel::NONE)
            << "x86_64 machines should support at least AVX2";
#elif defined(__aarch64__) || defined(_M_ARM64)
    // NEON is mandatory on aarch64
    EXPECT_NE(detected, faiss::SIMDLevel::NONE)
            << "ARM64 machines should support at least NEON";
#endif
}

TEST(CompileOptions, lists_expected_levels) {
    std::string options = faiss::get_compile_options();

    // All supported levels (except NONE) should be in compile options
    for (int i = 0; i < static_cast<int>(faiss::SIMDLevel::COUNT); ++i) {
        auto level = static_cast<faiss::SIMDLevel>(i);
        if (!faiss::SIMDConfig::is_simd_level_available(level)) {
            continue;
        }
        if (level == faiss::SIMDLevel::NONE) {
            continue; // NONE is not reported in options
        }
        std::string name = faiss::to_string(level);
        EXPECT_NE(options.find(name), std::string::npos)
                << "Supported level " << name
                << " should be in compile options: " << options;
    }

    // DD mode should have "DD" marker
    if (is_dd_mode()) {
        EXPECT_NE(options.find("DD"), std::string::npos)
                << "DD mode should have 'DD' in compile options: " << options;
    }
}
