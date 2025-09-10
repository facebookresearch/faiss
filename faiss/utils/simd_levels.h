/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <string>
#include <unordered_set>

namespace faiss {

#define COMPILE_SIMD_NONE

enum class SIMDLevel {
    NONE,
    // x86
    AVX2,
    AVX512,
    // arm & aarch64
    ARM_NEON,

    COUNT
};

std::optional<std::string> to_string(SIMDLevel level);

std::optional<SIMDLevel> to_simd_level(const std::string& level_str);

/* Current SIMD configuration. This static class manages the current SIMD level
 * and intializes it from the cpuid and the FAISS_SIMD_LEVEL
 * environment variable  */
struct SIMDConfig {
    static SIMDLevel level;
    static std::unordered_set<SIMDLevel>& supported_simd_levels();

    typedef SIMDLevel (*DetectSIMDLevelFunc)();
    static SIMDLevel auto_detect_simd_level();

    SIMDConfig(const char** faiss_simd_level_env = nullptr);

    static void set_level(SIMDLevel level);
    static SIMDLevel get_level();
    static std::string get_level_name();

    static bool is_simd_level_available(SIMDLevel level);
};

/*********************** x86 SIMD */

#ifdef COMPILE_SIMD_AVX2
#define DISPATCH_SIMDLevel_AVX2(f, ...) \
    case SIMDLevel::AVX2:               \
        return f<SIMDLevel::AVX2>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_AVX2(f, ...)
#endif

#ifdef COMPILE_SIMD_AVX512
#define DISPATCH_SIMDLevel_AVX512(f, ...) \
    case SIMDLevel::AVX512:               \
        return f<SIMDLevel::AVX512>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_AVX512(f, ...)
#endif

/* dispatch function f to f<SIMDLevel> */

#define DISPATCH_SIMDLevel(f, ...)                     \
    switch (SIMDConfig::level) {                       \
        case SIMDLevel::NONE:                          \
            return f<SIMDLevel::NONE>(__VA_ARGS__);    \
            DISPATCH_SIMDLevel_AVX2(f, __VA_ARGS__);   \
            DISPATCH_SIMDLevel_AVX512(f, __VA_ARGS__); \
        default:                                       \
            FAISS_ASSERT(!"Invalid SIMD level");       \
    }

} // namespace faiss
