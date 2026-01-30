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
    AVX512_SPR, // Sapphire Rapids: AVX512 + BF16 + FP16 + VNNI
    // arm & aarch64
    ARM_NEON,

    COUNT
};

std::optional<std::string> to_string(SIMDLevel level);

std::optional<SIMDLevel> to_simd_level(const std::string& level_str);

#ifdef FAISS_ENABLE_DD

/* Current SIMD configuration. This class manages the current SIMD level
 * and initializes it from the cpuid and the FAISS_SIMD_LEVEL
 * environment variable.
 *
 * NOTE: SIMDConfig is only available in Dynamic Dispatch (DD) builds.
 * In static builds, SIMD level is determined at compile time via
 * COMPILE_SIMD_* preprocessor flags. */
struct SIMDConfig {
    static SIMDLevel level;
    static std::unordered_set<SIMDLevel>& supported_simd_levels();

    using DetectSIMDLevelFunc = SIMDLevel (*)();
    static SIMDLevel auto_detect_simd_level();

    SIMDConfig(const char** faiss_simd_level_env = nullptr);

    static void set_level(SIMDLevel level);
    static SIMDLevel get_level();
    static std::string get_level_name();

    static bool is_simd_level_available(SIMDLevel level);
};

#endif // FAISS_ENABLE_DD

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

#ifdef COMPILE_SIMD_AVX512_SPR
#define DISPATCH_SIMDLevel_AVX512_SPR(f, ...) \
    case SIMDLevel::AVX512_SPR:               \
        return f<SIMDLevel::AVX512_SPR>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_AVX512_SPR(f, ...)
#endif

/*********************** ARM SIMD */

#ifdef COMPILE_SIMD_ARM_NEON
#define DISPATCH_SIMDLevel_ARM_NEON(f, ...) \
    case SIMDLevel::ARM_NEON:               \
        return f<SIMDLevel::ARM_NEON>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_ARM_NEON(f, ...)
#endif

/* dispatch function f to f<SIMDLevel> */

#ifdef FAISS_ENABLE_DD

// DD mode: runtime dispatch based on SIMDConfig::level
#define DISPATCH_SIMDLevel(f, ...)                         \
    switch (SIMDConfig::level) {                           \
        case SIMDLevel::NONE:                              \
            return f<SIMDLevel::NONE>(__VA_ARGS__);        \
            DISPATCH_SIMDLevel_AVX2(f, __VA_ARGS__);       \
            DISPATCH_SIMDLevel_AVX512(f, __VA_ARGS__);     \
            DISPATCH_SIMDLevel_AVX512_SPR(f, __VA_ARGS__); \
            DISPATCH_SIMDLevel_ARM_NEON(f, __VA_ARGS__);   \
        default:                                           \
            FAISS_ASSERT(!"Invalid SIMD level");           \
    }

#else // Static mode

// Static mode: direct call to compiled-in SIMD level (no runtime switch)
#if defined(COMPILE_SIMD_AVX512_SPR)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX512_SPR>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_AVX512)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX512>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_AVX2)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX2>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_ARM_NEON)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::ARM_NEON>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::NONE>(__VA_ARGS__)
#endif

#endif // FAISS_ENABLE_DD

} // namespace faiss
