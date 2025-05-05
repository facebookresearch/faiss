/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace faiss {

enum class SIMDLevel {
    NONE,
    AVX2,
    AVX512F,
    ARM_NEON,
    ARM_SVE,
    PPC_ALTIVEC,
};

struct SIMDConfig {
    static SIMDLevel level;
    static const char* level_names[];
    // initializes the simd_level from the cpuid and the FAISS_SIMD_LEVEL
    // environment variable
    SIMDConfig();
};

extern SIMDConfig simd_config;

/*********************** x86 SIMD */

#ifdef COMPILE_SIMD_AVX2
#define DISPATCH_SIMDLevel_AVX2(f, ...) \
    case SIMDLevel::AVX2:               \
        return f<SIMDLevel::AVX2>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_AVX2(f, ...)
#endif

#ifdef COMPILE_SIMD_AVX512F
#define DISPATCH_SIMDLevel_AVX512F(f, ...) \
    case SIMDLevel::AVX512F:               \
        return f<SIMDLevel::AVX512F>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_AVX512F(f, ...)
#endif

/*********************** ARM SIMD */

#ifdef COMPILE_SIMD_NEON
#define DISPATCH_SIMDLevel_ARM_NEON(f, ...) \
    case SIMDLevel::ARM_NEON:               \
        return f<SIMDLevel::ARM_NEON>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_ARM_NEON(f, ...)
#endif

#ifdef COMPILE_SIMD_SVE
#define DISPATCH_SIMDLevel_ARM_SVE(f, ...) \
    case SIMDLevel::ARM_SVE:               \
        return f<SIMDLevel::ARM_SVE>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_ARM_SVE(f, ...)
#endif

/* dispatch function f to f<SIMDLevel> */

#define DISPATCH_SIMDLevel(f, ...)                       \
    switch (simd_config::level) {                        \
        case SIMDLevel::NONE:                            \
            return f<SIMDLevel::NONE>(__VA_ARGS__);      \
            DISPATCH_SIMDLevel_AVX2(f, __VA_ARGS__);     \
            DISPATCH_SIMDLevel_AVX512F(f, __VA_ARGS__);  \
            DISPATCH_SIMDLevel_ARM_NEON(f, __VA_ARGS__); \
            DISPATCH_SIMDLevel_ARM_SVE(f, __VA_ARGS__);  \
        default:                                         \
            assert(!"invlalid SIMD level");              \
    }

} // namespace faiss
