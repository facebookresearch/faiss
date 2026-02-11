/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file simd_dispatch.h
 * @brief Internal dispatch macros for SIMD level selection.
 *
 * This is a PRIVATE header - do not include in public APIs or user code.
 * Only faiss internal .cpp files should include this header.
 *
 * For the public API (SIMDLevel enum, SIMDConfig class), use:
 *   #include <faiss/utils/simd_levels.h>
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

/*********************** x86 SIMD dispatch cases */

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

/*********************** ARM SIMD dispatch cases */

#ifdef COMPILE_SIMD_ARM_NEON
#define DISPATCH_SIMDLevel_ARM_NEON(f, ...) \
    case SIMDLevel::ARM_NEON:               \
        return f<SIMDLevel::ARM_NEON>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_ARM_NEON(f, ...)
#endif

#ifdef COMPILE_SIMD_ARM_SVE
#define DISPATCH_SIMDLevel_ARM_SVE(f, ...) \
    case SIMDLevel::ARM_SVE:               \
        return f<SIMDLevel::ARM_SVE>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_ARM_SVE(f, ...)
#endif

/*********************** Main dispatch macro */

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
            DISPATCH_SIMDLevel_ARM_SVE(f, __VA_ARGS__);    \
        default:                                           \
            FAISS_THROW_MSG("Invalid SIMD level");         \
    }

#else // Static mode

// Static mode: direct call to compiled-in SIMD level (no runtime switch)
#if defined(COMPILE_SIMD_AVX512_SPR)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX512_SPR>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_AVX512)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX512>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_AVX2)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX2>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_ARM_SVE)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::ARM_SVE>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_ARM_NEON)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::ARM_NEON>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::NONE>(__VA_ARGS__)
#endif

#endif // FAISS_ENABLE_DD

} // namespace faiss
