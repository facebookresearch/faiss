/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_levels.h>

#include <cstdlib>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

// Static member definitions - used in both DD and static modes
SIMDLevel SIMDConfig::level = SIMDLevel::NONE;

// Bitmask of supported SIMD levels (1 << SIMDLevel)
uint64_t SIMDConfig::supported_simd_levels = 0;

#ifdef FAISS_ENABLE_DD

// =============================================================================
// Dynamic Dispatch (DD) mode implementation
// =============================================================================

// Static initializer to run constructor at load time
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
static SIMDConfig simd_config_initializer;

SIMDConfig::SIMDConfig(const char** faiss_simd_level_env) {
    // Support dependency injection for testing
    const char* env_var = faiss_simd_level_env ? *faiss_simd_level_env
                                               : getenv("FAISS_SIMD_LEVEL");

    if (!env_var) {
        level = auto_detect_simd_level();
    } else {
        level = to_simd_level(env_var);
        supported_simd_levels = (1 << static_cast<int>(level));
    }
    supported_simd_levels |= (1 << static_cast<int>(SIMDLevel::NONE));
}

void SIMDConfig::set_level(SIMDLevel l) {
    if (!is_simd_level_available(l)) {
        FAISS_THROW_FMT(
                "SIMDConfig::set_level: level %s is not available",
                to_string(l).c_str());
    }
    level = l;
}

SIMDLevel SIMDConfig::get_level() {
    return level;
}

std::string SIMDConfig::get_level_name() {
    return to_string(level);
}

bool SIMDConfig::is_simd_level_available(SIMDLevel l) {
    return (supported_simd_levels & (1 << static_cast<int>(l))) != 0;
}

SIMDLevel SIMDConfig::auto_detect_simd_level() {
    SIMDLevel detected_level = SIMDLevel::NONE;

#if defined(__x86_64__) && \
        (defined(COMPILE_SIMD_AVX2) || defined(COMPILE_SIMD_AVX512))
    unsigned int eax, ebx, ecx, edx;

    eax = 1;
    ecx = 0;
    asm volatile("cpuid"
                 : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                 : "a"(eax), "c"(ecx));

    bool has_avx = (ecx & (1 << 28)) != 0;

    bool has_xsave_osxsave =
            (ecx & ((1 << 26) | (1 << 27))) == ((1 << 26) | (1 << 27));

    bool avx_supported = false;
    if (has_avx && has_xsave_osxsave) {
        unsigned int xcr0;
        asm volatile("xgetbv" : "=a"(xcr0), "=d"(edx) : "c"(0));
        avx_supported = (xcr0 & 6) == 6;
    }

    if (avx_supported) {
        eax = 7;
        ecx = 0;
        asm volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(eax), "c"(ecx));

        unsigned int xcr0;
        asm volatile("xgetbv" : "=a"(xcr0), "=d"(edx) : "c"(0));

#if defined(COMPILE_SIMD_AVX2) || defined(COMPILE_SIMD_AVX512)
        bool has_avx2 = (ebx & (1 << 5)) != 0;
        if (has_avx2) {
            supported_simd_levels |= (1 << static_cast<int>(SIMDLevel::AVX2));
            detected_level = SIMDLevel::AVX2;
        }

#if defined(COMPILE_SIMD_AVX512)
        bool cpu_has_avx512f = (ebx & (1 << 16)) != 0;
        bool os_supports_avx512 = (xcr0 & 0xE0) == 0xE0;
        bool has_avx512f = cpu_has_avx512f && os_supports_avx512;
        if (has_avx512f) {
            bool has_avx512cd = (ebx & (1 << 28)) != 0;
            bool has_avx512vl = (ebx & (1 << 31)) != 0;
            bool has_avx512dq = (ebx & (1 << 17)) != 0;
            bool has_avx512bw = (ebx & (1 << 30)) != 0;
            if (has_avx512bw && has_avx512cd && has_avx512vl && has_avx512dq) {
                detected_level = SIMDLevel::AVX512;
                supported_simd_levels |=
                        (1 << static_cast<int>(SIMDLevel::AVX512));

#if defined(COMPILE_SIMD_AVX512_SPR)
                // Check for Sapphire Rapids features (AVX512_BF16)
                // CPUID EAX=7, ECX=1: EAX bit 5 = AVX512_BF16
                unsigned int eax1, ebx1, ecx1, edx1;
                eax1 = 7;
                ecx1 = 1;
                asm volatile("cpuid"
                             : "=a"(eax1), "=b"(ebx1), "=c"(ecx1), "=d"(edx1)
                             : "a"(eax1), "c"(ecx1));
                bool has_avx512_bf16 = (eax1 & (1 << 5)) != 0;
                if (has_avx512_bf16) {
                    detected_level = SIMDLevel::AVX512_SPR;
                    supported_simd_levels |=
                            (1 << static_cast<int>(SIMDLevel::AVX512_SPR));
                }
#endif // defined(COMPILE_SIMD_AVX512_SPR)
            }
        }
#endif // defined(COMPILE_SIMD_AVX512)
#endif // defined(COMPILE_SIMD_AVX2) || defined(COMPILE_SIMD_AVX512)
    }
#endif // defined(__x86_64__) && ...

#if defined(__aarch64__) && defined(__ARM_NEON) && \
        defined(COMPILE_SIMD_ARM_NEON)
    // ARM NEON is standard on aarch64
    supported_simd_levels |= (1 << static_cast<int>(SIMDLevel::ARM_NEON));
    detected_level = SIMDLevel::ARM_NEON;
#endif

    return detected_level;
}

// Include private header for DISPATCH_SIMDLevel macro
#include <faiss/impl/simd_dispatch.h>

namespace {

template <SIMDLevel Level>
SIMDLevel get_dispatched_level_impl() {
    return Level;
}

} // namespace

SIMDLevel SIMDConfig::get_dispatched_level() {
    DISPATCH_SIMDLevel(get_dispatched_level_impl);
}

#else // Static mode

// =============================================================================
// Static mode implementation
// =============================================================================

// Static initializer to set up the single supported level
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
static SIMDConfig simd_config_initializer;

SIMDConfig::SIMDConfig(const char** /* faiss_simd_level_env */) {
    // In static mode, the level is fixed at compile time
    level = auto_detect_simd_level();
    supported_simd_levels = (1 << static_cast<int>(level));
}

void SIMDConfig::set_level(SIMDLevel l) {
    if (!is_simd_level_available(l)) {
        FAISS_THROW_FMT(
                "SIMDConfig::set_level: level %s is not available "
                "(static build only supports %s)",
                to_string(l).c_str(),
                to_string(level).c_str());
    }
    // In static mode, setting to the same level is a no-op
    level = l;
}

SIMDLevel SIMDConfig::get_level() {
    return level;
}

std::string SIMDConfig::get_level_name() {
    return to_string(level);
}

bool SIMDConfig::is_simd_level_available(SIMDLevel l) {
    return (supported_simd_levels & (1 << static_cast<int>(l))) != 0;
}

SIMDLevel SIMDConfig::auto_detect_simd_level() {
    // In static mode, return the compiled-in level
#if defined(COMPILE_SIMD_AVX512_SPR)
    return SIMDLevel::AVX512_SPR;
#elif defined(COMPILE_SIMD_AVX512)
    return SIMDLevel::AVX512;
#elif defined(COMPILE_SIMD_AVX2)
    return SIMDLevel::AVX2;
#elif defined(COMPILE_SIMD_ARM_NEON)
    return SIMDLevel::ARM_NEON;
#else
    return SIMDLevel::NONE;
#endif
}

SIMDLevel SIMDConfig::get_dispatched_level() {
    // In static mode, just return the current level (no dispatch)
    return get_level();
}

#endif // FAISS_ENABLE_DD

// =============================================================================
// Common functions (both modes)
// =============================================================================

std::string to_string(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::NONE:
            return "NONE";
        case SIMDLevel::AVX2:
            return "AVX2";
        case SIMDLevel::AVX512:
            return "AVX512";
        case SIMDLevel::AVX512_SPR:
            return "AVX512_SPR";
        case SIMDLevel::ARM_NEON:
            return "ARM_NEON";
        case SIMDLevel::COUNT:
        default:
            throw FaissException("Invalid SIMDLevel");
    }
}

SIMDLevel to_simd_level(const std::string& level_str) {
    if (level_str == "NONE") {
        return SIMDLevel::NONE;
    }
    if (level_str == "AVX2") {
        return SIMDLevel::AVX2;
    }
    if (level_str == "AVX512") {
        return SIMDLevel::AVX512;
    }
    if (level_str == "AVX512_SPR") {
        return SIMDLevel::AVX512_SPR;
    }
    if (level_str == "ARM_NEON") {
        return SIMDLevel::ARM_NEON;
    }

    throw FaissException("Invalid SIMD level string: " + level_str);
}

} // namespace faiss
