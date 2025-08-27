/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_levels.h>

#include <faiss/impl/FaissAssert.h>
#include <cstdlib>

namespace faiss {

SIMDLevel SIMDConfig::level = SIMDLevel::NONE;
std::unordered_set<SIMDLevel>& SIMDConfig::supported_simd_levels() {
    static std::unordered_set<SIMDLevel> levels;
    return levels;
}

// it is there to make sure the constructor runs
static SIMDConfig dummy_config;

SIMDConfig::SIMDConfig(const char** faiss_simd_level_env) {
    // added to support dependency injection
    const char* env_var = faiss_simd_level_env ? *faiss_simd_level_env
                                               : getenv("FAISS_SIMD_LEVEL");

    // check environment variable for SIMD level is explicitly set
    if (!env_var) {
        level = auto_detect_simd_level();
    } else {
        auto matched_level = to_simd_level(env_var);
        if (matched_level.has_value()) {
            set_level(matched_level.value());
            supported_simd_levels().clear();
            supported_simd_levels().insert(matched_level.value());
        } else {
            fprintf(stderr,
                    "FAISS_SIMD_LEVEL is set to %s, which is unknown\n",
                    env_var);
            exit(1);
        }
    }
    supported_simd_levels().insert(SIMDLevel::NONE);
}

void SIMDConfig::set_level(SIMDLevel l) {
    level = l;
}

SIMDLevel SIMDConfig::get_level() {
    return level;
}

std::string SIMDConfig::get_level_name() {
    return to_string(level).value_or("");
}

bool SIMDConfig::is_simd_level_available(SIMDLevel l) {
    return supported_simd_levels().find(l) != supported_simd_levels().end();
}

SIMDLevel SIMDConfig::auto_detect_simd_level() {
    SIMDLevel level = SIMDLevel::NONE;

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
            SIMDConfig::supported_simd_levels().insert(SIMDLevel::AVX2);
            level = SIMDLevel::AVX2;
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
                level = SIMDLevel::AVX512;
                supported_simd_levels().insert(SIMDLevel::AVX512);
            }
        }
#endif // defined(COMPILE_SIMD_AVX512)
#endif // defined(COMPILE_SIMD_AVX2)|| defined(COMPILE_SIMD_AVX512)
    }
#endif // defined(__x86_64__) && (defined(COMPILE_SIMD_AVX2) ||
       // defined(COMPILE_SIMD_AVX512))

#if defined(__aarch64__) && defined(__ARM_NEON) && \
        defined(COMPILE_SIMD_ARM_NEON)
    // ARM NEON is standard on aarch64
    supported_simd_levels().insert(SIMDLevel::ARM_NEON);
    level = SIMDLevel::ARM_NEON;
    // TODO: Add ARM SVE detection when needed
    // For now, we default to ARM_NEON as it's universally supported on aarch64
#endif

    return level;
}

std::optional<std::string> to_string(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::NONE:
            return "NONE";
        case SIMDLevel::AVX2:
            return "AVX2";
        case SIMDLevel::AVX512:
            return "AVX512";
        case SIMDLevel::ARM_NEON:
            return "ARM_NEON";
        default:
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<SIMDLevel> to_simd_level(const std::string& level_str) {
    if (level_str == "NONE") {
        return SIMDLevel::NONE;
    }
    if (level_str == "AVX2") {
        return SIMDLevel::AVX2;
    }
    if (level_str == "AVX512") {
        return SIMDLevel::AVX512;
    }
    if (level_str == "ARM_NEON") {
        return SIMDLevel::ARM_NEON;
    }

    return std::nullopt;
}

} // namespace faiss
