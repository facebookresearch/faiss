/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_levels.h>

#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER)
// __cpuidex, _xgetbv
#include <intrin.h>
#endif

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>

namespace faiss {

// Static member definitions - used in both DD and static modes
SIMDLevel SIMDConfig::level = SIMDLevel::NONE;

// Bitmask of supported SIMD levels (1 << SIMDLevel)
uint64_t SIMDConfig::supported_simd_levels = 0;

// Microarchitecture flags (x86). Default false; set by
// detect_x86_uarch_flags() at load time.
bool SIMDConfig::avx512_split = false;

// ARM SVE runtime detection
#if defined(__aarch64__) || defined(_M_ARM64)

#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif

static bool has_sve() {
    return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
}

#elif defined(__APPLE__)
// Apple Silicon does NOT support SVE
static bool has_sve() {
    return false;
}

#else
// Other aarch64 platforms: conservatively report no SVE
static bool has_sve() {
    return false;
}

#endif // __linux__ / __APPLE__ / other

#else // Not ARM64
[[maybe_unused]] static bool has_sve() {
    return false;
}
#endif

namespace {

#if defined(__x86_64__) || defined(_M_X64)

// MSVC and clang-cl do not support GNU-style
// 64-bit inline assembly, MSVC defines _M_X64 instead of __x86_64__

#if defined(_MSC_VER)

[[maybe_unused]] void cpuid_count(
        unsigned int leaf,
        unsigned int subleaf,
        unsigned int regs[4]) {
    int r[4];
    __cpuidex(r, static_cast<int>(leaf), static_cast<int>(subleaf));
    for (int i = 0; i < 4; i++) {
        regs[i] = static_cast<unsigned int>(r[i]);
    }
}

[[maybe_unused]] uint64_t xgetbv0() {
    return static_cast<uint64_t>(_xgetbv(0));
}

#else // GCC / Clang

[[maybe_unused]] void cpuid_count(
        unsigned int leaf,
        unsigned int subleaf,
        unsigned int regs[4]) {
    asm volatile("cpuid"
                 : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                 : "a"(leaf), "c"(subleaf));
}

[[maybe_unused]] uint64_t xgetbv0() {
    unsigned int eax, edx;
    asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return eax | (static_cast<uint64_t>(edx) << 32);
}

#endif // _MSC_VER

// Detect x86 microarchitecture flags used for kernel routing. Uses raw
// cpuid so it is safe to run on any CPU regardless of compiled SIMD level.
void detect_x86_uarch_flags() {
    unsigned int regs[4];

    // Vendor string (CPUID.0): "AuthenticAMD" is EBX="Auth", EDX="enti",
    // ECX="cAMD".
    cpuid_count(0, 0, regs);
    const bool is_amd = regs[1] == 0x68747541u && regs[3] == 0x69746e65u &&
            regs[2] == 0x444d4163u;

    // Family/model (CPUID.1 EAX).
    cpuid_count(1, 0, regs);
    const unsigned int eax1 = regs[0];
    const unsigned int base_family = (eax1 >> 8) & 0xfu;
    const unsigned int display_family =
            base_family + (base_family == 0xfu ? ((eax1 >> 20) & 0xffu) : 0u);
    // AMD Zen 4 / Zen 4c (Bergamo) is family 0x19 and splits AVX-512.
    // (Zen 5, family 0x1A, has a native 512-bit datapath and is excluded.)
    SIMDConfig::avx512_split = is_amd && display_family == 0x19u;
}

#else // Not x86-64

void detect_x86_uarch_flags() {}

#endif // defined(__x86_64__) || defined(_M_X64)

} // namespace

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

    detect_x86_uarch_flags();

#if (defined(__x86_64__) || defined(_M_X64)) && \
        (defined(COMPILE_SIMD_AVX2) || defined(COMPILE_SIMD_AVX512))
    unsigned int regs[4];

    cpuid_count(1, 0, regs);
    unsigned int ecx1 = regs[2];

    bool has_avx = (ecx1 & (1 << 28)) != 0;

    bool has_xsave_osxsave =
            (ecx1 & ((1 << 26) | (1 << 27))) == ((1 << 26) | (1 << 27));

    bool avx_supported = false;
    if (has_avx && has_xsave_osxsave) {
        avx_supported = (xgetbv0() & 6) == 6;
    }

    if (avx_supported) {
        cpuid_count(7, 0, regs);
        unsigned int ebx7 = regs[1];
        // EDX of CPUID leaf 7 subleaf 0 carries AVX512_FP16 (bit 23),
        // needed for the SPR detection below. Kept in a local so a later
        // xgetbv cannot clobber it.
        unsigned int cpuid7_edx = regs[3];

        uint64_t xcr0 = xgetbv0();

#if defined(COMPILE_SIMD_AVX2) || defined(COMPILE_SIMD_AVX512)
        bool has_avx2 = (ebx7 & (1 << 5)) != 0;
        if (has_avx2) {
            supported_simd_levels |= (1 << static_cast<int>(SIMDLevel::AVX2));
            detected_level = SIMDLevel::AVX2;
        }

#if defined(COMPILE_SIMD_AVX512)
        bool cpu_has_avx512f = (ebx7 & (1 << 16)) != 0;
        bool os_supports_avx512 = (xcr0 & 0xE0) == 0xE0;
        bool has_avx512f = cpu_has_avx512f && os_supports_avx512;
        if (has_avx512f) {
            bool has_avx512cd = (ebx7 & (1 << 28)) != 0;
            bool has_avx512vl = (ebx7 & (1 << 31)) != 0;
            bool has_avx512dq = (ebx7 & (1 << 17)) != 0;
            bool has_avx512bw = (ebx7 & (1 << 30)) != 0;
            if (has_avx512bw && has_avx512cd && has_avx512vl && has_avx512dq) {
                detected_level = SIMDLevel::AVX512;
                supported_simd_levels |=
                        (1 << static_cast<int>(SIMDLevel::AVX512));

#if defined(COMPILE_SIMD_AVX512_SPR)
                // Check for Sapphire Rapids features.
                // The SPR code path is compiled with -mavx512fp16, so we
                // must verify both AVX512_BF16 and AVX512_FP16 before
                // dispatching to it. AMD Zen 4 (bergamo) has BF16 but
                // not FP16 — using SPR code there causes SIGILL.
                // CPUID EAX=7, ECX=1: EAX bit 5 = AVX512_BF16
                // CPUID EAX=7, ECX=0: EDX bit 23 = AVX512_FP16
                // (Linux: X86_FEATURE_AVX512_FP16 = 18*32+23)
                bool has_avx512_fp16 = (cpuid7_edx & (1 << 23)) != 0;
                cpuid_count(7, 1, regs);
                const bool has_avx512_bf16 = (regs[0] & (1 << 5)) != 0;
                if (has_avx512_bf16 && has_avx512_fp16) {
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
#endif // defined(__x86_64__) || defined(_M_X64)

#ifdef COMPILE_SIMD_ARM_NEON
    // ARM NEON is standard on aarch64
    supported_simd_levels |= (1 << static_cast<int>(SIMDLevel::ARM_NEON));
    detected_level = SIMDLevel::ARM_NEON;
#endif

#ifdef COMPILE_SIMD_ARM_SVE
    if (has_sve()) {
        supported_simd_levels |= (1 << static_cast<int>(SIMDLevel::ARM_SVE));
        detected_level = SIMDLevel::ARM_SVE;
    }
#endif

#if defined(__riscv) && defined(COMPILE_SIMD_RISCV_RVV)
    // RVV is always available on RISC-V builds compiled with rv64gcv.
    supported_simd_levels |= (1 << static_cast<int>(SIMDLevel::RISCV_RVV));
    detected_level = SIMDLevel::RISCV_RVV;
#endif

    return detected_level;
}

namespace {

template <SIMDLevel Level>
SIMDLevel get_dispatched_level_impl() {
    return Level;
}

} // namespace

SIMDLevel SIMDConfig::get_dispatched_level() {
    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_ALL>(
            [&]<SIMDLevel SL>() { return get_dispatched_level_impl<SL>(); });
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
    detect_x86_uarch_flags();
    // In static mode, return the compiled-in level
#if defined(COMPILE_SIMD_AVX512_SPR)
    return SIMDLevel::AVX512_SPR;
#elif defined(COMPILE_SIMD_AVX512)
    return SIMDLevel::AVX512;
#elif defined(COMPILE_SIMD_AVX2)
    return SIMDLevel::AVX2;
#elif defined(COMPILE_SIMD_ARM_SVE)
    return SIMDLevel::ARM_SVE;
#elif defined(COMPILE_SIMD_ARM_NEON)
    return SIMDLevel::ARM_NEON;
#elif defined(COMPILE_SIMD_RISCV_RVV)
    return SIMDLevel::RISCV_RVV;
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
        case SIMDLevel::ARM_SVE:
            return "ARM_SVE";
        case SIMDLevel::RISCV_RVV:
            return "RISCV_RVV";
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
    if (level_str == "ARM_SVE") {
        return SIMDLevel::ARM_SVE;
    }
    if (level_str == "RISCV_RVV") {
        return SIMDLevel::RISCV_RVV;
    }

    throw FaissException("Invalid SIMD level string: " + level_str);
}

} // namespace faiss
