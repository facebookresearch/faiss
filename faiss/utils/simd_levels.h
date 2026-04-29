/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#include <faiss/impl/platform_macros.h>

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
    ARM_SVE, // Scalable Vector Extension (ARMv8.2+)
    // riscv
    RISCV_RVV, // RISC-V Vector Extension (rv64gcv)

    COUNT
};

/***************************************************************
 * SINGLE_SIMD_LEVEL: the SIMD level for code without explicit SL context.
 *
 * In static mode: resolves to the compiled-in level (zero overhead).
 * In DD mode: resolves to NONE (emulated scalar). Code using
 * SINGLE_SIMD_LEVEL is meant to be incrementally migrated to use
 * proper SL dispatch — SINGLE_SIMD_LEVEL is migration scaffolding,
 * not permanent API.
 ***************************************************************/
#ifdef FAISS_ENABLE_DD
// DD dispatches to the highest optional SIMD level at runtime.
// On ARM64, NEON is mandatory (always available via COMPILE_SIMD_ARM_NEON),
// so the baseline is ARM_NEON. On x86, the baseline is NONE.
#if defined(COMPILE_SIMD_ARM_NEON)
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::ARM_NEON;
#else
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::NONE;
#endif
#else
#if defined(COMPILE_SIMD_AVX512_SPR)
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::AVX512_SPR;
#elif defined(COMPILE_SIMD_AVX512)
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::AVX512;
#elif defined(COMPILE_SIMD_AVX2)
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::AVX2;
#elif defined(COMPILE_SIMD_ARM_SVE)
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::ARM_SVE;
#elif defined(COMPILE_SIMD_ARM_NEON)
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::ARM_NEON;
#elif defined(COMPILE_SIMD_RISCV_RVV)
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::RISCV_RVV;
#else
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL = SIMDLevel::NONE;
#endif
#endif

/***************************************************************
 * Helper to select the appropriate 256-bit SIMD level.
 *
 * For 256-bit SIMD types (simd16uint16, simd32uint8, etc.), maps:
 *   AVX512/AVX512_SPR → AVX2 (256-bit ops use AVX2 instructions)
 *   AVX2 → AVX2
 *   ARM_NEON/ARM_SVE → ARM_NEON
 *   NONE → NONE
 ***************************************************************/
template <SIMDLevel SL>
struct simd256_level_selector {
    static constexpr SIMDLevel value =
            (SL == SIMDLevel::AVX512 || SL == SIMDLevel::AVX512_SPR)
            ? SIMDLevel::AVX2
            : (SL == SIMDLevel::ARM_SVE ? SIMDLevel::ARM_NEON : SL);
};

/// SINGLE_SIMD_LEVEL mapped to 256-bit: use this for 256-bit simd types
/// (simd16uint16, simd32uint8, etc.) which don't have AVX512/SVE
/// specializations.
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL_256 =
        simd256_level_selector<SINGLE_SIMD_LEVEL>::value;

/***************************************************************
 * Helper to select the appropriate 512-bit SIMD level.
 *
 * For 512-bit SIMD types (simd32uint16, simd64uint8, etc.), maps:
 *   AVX512_SPR → AVX512 (512-bit ops share the same instructions)
 *   AVX512 → AVX512
 *   NONE → NONE
 ***************************************************************/
template <SIMDLevel SL>
struct simd512_level_selector {
    static constexpr SIMDLevel value =
            (SL == SIMDLevel::AVX512_SPR) ? SIMDLevel::AVX512 : SL;
};

/// SINGLE_SIMD_LEVEL mapped to 512-bit: use this for 512-bit simd types
/// (simd32uint16, simd64uint8, etc.) which don't have AVX512_SPR
/// specializations (AVX512_SPR uses the same 512-bit integer ops as AVX512).
inline constexpr SIMDLevel SINGLE_SIMD_LEVEL_512 =
        simd512_level_selector<SINGLE_SIMD_LEVEL>::value;

/// Number of float32 lanes for a given SIMD level.
/// ARM_SVE is variable-width (128–2048 bits); no single constant is correct.
template <SIMDLevel SL>
constexpr int simd_width() {
    static_assert(
            SL != SIMDLevel::ARM_SVE,
            "simd_width<ARM_SVE> is not supported: SVE is variable-width");
    static_assert(
            SL != SIMDLevel::RISCV_RVV,
            "simd_width<RISCV_RVV> is not supported: RVV is variable-width");
    if constexpr (SL == SIMDLevel::AVX512 || SL == SIMDLevel::AVX512_SPR)
        return 16;
    else if constexpr (SL == SIMDLevel::AVX2 || SL == SIMDLevel::ARM_NEON)
        return 8;
    else
        return 1;
}

/// Convert SIMDLevel to string. Throws FaissException for invalid level.
std::string to_string(SIMDLevel level);

/// Parse string to SIMDLevel. Throws FaissException for invalid strings.
SIMDLevel to_simd_level(const std::string& level_str);

/**
 * Current SIMD configuration.
 *
 * This class provides a uniform API for querying and setting the SIMD level,
 * regardless of whether faiss was built with Dynamic Dispatch (DD) or static
 * SIMD selection.
 *
 * In DD mode:
 *   - get_level() returns the runtime-detected or user-set level
 *   - set_level() changes the runtime level (if level is supported)
 *   - supported_simd_levels() returns bitmask of all compiled-in levels
 *
 * In static mode:
 *   - get_level() returns the compiled-in level
 *   - set_level() succeeds only if level matches compiled-in level
 *   - supported_simd_levels() returns bitmask with single level
 */
struct FAISS_API SIMDConfig {
    static SIMDLevel level;

    /// Returns bitmask of supported SIMD levels (1 << SIMDLevel).
    static uint64_t supported_simd_levels;

    static SIMDLevel auto_detect_simd_level();

    SIMDConfig(const char** faiss_simd_level_env = nullptr);

    /// Set the SIMD level. Throws FaissException if level is not supported.
    static void set_level(SIMDLevel level);
    static SIMDLevel get_level();
    static std::string get_level_name();

    /// Check if a SIMD level is available (compiled in).
    static bool is_simd_level_available(SIMDLevel level);

    /// Returns the SIMD level via the dispatch mechanism.
    /// In DD mode, uses with_simd_level internally.
    /// In static mode, returns the compiled-in level.
    /// Useful for verification: get_level() == get_dispatched_level()
    static SIMDLevel get_dispatched_level();
};

} // namespace faiss
