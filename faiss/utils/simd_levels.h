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

    COUNT
};

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
    /// In DD mode, uses DISPATCH_SIMDLevel internally.
    /// In static mode, returns the compiled-in level.
    /// Useful for verification: get_level() == get_dispatched_level()
    static SIMDLevel get_dispatched_level();
};

} // namespace faiss
