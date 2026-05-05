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

/** Defining which SIMD levels are available for a given function is via a
 * binary mask. Here we predefine the most common masks.
 *  */

constexpr int AVAILABLE_SIMD_LEVELS_NONE = (1 << int(SIMDLevel::NONE));

constexpr int AVAILABLE_SIMD_LEVELS_AVX2_NEON = AVAILABLE_SIMD_LEVELS_NONE |
        (1 << int(SIMDLevel::AVX2)) | (1 << int(SIMDLevel::ARM_NEON));

// A0: same + AVX512
constexpr int AVAILABLE_SIMD_LEVELS_A0 =
        AVAILABLE_SIMD_LEVELS_AVX2_NEON | (1 << int(SIMDLevel::AVX512));

// A1: same + ARM_SVE (for functions with dedicated SVE implementations)
constexpr int AVAILABLE_SIMD_LEVELS_A1 =
        AVAILABLE_SIMD_LEVELS_A0 | (1 << int(SIMDLevel::ARM_SVE));

// A2: NONE + AVX2 + ARM_SVE only (for functions with only these
// implementations)
constexpr int AVAILABLE_SIMD_LEVELS_A2 = AVAILABLE_SIMD_LEVELS_NONE |
        (1 << int(SIMDLevel::AVX2)) | (1 << int(SIMDLevel::ARM_SVE));

constexpr int AVAILABLE_SIMD_LEVELS_ALL = -1;

/** The complete dispatching function. It takes into account:
 * - the currently selected SIMD level
 * - the compiled in SIMD levels (given by COMPILE_SIMD_XXX)
 * - the available SIMD implementations for that particular function (given by
 * available_levels)
 */

template <int available_levels, typename LambdaType>
inline auto with_selected_simd_levels(LambdaType&& action) {
#ifdef FAISS_ENABLE_DD
    switch (SIMDConfig::level) {
        // For x86 -- try from highest to lowest level

#ifdef COMPILE_SIMD_AVX512_SPR
        case SIMDLevel::AVX512_SPR:
            if constexpr (
                    available_levels & (1 << int(SIMDLevel::AVX512_SPR))) {
                return action.template operator()<SIMDLevel::AVX512_SPR>();
            }
            [[fallthrough]];
#endif

#ifdef COMPILE_SIMD_AVX512
        case SIMDLevel::AVX512:
            if constexpr (available_levels & (1 << int(SIMDLevel::AVX512))) {
                return action.template operator()<SIMDLevel::AVX512>();
            }
            [[fallthrough]];
#endif

#ifdef COMPILE_SIMD_AVX2
        case SIMDLevel::AVX2:
            if constexpr (available_levels & (1 << int(SIMDLevel::AVX2))) {
                return action.template operator()<SIMDLevel::AVX2>();
            }
            [[fallthrough]];
#endif

            // For ARM, try from highest to lowest level
#ifdef COMPILE_SIMD_ARM_SVE
        case SIMDLevel::ARM_SVE:
            if constexpr (available_levels & (1 << int(SIMDLevel::ARM_SVE))) {
                return action.template operator()<SIMDLevel::ARM_SVE>();
            }
            [[fallthrough]];
#endif

#ifdef COMPILE_SIMD_ARM_NEON
        case SIMDLevel::ARM_NEON:
            if constexpr (available_levels & (1 << int(SIMDLevel::ARM_NEON))) {
                return action.template operator()<SIMDLevel::ARM_NEON>();
            }
            [[fallthrough]];
#endif

#ifdef COMPILE_SIMD_RISCV_RVV
        case SIMDLevel::RISCV_RVV:
            if constexpr (available_levels & (1 << int(SIMDLevel::RISCV_RVV))) {
                return action.template operator()<SIMDLevel::RISCV_RVV>();
            }
            [[fallthrough]];
#endif
        default:
            return action.template operator()<SIMDLevel::NONE>();
    }
#else // static dispatch
    // In static mode, SINGLE_SIMD_LEVEL is a constexpr resolved at compile
    // time. If the compiled level is not in the available set, fall through
    // to NONE (mirroring the DD fallthrough behavior). Only SINGLE_SIMD_LEVEL
    // and NONE have compiled specializations.
    if constexpr (available_levels & (1 << int(SINGLE_SIMD_LEVEL))) {
        return action.template operator()<SINGLE_SIMD_LEVEL>();
    } else {
        return action.template operator()<SIMDLevel::NONE>();
    }
#endif
}

/**
 * Dispatch to a lambda with SIMDLevel as a compile-time constant.
 *
 * This function calls the provided templated lambda with the current
 * runtime SIMD level (from SIMDConfig::level) as a compile-time template
 * argument. This enables SIMD-specialized code paths while keeping the
 * dispatch logic centralized.
 *
 * The key benefit is that the SIMD dispatch happens once, outside any loops,
 * so the loop body runs with the optimal SIMD implementation without
 * per-iteration dispatch overhead.
 *
 * Example with a loop (the dispatch happens once, not per iteration):
 *
 *   std::vector<float> distances(n);
 *   with_simd_level([&]<SIMDLevel level>() {
 *       for (size_t i = 0; i < n; i++) {
 *           distances[i] = fvec_L2sqr<level>(query, vectors + i * d, d);
 *       }
 *   });
 *
 * The lambda must be a generic lambda with a SIMDLevel template parameter.
 * By default, the lambda uses levels AVX2 + AVX512 + NEON, since these are the
 * most common cases.
 *
 * @param action A generic lambda with signature `template<SIMDLevel> T
 * operator()()`
 * @return The return value of the lambda
 */
template <typename LambdaType>
inline auto with_simd_level(LambdaType&& action) {
    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_A0>(
            std::forward<LambdaType>(action));
}

/**
 * Use for functions implemented with simdXintY (256-bit) operations
 * that don't have dedicated AVX512 or SVE implementations.
 */
template <typename LambdaType>
inline auto with_simd_level_256bit(LambdaType&& action) {
    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_AVX2_NEON>(
            std::forward<LambdaType>(action));
}

} // namespace faiss
