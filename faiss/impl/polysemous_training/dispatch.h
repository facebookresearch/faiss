/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/utils/simd_levels.h>

namespace faiss {

struct ReproduceDistancesObjective;

namespace polysemous_training {

// Levels with a dedicated kernel: scalar (NONE) plus AVX-512. Any other runtime
// level (AVX2, NEON, ...) falls back to NONE via with_selected_simd_levels /
// get_simd_fallback, and AVX512_SPR reuses the AVX-512 kernel.
constexpr int SIMD_LEVELS =
        (1 << int(SIMDLevel::NONE)) | (1 << int(SIMDLevel::AVX512));

/// compute_cost for ReproduceWithHammingObjective.
/// Parameters mirror the objective's fields to avoid exposing the
/// anonymous-namespace struct.
template <SIMDLevel SL>
double hamming_compute_cost(
        int n,
        const int* perm,
        const double* target_dis,
        const double* weights);

/// cost_update for ReproduceWithHammingObjective.
template <SIMDLevel SL>
double hamming_cost_update(
        int n,
        const int* perm,
        int iw,
        int jw,
        const double* target_dis,
        const double* weights);

/// compute_cost for ReproduceDistancesObjective.
template <SIMDLevel SL>
double distances_compute_cost(
        const ReproduceDistancesObjective& obj,
        const int* perm);

/// cost_update for ReproduceDistancesObjective.
template <SIMDLevel SL>
double distances_cost_update(
        const ReproduceDistancesObjective& obj,
        const int* perm,
        int iw,
        int jw);

// The scalar (NONE) specializations are defined in PolysemousTraining.cpp; the
// AVX-512 specializations are defined in polysemous_training/avx512.cpp. Both
// are declared here so callers instantiate the out-of-line definition rather
// than implicitly instantiating the (undefined) primary template.
template <>
double hamming_compute_cost<SIMDLevel::NONE>(
        int n,
        const int* perm,
        const double* target_dis,
        const double* weights);
template <>
double hamming_compute_cost<SIMDLevel::AVX512>(
        int n,
        const int* perm,
        const double* target_dis,
        const double* weights);

template <>
double hamming_cost_update<SIMDLevel::NONE>(
        int n,
        const int* perm,
        int iw,
        int jw,
        const double* target_dis,
        const double* weights);
template <>
double hamming_cost_update<SIMDLevel::AVX512>(
        int n,
        const int* perm,
        int iw,
        int jw,
        const double* target_dis,
        const double* weights);

template <>
double distances_compute_cost<SIMDLevel::NONE>(
        const ReproduceDistancesObjective& obj,
        const int* perm);
template <>
double distances_compute_cost<SIMDLevel::AVX512>(
        const ReproduceDistancesObjective& obj,
        const int* perm);

template <>
double distances_cost_update<SIMDLevel::NONE>(
        const ReproduceDistancesObjective& obj,
        const int* perm,
        int iw,
        int jw);
template <>
double distances_cost_update<SIMDLevel::AVX512>(
        const ReproduceDistancesObjective& obj,
        const int* perm,
        int iw,
        int jw);

} // namespace polysemous_training
} // namespace faiss
