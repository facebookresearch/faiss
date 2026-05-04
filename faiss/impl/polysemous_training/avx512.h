/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace faiss {

class ReproduceDistancesObjective;

namespace polysemous_avx512 {

/// AVX-512 compute_cost for ReproduceWithHammingObjective.
/// Parameters mirror the objective's fields to avoid exposing the
/// anonymous-namespace struct.
double hamming_compute_cost_avx512(
        int n,
        const int* perm,
        const double* target_dis,
        const double* weights);

/// AVX-512 cost_update for ReproduceWithHammingObjective.
double hamming_cost_update_avx512(
        int n,
        const int* perm,
        int iw,
        int jw,
        const double* target_dis,
        const double* weights);

/// AVX-512 compute_cost for ReproduceDistancesObjective.
double distances_compute_cost_avx512(
        const ReproduceDistancesObjective& obj,
        const int* perm);

/// AVX-512 cost_update for ReproduceDistancesObjective.
double distances_cost_update_avx512(
        const ReproduceDistancesObjective& obj,
        const int* perm,
        int iw,
        int jw);

} // namespace polysemous_avx512
} // namespace faiss
