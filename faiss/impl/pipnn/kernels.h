/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <cstddef>
#include <cstdint>

/*
 * Each kernel makes one BLAS call (sgemm_ or ssyrk_), and results are bitwise
 * reproducible across thread counts only if that call runs single-threaded.
 * So call the kernels inside an OpenMP parallel region, or from serial code
 * only when omp_get_max_threads() == 1, and keep BLAS from threading on its
 * own: MKL_NUM_THREADS unset or 1, OPENBLAS_NUM_THREADS=1 for pthreads
 * OpenBLAS, VECLIB_MAXIMUM_THREADS=1 on macOS, OMP_NUM_THREADS=1 for an
 * OpenMP BLAS in a Faiss build without OpenMP. Inputs should be 64-byte
 * aligned: some BLAS round differently on differently aligned buffers.
 */

namespace faiss {
namespace pipnn {

/// Points per stripe item of the partition kernel. Changing it changes graphs.
constexpr size_t kStripe = 1024;

enum class Ranking : uint8_t { L2 = 0, IP = 1, Angle = 2 };

/// out (S x fanout): each point's nearest leaders in (rank, index) order, with
/// rank L2 ||l||^2 - 2 p.l, IP -p.l, Angle -p.l / ||l|| (+inf if ||l|| = 0).
/// NaN ranks +inf. scratch: S * L floats; leader_norms may be null for IP.
void stripe_assign(
        const float* points,
        size_t S,
        const float* leaders,
        const float* leader_norms,
        size_t L,
        size_t d,
        Ranking ranking,
        int fanout,
        float* scratch,
        uint16_t* out);

/// k nearest neighbours (s x k) of each leaf point in (dist, index) order;
/// unfilled slots are -1 / +inf. gram: s * s floats; norms null for IP only.
void leaf_knn(
        const float* X,
        const float* norms,
        size_t s,
        size_t d,
        MetricType metric,
        int k,
        float* gram,
        int32_t* nn_idx,
        float* nn_dist);

/// n x n distances (L2 clamped at 0, IP -dot); out is also the Gram scratch.
/// The L2 diagonal is small and non-negative, but not necessarily 0.
void pairwise_distances(
        const float* X,
        const float* norms,
        size_t n,
        size_t d,
        MetricType metric,
        float* out);

} // namespace pipnn
} // namespace faiss
