// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, 6sense Insights Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include <faiss/gpu/impl/GpuHnswSearchKernel.cuh>

namespace faiss {
namespace gpu {
namespace hnsw_kernel {

// Fixed upper bound on the launch block size for the brute-force kernel; the
// tree reduction requires a power-of-two block, so callers must launch with a
// power-of-two blockDim <= kBruteForceMaxBlock.
constexpr int kBruteForceMaxBlock = 256;

// Brute-force top-k over the live (non-filtered) rows for a set of queries,
// operating entirely from GPU-resident vectors (Knowhere frees the CPU HNSW
// index after upload, so the fallback cannot call it). Distances use the same
// "smaller == closer" internal convention as the graph kernel (IP negated),
// and are negated back on copy-out so callers receive the true similarity.
//
// One block per work item. The work item maps to a query index via d_worklist,
// or identity when d_worklist == nullptr (the up-front all-queries path). Rows
// filtered out by the bitset are skipped. Selection is iterative in
// (distance, id) lexicographic order: k block-wide reductions, each finding the
// smallest live candidate strictly after the previous winner. This needs no
// per-query top-k buffer and is duplicate-free because (dist, id) is a total
// order. Cost is O(k * N / blockDim) per query. When fewer than k live rows
// exist the remaining slots are filled with sentinels (-1 id, worst score).
//
// Requires: blockDim.x is a power of two, <= kBruteForceMaxBlock. For the DP4A
// int8 path (USE_DP4A) dim must be a multiple of 4, matching the graph kernel.
template <typename DataT, typename QueryT, bool USE_DP4A>
__global__ void brute_force_topk_kernel(
        const QueryT* __restrict__ d_queries,
        const DataT* __restrict__ d_dataset,
        const float* __restrict__ d_inv_norms,
        const uint8_t* __restrict__ d_bitset,
        const uint32_t* __restrict__ d_worklist, // null => identity mapping
        int num_items,
        const int* __restrict__ d_num_items, // null => use num_items directly
        int N,
        int dim,
        int k,
        bool use_inner_product,
        uint64_t* __restrict__ d_neighbors,
        float* __restrict__ d_distances) {
    // The per-query fallback launches a full num_queries grid and reads the
    // device-side worklist length here, so no host sync is needed between the
    // graph kernel and this pass. The up-front path passes num_items directly.
    int count = (d_num_items != nullptr) ? *d_num_items : num_items;
    int item = blockIdx.x;
    if (item >= count)
        return;
    int query_idx = (d_worklist != nullptr)
            ? static_cast<int>(d_worklist[item])
            : item;

    const QueryT* query = d_queries + static_cast<int64_t>(query_idx) * dim;
    int dim4 = dim / 4;
    int tid = threadIdx.x;

    __shared__ float s_dist[kBruteForceMaxBlock];
    __shared__ uint32_t s_id[kBruteForceMaxBlock];
    __shared__ float win_dist_s;
    __shared__ uint32_t win_id_s;

    // (-inf, 0): the first round selects the global minimum since any finite
    // distance is strictly greater than -FLT_MAX.
    float prev_dist = -FLT_MAX;
    uint32_t prev_id = 0;

    for (int t = 0; t < k; t++) {
        float local_best = FLT_MAX;
        uint32_t local_id = UINT32_MAX;
        for (int row = tid; row < N; row += blockDim.x) {
            uint32_t r = static_cast<uint32_t>(row);
            if (is_bitset_filtered(d_bitset, r))
                continue;
            float d = layer0_distance<DataT, QueryT, USE_DP4A>(
                    query, d_dataset, d_inv_norms, r, dim, dim4,
                    use_inner_product);
            bool after = (d > prev_dist) || (d == prev_dist && r > prev_id);
            if (!after)
                continue;
            if (d < local_best || (d == local_best && r < local_id)) {
                local_best = d;
                local_id = r;
            }
        }
        s_dist[tid] = local_best;
        s_id[tid] = local_id;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float od = s_dist[tid + stride];
                uint32_t oi = s_id[tid + stride];
                if (od < s_dist[tid] ||
                    (od == s_dist[tid] && oi < s_id[tid])) {
                    s_dist[tid] = od;
                    s_id[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            win_dist_s = s_dist[0];
            win_id_s = s_id[0];
        }
        __syncthreads();
        float win_dist = win_dist_s;
        uint32_t win_id = win_id_s;

        int64_t out = static_cast<int64_t>(query_idx) * k + t;
        if (win_id == UINT32_MAX) {
            // No live candidate remains: sentinel-fill this and the rest.
            for (int tt = t + tid; tt < k; tt += blockDim.x) {
                int64_t o2 = static_cast<int64_t>(query_idx) * k + tt;
                d_neighbors[o2] = UINT64_MAX;
                d_distances[o2] = use_inner_product ? -FLT_MAX : FLT_MAX;
            }
            break;
        }
        if (tid == 0) {
            d_neighbors[out] = static_cast<uint64_t>(win_id);
            d_distances[out] = use_inner_product ? -win_dist : win_dist;
        }
        prev_dist = win_dist;
        prev_id = win_id;
        __syncthreads();
    }
}

} // namespace hnsw_kernel
} // namespace gpu
} // namespace faiss
