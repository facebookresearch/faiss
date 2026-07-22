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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/gpu/impl/GpuHnswTypes.h>
#include <faiss/gpu/impl/GpuHnswBruteForce.cuh>
#include <faiss/gpu/impl/GpuHnswSearchKernel.cuh>

#define GPU_HNSW_CUDA_CHECK(expr)                                     \
    do {                                                              \
        cudaError_t _e = (expr);                                      \
        if (_e != cudaSuccess) {                                      \
            throw std::runtime_error(                                 \
                    std::string("CUDA error: ") +                     \
                    cudaGetErrorString(_e) + " at " + __FILE__ + ":" + \
                    std::to_string(__LINE__));                         \
        }                                                             \
    } while (0)

namespace faiss {
namespace gpu {

inline void gpu_hnsw_search(
        cudaStream_t stream,
        const GpuHnswSearchParams& params,
        const GpuHnswDeviceIndex& idx,
        GpuHnswSearchScratch& sc,
        int num_queries,
        int k) {

    int ef = params.ef;
    int sw = params.search_width;
    // Reject degenerate params up front: ef and search_width feed the
    // shared-memory sizing and the auto-iteration bound (2*ef/sw below), so a
    // value of 0 would divide by zero and a negative value would under-size
    // the buffers.
    if (ef <= 0 || sw <= 0) {
        throw std::runtime_error(
                std::string("gpu_hnsw: ef and search_width must be > 0 (ef=") +
                std::to_string(ef) + ", search_width=" + std::to_string(sw) +
                ")");
    }
    // Auto-clamp ef up to k. The kernel tracks exactly ef candidates in the
    // result beam, so ef < k could only ever return ef real neighbors with the
    // remaining k-ef slots padded with sentinels. Raising ef to k matches the
    // CPU / Knowhere HNSW contract (always return k valid results). The
    // shared-memory-fit check below still throws if even k slots cannot fit.
    if (k > ef) {
        ef = k;
    }
    int max_iter = params.max_iterations > 0
            ? params.max_iterations
            : 2 * ef / sw + 10;
    int dim = static_cast<int>(idx.dim);
    int num_upper_layers = idx.num_upper_layers_built;

    // --- Filtered-search setup (deletes / TTL / partition bitset) ---
    // The bitset bytes are uploaded into sc.d_bitset by the caller (searchHost/
    // searchHostInt8) on this same stream before gpu_hnsw_search runs. When
    // bitset_data is null we take the unfiltered fast path.
    bool has_filter = (params.bitset_data != nullptr);
    int64_t nbits = params.bitset_nbits;
    int64_t filtered_count = params.bitset_filtered_count;
    int64_t live_count = has_filter ? (nbits - filtered_count)
                                    : static_cast<int64_t>(idx.n_rows);
    if (live_count < 0) {
        live_count = 0;
    }
    // kAlpha rate-limits how often filtered nodes become waypoints; it scales
    // with the delete ratio, matching CPU HNSW (kAlpha = filter_ratio * 0.7).
    float filter_ratio = (has_filter && nbits > 0)
            ? static_cast<float>(filtered_count) / static_cast<float>(nbits)
            : 0.0f;
    float kAlpha = filter_ratio * 0.7f;
    bool disable_bf = params.disable_fallback_brute_force;

    // Up-front brute force: when almost everything is filtered, or k is a large
    // fraction of the live set, the graph walk cannot beat a full scan and
    // risks returning fewer than k live results. Mirrors CPU HNSW's
    // kHnswSearchKnnBFFilterThreshold (0.93) and kHnswSearchBFTopkThreshold
    // (0.5, applied to the live count). Disabled when the caller disables the
    // brute-force fallback.
    bool up_front_bf = has_filter && !disable_bf && nbits > 0 &&
            (static_cast<double>(filtered_count) >=
                     0.93 * static_cast<double>(nbits) ||
             static_cast<double>(k) >= 0.5 * static_cast<double>(live_count));

    // Layer-0 is templated on the dataset type (DataT), the layer-0 query type
    // (QueryT: float generic, int8_t for the native DP4A path) and USE_DP4A.
    // The upper-layer greedy descent always uses the fp32 queries (sc.d_queries).
    auto launch_kernels = [&]<typename DataT, typename QueryT, bool USE_DP4A>(
                                  const DataT* d_data,
                                  const float* d_inv_norms,
                                  const QueryT* d_layer0_queries) {
        int N_int = static_cast<int>(idx.n_rows);

        // Brute-force launcher (shares the current dtype specialization). It
        // operates on a chunk of `num_items` queries whose scratch pointers
        // (q0/nb0/ds0) are already offset by the caller; when non-null the
        // worklist holds chunk-local query indices, matching the offset outputs.
        // Grid is num_items: the up-front path uses the identity mapping, the
        // per-query fallback reads its worklist length from d_num on the device
        // (no host sync between graph and BF). Block must be a power of two <=
        // kBruteForceMaxBlock for the tree reduction.
        int bf_block = 128;
        auto launch_bf = [&](const QueryT* q0,
                             uint64_t* nb0,
                             float* ds0,
                             const uint32_t* worklist,
                             const int* d_num,
                             int num_items) {
            hnsw_kernel::brute_force_topk_kernel<DataT, QueryT, USE_DP4A>
                    <<<num_items, bf_block, 0, stream>>>(
                            q0,
                            d_data,
                            d_inv_norms,
                            sc.d_bitset,
                            worklist,
                            num_items,
                            d_num,
                            N_int,
                            dim,
                            k,
                            idx.use_ip,
                            nb0,
                            ds0);
            GPU_HNSW_CUDA_CHECK(cudaGetLastError());
        };

        // --- Shared-memory sizing / ef finalization ---
        // Query-count independent (depends only on ef/search_width/M and the
        // device smem limit), so compute it once before the chunk loop.
        int block_size =
                params.thread_block_size > 0 ? params.thread_block_size : 128;

        // Per-block dynamic shared-memory budget for this device. The default
        // limit is 48 KiB, but Volta+ GPUs can opt into more via
        // cudaFuncSetAttribute; query the real limit instead of assuming 48 KiB.
        int smem_max = 49152;
        {
            int device = 0;
            int optin = 0;
            if (cudaGetDevice(&device) == cudaSuccess &&
                cudaDeviceGetAttribute(
                        &optin,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin,
                        device) == cudaSuccess &&
                optin > smem_max) {
                smem_max = optin;
            }
        }

        // The bitonic sort in the parallel merge needs a power-of-two staging
        // capacity, one thread per slot. Pad up to the next power of two
        // (handles non-power-of-two 2*M) and grow the block so every staging
        // slot is owned by a thread.
        int max_staging =
                hnsw_kernel::padded_staging_capacity(sw, idx.max_degree0);
        {
            if (block_size < max_staging) {
                block_size = max_staging;
            }
            // The layer-0 expansion is warp-cooperative (one warp per candidate
            // edge, 32 lanes striding the vector for coalesced loads), so the
            // block must be a whole number of warps for the full-mask __shfl
            // reductions to be well-defined. The default (128) and the
            // power-of-two staging bump are already multiples of 32; this only
            // rounds up a caller-supplied non-multiple thread_block_size.
            block_size = ((block_size + 31) / 32) * 32;
            if (block_size > 1024) {
                throw std::runtime_error(
                        std::string("gpu_hnsw: padded staging capacity ") +
                        std::to_string(max_staging) +
                        " exceeds the max CUDA block size (1024); reduce "
                        "search_width or M (max_degree0=" +
                        std::to_string(idx.max_degree0) + ")");
            }
            // Fixed overhead: staging (max_staging*8) + parent_ids (sw*4)
            // + meta (12, or 24 on the filtered path with 6 slots).
            int meta_bytes = has_filter ? 24 : 12;
            int smem_overhead = max_staging * 8 + sw * 4 + meta_bytes;
            // Per-ef cost: 3 result arrays + 3 merge arrays = 6 × 4 = 24
            // bytes/slot. The filtered path adds the invalid frontier (3 more
            // arrays); with ef_inv capped at ef (the default) that is a further
            // 12 bytes/slot, so divide by 36 to bound ef conservatively.
            int per_ef = has_filter ? 36 : 24;
            int max_ef = (smem_max - smem_overhead) / per_ef;
            if (max_ef < 1) {
                throw std::runtime_error(
                        std::string("gpu_hnsw: search_width=") +
                        std::to_string(sw) +
                        " too large for device shared memory (" +
                        std::to_string(smem_max) +
                        " bytes); reduce search_width");
            }
            // ef was auto-clamped up to at least k above. If even k candidate
            // slots cannot fit in shared memory, fail loudly rather than
            // silently returning fewer than k valid results.
            if (k > max_ef) {
                throw std::runtime_error(
                        std::string("gpu_hnsw: k=") + std::to_string(k) +
                        " needs k candidate slots (ef>=k), only " +
                        std::to_string(max_ef) +
                        " fit in device shared memory (" +
                        std::to_string(smem_max) +
                        " bytes); reduce k or raise thread_block_size");
            }
            if (ef > max_ef) {
                fprintf(stderr,
                        "[gpu_hnsw] warning: ef=%d exceeds the per-block "
                        "shared-memory budget (%d bytes); clamping ef to %d. "
                        "Recall may be reduced; raise thread_block_size or "
                        "lower search_width to restore the requested ef.\n",
                        ef, smem_max, max_ef);
                ef = max_ef;
            }
        }

        // Invalid-frontier capacity: default to the (now finalized) ef, but
        // allow the caller to cap it smaller to save shared memory. Capped at
        // ef so the /36 clamp above stays conservative. Only used on the
        // filtered path.
        int ef_inv = 0;
        if (has_filter) {
            ef_inv = (params.ef_inv > 0) ? std::min(params.ef_inv, ef) : ef;
        }

        size_t smem_size = hnsw_kernel::calc_layer0_smem_size(
                ef, sw, idx.max_degree0, has_filter ? ef_inv : 0);

        // Graph search + per-query brute-force fallback for one chunk of `cnq`
        // queries. The scratch pointers (q0/ep0/nb0/ds0) are offset by the
        // caller so the kernels' chunk-local blockIdx.x maps to the right
        // global query; the visited bitmap is the base buffer, indexed by the
        // same chunk-local index. The graph kernel is templated on HAS_FILTER;
        // the filtered and unfiltered instantiations are distinct __global__
        // functions, so each needs its own cudaFuncSetAttribute high-water
        // tracking (the statics below are per-<...,HAS_FILTER>).
        auto run_graph = [&]<bool HAS_FILTER>(
                                 const QueryT* q0,
                                 uint32_t* ep0,
                                 uint64_t* nb0,
                                 float* ds0,
                                 int cnq) {
            // Opt into >48 KiB dynamic shared memory when the device supports
            // it; without this the kernel launch would fail for large ef. This
            // is cached per kernel instantiation: cudaFuncSetAttribute
            // configures the kernel's max dynamic shared memory globally, so it
            // only needs to run once per (kernel, high-water size).
            //
            // The mutex makes the check-set-store atomic and monotonic: without
            // it, two concurrent searches with different smem sizes can
            // interleave so a smaller-ef search's cudaFuncSetAttribute runs
            // *after* a larger one, downgrading the kernel's global attribute
            // below what a recorded high-water mark implies, and a later
            // intermediate search then skips the set (thinks it's configured)
            // and fails to launch. Under the lock the attribute only ever grows,
            // and is always >= the current launch's requirement before we
            // proceed.
            if (smem_size > 49152) {
                static std::mutex configured_smem_mutex;
                static size_t configured_smem = 0;
                std::lock_guard<std::mutex> lock(configured_smem_mutex);
                if (smem_size > configured_smem) {
                    GPU_HNSW_CUDA_CHECK(cudaFuncSetAttribute(
                            hnsw_kernel::layer0_beam_search_kernel<
                                    DataT,
                                    QueryT,
                                    USE_DP4A,
                                    HAS_FILTER>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(smem_size)));
                    configured_smem = smem_size;
                }
            }

            // Zero the per-query BF worklist counter before the graph launch
            // (same stream, so ordered ahead of the kernel that appends to it).
            if constexpr (HAS_FILTER) {
                if (!disable_bf) {
                    GPU_HNSW_CUDA_CHECK(cudaMemsetAsync(
                            sc.d_needs_bf_count, 0, sizeof(int), stream));
                }
            }

            const uint8_t* d_bitset = HAS_FILTER ? sc.d_bitset : nullptr;
            uint32_t* d_needs_bf = HAS_FILTER ? sc.d_needs_bf : nullptr;
            int* d_needs_bf_count = HAS_FILTER ? sc.d_needs_bf_count : nullptr;

            hnsw_kernel::layer0_beam_search_kernel<
                    DataT,
                    QueryT,
                    USE_DP4A,
                    HAS_FILTER><<<cnq, block_size, smem_size, stream>>>(
                    q0,
                    d_data,
                    d_inv_norms,
                    idx.d_layer0_graph,
                    ep0,
                    sc.d_visited_bitmaps,
                    nb0,
                    ds0,
                    cnq,
                    N_int,
                    dim,
                    idx.max_degree0,
                    k,
                    ef,
                    sw,
                    max_iter,
                    idx.use_ip,
                    d_bitset,
                    ef_inv,
                    kAlpha,
                    static_cast<int>(live_count),
                    disable_bf,
                    d_needs_bf,
                    d_needs_bf_count);
            GPU_HNSW_CUDA_CHECK(cudaGetLastError());

            // Per-query brute-force fallback: queries whose graph search fell
            // short (fewer than k live results) were appended to sc.d_needs_bf
            // (chunk-local indices) by the graph kernel. Launch a full-width
            // grid; each block reads the device worklist length and early-exits
            // when out of range, so no host sync is needed between the two
            // kernels.
            if constexpr (HAS_FILTER) {
                if (!disable_bf) {
                    launch_bf(q0, nb0, ds0, sc.d_needs_bf,
                              sc.d_needs_bf_count, cnq);
                }
            }
        };

        // --- Query chunking to bound the visited-bitmap VRAM ---
        // Process the batch in chunks of at most chunk_nq queries so the visited
        // bitmap (sized in GpuHnswSearchScratch::ensure with the same chunk)
        // stays within the VRAM cap regardless of nq / search concurrency. For
        // small segments chunk_nq == num_queries -> a single pass, identical to
        // the pre-chunk behavior.
        int chunk_nq = gpu_hnsw_bitmap_chunk(num_queries, N_int);
        for (int c0 = 0; c0 < num_queries; c0 += chunk_nq) {
            int cnq = std::min(chunk_nq, num_queries - c0);
            // Chunk-offset scratch pointers. q0 is the layer-0 query buffer
            // (int8 on the DP4A path, else fp32); fq0 is always the fp32 buffer
            // used by the upper-layer greedy descent.
            const QueryT* q0 =
                    d_layer0_queries + static_cast<int64_t>(c0) * dim;
            const float* fq0 = sc.d_queries + static_cast<int64_t>(c0) * dim;
            uint32_t* ep0 = sc.d_entry_points + c0;
            uint64_t* nb0 = sc.d_neighbors + static_cast<int64_t>(c0) * k;
            float* ds0 = sc.d_distances + static_cast<int64_t>(c0) * k;

            // Up-front brute force skips the graph search entirely (does not
            // touch the visited bitmap, but still runs per chunk so its outputs
            // land at the right offsets).
            if (up_front_bf) {
                launch_bf(q0, nb0, ds0, nullptr, nullptr, cnq);
                continue;
            }

            if (num_upper_layers > 0) {
                auto* d_layer_ptrs =
                        static_cast<hnsw_kernel::upper_layer_ptrs*>(
                                idx.d_upper_layer_ptrs);

                int warps_per_block = 4;
                int threads_per_block = warps_per_block * 32;
                int num_blocks =
                        (cnq + warps_per_block - 1) / warps_per_block;

                hnsw_kernel::upper_layer_search_kernel<DataT>
                        <<<num_blocks, threads_per_block, 0, stream>>>(
                                fq0,
                                d_data,
                                d_inv_norms,
                                d_layer_ptrs,
                                ep0,
                                idx.entry_point,
                                cnq,
                                dim,
                                num_upper_layers,
                                idx.use_ip);
                GPU_HNSW_CUDA_CHECK(cudaGetLastError());
            } else {
                std::vector<uint32_t> h_eps(cnq, idx.entry_point);
                GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
                        ep0,
                        h_eps.data(),
                        static_cast<size_t>(cnq) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        stream));
                // h_eps is stack-local; synchronize before it is destroyed so
                // the copy never reads freed memory (safe even if the source
                // buffer is ever switched to pinned host memory).
                GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // Zero this chunk's visited bitmap (reused across chunks).
            size_t bitmap_bytes =
                    hnsw_kernel::calc_visited_bitmap_size(cnq, N_int);
            GPU_HNSW_CUDA_CHECK(cudaMemsetAsync(
                    sc.d_visited_bitmaps, 0, bitmap_bytes, stream));

            if (has_filter) {
                run_graph.template operator()<true>(q0, ep0, nb0, ds0, cnq);
            } else {
                run_graph.template operator()<false>(q0, ep0, nb0, ds0, cnq);
            }
        }
    };

    switch (idx.dataset_type) {
        case GpuHnswDatasetType::INT8:
            // Native DP4A path: requires int8 queries staged on device, an
            // inner-product/cosine metric (DP4A only computes dot products) and
            // dim % 4 == 0. Otherwise fall back to the generic fp32-query path.
            if (sc.d_queries_i8 != nullptr && idx.use_ip && (dim % 4 == 0)) {
                launch_kernels.template operator()<int8_t, int8_t, true>(
                        static_cast<const int8_t*>(idx.d_dataset),
                        idx.d_inv_norms,
                        sc.d_queries_i8);
            } else {
                launch_kernels.template operator()<int8_t, float, false>(
                        static_cast<const int8_t*>(idx.d_dataset),
                        idx.d_inv_norms,
                        sc.d_queries);
            }
            break;
        case GpuHnswDatasetType::FP16:
            launch_kernels.template operator()<half, float, false>(
                    static_cast<const half*>(idx.d_dataset),
                    idx.d_inv_norms,
                    sc.d_queries);
            break;
        case GpuHnswDatasetType::BF16:
            launch_kernels.template operator()<__nv_bfloat16, float, false>(
                    static_cast<const __nv_bfloat16*>(idx.d_dataset),
                    idx.d_inv_norms,
                    sc.d_queries);
            break;
        case GpuHnswDatasetType::FP32:
        default:
            launch_kernels.template operator()<float, float, false>(
                    static_cast<const float*>(idx.d_dataset),
                    idx.d_inv_norms,
                    sc.d_queries);
            break;
    }
}

} // namespace gpu
} // namespace faiss
