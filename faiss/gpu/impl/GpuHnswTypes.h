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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

namespace faiss {
namespace gpu {

// Test-only fault injection for the device-upload path (consulted by
// GPU_HNSW_BUILD_CUDA_CHECK in GpuHnswBuildCommon.cuh). Production code never
// arms it (the countdown stays 0), so the check is a single relaxed atomic
// load per wrapped CUDA call with no runtime effect. Unit tests call arm(n)
// so the n-th
// subsequent wrapped CUDA call reports a simulated failure, exercising
// Deserialize()'s upload error / partial-allocation cleanup / retry path
// without a real OOM. Defined here (host-safe header) rather than in the
// .cuh so host-compiled tests can arm it without pulling in device kernels.
// The static
// lives in an inline function, so all translation units share one instance.
struct GpuHnswUploadFaultInjection {
    static std::atomic<int>& countdown() {
        static std::atomic<int> c{0};
        return c;
    }
    // Arm so the n-th following wrapped CUDA call fails (n >= 1). 0 disarms.
    static void arm(int n) {
        countdown().store(n, std::memory_order_relaxed);
    }
    static void disarm() {
        countdown().store(0, std::memory_order_relaxed);
    }
    // Returns true exactly once, on the n-th call after arm(n); self-disarms.
    static bool should_fail() {
        int cur = countdown().load(std::memory_order_relaxed);
        if (cur <= 0)
            return false;
        return countdown().fetch_sub(1, std::memory_order_relaxed) == 1;
    }
};

// Element type of the device-resident dataset. The graph walk kernel is
// templated on this; each value selects a load_elem specialization so the
// vectors stay in their native precision on the GPU (no up-conversion to
// fp32 at upload time).
enum class GpuHnswDatasetType {
    FP32 = 0,
    INT8 = 1,
    FP16 = 2,
    BF16 = 3,
};

// Cap (bytes) on the per-search visited bitmap for a single scratch slot. The
// bitmap is nq * ceil(N/32) * 4 bytes and is grow-only per slot, so a large
// query batch or high search concurrency (one bitmap per pool slot) can grow it
// until it exhausts device memory. Observed regression: 16 concurrent batch=512
// searches on a 538M-row segment grew the pool to ~97 of ~98 GB and OOM'd every
// subsequent allocation. Bounding the bitmap and processing queries in
// nq-chunks caps it regardless of batch size / concurrency.
//
// Tunable via environment (re-read on each call so tests can toggle it; getenv
// is negligible next to a GPU search): GPU_HNSW_BITMAP_BYTES sets the cap in
// bytes (used by tests to force the multi-chunk path on tiny inputs) and takes
// precedence over GPU_HNSW_BITMAP_MB (megabytes). Default 256 MiB. The value is
// only a chunking bound, so any positive value is safe (the chunk is clamped to
// >= 1 query regardless).
inline size_t gpu_hnsw_bitmap_cap_bytes() {
    if (const char* eb = std::getenv("GPU_HNSW_BITMAP_BYTES")) {
        long long v = std::atoll(eb);
        if (v > 0) {
            return static_cast<size_t>(v);
        }
    }
    size_t mb = 256;
    if (const char* e = std::getenv("GPU_HNSW_BITMAP_MB")) {
        long v = std::atol(e);
        if (v > 0) {
            mb = static_cast<size_t>(v);
        }
    }
    return mb * (static_cast<size_t>(1) << 20);
}

// Number of queries to process per launch so the visited bitmap stays within
// gpu_hnsw_bitmap_cap_bytes(). Always in [1, nq]; returns nq when the whole
// batch already fits, so small segments are chunked into a single pass and see
// no behavior change. Queries are independent in HNSW search, so chunking does
// not change per-query results. Must stay in lockstep with the bitmap sizing in
// GpuHnswSearchScratch::ensure() and the launch loop in gpu_hnsw_search().
inline int gpu_hnsw_bitmap_chunk(int nq, int N) {
    if (nq <= 0) {
        return nq;
    }
    size_t per_query =
            static_cast<size_t>((N + 31) / 32) * sizeof(uint32_t);
    if (per_query == 0) {
        return nq;
    }
    size_t cap_q = gpu_hnsw_bitmap_cap_bytes() / per_query;
    if (cap_q < 1) {
        cap_q = 1;
    }
    return (cap_q >= static_cast<size_t>(nq)) ? nq
                                              : static_cast<int>(cap_q);
}

struct GpuHnswSearchParams {
    int ef = 200;
    int search_width = 4;
    int max_iterations = 0;
    int thread_block_size = 0;

    // --- Filtered search (deletes / TTL / partition / visibility bitset) ---
    //
    // When bitset_data == nullptr the search runs the unfiltered fast path,
    // which is codegen-identical to the append-only kernel (the filter
    // branches are compiled out via `if constexpr`). When non-null it enables
    // CPU-HNSW-parity filtered search: filtered nodes stay graph waypoints but
    // are never emitted, an alpha gate rate-limits their expansion, and a
    // brute-force fallback guarantees k live results under heavy deletes.
    //
    // Semantics match Knowhere's BitsetView: a set bit means the row at that
    // index is filtered OUT (deleted / not visible). The bit order is
    // LSB-first per byte: row r is byte r/8, bit r%8. The index space is the
    // storage add-order row id, which the GPU returns directly (see the design
    // doc's ID-mapping section and the gpu_hnsw_id_mapping test).
    const uint8_t* bitset_data = nullptr; // host ptr, uploaded per search batch
    int64_t bitset_nbits = 0;             // rows the bitset covers (== n_rows)
    int64_t bitset_filtered_count = 0;    // popcount: rows filtered out
    // Invalid-frontier capacity. 0 => default to ef. Cappable to bound the
    // extra shared memory (12 B / ef_inv on top of the 24 B / ef base).
    int ef_inv = 0;
    // Mirrors Knowhere's hnsw_cfg.disable_fallback_brute_force. When true the
    // short-result BF fallback is skipped and short queries keep padded
    // sentinels, matching CPU HNSW with fallback disabled.
    bool disable_fallback_brute_force = false;
};

struct GpuHnswDeviceUpperLayer {
    uint32_t* d_node_ids = nullptr;
    uint32_t* d_neighbors = nullptr;
    uint32_t num_nodes = 0;
    uint32_t max_degree = 0;
};

struct GpuHnswSearchScratch {
    float* d_queries = nullptr;
    uint64_t* d_neighbors = nullptr;
    float* d_distances = nullptr;
    uint32_t* d_entry_points = nullptr;
    uint32_t* d_visited_bitmaps = nullptr;
    int8_t* d_queries_i8 = nullptr; // int8 queries for the native DP4A path
    // True only when int8 queries were staged into d_queries_i8 *this* search
    // (set in ensure() from use_i8_queries). Scratch slots are pooled and
    // reused, so d_queries_i8 stays allocated after an int8 search; gate the
    // DP4A path on this flag, not on d_queries_i8 != nullptr, or a later
    // fp32-query search on the same slot would score against stale int8 data.
    bool i8_queries_staged = false;

    // Filtered-search scratch. d_bitset holds the uploaded BitsetView bytes
    // (ceil(nbits/8)); d_needs_bf is the device-side worklist of query indices
    // whose graph search returned fewer than k live results (per-query BF
    // fallback), and d_needs_bf_count is its atomic length. All allocated
    // lazily and only when a filtered search runs.
    uint8_t* d_bitset = nullptr;
    uint32_t* d_needs_bf = nullptr;
    int* d_needs_bf_count = nullptr;

    size_t queries_bytes = 0;
    size_t neighbors_bytes = 0;
    size_t distances_bytes = 0;
    int entry_cap = 0;
    size_t bitmap_bytes = 0;
    size_t queries_i8_bytes = 0;
    size_t bitset_bytes = 0;
    int needs_bf_cap = 0;

    // Device this scratch's allocations live on; used to set the CUDA device
    // context before freeing in the destructor (multi-GPU correctness).
    int device = 0;

    void ensure(int nq, int k, int dim, int N, bool use_i8_queries = false);

    // Ensure the per-search filter scratch is large enough: a device bitset of
    // `bitset_bytes_needed` bytes (reallocated when n_rows grows) and a
    // needs_bf worklist sized for `nq` queries plus its counter. Separate from
    // ensure() because filtering is optional and the bitset size tracks the
    // segment row count, not nq/k/dim.
    void ensure_filter(int nq, size_t bitset_bytes_needed);

    ~GpuHnswSearchScratch();

    GpuHnswSearchScratch() = default;
    GpuHnswSearchScratch(const GpuHnswSearchScratch&) = delete;
    GpuHnswSearchScratch& operator=(const GpuHnswSearchScratch&) = delete;
};

/// One slot in the scratch pool: a scratch buffer + its own CUDA stream.
struct GpuHnswScratchSlot {
    GpuHnswSearchScratch scratch;
    cudaStream_t stream = nullptr;

    ~GpuHnswScratchSlot();
    GpuHnswScratchSlot() = default;
    GpuHnswScratchSlot(const GpuHnswScratchSlot&) = delete;
    GpuHnswScratchSlot& operator=(const GpuHnswScratchSlot&) = delete;
};

/// Pool of scratch buffers allowing concurrent GPU searches on the same
/// segment.
/// Each acquire() returns a slot with its own scratch + CUDA stream.
/// Callers block if all slots are in use.
class GpuHnswScratchPool {
 public:
    /// Create a pool. CUDA streams are allocated lazily on first acquire().
    explicit GpuHnswScratchPool(int pool_size = 4, int device = 0);
    ~GpuHnswScratchPool() = default;

    GpuHnswScratchPool(const GpuHnswScratchPool&) = delete;
    GpuHnswScratchPool& operator=(const GpuHnswScratchPool&) = delete;

    /// Acquire a scratch slot (blocks until one is available).
    GpuHnswScratchSlot* acquire();
    /// Release a previously acquired scratch slot back to the pool.
    void release(GpuHnswScratchSlot* slot);

    int pool_size() const { return pool_size_; }

 private:
    void init_once();

    std::mutex mutex_;
    std::condition_variable cv_;
    int pool_size_;
    int device_;
    bool initialized_ = false;
    std::vector<std::unique_ptr<GpuHnswScratchSlot>> slots_;
    std::vector<GpuHnswScratchSlot*> available_;
};

/// RAII guard: acquires a scratch slot on construction, releases on
/// destruction.
class ScratchPoolGuard {
 public:
    ScratchPoolGuard(GpuHnswScratchPool& pool)
            : pool_(pool), slot_(pool.acquire()) {}
    ~ScratchPoolGuard() { pool_.release(slot_); }
    GpuHnswScratchSlot* get() const { return slot_; }

    ScratchPoolGuard(const ScratchPoolGuard&) = delete;
    ScratchPoolGuard& operator=(const ScratchPoolGuard&) = delete;

 private:
    GpuHnswScratchPool& pool_;
    GpuHnswScratchSlot* slot_;
};

struct GpuHnswDeviceIndex {
    void* d_dataset = nullptr;
    GpuHnswDatasetType dataset_type = GpuHnswDatasetType::FP32;
    float* d_inv_norms = nullptr;
    uint32_t* d_layer0_graph = nullptr;
    std::vector<GpuHnswDeviceUpperLayer> upper_layers;

    int64_t n_rows = 0;
    int64_t dim = 0;
    uint32_t entry_point = 0;
    // Total layer count (max_level + 1), informational only; the search path
    // uses num_upper_layers_built (below) for the number of uploaded upper
    // layers. Kept for diagnostics/logging.
    int num_layers = 0;
    int M = 0;
    int max_degree0 = 0;
    bool use_ip = false;

    void* d_upper_layer_ptrs = nullptr;
    int num_upper_layers_built = 0;

    // Device this index's allocations live on; used to set the CUDA device
    // context before freeing in the destructor (multi-GPU correctness).
    int device = 0;

    mutable std::unique_ptr<GpuHnswScratchPool> scratch_pool;

    ~GpuHnswDeviceIndex();
};

} // namespace gpu
} // namespace faiss
