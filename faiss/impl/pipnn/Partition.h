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
#include <functional>
#include <vector>

namespace faiss {

struct Index;

namespace pipnn {

enum class Ranking : uint8_t;

/// Randomized Ball Carving settings (paper Alg. 5).
struct PartitionParams {
    int c_max = 1024;                  // largest leaf
    int c_min = 128;                   // smaller children are merged
    int leader_cap = 1000;             // most leaders per split
    float leader_fraction = 0.005f;    // leaders per split, as a fraction of it
    std::vector<int> fanout = {10, 3}; // leaders joined per depth, then 1
    MetricType metric = METRIC_L2;
    bool ip_partition_by_angle = false;
    size_t wave_budget = size_t(1) << 30; // point instances per wave
    int max_depth = 32; // children at this depth are cut into id slices
    int fanout_at(int depth) const;
};

struct PartitionStats {
    size_t n_leaves = 0;
    size_t n_point_instances = 0; // sum of leaf sizes
    size_t max_leaf = 0;
    size_t n_slice_fallbacks = 0, n_l2_repartitions = 0;
    int max_depth_seen = 0;
    /// leaf_size_histogram[i] counts leaves of size in [64 i + 1, 64 (i + 1)].
    std::vector<size_t> leaf_size_histogram;
    size_t n_waves = 0;
};

/// Called once per leaf with its sorted, unique ids (s <= c_max), concurrently
/// from partition_and_visit's OpenMP region: omp_get_thread_num() identifies
/// the calling thread. `ids` is valid only during the call.
using LeafVisitor = std::function<void(const int32_t* ids, size_t s)>;

/// Randomized Ball Carving with multi-level fanout. Calls `on_leaf` exactly
/// once per leaf and fills `*stats` (if not null). On an exception or an
/// interrupt it rethrows with some leaves unvisited and `*stats` unchanged.
void partition_and_visit(
        const Index& storage,
        const PartitionParams& p,
        uint64_t root_seed,
        const LeafVisitor& on_leaf,
        PartitionStats* stats);

/// Deterministic hash of (a, b) for derived seeds. For a fixed a, distinct b
/// give distinct results.
uint64_t mix_seed(uint64_t a, uint64_t b);

/// Building blocks of partition_and_visit, exposed for unit tests.
namespace detail {

/// min(max(ceil(leader_fraction * size), 4 * fanout(depth)), leader_cap, size)
size_t leader_count(size_t size, const PartitionParams& p, int depth);

/// k distinct positions of [0, N), ascending (Floyd's algorithm).
void sample_positions(
        size_t N,
        size_t k,
        uint64_t seed,
        std::vector<size_t>& out);

Ranking ranking_for(const PartitionParams& p, bool force_l2);

} // namespace detail

} // namespace pipnn
} // namespace faiss
