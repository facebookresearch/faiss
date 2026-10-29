/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

struct DistanceComputer;
struct IDSelector;
struct Index;
struct PiPNN;
struct VisitedTable;

namespace pipnn {

// Internal build pieces, exposed for tests; hidden from SWIG.
#ifndef SWIG

/// Points per medoid block. It fixes the summation order of the mean, so
/// changing it changes the entry point.
constexpr size_t kMedoidBlock = size_t(1) << 16;

/// Throws unless every vector is finite with squared norm <= FLT_MAX / 8, and
/// returns the L2 medoid: the point closest to the mean, lowest id on ties.
int32_t medoid_and_validate(const Index& storage, float* norm_cv = nullptr);

/// RobustPrune of x (row 0 of the m x m D) over cand[t] (row 1 + t) in
/// (D, id) order; writes <= R ids to out, -1 padded. order: m - 1 ints.
void robust_prune(
        const float* D,
        size_t m,
        const int32_t* cand,
        MetricType metric,
        float alpha,
        int R,
        int* order,
        int32_t* out);

namespace detail {

/// PiPNN::build with a given reservoir chunk size, for tests.
void build_with_chunk(
        PiPNN& p,
        const Index& storage,
        size_t reservoir_chunk,
        bool verbose);

} // namespace detail

#endif // SWIG

} // namespace pipnn

/// PiPNN graph (Rubel et al., KDD '26, arXiv:2602.21247).
struct PiPNN {
    // build parameters
    int R = 64;              ///< max out-degree
    int leaf_k = 2;          ///< neighbours each point offers per leaf
    int hash_bits = 12;      ///< HashPrune keeps one candidate per hash bucket
    int reservoir_size = 64; ///< candidates kept per point
    int c_max = 1024;        ///< largest leaf
    int c_min = 128;         ///< smaller partition children are merged
    int leader_cap = 1000;   ///< most leaders per split
    float leader_fraction = 0.005f; ///< leaders per split, fraction of its size
    float alpha = 0;                ///< 0 = 1.2
    std::vector<int> fanout = {10, 3}; ///< leaders joined per depth, then 1
    int64_t seed = 1234;
    /// IP: rank partition leaders by angle, not inner product
    bool ip_partition_by_angle = false;
    /// point instances per partition wave, which bounds memory
    size_t partition_wave_budget = size_t(1) << 30;

    int search_L = 64; ///< default beam width of search

    // result
    MaybeOwnedVector<int32_t> graph; ///< ntotal x R, -1 padded
    /// next higher row with the same code, or -1; empty if all codes differ.
    /// Such copies have empty graph rows; search returns them with the first.
    MaybeOwnedVector<int32_t> next_copy;
    int32_t entry_point = -1; ///< L2 medoid
    idx_t ntotal = 0;

    PiPNN() = default;
    explicit PiPNN(int R) : R(R) {}

    float effective_alpha(MetricType metric) const;

    void check_params() const;

#ifndef SWIG
    /// Builds over the vectors and metric of storage. Strong guarantee: the
    /// results are unchanged if it throws.
    void build(const Index& storage, bool verbose = false);

    /// Beam max(L, k); qdis smaller-is-better; vt: ntotal slots, clean in/out.
    /// Writes the k nearest in sel by (dis, id), padded; sel does not prune.
    void search(
            DistanceComputer& qdis,
            int L,
            idx_t k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            const IDSelector* sel) const;
#endif

    /// Empties the graph; entry_point = -1, ntotal = 0.
    void reset();
};

} // namespace faiss
