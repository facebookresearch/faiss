/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Per-SIMD-level cross-equivalence tests for approx_topk_by_mode and
// HeapWithBucketsCMaxFloat::bs_addn (defined in approx_topk.h).
//
// The functions dispatch via with_simd_level_256bit. test_approx_topk.cpp
// exercises them only at the binary's compiled/dispatched level. This test
// runs the same input at every available SIMD level via SIMDConfig::set_level
// and combines two assertions:
// (a) Sanity vs ground truth at each level. Every returned value comes from
//     the input array; every returned id is a valid index; the returned set
//     does not include sentinels (FLT_MAX / -1). For EXACT_TOPK it must
//     equal the true top-k (modulo permutation within ties). For the
//     approximate bucket modes it must be a subset of the true top-k
//     candidates with bounded slack -- catches no-op or trivially broken
//     impls that the cross-level comparison alone would miss.
// (b) Cross-level equivalence to the SIMDLevel::NONE reference. The bucket
//     modes are documented as deliberate approximations (see
//     simdlib256-inl.h's bucket selection comment); we still expect the
//     concrete chosen set to match level-by-level for the inputs used here,
//     but if a future SIMD impl legitimately diverges within the
//     approximation contract, the (a) sanity checks remain protective.

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/impl/approx_topk/approx_topk.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/simd_levels.h>

using namespace faiss;

namespace {

struct Result {
    std::vector<float> val;
    std::vector<int32_t> ids;
    bool operator==(const Result& other) const {
        return val == other.val && ids == other.ids;
    }
};

template <uint32_t NBUCKETS, uint32_t N>
Result run_bs_addn(
        SIMDLevel level,
        uint32_t beam_size,
        uint32_t n_per_beam,
        uint32_t k,
        const std::vector<float>& distances) {
    SIMDConfig::set_level(level);
    Result out;
    out.val.assign(k, std::numeric_limits<float>::max());
    out.ids.assign(k, -1);
    HeapWithBuckets<CMax<float, int>, NBUCKETS, N>::bs_addn(
            beam_size,
            n_per_beam,
            distances.data(),
            k,
            out.val.data(),
            out.ids.data());
    return out;
}

Result run_approx_topk_by_mode(
        SIMDLevel level,
        ApproxTopK_mode_t mode,
        uint32_t beam_size,
        uint32_t n_per_beam,
        uint32_t k,
        const std::vector<float>& distances) {
    SIMDConfig::set_level(level);
    Result out;
    out.val.assign(k, std::numeric_limits<float>::max());
    out.ids.assign(k, -1);
    approx_topk_by_mode(
            mode,
            beam_size,
            n_per_beam,
            distances.data(),
            k,
            out.val.data(),
            out.ids.data());
    return out;
}

std::vector<float> make_random(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = u(rng);
    }
    return v;
}

std::vector<SIMDLevel> available_levels() {
    static const std::vector<SIMDLevel> all = {
            SIMDLevel::NONE,
            SIMDLevel::AVX2,
            SIMDLevel::AVX512,
            SIMDLevel::AVX512_SPR,
            SIMDLevel::ARM_NEON,
            SIMDLevel::ARM_SVE,
    };
    std::vector<SIMDLevel> out;
    for (auto lv : all) {
        if (SIMDConfig::is_simd_level_available(lv)) {
            out.push_back(lv);
        }
    }
    return out;
}

class SIMDLevelGuard {
   public:
    SIMDLevelGuard() : prev_(SIMDConfig::get_level()) {}
    ~SIMDLevelGuard() {
        SIMDConfig::set_level(prev_);
    }

   private:
    SIMDLevel prev_;
};

// True top-k by full sort (smallest-distance-first), returning the k chosen
// indices (in arbitrary order) and the k-th smallest distance value for
// recall-bound checks.
struct GroundTruth {
    std::unordered_set<int> top_k_ids;
    float kth_smallest;
};

GroundTruth ground_truth(const std::vector<float>& distances, uint32_t k) {
    std::vector<int> idx(distances.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
            idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
                return distances[a] < distances[b];
            });
    GroundTruth gt;
    for (uint32_t i = 0; i < k; ++i) {
        gt.top_k_ids.insert(idx[i]);
    }
    gt.kth_smallest = distances[idx[k - 1]];
    return gt;
}

// Sanity-check a Result: every entry's value comes from the input at the
// reported id, no entry is a sentinel, and ids are unique.
::testing::AssertionResult check_result_well_formed(
        const Result& r,
        const std::vector<float>& distances) {
    std::unordered_set<int> seen_ids;
    for (size_t i = 0; i < r.val.size(); ++i) {
        if (r.ids[i] < 0 || r.ids[i] >= static_cast<int>(distances.size())) {
            return ::testing::AssertionFailure()
                    << "id[" << i << "] = " << r.ids[i] << " out of range";
        }
        if (r.val[i] == std::numeric_limits<float>::max()) {
            return ::testing::AssertionFailure()
                    << "val[" << i << "] is FLT_MAX sentinel";
        }
        if (r.val[i] != distances[r.ids[i]]) {
            return ::testing::AssertionFailure()
                    << "val[" << i << "] = " << r.val[i]
                    << " does not match distances[ids[" << i << "]=" << r.ids[i]
                    << "] = " << distances[r.ids[i]];
        }
        if (!seen_ids.insert(r.ids[i]).second) {
            return ::testing::AssertionFailure() << "duplicate id " << r.ids[i];
        }
    }
    return ::testing::AssertionSuccess();
}

} // namespace

TEST(ApproxTopKDispatch, BsAddn_AllLevelsMatchNone) {
    SIMDLevelGuard guard;
    const auto levels = available_levels();
    if (levels.size() <= 1) {
        GTEST_SKIP() << "only one SIMD level available; nothing to compare";
    }

    constexpr uint32_t beam_size = 3;
    constexpr uint32_t n_per_beam = 200;
    constexpr uint32_t k = 16;
    auto distances = make_random(beam_size * n_per_beam, 42);

    // Approximation slack: HeapWithBuckets<NBUCKETS=8, N=3> retains the
    // top-3 of each of 8 buckets, i.e., k_bucketed = 24 candidates.
    // At k = 16 the result must lie within those 24, so every returned
    // value should be among the true top-24 distances. We use 2*k as a
    // generous upper bound for that pool size.
    constexpr uint32_t approx_k = 2 * k;
    auto gt = ground_truth(distances, approx_k);

    auto ref = run_bs_addn<8, 3>(
            SIMDLevel::NONE, beam_size, n_per_beam, k, distances);
    EXPECT_TRUE(check_result_well_formed(ref, distances));
    for (uint32_t i = 0; i < k; ++i) {
        EXPECT_LE(ref.val[i], gt.kth_smallest)
                << "ref entry " << i << " (val=" << ref.val[i]
                << ") exceeds true approx-top-" << approx_k;
    }
    for (auto lv : levels) {
        if (lv == SIMDLevel::NONE) {
            continue;
        }
        auto got = run_bs_addn<8, 3>(lv, beam_size, n_per_beam, k, distances);
        EXPECT_TRUE(check_result_well_formed(got, distances));
        EXPECT_EQ(got, ref)
                << "bs_addn diverged at level " << static_cast<int>(lv);
    }
}

TEST(ApproxTopKDispatch, ApproxTopKByMode_AllLevelsMatchNone) {
    SIMDLevelGuard guard;
    const auto levels = available_levels();
    if (levels.size() <= 1) {
        GTEST_SKIP() << "only one SIMD level available; nothing to compare";
    }

    constexpr uint32_t beam_size = 2;
    constexpr uint32_t n_per_beam = 500;
    constexpr uint32_t k = 32;
    auto distances = make_random(beam_size * n_per_beam, 7);

    const ApproxTopK_mode_t modes[] = {
            ApproxTopK_mode_t::EXACT_TOPK,
            ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3,
            ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D2,
            ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B16_D2,
            ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B32_D2,
    };
    auto exact_gt = ground_truth(distances, k);
    // For approximate modes the chosen set lies within a larger pool of
    // top candidates per bucket; 2*k is a generous upper bound.
    auto approx_gt = ground_truth(distances, 2 * k);

    for (auto mode : modes) {
        auto ref = run_approx_topk_by_mode(
                SIMDLevel::NONE, mode, beam_size, n_per_beam, k, distances);
        EXPECT_TRUE(check_result_well_formed(ref, distances))
                << "mode " << (int)mode;
        if (mode == ApproxTopK_mode_t::EXACT_TOPK) {
            // Must equal the true top-k as a set.
            std::unordered_set<int> ref_ids(ref.ids.begin(), ref.ids.end());
            EXPECT_EQ(ref_ids, exact_gt.top_k_ids);
        } else {
            // Each returned value must be at least as small as the true
            // 2k-th distance (an upper bound for any sensible bucketed
            // top-k).
            for (uint32_t i = 0; i < k; ++i) {
                EXPECT_LE(ref.val[i], approx_gt.kth_smallest)
                        << "mode " << (int)mode << " entry " << i;
            }
        }
        for (auto lv : levels) {
            if (lv == SIMDLevel::NONE) {
                continue;
            }
            auto got = run_approx_topk_by_mode(
                    lv, mode, beam_size, n_per_beam, k, distances);
            EXPECT_TRUE(check_result_well_formed(got, distances))
                    << "mode " << (int)mode << " level "
                    << static_cast<int>(lv);
            EXPECT_EQ(got, ref)
                    << "approx_topk_by_mode diverged at level "
                    << static_cast<int>(lv) << " mode " << (int)mode;
        }
    }
}
