/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Correctness tests for the RISC-V Vector (RVV) uint16 partition and
// histogram kernels in utils/simd_impl/partitioning_rvv.cpp.
//
// with_simd_level_256bit() selects RISCV_RVV, so the call sites in
// partitioning.cpp reach these kernels for aligned uint16 arrays. The
// partition result is checked against its contract (kept prefix on the
// threshold's side, exact extreme-value multiset, ids still paired with
// their values) and the histograms against a local scalar implementation.
//
// This TU is compiled only on RISC-V, with the same -march as the
// library's RVV sources (see tests/CMakeLists.txt).

#ifdef COMPILE_SIMD_RISCV_RVV

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/partitioning.h>
#include <faiss/utils/simd_levels.h>

namespace {

using faiss::SIMDLevel;

constexpr SIMDLevel kRVV = SIMDLevel::RISCV_RVV;

// ---------------------------------------------------------------------------
// Partitioning / histogram
//
// with_simd_level_256bit() now selects RISCV_RVV, so the uint16 partition and
// histogram call sites in partitioning.cpp reach the kernels in
// utils/simd_impl/partitioning_rvv.cpp. They are checked against the
// partition invariant, the exact extreme-value multiset, and a local scalar
// histogram.
// ---------------------------------------------------------------------------

template <class C, typename IdT>
void expect_partition_property(
        size_t n,
        size_t q_min,
        size_t q_max,
        uint32_t seed,
        uint16_t max_value = 1023) {
    faiss::AlignedTable<uint16_t> vals(n);
    std::vector<IdT> ids(n);
    std::mt19937 rng(seed);
    for (size_t i = 0; i < n; i++) {
        vals[i] = max_value == 0 ? uint16_t(0)
                                 : uint16_t(rng() % (uint32_t(max_value) + 1));
        ids[i] = IdT(i);
    }
    const std::vector<uint16_t> original(vals.get(), vals.get() + n);

    size_t q_out = 0;
    const uint16_t thresh = faiss::partition_fuzzy_simd<kRVV, C>(
            vals.get(), ids.data(), n, q_min, q_max, &q_out);

    ASSERT_GE(q_out, q_min);
    ASSERT_LE(q_out, q_max);
    ASSERT_LE(q_out, n);

    // Partition invariant for the kept prefix: every value is on the
    // threshold's kept side (CMax keeps the small values, CMin the large
    // ones). The region past q_out is documented as destroyed, so it is
    // not inspected.
    for (size_t i = 0; i < q_out; i++) {
        EXPECT_FALSE(C::cmp(vals[i], thresh))
                << "prefix i=" << i << " v=" << vals[i] << " t=" << thresh;
    }

    // Ids must map back to the value that was moved with them, and the
    // prefix must be a set of distinct elements.
    std::vector<char> seen(n, 0);
    for (size_t i = 0; i < q_out; i++) {
        ASSERT_GE(ids[i], 0);
        ASSERT_LT(size_t(ids[i]), n);
        EXPECT_EQ(original[size_t(ids[i])], vals[i]);
        EXPECT_FALSE(seen[size_t(ids[i])]) << "duplicate id " << ids[i];
        seen[size_t(ids[i])] = 1;
    }

    // The prefix must be exactly the q_out smallest (CMax) / largest (CMin)
    // input values.
    std::vector<uint16_t> got(vals.get(), vals.get() + q_out);
    std::sort(got.begin(), got.end());
    std::vector<uint16_t> expected = original;
    std::sort(expected.begin(), expected.end());
    if (C::is_max) {
        expected.resize(q_out);
    } else {
        expected.erase(expected.begin(), expected.end() - q_out);
    }
    EXPECT_EQ(got, expected);
}

TEST(PartitioningRVV, PartitionFuzzy) {
    // Shapes around one vector and several full vectors, both id widths,
    // both comparators, exact and fuzzy q.
    expect_partition_property<faiss::CMax<uint16_t, int64_t>>(17, 5, 5, 200);
    expect_partition_property<faiss::CMax<uint16_t, int64_t>>(
            1000, 100, 100, 201);
    expect_partition_property<faiss::CMax<uint16_t, int>>(1025, 333, 333, 202);
    expect_partition_property<faiss::CMin<uint16_t, int64_t>>(
            1000, 100, 100, 203);
    expect_partition_property<faiss::CMin<uint16_t, int>>(1025, 512, 512, 204);

    // q_min < q_max: any q_out in the range is a valid partition.
    expect_partition_property<faiss::CMax<uint16_t, int64_t>>(
            1000, 100, 300, 205);
    expect_partition_property<faiss::CMin<uint16_t, int>>(1000, 100, 300, 206);

    // All values equal: the n_eq handling must keep exactly q of them.
    expect_partition_property<faiss::CMax<uint16_t, int64_t>>(
            1024, 300, 300, 207, /*max_value=*/0);
}

TEST(PartitioningRVV, PartitionDispatch) {
    // The public entry point must hit the same kernel (aligned uint16
    // input, with_simd_level_256bit now selects RISCV_RVV).
    using C = faiss::CMax<uint16_t, int64_t>;
    const size_t n = 512, q = 77;
    faiss::AlignedTable<uint16_t> vals(n);
    std::vector<int64_t> ids(n);
    std::mt19937 rng(210);
    for (size_t i = 0; i < n; i++) {
        vals[i] = uint16_t(rng() % 4096);
        ids[i] = int64_t(i);
    }

    size_t q_out_public = 0;
    const uint16_t thresh_public = faiss::partition_fuzzy<C>(
            vals.get(), ids.data(), n, q, q, &q_out_public);

    faiss::AlignedTable<uint16_t> vals_ref(n);
    std::vector<int64_t> ids_ref(n);
    for (size_t i = 0; i < n; i++) {
        vals_ref[i] = vals[i];
        ids_ref[i] = ids[i];
    }
    size_t q_out_direct = 0;
    const uint16_t thresh_direct = faiss::partition_fuzzy_simd<kRVV, C>(
            vals_ref.get(), ids_ref.data(), n, q, q, &q_out_direct);

    EXPECT_EQ(q_out_public, q_out_direct);
    EXPECT_EQ(thresh_public, thresh_direct);
    EXPECT_EQ(
            std::vector<uint16_t>(vals.get(), vals.get() + q),
            std::vector<uint16_t>(vals_ref.get(), vals_ref.get() + q));
}

void scalar_histogram_8(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    std::fill(hist, hist + 8, 0);
    for (int i = 0; i < n; i++) {
        if (data[i] < min) {
            continue;
        }
        const uint16_t v = uint16_t((data[i] - min) >> shift);
        if (v < 8) {
            hist[v]++;
        }
    }
}

void scalar_histogram_16(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    std::fill(hist, hist + 16, 0);
    const int vmax0 = std::min((16 << shift) + int(min), 65536);
    const uint16_t vmax = uint16_t(vmax0 - 1 - int(min));
    for (int i = 0; i < n; i++) {
        uint16_t v = uint16_t(data[i] - min);
        if (!(v <= vmax)) {
            continue;
        }
        v = uint16_t(v >> shift);
        hist[v]++;
    }
}

TEST(PartitioningRVV, SimdHistogram) {
    // Call the RVV specializations directly: the plain simd_histogram_16()
    // has a GCC 12 scalar fallback, and the dispatch itself is covered by
    // DispatchSelectsRVV / PartitionDispatch.
    const size_t n = 5000;
    faiss::AlignedTable<uint16_t> tab(n);
    std::mt19937 rng(220);
    for (size_t i = 0; i < n; i++) {
        tab[i] = uint16_t(rng() & 0xFFFF);
    }

    for (int shift : {0, 3, 7, 12}) {
        int rvv[16] = {0};
        int ref[16] = {0};

        faiss::simd_histogram_8<kRVV>(tab.get(), int(n), 0, shift, rvv);
        scalar_histogram_8(tab.get(), int(n), 0, shift, ref);
        EXPECT_EQ(
                std::vector<int>(rvv, rvv + 8), std::vector<int>(ref, ref + 8))
                << "histogram_8 shift=" << shift;

        faiss::simd_histogram_16<kRVV>(tab.get(), int(n), 0, shift, rvv);
        scalar_histogram_16(tab.get(), int(n), 0, shift, ref);
        EXPECT_EQ(
                std::vector<int>(rvv, rvv + 16),
                std::vector<int>(ref, ref + 16))
                << "histogram_16 shift=" << shift;
    }

    // Non-zero min: values below it are ignored.
    const uint16_t min = 1000;
    for (int shift : {0, 5}) {
        int rvv[16] = {0};
        int ref[16] = {0};
        faiss::simd_histogram_8<kRVV>(tab.get(), int(n), min, shift, rvv);
        scalar_histogram_8(tab.get(), int(n), min, shift, ref);
        EXPECT_EQ(
                std::vector<int>(rvv, rvv + 8), std::vector<int>(ref, ref + 8))
                << "histogram_8 min shift=" << shift;

        faiss::simd_histogram_16<kRVV>(tab.get(), int(n), min, shift, rvv);
        scalar_histogram_16(tab.get(), int(n), min, shift, ref);
        EXPECT_EQ(
                std::vector<int>(rvv, rvv + 16),
                std::vector<int>(ref, ref + 16))
                << "histogram_16 min shift=" << shift;
    }
}

} // namespace

#endif // COMPILE_SIMD_RISCV_RVV
