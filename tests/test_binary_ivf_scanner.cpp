/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/utils/Heap.h>

namespace {

using C = faiss::CMax<int32_t, faiss::idx_t>;

// 160-bit codes, the width HammingComputer20 specializes.
constexpr int kCodeSize = 20;
constexpr int kDim = kCodeSize * 8;

std::vector<uint8_t> random_codes(std::mt19937& rng, size_t n) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> codes(n * kCodeSize);
    for (auto& b : codes) {
        b = static_cast<uint8_t>(byte(rng));
    }
    return codes;
}

std::vector<faiss::idx_t> sequential_ids(size_t n) {
    std::vector<faiss::idx_t> ids(n);
    for (size_t i = 0; i < n; i++) {
        ids[i] = static_cast<faiss::idx_t>(i) * 3 + 100;
    }
    return ids;
}

// A Hamming distance written from the definition, so it can disagree with the
// computer the scanner uses.
int hamming_of(const uint8_t* a, const uint8_t* b) {
    int dis = 0;
    for (int i = 0; i < kCodeSize; i++) {
        dis += std::popcount(static_cast<unsigned>(a[i] ^ b[i]));
    }
    return dis;
}

std::vector<int> expected_distances(
        const std::vector<uint8_t>& codes,
        const std::vector<uint8_t>& query,
        size_t n,
        size_t k,
        uint32_t radius) {
    std::vector<int> all;
    for (size_t i = 0; i < n; i++) {
        const int dis = hamming_of(query.data(), &codes[i * kCodeSize]);
        if (static_cast<uint32_t>(dis) < radius) {
            all.push_back(dis);
        }
    }
    std::sort(all.begin(), all.end());
    if (all.size() > k) {
        all.resize(k);
    }
    return all;
}

struct ScannerFixture {
    faiss::IndexBinaryFlat quantizer{kDim};
    std::unique_ptr<faiss::IndexBinaryIVF> ivf;
    std::unique_ptr<faiss::BinaryInvertedListScanner> scanner;

    ScannerFixture(std::mt19937& rng, const std::vector<uint8_t>& query) {
        const auto centroid = random_codes(rng, 1);
        quantizer.add(1, centroid.data());
        ivf = std::make_unique<faiss::IndexBinaryIVF>(&quantizer, kDim, 1);
        ivf->is_trained = true;
        scanner.reset(ivf->get_InvertedListScanner(false));
        scanner->set_query(query.data());
        scanner->set_list(0, 0);
    }
};

std::vector<int> run_scan(
        faiss::BinaryInvertedListScanner& scanner,
        const std::vector<uint8_t>& codes,
        const std::vector<faiss::idx_t>& ids,
        const std::vector<size_t>& slice_sizes,
        size_t k,
        std::optional<int32_t> radius = std::nullopt) {
    std::vector<int32_t> simi(k);
    std::vector<faiss::idx_t> idxi(k);
    faiss::heap_heapify<C>(k, simi.data(), idxi.data());
    if (radius.has_value()) {
        std::fill(simi.begin(), simi.end(), *radius);
    }

    size_t offset = 0;
    for (const size_t count : slice_sizes) {
        scanner.scan_codes(
                count,
                codes.data() + offset * kCodeSize,
                ids.data() + offset,
                simi.data(),
                idxi.data(),
                k);
        offset += count;
    }
    faiss::heap_reorder<C>(k, simi.data(), idxi.data());

    std::vector<int> found;
    for (size_t i = 0; i < k; i++) {
        if (idxi[i] >= 0) {
            found.push_back(simi[i]);
        }
    }
    return found;
}

TEST(BinaryIVFScanner, AnUnseededHeapKeepsTheKNearest) {
    std::mt19937 rng(1234);
    const size_t n = 2000;
    const auto query = random_codes(rng, 1);
    const auto codes = random_codes(rng, n);
    const auto ids = sequential_ids(n);
    ScannerFixture fixture(rng, query);

    const size_t k = 10;
    EXPECT_EQ(
            run_scan(*fixture.scanner, codes, ids, {n}, k),
            expected_distances(
                    codes, query, n, k, std::numeric_limits<uint32_t>::max()));
}

TEST(BinaryIVFScanner, ASeededHeapAppliesTheRadiusAndKTogether) {
    std::mt19937 rng(21);
    const size_t n = 4000;
    const auto query = random_codes(rng, 1);
    const auto codes = random_codes(rng, n);
    const auto ids = sequential_ids(n);

    for (const auto [k, radius] :
         {std::pair<size_t, int32_t>{64, 60}, {5, 70}}) {
        ScannerFixture fixture(rng, query);
        EXPECT_EQ(
                run_scan(*fixture.scanner, codes, ids, {n}, k, radius),
                expected_distances(
                        codes, query, n, k, static_cast<uint32_t>(radius)))
                << "k=" << k << " radius=" << radius;
    }
}

TEST(BinaryIVFScanner, ResultsAccumulateAcrossCalls) {
    std::mt19937 rng(7);
    const size_t n = 3000;
    const auto query = random_codes(rng, 1);
    const auto codes = random_codes(rng, n);
    const auto ids = sequential_ids(n);

    const size_t k = 16;
    const std::vector<std::vector<size_t>> layouts = {
            {1, n - 1}, {0, n}, {1000, 0, 1000, 1, 999}};

    for (const auto& layout : layouts) {
        ScannerFixture fixture(rng, query);
        EXPECT_EQ(
                run_scan(*fixture.scanner, codes, ids, layout, k, 75),
                expected_distances(codes, query, n, k, 75))
                << "slice count " << layout.size();
    }
}

TEST(BinaryIVFScanner, ASeededHeapIsExclusive) {
    std::mt19937 rng(31);
    const std::vector<uint8_t> query(kCodeSize, 0);
    const size_t n = 2;
    std::vector<uint8_t> codes(n * kCodeSize, 0);
    codes[0] = 0x0f;         // 4 bits set
    codes[kCodeSize] = 0x1f; // 5 bits set
    const auto ids = sequential_ids(n);

    ScannerFixture fixture(rng, query);
    const std::vector<int> expected{4};
    EXPECT_EQ(run_scan(*fixture.scanner, codes, ids, {n}, 10, 5), expected);

    ScannerFixture none(rng, query);
    EXPECT_TRUE(run_scan(*none.scanner, codes, ids, {n}, 10, 0).empty());

    // A seed below zero admits nothing rather than everything.
    ScannerFixture negative(rng, query);
    EXPECT_TRUE(run_scan(*negative.scanner, codes, ids, {n}, 10, -5).empty());
}

TEST(BinaryIVFScanner, ZeroKScansNothing) {
    std::mt19937 rng(51);
    const size_t n = 64;
    const auto query = random_codes(rng, 1);
    const auto codes = random_codes(rng, n);
    const auto ids = sequential_ids(n);

    // Real buffers, so dropping the guard fails the expectation below
    // instead of reading through a null pointer.
    std::vector<int32_t> simi{-1};
    std::vector<faiss::idx_t> idxi{-1};

    ScannerFixture fixture(rng, query);
    EXPECT_EQ(
            fixture.scanner->scan_codes(
                    n, codes.data(), ids.data(), simi.data(), idxi.data(), 0),
            0u);
    EXPECT_EQ(simi[0], -1);
    EXPECT_EQ(idxi[0], -1);
}

} // namespace
