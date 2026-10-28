/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/impl/approx_topk/approx_topk.h>

#include <faiss/utils/Heap.h>

using namespace faiss;

template <uint32_t NBUCKETS, uint32_t N>
void test_approx_topk(
        const uint32_t beamSize,
        const uint32_t nPerBeam,
        const uint32_t k,
        const uint32_t nDatasetsToTest) {
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // the element was completely missed in approx version.
    size_t nMissed = 0;
    // the element is available
    size_t nAvailable = 0;

    for (size_t iDataset = 0; iDataset < nDatasetsToTest; iDataset++) {
        const size_t n = (size_t)(nPerBeam)*beamSize;
        std::vector<float> distances(n, 0);
        for (size_t i = 0; i < n; i++) {
            distances[i] = u(rng);
        }

        using C = CMax<float, int>;

        std::vector<float> baselineDistances(k, C::neutral());
        std::vector<int> baselineIndices(k, -1);

        heap_addn<C>(
                k,
                baselineDistances.data(),
                baselineIndices.data(),
                distances.data(),
                nullptr,
                nPerBeam * beamSize);

        std::vector<float> approxDistances(k, C::neutral());
        std::vector<int> approxIndices(k, -1);

        HeapWithBuckets<C, NBUCKETS, N>::bs_addn(
                beamSize,
                nPerBeam,
                distances.data(),
                k,
                approxDistances.data(),
                approxIndices.data());

        std::unordered_set<int> bsIndicesHS(
                baselineIndices.cbegin(), baselineIndices.cend());
        for (uint32_t i = 0; i < k; i++) {
            auto itr = bsIndicesHS.find(approxIndices[i]);
            if (itr != bsIndicesHS.cend()) {
                nAvailable += 1;
            } else {
                nMissed += 1;
            }
        }
    }

    // just confirm that the error is not crazy
    if (NBUCKETS * N * beamSize >= k) {
        EXPECT_TRUE(nAvailable > nMissed);
    }
}

TEST(testApproxTopk, COMMON) {
    const uint32_t nDifferentDatasets = 8;

    uint32_t kValues[] = {1, 2, 3, 5, 8, 13, 21, 34};

    const uint32_t codebookSize = 256;
    for (const auto k : kValues) {
        test_approx_topk<1 * 8, 3>(1, codebookSize, k, nDifferentDatasets);
        test_approx_topk<1 * 8, 3>(k, codebookSize, k, nDifferentDatasets);

        test_approx_topk<1 * 8, 2>(1, codebookSize, k, nDifferentDatasets);
        test_approx_topk<1 * 8, 2>(k, codebookSize, k, nDifferentDatasets);

        test_approx_topk<2 * 8, 2>(1, codebookSize, k, nDifferentDatasets);
        test_approx_topk<2 * 8, 2>(k, codebookSize, k, nDifferentDatasets);

        test_approx_topk<4 * 8, 2>(1, codebookSize, k, nDifferentDatasets);
        test_approx_topk<4 * 8, 2>(k, codebookSize, k, nDifferentDatasets);
    }
}
