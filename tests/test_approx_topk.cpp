/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <faiss/utils/approx_topk/approx_topk.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/Heap.h>

//
using namespace faiss;

//
template <uint32_t NBUCKETS, uint32_t N>
void test_approx_topk(
        const uint32_t beamSize,
        const uint32_t nPerBeam,
        const uint32_t k,
        const uint32_t nDatasetsToTest,
        const bool verbose) {
    if (verbose) {
        printf("-----------\n");
    }

    // generate random data
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // matches
    size_t nMatches = 0;
    // the element was completely missed in approx version.
    size_t nMissed = 0;
    // the element is available
    size_t nAvailable = 0;
    // the distance is the same, but the index is different.
    size_t nSoftMismatches = 0;
    // the distances are different
    size_t nHardMismatches = 0;
    // error of distances
    double sqrError = 0.0;

    //
    double timeBaseline = 0.0;
    double timeApprox = 0.0;

    for (size_t iDataset = 0; iDataset < nDatasetsToTest; iDataset++) {
        const size_t n = (size_t)(nPerBeam)*beamSize;
        std::vector<float> distances(n, 0);
        for (size_t i = 0; i < n; i++) {
            distances[i] = u(rng);
        }

        //
        using C = CMax<float, int>;

        // do a regular beam search
        std::vector<float> baselineDistances(k, C::neutral());
        std::vector<int> baselineIndices(k, -1);

        auto startBaseline = std::chrono::high_resolution_clock::now();
        heap_addn<C>(
                k,
                baselineDistances.data(),
                baselineIndices.data(),
                distances.data(),
                nullptr,
                nPerBeam * beamSize);
        auto endBaseline = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffBaseline =
                endBaseline - startBaseline;
        timeBaseline += diffBaseline.count();

        heap_reorder<C>(k, baselineDistances.data(), baselineIndices.data());

        // do an approximate beam search
        std::vector<float> approxDistances(k, C::neutral());
        std::vector<int> approxIndices(k, -1);

        auto startApprox = std::chrono::high_resolution_clock::now();
        try {
            HeapWithBuckets<C, NBUCKETS, N>::bs_addn(
                    beamSize,
                    nPerBeam,
                    distances.data(),
                    k,
                    approxDistances.data(),
                    approxIndices.data());
        } catch (const faiss::FaissException&) {
            //
            if (verbose) {
                printf("Skipping the case.\n");
            }
            return;
        }

        auto endApprox = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffApprox = endApprox - startApprox;
        timeApprox += diffApprox.count();

        heap_reorder<C>(k, approxDistances.data(), approxIndices.data());

        bool bGotMismatches = false;

        // the error
        for (uint32_t i = 0; i < k; i++) {
            if (baselineDistances[i] != approxDistances[i]) {
                nHardMismatches += 1;

                double diff = baselineDistances[i] - approxDistances[i];
                sqrError += diff * diff;

                bGotMismatches = true;

                if (verbose) {
                    printf("i=%d, bs.d=%f, bs.i=%d, app.d=%f, app.i=%d\n",
                           i,
                           baselineDistances[i],
                           baselineIndices[i],
                           approxDistances[i],
                           approxIndices[i]);
                }
            } else {
                if (baselineIndices[i] != approxIndices[i]) {
                    nSoftMismatches += 1;
                } else {
                    nMatches += 1;
                }
            }
        }

        if (bGotMismatches) {
            if (verbose) {
                printf("\n");
            }
        }

        //
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

    if (verbose) {
        printf("%d, %d, %d, %d, %d, %d: %ld, %ld, %ld, %f, %ld, %ld, %f, %f\n",
               NBUCKETS,
               N,
               beamSize,
               nPerBeam,
               k,
               nDatasetsToTest,
               nMatches,
               nSoftMismatches,
               nHardMismatches,
               sqrError,
               nAvailable,
               nMissed,
               timeBaseline,
               timeApprox);
    }

    // just confirm that the error is not crazy
    if (NBUCKETS * N * beamSize >= k) {
        EXPECT_TRUE(nAvailable > nMissed);
    } else {
        // it is possible that the results are crazy here. Skip it.
    }
}

//
TEST(testApproxTopk, COMMON) {
    constexpr bool verbose = false;

    //
    const uint32_t nDifferentDatasets = 8;

    uint32_t kValues[] = {1, 2, 3, 5, 8, 13, 21, 34};

    for (size_t codebookBitSize = 8; codebookBitSize <= 10; codebookBitSize++) {
        const uint32_t codebookSize = 1 << codebookBitSize;
        for (const auto k : kValues) {
            test_approx_topk<1 * 8, 3>(
                    1, codebookSize, k, nDifferentDatasets, verbose);
            test_approx_topk<1 * 8, 3>(
                    k, codebookSize, k, nDifferentDatasets, verbose);

            test_approx_topk<1 * 8, 2>(
                    1, codebookSize, k, nDifferentDatasets, verbose);
            test_approx_topk<1 * 8, 2>(
                    k, codebookSize, k, nDifferentDatasets, verbose);

            test_approx_topk<2 * 8, 2>(
                    1, codebookSize, k, nDifferentDatasets, verbose);
            test_approx_topk<2 * 8, 2>(
                    k, codebookSize, k, nDifferentDatasets, verbose);

            test_approx_topk<4 * 8, 2>(
                    1, codebookSize, k, nDifferentDatasets, verbose);
            test_approx_topk<4 * 8, 2>(
                    k, codebookSize, k, nDifferentDatasets, verbose);
        }
    }
}

//
