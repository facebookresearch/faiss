/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/pq_code_distance.h>

size_t nMismatches(
        const std::vector<float>& ref,
        const std::vector<float>& candidate) {
    size_t count = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double abs = std::abs(ref[i] - candidate[i]);
        if (abs >= 1e-5) {
            count += 1;
        }
    }

    return count;
}

void test(
        // dimensionality of the data
        const size_t dim,
        // number of subquantizers
        const size_t subq,
        // bits per subquantizer
        const size_t nbits,
        // number of codes to process
        const size_t n) {
    FAISS_THROW_IF_NOT(nbits == 8);

    // remove if benchmarking is needed
    omp_set_num_threads(1);

    // rng
    std::minstd_rand rng(123);
    std::uniform_int_distribution<uint8_t> u(0, 255);
    std::uniform_real_distribution<float> uf(0, 1);

    // initialize lookup
    std::vector<float> lookup(256 * subq, 0);
    for (size_t i = 0; i < lookup.size(); i++) {
        lookup[i] = uf(rng);
    }

    // initialize codes
    std::vector<uint8_t> codes(n * subq);
#pragma omp parallel
    {
        std::minstd_rand rng0(123);
        std::uniform_int_distribution<uint8_t> u1(0, 255);

#pragma omp for schedule(guided)
        for (size_t i = 0; i < codes.size(); i++) {
            codes[i] = u1(rng0);
        }
    }

    // warmup. compute reference results using explicit scalar
    std::vector<float> resultsRef(n, 0);
    for (size_t k = 0; k < 10; k++) {
#pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < n; i++) {
            resultsRef[i] = faiss::PQCodeDistance<
                    faiss::PQDecoder8,
                    faiss::SIMDLevel::NONE>::
                    distance_single_code(
                            subq, 8, lookup.data(), codes.data() + subq * i);
        }
    }

    // scalar, 1 code per step
    std::vector<float> resultsScalar1x(n, 0);
    double scalar1xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i++) {
                resultsScalar1x[i] = faiss::PQCodeDistance<
                        faiss::PQDecoder8,
                        faiss::SIMDLevel::NONE>::
                        distance_single_code(
                                subq,
                                8,
                                lookup.data(),
                                codes.data() + subq * i);
            }
        }
        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        scalar1xMsec = (duration.count() * 1000.0);
    }

    // scalar, 4 codes per step
    std::vector<float> resultsScalar4x(n, 0);
    double scalar4xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i += 4) {
                faiss::PQCodeDistance<
                        faiss::PQDecoder8,
                        faiss::SIMDLevel::NONE>::
                        distance_four_codes(
                                subq,
                                8,
                                lookup.data(),
                                codes.data() + subq * (i + 0),
                                codes.data() + subq * (i + 1),
                                codes.data() + subq * (i + 2),
                                codes.data() + subq * (i + 3),
                                resultsScalar4x[i + 0],
                                resultsScalar4x[i + 1],
                                resultsScalar4x[i + 2],
                                resultsScalar4x[i + 3]);
            }
        }

        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        scalar4xMsec = (duration.count() * 1000.0);
    }

    // dispatched, 1 code per step (best available SIMD level)
    std::vector<float> resultsDispatched1x(n, 0);
    double dispatched1xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i++) {
                resultsDispatched1x[i] = faiss::pq_code_distance_single(
                        subq, 8, lookup.data(), codes.data() + subq * i);
            }
        }
        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        dispatched1xMsec = (duration.count() * 1000.0);
    }

    // dispatched, 4 codes per step (best available SIMD level)
    std::vector<float> resultsDispatched4x(n, 0);
    double dispatched4xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i += 4) {
                faiss::pq_code_distance_four(
                        subq,
                        8,
                        lookup.data(),
                        codes.data() + subq * (i + 0),
                        codes.data() + subq * (i + 1),
                        codes.data() + subq * (i + 2),
                        codes.data() + subq * (i + 3),
                        resultsDispatched4x[i + 0],
                        resultsDispatched4x[i + 1],
                        resultsDispatched4x[i + 2],
                        resultsDispatched4x[i + 3]);
            }
        }

        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        dispatched4xMsec = (duration.count() * 1000.0);
    }

    const size_t nMismatchesS1 = nMismatches(resultsRef, resultsScalar1x);
    const size_t nMismatchesS4 = nMismatches(resultsRef, resultsScalar4x);
    const size_t nMismatchesD1 = nMismatches(resultsRef, resultsDispatched1x);
    const size_t nMismatchesD4 = nMismatches(resultsRef, resultsDispatched4x);

    std::cout << "Dim = " << dim << ", subq = " << subq << ", nbits = " << nbits
              << ", n = " << n << std::endl;
    std::cout << "Scalar 1x code: " << scalar1xMsec << " msec, "
              << nMismatchesS1 << " mismatches" << std::endl;
    std::cout << "Scalar 4x code: " << scalar4xMsec << " msec, "
              << nMismatchesS4 << " mismatches" << std::endl;
    std::cout << "Dispatched 1x code: " << dispatched1xMsec << " msec, "
              << nMismatchesD1 << " mismatches" << std::endl;
    std::cout << "Dispatched 4x code: " << dispatched4xMsec << " msec, "
              << nMismatchesD4 << " mismatches" << std::endl;
    std::cout << std::endl;

    ASSERT_EQ(nMismatchesS1, 0);
    ASSERT_EQ(nMismatchesS4, 0);
    ASSERT_EQ(nMismatchesD1, 0);
    ASSERT_EQ(nMismatchesD4, 0);
}

// this test can be used as a benchmark.
// 1. Increase the value of NELEMENTS
// 2. Remove omp_set_num_threads()

constexpr size_t NELEMENTS = 10000;

TEST(TestCodeDistance, SUBQ4_NBITS8) {
    test(256, 4, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ8_NBITS8) {
    test(256, 8, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ16_NBITS8) {
    test(256, 16, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ32_NBITS8) {
    test(256, 32, 8, NELEMENTS);
}

// Odd M values exercise the scalar tail within SIMD implementations.
// AVX2/AVX512 process M in fixed-size chunks; these values ensure the
// remainder path is correct.

TEST(TestCodeDistance, SUBQ1_NBITS8) {
    test(256, 1, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ3_NBITS8) {
    test(256, 3, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ5_NBITS8) {
    test(256, 5, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ7_NBITS8) {
    test(256, 7, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ9_NBITS8) {
    test(256, 9, 8, NELEMENTS);
}

TEST(TestCodeDistance, SUBQ15_NBITS8) {
    test(256, 15, 8, NELEMENTS);
}
