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
#include <memory>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/code_distance/code_distance.h>

size_t nMismatches(
        const std::vector<float>& ref,
        const std::vector<float>& candidate) {
    size_t count = 0;
    for (size_t i = 0; i < count; i++) {
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

    // warmup. compute reference results
    std::vector<float> resultsRef(n, 0);
    for (size_t k = 0; k < 10; k++) {
#pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < n; i++) {
            resultsRef[i] =
                    faiss::distance_single_code_generic<faiss::PQDecoder8>(
                            subq, 8, lookup.data(), codes.data() + subq * i);
        }
    }

    // generic, 1 code per step
    std::vector<float> resultsNewGeneric1x(n, 0);
    double generic1xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i++) {
                resultsNewGeneric1x[i] =
                        faiss::distance_single_code_generic<faiss::PQDecoder8>(
                                subq,
                                8,
                                lookup.data(),
                                codes.data() + subq * i);
            }
        }
        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        generic1xMsec = (duration.count() * 1000.0);
    }

    // generic, 4 codes per step
    std::vector<float> resultsNewGeneric4x(n, 0);
    double generic4xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i += 4) {
                faiss::distance_four_codes_generic<faiss::PQDecoder8>(
                        subq,
                        8,
                        lookup.data(),
                        codes.data() + subq * (i + 0),
                        codes.data() + subq * (i + 1),
                        codes.data() + subq * (i + 2),
                        codes.data() + subq * (i + 3),
                        resultsNewGeneric4x[i + 0],
                        resultsNewGeneric4x[i + 1],
                        resultsNewGeneric4x[i + 2],
                        resultsNewGeneric4x[i + 3]);
            }
        }

        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        generic4xMsec = (duration.count() * 1000.0);
    }

    // generic, 1 code per step
    std::vector<float> resultsNewCustom1x(n, 0);
    double custom1xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i++) {
                resultsNewCustom1x[i] =
                        faiss::distance_single_code<faiss::PQDecoder8>(
                                subq,
                                8,
                                lookup.data(),
                                codes.data() + subq * i);
            }
        }
        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        custom1xMsec = (duration.count() * 1000.0);
    }

    // generic, 4 codes per step
    std::vector<float> resultsNewCustom4x(n, 0);
    double custom4xMsec = 0;
    {
        const auto startingTimepoint = std::chrono::steady_clock::now();
        for (size_t k = 0; k < 1000; k++) {
#pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < n; i += 4) {
                faiss::distance_four_codes<faiss::PQDecoder8>(
                        subq,
                        8,
                        lookup.data(),
                        codes.data() + subq * (i + 0),
                        codes.data() + subq * (i + 1),
                        codes.data() + subq * (i + 2),
                        codes.data() + subq * (i + 3),
                        resultsNewCustom4x[i + 0],
                        resultsNewCustom4x[i + 1],
                        resultsNewCustom4x[i + 2],
                        resultsNewCustom4x[i + 3]);
            }
        }

        const auto endingTimepoint = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration =
                endingTimepoint - startingTimepoint;
        custom4xMsec = (duration.count() * 1000.0);
    }

    const size_t nMismatchesG1 = nMismatches(resultsRef, resultsNewGeneric1x);
    const size_t nMismatchesG4 = nMismatches(resultsRef, resultsNewGeneric4x);
    const size_t nMismatchesCustom1 =
            nMismatches(resultsRef, resultsNewCustom1x);
    const size_t nMismatchesCustom4 =
            nMismatches(resultsRef, resultsNewCustom4x);

    std::cout << "Dim = " << dim << ", subq = " << subq << ", nbits = " << nbits
              << ", n = " << n << std::endl;
    std::cout << "Generic 1x code: " << generic1xMsec << " msec, "
              << nMismatchesG1 << " mismatches" << std::endl;
    std::cout << "Generic 4x code: " << generic4xMsec << " msec, "
              << nMismatchesG4 << " mismatches" << std::endl;
    std::cout << "custom 1x code: " << custom1xMsec << " msec, "
              << nMismatchesCustom1 << " mismatches" << std::endl;
    std::cout << "custom 4x code: " << custom4xMsec << " msec, "
              << nMismatchesCustom4 << " mismatches" << std::endl;
    std::cout << std::endl;

    ASSERT_EQ(nMismatchesG1, 0);
    ASSERT_EQ(nMismatchesG4, 0);
    ASSERT_EQ(nMismatchesCustom1, 0);
    ASSERT_EQ(nMismatchesCustom4, 0);
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
