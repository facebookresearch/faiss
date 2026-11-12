/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

// Pull in inline NONE specializations so the test can call
// PQCodeDistance<PQDecoder8, SIMDLevel::NONE> directly.
#include <faiss/impl/pq_code_distance/pq_code_distance-generic.h>

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
        // number of subquantizers
        const size_t subq,
        // number of codes to process
        const size_t n) {
    std::minstd_rand rng(123);
    std::uniform_real_distribution<float> uf(0, 1);

    std::vector<float> lookup(256 * subq, 0);
    for (size_t i = 0; i < lookup.size(); i++) {
        lookup[i] = uf(rng);
    }

    std::vector<uint8_t> codes(n * subq);
#pragma omp parallel
    {
        std::minstd_rand rng0(123);
        std::uniform_int_distribution<int> u1(0, 255);

#pragma omp for schedule(guided)
        for (int64_t i = 0; i < (int64_t)codes.size(); i++) {
            codes[i] = u1(rng0);
        }
    }

    // reference results using explicit scalar
    std::vector<float> resultsRef(n, 0);
#pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < (int64_t)n; i++) {
        resultsRef[i] = faiss::PQCodeDistance<
                faiss::PQDecoder8,
                faiss::SIMDLevel::NONE>::
                distance_single_code(
                        subq, 8, lookup.data(), codes.data() + subq * i);
    }

    // scalar, 4 codes per step
    std::vector<float> resultsScalar4x(n, 0);
#pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < (int64_t)n; i += 4) {
        faiss::PQCodeDistance<faiss::PQDecoder8, faiss::SIMDLevel::NONE>::
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

    // dispatched, 1 code per step (best available SIMD level)
    std::vector<float> resultsDispatched1x(n, 0);
#pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < (int64_t)n; i++) {
        resultsDispatched1x[i] = faiss::pq_code_distance_8bit_single(
                subq, lookup.data(), codes.data() + subq * i);
    }

    // dispatched, 4 codes per step (best available SIMD level)
    std::vector<float> resultsDispatched4x(n, 0);
#pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < (int64_t)n; i += 4) {
        faiss::pq_code_distance_8bit_four(
                subq,
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

    ASSERT_EQ(nMismatches(resultsRef, resultsScalar4x), 0);
    ASSERT_EQ(nMismatches(resultsRef, resultsDispatched1x), 0);
    ASSERT_EQ(nMismatches(resultsRef, resultsDispatched4x), 0);
}

// M = 4 and 8 have dedicated AVX2 kernels, 16 and 32 fill the 16-wide blocks,
// and the other values exercise the leftover loops of the SIMD kernels.
TEST(TestCodeDistance, NBITS8) {
    for (size_t subq : {1, 4, 5, 8, 9, 15, 16, 32}) {
        SCOPED_TRACE(subq);
        test(subq, 10000);
    }
}
