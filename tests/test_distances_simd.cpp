/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <faiss/utils/distances.h>

// reference implementations
void fvec_inner_products_ny_ref(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        ip[i] = faiss::fvec_inner_product(x, y, d);
        y += d;
    }
}

void fvec_L2sqr_ny_ref(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = faiss::fvec_L2sqr(x, y, d);
        y += d;
    }
}

// test templated versions of fvec_L2sqr_ny
TEST(TestFvecL2sqrNy, D2) {
    // we're using int values in order to get 100% accurate
    // results with floats.
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> u(0, 32);

    for (const auto dim : {2, 4, 8, 12}) {
        std::vector<float> x(dim, 0);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = u(rng);
        }

        for (const auto nrows : {1, 2, 5, 10, 15, 20, 25}) {
            std::vector<float> y(nrows * dim);
            for (size_t i = 0; i < y.size(); i++) {
                y[i] = u(rng);
            }

            std::vector<float> distances(nrows, 0);
            faiss::fvec_L2sqr_ny(
                    distances.data(), x.data(), y.data(), dim, nrows);

            std::vector<float> distances_ref(nrows, 0);
            fvec_L2sqr_ny_ref(
                    distances_ref.data(), x.data(), y.data(), dim, nrows);

            ASSERT_EQ(distances, distances_ref)
                    << "Mismatching results for dim = " << dim
                    << ", nrows = " << nrows;
        }
    }
}

// fvec_inner_products_ny
TEST(TestFvecInnerProductsNy, D2) {
    // we're using int values in order to get 100% accurate
    // results with floats.
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> u(0, 32);

    for (const auto dim : {2, 4, 8, 12}) {
        std::vector<float> x(dim, 0);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = u(rng);
        }

        for (const auto nrows : {1, 2, 5, 10, 15, 20, 25}) {
            std::vector<float> y(nrows * dim);
            for (size_t i = 0; i < y.size(); i++) {
                y[i] = u(rng);
            }

            std::vector<float> distances(nrows, 0);
            faiss::fvec_inner_products_ny(
                    distances.data(), x.data(), y.data(), dim, nrows);

            std::vector<float> distances_ref(nrows, 0);
            fvec_inner_products_ny_ref(
                    distances_ref.data(), x.data(), y.data(), dim, nrows);

            ASSERT_EQ(distances, distances_ref)
                    << "Mismatching results for dim = " << dim
                    << ", nrows = " << nrows;
        }
    }
}
