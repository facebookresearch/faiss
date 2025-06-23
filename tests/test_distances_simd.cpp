/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
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

TEST(TestFvecL2sqr, distances_L2_squared_y_transposed) {
    // ints instead of floats for 100% accuracy
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    // modulo 8 results - 16 is to repeat the loop in the function
    int ny = 11; // this value will hit all the codepaths
    for (const auto d : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
        // initialize inputs
        std::vector<float> x(d);
        float x_sqlen = 0;
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = uniform(rng);
            x_sqlen += x[i] * x[i];
        }
        std::vector<float> y(d * ny);
        std::vector<float> y_sqlens(ny, 0);
        for (size_t i = 0; i < ny; i++) {
            for (size_t j = 0; j < y.size(); j++) {
                y[j] = uniform(rng);
                y_sqlens[i] += y[j] * y[j];
            }
        }

        // perform function
        std::vector<float> true_distances(ny, 0);
        for (size_t i = 0; i < ny; i++) {
            float dp = 0;
            for (size_t j = 0; j < d; j++) {
                dp += x[j] * y[i + j * ny];
            }
            true_distances[i] = x_sqlen + y_sqlens[i] - 2 * dp;
        }

        std::vector<float> distances(ny);
        faiss::fvec_L2sqr_ny_transposed(
                distances.data(),
                x.data(),
                y.data(),
                y_sqlens.data(),
                d,
                ny, // no need for special offset to test all lines of code
                ny);

        ASSERT_EQ(distances, true_distances)
                << "Mismatching fvec_L2sqr_ny_transposed results for d = " << d;
    }
}

TEST(TestFvecL2sqr, nearest_L2_squared_y_transposed) {
    // ints instead of floats for 100% accuracy
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    // modulo 8 results - 16 is to repeat the loop in the function
    int ny = 11; // this value will hit all the codepaths
    for (const auto d : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
        // initialize inputs
        std::vector<float> x(d);
        float x_sqlen = 0;
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = uniform(rng);
            x_sqlen += x[i] * x[i];
        }
        std::vector<float> y(d * ny);
        std::vector<float> y_sqlens(ny, 0);
        for (size_t i = 0; i < ny; i++) {
            for (size_t j = 0; j < y.size(); j++) {
                y[j] = uniform(rng);
                y_sqlens[i] += y[j] * y[j];
            }
        }

        // get distances
        std::vector<float> distances(ny, 0);
        for (size_t i = 0; i < ny; i++) {
            float dp = 0;
            for (size_t j = 0; j < d; j++) {
                dp += x[j] * y[i + j * ny];
            }
            distances[i] = x_sqlen + y_sqlens[i] - 2 * dp;
        }
        // find nearest
        size_t true_nearest_idx = 0;
        float min_dis = HUGE_VALF;
        for (size_t i = 0; i < ny; i++) {
            if (distances[i] < min_dis) {
                min_dis = distances[i];
                true_nearest_idx = i;
            }
        }

        std::vector<float> buffer(ny);
        size_t nearest_idx = faiss::fvec_L2sqr_ny_nearest_y_transposed(
                buffer.data(),
                x.data(),
                y.data(),
                y_sqlens.data(),
                d,
                ny, // no need for special offset to test all lines of code
                ny);

        ASSERT_EQ(nearest_idx, true_nearest_idx)
                << "Mismatching fvec_L2sqr_ny_nearest_y_transposed results for d = "
                << d;
    }
}

TEST(TestFvecL1, manhattan_distance) {
    // ints instead of floats for 100% accuracy
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    // modulo 8 results - 16 is to repeat the while loop in the function
    for (const auto nrows : {8, 9, 10, 11, 12, 13, 14, 15, 16}) {
        std::vector<float> x(nrows);
        std::vector<float> y(nrows);
        float true_distance = 0;
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = uniform(rng);
            y[i] = uniform(rng);
            true_distance += std::abs(x[i] - y[i]);
        }

        auto distance = faiss::fvec_L1(x.data(), y.data(), x.size());

        ASSERT_EQ(distance, true_distance)
                << "Mismatching fvec_Linf results for nrows = " << nrows;
    }
}

TEST(TestFvecLinf, chebyshev_distance) {
    // ints instead of floats for 100% accuracy
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    // modulo 8 results - 16 is to repeat the while loop in the function
    for (const auto nrows : {8, 9, 10, 11, 12, 13, 14, 15, 16}) {
        std::vector<float> x(nrows);
        std::vector<float> y(nrows);
        float true_distance = 0;
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = uniform(rng);
            y[i] = uniform(rng);
            true_distance = std::max(true_distance, std::abs(x[i] - y[i]));
        }

        auto distance = faiss::fvec_Linf(x.data(), y.data(), x.size());

        ASSERT_EQ(distance, true_distance)
                << "Mismatching fvec_Linf results for nrows = " << nrows;
    }
}

TEST(TestFvecMadd, multiple_add) {
    // ints instead of floats for 100% accuracy
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    // modulo 8 results - 16 is to repeat the while loop in the function
    for (const auto nrows : {8, 9, 10, 11, 12, 13, 14, 15, 16}) {
        std::vector<float> a(nrows);
        std::vector<float> b(nrows);
        const float bf = uniform(rng);
        std::vector<float> true_distances(nrows);
        for (size_t i = 0; i < a.size(); i++) {
            a[i] = uniform(rng);
            b[i] = uniform(rng);
            true_distances[i] = a[i] + bf * b[i];
        }

        std::vector<float> distances(nrows);
        faiss::fvec_madd(a.size(), a.data(), bf, b.data(), distances.data());

        ASSERT_EQ(distances, true_distances)
                << "Mismatching fvec_madd results for nrows = " << nrows;
    }
}

TEST(TestFvecAdd, add_array) {
    // ints instead of floats for 100% accuracy
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    for (const auto nrows : {1, 2, 5, 10, 15, 20, 25}) {
        std::vector<float> a(nrows);
        std::vector<float> b(nrows);
        std::vector<float> true_distances(nrows);
        for (size_t i = 0; i < a.size(); i++) {
            a[i] = uniform(rng);
            b[i] = uniform(rng);
            true_distances[i] = a[i] + b[i];
        }

        std::vector<float> distances(nrows);
        faiss::fvec_add(a.size(), a.data(), b.data(), distances.data());

        ASSERT_EQ(distances, true_distances)
                << "Mismatching array-array fvec_add results for nrows = "
                << nrows;
    }
}

TEST(TestFvecAdd, add_value) {
    // ints instead of floats for 100% accuracy
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    for (const auto nrows : {1, 2, 5, 10, 15, 20, 25}) {
        std::vector<float> a(nrows);
        const float b = uniform(rng); // value to add
        std::vector<float> true_distances(nrows);
        for (size_t i = 0; i < a.size(); i++) {
            a[i] = uniform(rng);
            true_distances[i] = a[i] + b;
        }

        std::vector<float> distances(nrows);
        faiss::fvec_add(a.size(), a.data(), b, distances.data());

        ASSERT_EQ(distances, true_distances)
                << "Mismatching array-value fvec_add results for nrows = "
                << nrows;
    }
}
