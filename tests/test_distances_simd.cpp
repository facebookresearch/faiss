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
static void fvec_inner_products_ny_ref(
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

static void fvec_L2sqr_ny_ref(
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
                << "Mismatching fvec_L1 results for nrows = " << nrows;
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


// Test batched L2 distance with threshold - correctness without early abort
TEST(TestFvecL2sqrBatched, correctness_no_abort) {
    std::default_random_engine rng(456);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    for (const auto d : {16, 32, 64, 128, 256, 512, 1024, 2048}) {
        for (const auto batch_size : {4, 8, 16, 32, 64}) {
            std::vector<float> x(d);
            std::vector<float> y(d);
            for (size_t i = 0; i < d; i++) {
                x[i] = uniform(rng);
                y[i] = uniform(rng);
            }
            float expected = faiss::fvec_L2sqr(x.data(), y.data(), d);
            float result = faiss::fvec_L2sqr_batched(
                    x.data(), y.data(), d, batch_size, 1e10f);
            ASSERT_FLOAT_EQ(result, expected)
                    << "d=" << d << ", batch_size=" << batch_size;
        }
    }
}

// Test that early abort triggers and returns a partial result
TEST(TestFvecL2sqrBatched, early_abort) {
    std::default_random_engine rng(789);
    std::uniform_int_distribution<int32_t> uniform(10, 50);
    const size_t d = 1024;
    const size_t batch_size = 16;
    std::vector<float> x(d);
    std::vector<float> y(d);
    for (size_t i = 0; i < d; i++) {
        x[i] = uniform(rng);
        y[i] = uniform(rng) + 100.0f;
    }
    float full_distance = faiss::fvec_L2sqr(x.data(), y.data(), d);
    float threshold = full_distance * 0.1f;
    float partial_distance = faiss::fvec_L2sqr_batched(
            x.data(), y.data(), d, batch_size, threshold);
    ASSERT_GT(partial_distance, threshold);
    ASSERT_LT(partial_distance, full_distance);
}

// Test with zero vectors (distance should be 0)
TEST(TestFvecL2sqrBatched, zero_vectors) {
    const size_t d = 64;
    const size_t batch_size = 16;
    std::vector<float> zeros(d, 0.0f);
    float result = faiss::fvec_L2sqr_batched(
            zeros.data(), zeros.data(), d, batch_size, 1e10f);
    ASSERT_FLOAT_EQ(result, 0.0f);
}


// Test batched inner product - correctness without early abort (normalized vecs)
TEST(TestFvecInnerProductBatched, correctness_no_abort) {
    std::default_random_engine rng(101);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (const auto d : {16, 32, 64, 128, 256, 512, 1024}) {
        for (const auto batch_size : {4, 8, 16, 32, 64}) {
            std::vector<float> x(d);
            std::vector<float> y(d);
            // Generate and normalize vectors
            float norm_x = 0.0f, norm_y = 0.0f;
            for (size_t i = 0; i < d; i++) {
                x[i] = normal(rng);
                y[i] = normal(rng);
                norm_x += x[i] * x[i];
                norm_y += y[i] * y[i];
            }
            norm_x = std::sqrt(norm_x);
            norm_y = std::sqrt(norm_y);
            for (size_t i = 0; i < d; i++) {
                x[i] /= norm_x;
                y[i] /= norm_y;
            }

            float expected =
                    faiss::fvec_inner_product(x.data(), y.data(), d);
            // Use very low threshold so no abort happens
            float result = faiss::fvec_inner_product_batched(
                    x.data(), y.data(), d, batch_size, -1e10f);
            ASSERT_NEAR(result, expected, 1e-5)
                    << "d=" << d << ", batch_size=" << batch_size;
        }
    }
}

// Test that early abort triggers for dissimilar normalized vectors
TEST(TestFvecInnerProductBatched, early_abort) {
    const size_t d = 1024;
    const size_t batch_size = 16;
    // Create two orthogonal-ish normalized vectors
    std::vector<float> x(d, 0.0f);
    std::vector<float> y(d, 0.0f);
    // x is concentrated in first half, y in second half
    float norm = std::sqrt(static_cast<float>(d / 2));
    for (size_t i = 0; i < d / 2; i++) {
        x[i] = 1.0f / norm;
    }
    for (size_t i = d / 2; i < d; i++) {
        y[i] = 1.0f / norm;
    }

    float full_ip = faiss::fvec_inner_product(x.data(), y.data(), d);
    // Set threshold high enough that the optimistic bound fails early
    float threshold = 0.9f;
    float partial_ip = faiss::fvec_inner_product_batched(
            x.data(), y.data(), d, batch_size, threshold);
    // Should have aborted: partial result + remaining can't reach 0.9
    ASSERT_LT(partial_ip, full_ip + 1e-6);
    // The full IP of these near-orthogonal vectors is ~0
    ASSERT_NEAR(full_ip, 0.0f, 1e-6);
}

// Test with identical normalized vectors (IP should be ~1.0)
TEST(TestFvecInnerProductBatched, identical_vectors) {
    const size_t d = 128;
    const size_t batch_size = 16;
    std::vector<float> x(d);
    float norm = std::sqrt(static_cast<float>(d));
    for (size_t i = 0; i < d; i++) {
        x[i] = 1.0f / norm;
    }
    float result = faiss::fvec_inner_product_batched(
            x.data(), x.data(), d, batch_size, -1e10f);
    ASSERT_NEAR(result, 1.0f, 1e-5);
}


// Test batched Linf distance - correctness without early abort
TEST(TestFvecLinfBatched, correctness_no_abort) {
    std::default_random_engine rng(202);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    for (const auto d : {16, 32, 64, 128, 256, 512, 1024, 2048}) {
        for (const auto batch_size : {4, 8, 16, 32, 64}) {
            std::vector<float> x(d);
            std::vector<float> y(d);
            for (size_t i = 0; i < d; i++) {
                x[i] = uniform(rng);
                y[i] = uniform(rng);
            }
            float expected = faiss::fvec_Linf(x.data(), y.data(), d);
            float result = faiss::fvec_Linf_batched(
                    x.data(), y.data(), d, batch_size, 1e10f);
            ASSERT_FLOAT_EQ(result, expected)
                    << "d=" << d << ", batch_size=" << batch_size;
        }
    }
}

// Test that early abort triggers when max diff exceeds threshold
TEST(TestFvecLinfBatched, early_abort) {
    const size_t d = 1024;
    const size_t batch_size = 16;
    // Place a large difference early in the vectors
    std::vector<float> x(d, 0.0f);
    std::vector<float> y(d, 0.0f);
    x[5] = 100.0f; // Large difference at dimension 5
    y[5] = 0.0f;
    // Also place smaller differences later
    for (size_t i = 64; i < d; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    float full_distance = faiss::fvec_Linf(x.data(), y.data(), d);
    ASSERT_FLOAT_EQ(full_distance, 100.0f);
    // Threshold below the max diff — should abort in first batch
    float result = faiss::fvec_Linf_batched(
            x.data(), y.data(), d, batch_size, 50.0f);
    ASSERT_GT(result, 50.0f);
    ASSERT_FLOAT_EQ(result, 100.0f);
}

// Test with identical vectors (Linf should be 0)
TEST(TestFvecLinfBatched, identical_vectors) {
    const size_t d = 128;
    const size_t batch_size = 16;
    std::vector<float> x(d, 42.0f);
    float result = faiss::fvec_Linf_batched(
            x.data(), x.data(), d, batch_size, 1e10f);
    ASSERT_FLOAT_EQ(result, 0.0f);
}

// ============================================================
// Tests that verify the batched sequential path produces the
// same knn results as the BLAS path. The batched code in
// exhaustive_L2sqr_seq / exhaustive_inner_product_seq only runs
// when nx < distance_compute_blas_threshold or an IDSelector is
// active. These tests force both paths and compare results.
// ============================================================

#include <algorithm>
#include <cmath>
#include <faiss/IndexFlat.h>

// Helper: run knn search with a specific blas threshold, return distances+ids
static void run_knn_L2(
        const float* xq,
        const float* xb,
        size_t d,
        size_t nq,
        size_t nb,
        size_t k,
        int blas_threshold,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& ids) {
    int saved = faiss::distance_compute_blas_threshold;
    faiss::distance_compute_blas_threshold = blas_threshold;

    distances.resize(nq * k);
    ids.resize(nq * k);
    faiss::float_maxheap_array_t res = {nq, k, ids.data(), distances.data()};
    faiss::knn_L2sqr(xq, xb, d, nq, nb, &res);

    faiss::distance_compute_blas_threshold = saved;
}

static void run_knn_IP(
        const float* xq,
        const float* xb,
        size_t d,
        size_t nq,
        size_t nb,
        size_t k,
        int blas_threshold,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& ids) {
    int saved = faiss::distance_compute_blas_threshold;
    faiss::distance_compute_blas_threshold = blas_threshold;

    distances.resize(nq * k);
    ids.resize(nq * k);
    faiss::float_minheap_array_t res = {nq, k, ids.data(), distances.data()};
    faiss::knn_inner_product(xq, xb, d, nq, nb, &res);

    faiss::distance_compute_blas_threshold = saved;
}

// Force BLAS path (threshold=0 means nx is always >= threshold)
// vs sequential batched path (threshold=INT_MAX means always sequential).
// Compare that both produce the same top-k result set.
TEST(TestBatchedKnn, L2_blas_vs_sequential) {
    const size_t d = 128;
    const size_t nb = 5000;
    const size_t nq = 50;
    const size_t k = 10;

    std::default_random_engine rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> xb(nb * d), xq(nq * d);
    for (auto& v : xb) v = normal(rng);
    for (auto& v : xq) v = normal(rng);

    std::vector<float> d_blas, d_seq;
    std::vector<faiss::idx_t> i_blas, i_seq;

    // BLAS path: set threshold to 0 so nq >= 0 always takes BLAS
    run_knn_L2(xq.data(), xb.data(), d, nq, nb, k, 0, d_blas, i_blas);
    // Sequential (batched) path: threshold very high so always sequential
    run_knn_L2(xq.data(), xb.data(), d, nq, nb, k,
               INT_MAX, d_seq, i_seq);

    for (size_t q = 0; q < nq; q++) {
        for (size_t j = 0; j < k; j++) {
            size_t idx = q * k + j;
            ASSERT_EQ(i_blas[idx], i_seq[idx])
                    << "L2 mismatch at query=" << q << " rank=" << j;
            ASSERT_NEAR(d_blas[idx], d_seq[idx], 1e-3)
                    << "L2 distance mismatch at query=" << q
                    << " rank=" << j;
        }
    }
}

TEST(TestBatchedKnn, IP_blas_vs_sequential) {
    const size_t d = 128;
    const size_t nb = 5000;
    const size_t nq = 50;
    const size_t k = 10;

    std::default_random_engine rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> xb(nb * d), xq(nq * d);
    for (auto& v : xb) v = normal(rng);
    for (auto& v : xq) v = normal(rng);
    // Normalize for inner product
    auto normalize = [&](std::vector<float>& vecs, size_t n) {
        for (size_t i = 0; i < n; i++) {
            float norm = 0;
            for (size_t j = 0; j < d; j++)
                norm += vecs[i * d + j] * vecs[i * d + j];
            norm = std::sqrt(norm);
            for (size_t j = 0; j < d; j++)
                vecs[i * d + j] /= norm;
        }
    };
    normalize(xb, nb);
    normalize(xq, nq);

    std::vector<float> d_blas, d_seq;
    std::vector<faiss::idx_t> i_blas, i_seq;

    run_knn_IP(xq.data(), xb.data(), d, nq, nb, k, 0, d_blas, i_blas);
    run_knn_IP(xq.data(), xb.data(), d, nq, nb, k,
               INT_MAX, d_seq, i_seq);

    for (size_t q = 0; q < nq; q++) {
        for (size_t j = 0; j < k; j++) {
            size_t idx = q * k + j;
            ASSERT_EQ(i_blas[idx], i_seq[idx])
                    << "IP mismatch at query=" << q << " rank=" << j;
            ASSERT_NEAR(d_blas[idx], d_seq[idx], 1e-4)
                    << "IP distance mismatch at query=" << q
                    << " rank=" << j;
        }
    }
}
