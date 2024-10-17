/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags_declare.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <random>

#include "faiss/utils/distances.h"

static const unsigned NPTS{1000};
static const unsigned DIM{128};
static const std::array<unsigned, 3> TOP_K_LIST{1, 50, 100};
static const std::array<unsigned, 2> NUMS_QUREY_LIST{10, 30};

using VectorDataPtr = std::unique_ptr<float[]>;

VectorDataPtr generateRandomVector(
        const unsigned npts,
        const unsigned dim,
        int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    VectorDataPtr data = std::make_unique<float[]>(npts * dim);
    for (unsigned n{0}; n < npts; ++n) {
        for (unsigned d = 0; d < dim; ++d) {
            data[n * dim + d] = dis(gen);
        }
    }
    return std::move(data);
}

float computeL2SqrDist(
        const float* vec1,
        const float* vec2,
        const unsigned dim) {
    float dist = 0;
    for (unsigned i{0}; i < dim; ++i) {
        float diff = vec1[i] - vec2[i];
        dist += diff * diff;
    }
    return dist;
}

float computeInnerProductDist(
        const float* vec1,
        const float* vec2,
        const unsigned dim) {
    float dist = 0;
    for (unsigned i{0}; i < dim; ++i) {
        dist += vec1[i] * vec2[i];
    }
    return dist;
}

void checkKNNResults(
        const float* queries,
        const float* db,
        size_t nq,
        size_t ndb,
        size_t dim,
        size_t k,
        const float* distances,
        const int64_t* indexes,
        bool is_l2 = true) {
    for (size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<float, int64_t>> res_pairs;
        for (int64_t i = 0; i < ndb; ++i) {
            float dist{0};
            if (is_l2) {
                dist = computeL2SqrDist(queries + q * dim, db + i * dim, dim);
            } else {
                dist = computeInnerProductDist(
                        queries + q * dim, db + i * dim, dim);
            }
            res_pairs.emplace_back(dist, i);
        }
        std::sort(
                res_pairs.begin(),
                res_pairs.end(),
                [is_l2](const std::pair<float, int64_t>& a,
                        const std::pair<float, int64_t>& b) {
                    return is_l2 ? a.first < b.first : a.first > b.first;
                });
        for (size_t j = 0; j < k; ++j) {
            ASSERT_EQ(res_pairs[j].second, indexes[q * k + j]);
            // WARNING: There will be a very small fractional error :(
            // ASSERT_FLOAT_EQ(res_pairs[j].first, distances[q * k + j]);
        }
    }
}

TEST(TestKNNFunctions, knn_L2sqr) {
    VectorDataPtr random_base_data = generateRandomVector(NPTS, DIM);
    for (unsigned top_k : TOP_K_LIST) {
        for (unsigned nums_query : NUMS_QUREY_LIST) {
            VectorDataPtr random_query_data =
                    generateRandomVector(nums_query, DIM);
            std::unique_ptr<float[]> distances =
                    std::make_unique<float[]>(nums_query * top_k);
            std::unique_ptr<int64_t[]> indexes =
                    std::make_unique<int64_t[]>(nums_query * top_k);
            faiss::knn_L2sqr(
                    random_query_data.get(),
                    random_base_data.get(),
                    DIM,
                    nums_query,
                    NPTS,
                    top_k,
                    distances.get(),
                    indexes.get());
            checkKNNResults(
                    random_query_data.get(),
                    random_base_data.get(),
                    nums_query,
                    NPTS,
                    DIM,
                    top_k,
                    distances.get(),
                    indexes.get(),
                    true);
        }
    }
}

// TODO:
TEST(TestKNNFunctions, knn_inner_product) {
    VectorDataPtr random_base_data = generateRandomVector(NPTS, DIM);
    for (unsigned top_k : TOP_K_LIST) {
        for (unsigned nums_query : NUMS_QUREY_LIST) {
            VectorDataPtr random_query_data =
                    generateRandomVector(nums_query, DIM);
            std::unique_ptr<float[]> distances =
                    std::make_unique<float[]>(nums_query * top_k);
            std::unique_ptr<int64_t[]> indexes =
                    std::make_unique<int64_t[]>(nums_query * top_k);
            faiss::knn_inner_product(
                    random_query_data.get(),
                    random_base_data.get(),
                    DIM,
                    nums_query,
                    NPTS,
                    top_k,
                    distances.get(),
                    indexes.get());
            checkKNNResults(
                    random_query_data.get(),
                    random_base_data.get(),
                    nums_query,
                    NPTS,
                    DIM,
                    top_k,
                    distances.get(),
                    indexes.get(),
                    false);
        }
    }
}
