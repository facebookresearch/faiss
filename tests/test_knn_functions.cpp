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
#include <cstdlib>
#include <memory>
#include <random>

#include "faiss/utils/distances.h"

static const unsigned db_round = 3;
static const unsigned nb{10000};
static const std::array<unsigned, 3> nb_list{1000, 10000, 100000};
static const std::array<unsigned, 3> d_list{32, 128, 1024};
static const std::array<unsigned, 3> k_list{1, 50, 100};
static const std::array<unsigned, 2> nq_list{10, 30};

using VectorDataPtr = std::unique_ptr<float[]>;

VectorDataPtr generateRandomVector(
        const unsigned nx,
        const unsigned d,
        int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    VectorDataPtr data = std::make_unique<float[]>(nx * d);
    for (unsigned n{0}; n < nx; ++n) {
        for (unsigned i = 0; i < d; ++i) {
            data[n * d + i] = dis(gen);
        }
    }
    return data;
}

float computeL2SqrDist(const float* vec1, const float* vec2, const unsigned d) {
    float dist = 0;
    for (unsigned i{0}; i < d; ++i) {
        float diff = vec1[i] - vec2[i];
        dist += diff * diff;
    }
    return dist;
}

float computeInnerProductDist(
        const float* vec1,
        const float* vec2,
        const unsigned d) {
    float dist = 0;
    for (unsigned i{0}; i < d; ++i) {
        dist += vec1[i] * vec2[i];
    }
    return dist;
}

void checkKNNResults(
        const float* x,
        const float* y,
        size_t nx,
        size_t ny,
        size_t d,
        size_t k,
        const float* distances,
        const int64_t* indexes,
        bool is_l2 = true) {
    for (size_t q = 0; q < nx; ++q) {
        std::vector<std::pair<float, int64_t>> res_pairs;
        for (int64_t i = 0; i < ny; ++i) {
            float dist{0};
            if (is_l2) {
                dist = faiss::fvec_L2sqr(x + q * d, y + i * d, d);
            } else {
                dist = faiss::fvec_inner_product(x + q * d, y + i * d, d);
            }
            res_pairs.emplace_back(dist, i);
        }
        std::sort(
                res_pairs.begin(),
                res_pairs.end(),
                [is_l2](const std::pair<float, int64_t>& a,
                        const std::pair<float, int64_t>& b) {
                    // return is_l2 ? (a.first < b.first) : (a.first > b.first);
                    if (is_l2) {
                        return a.first != b.first ? a.first < b.first
                                                  : a.second < b.second;
                    } else {
                        return a.first != b.first ? a.first > b.first
                                                  : a.second < b.second;
                    }
                });
        for (size_t j = 0; j < k; ++j) {
            // ASSERT_EQ(res_pairs[j].second, indexes[q * k + j]);
            EXPECT_EQ(res_pairs[j].second, indexes[q * k + j]);
            if (res_pairs[j].second != indexes[q * k + j]) {
                std::cout << "location: " << j << std::endl;
                std::cout << "distances: " << res_pairs.at(j).first << "|"
                          << distances[q * k + j] << std::endl;
            }

            float local_dist{0}, ref_dist{distances[q * k + j]};
            if (is_l2) {
                local_dist = faiss::fvec_L2sqr(
                        x + q * d, y + res_pairs[j].second * d, d);
            } else {
                local_dist = faiss::fvec_inner_product(
                        x + q * d, y + res_pairs[j].second * d, d);
            }
            float relative_error = std::abs((local_dist - ref_dist)) / ref_dist;
            ASSERT_LE(relative_error, 0.0001);
        }
    }
}

TEST(TestKNNFunctions, knn_L2sqr) {
    for (unsigned r{0}; r < db_round; ++r) {
        unsigned nb = nb_list.at(r);
        unsigned d = d_list.at(r);
        VectorDataPtr random_base_data = generateRandomVector(nb, d, d);
        for (unsigned k : k_list) {
            for (unsigned nq : nq_list) {
                VectorDataPtr random_query_data =
                        generateRandomVector(nq, d, nq);
                std::unique_ptr<float[]> distances =
                        std::make_unique<float[]>(nq * k);
                std::unique_ptr<int64_t[]> indexes =
                        std::make_unique<int64_t[]>(nq * k);
                faiss::knn_L2sqr(
                        random_query_data.get(),
                        random_base_data.get(),
                        d,
                        nq,
                        nb,
                        k,
                        distances.get(),
                        indexes.get());
                checkKNNResults(
                        random_query_data.get(),
                        random_base_data.get(),
                        nq,
                        nb,
                        d,
                        k,
                        distances.get(),
                        indexes.get(),
                        true);
            }
        }
    }
}

TEST(TestKNNFunctions, knn_inner_product) {
    for (unsigned r{0}; r < db_round; ++r) {
        unsigned nb = nb_list.at(r);
        unsigned d = d_list.at(r);
        VectorDataPtr random_base_data = generateRandomVector(nb, d, d);
        for (unsigned k : k_list) {
            for (unsigned nq : nq_list) {
                VectorDataPtr random_query_data =
                        generateRandomVector(nq, d, nq);
                std::unique_ptr<float[]> distances =
                        std::make_unique<float[]>(nq * k);
                std::unique_ptr<int64_t[]> indexes =
                        std::make_unique<int64_t[]>(nq * k);
                faiss::knn_inner_product(
                        random_query_data.get(),
                        random_base_data.get(),
                        d,
                        nq,
                        nb,
                        k,
                        distances.get(),
                        indexes.get());
                checkKNNResults(
                        random_query_data.get(),
                        random_base_data.get(),
                        nq,
                        nb,
                        d,
                        k,
                        distances.get(),
                        indexes.get(),
                        false);
            }
        }
    }
}
