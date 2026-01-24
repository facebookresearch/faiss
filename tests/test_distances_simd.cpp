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

void remove_simd_level_if_exists(
        std::unordered_set<faiss::SIMDLevel>& levels,
        faiss::SIMDLevel level) {
    std::erase_if(
            levels, [level](faiss::SIMDLevel elem) { return elem == level; });
}

class DistancesSIMDTest : public ::testing::TestWithParam<faiss::SIMDLevel> {
   protected:
    void SetUp() override {
        original_simd_level = faiss::SIMDConfig::get_level();
        std::iota(dims.begin(), dims.end(), 1);

        ntests = 4;

        simd_level = GetParam();
        faiss::SIMDConfig::set_level(simd_level);

        EXPECT_EQ(faiss::SIMDConfig::get_level(), simd_level);

        rng = std::default_random_engine(123);
        uniform = std::uniform_int_distribution<int32_t>(0, 32);
    }

    void TearDown() override {
        faiss::SIMDConfig::set_level(original_simd_level);
    }

    std::tuple<std::vector<float>, std::vector<std::vector<float>>>
    SetupTestData(int dims, int ny) {
        std::vector<float> x(dims);
        std::vector<std::vector<float>> y(ny, std::vector<float>(dims));

        for (size_t i = 0; i < dims; i++) {
            x[i] = uniform(rng);
            for (size_t j = 0; j < ny; j++) {
                y[j][i] = uniform(rng);
            }
        }
        return std::make_tuple(x, y);
    }

    std::vector<float> flatten_2d_vector(
            const std::vector<std::vector<float>>& v) {
        std::vector<float> flat_v;
        for (const auto& vec : v) {
            flat_v.insert(flat_v.end(), vec.begin(), vec.end());
        }
        return flat_v;
    }

    faiss::SIMDLevel simd_level = faiss::SIMDLevel::NONE;
    faiss::SIMDLevel original_simd_level = faiss::SIMDLevel::NONE;
    std::default_random_engine rng;
    std::uniform_int_distribution<int32_t> uniform;

    std::vector<int> dims = {128};
    int ntests = 1;
};

TEST_P(DistancesSIMDTest, LinfDistance_chebyshev_distance) {
    for (int i = 0; i < ntests; ++i) { // repeat tests
        for (const auto dim : dims) {  // test different dimensions
            int ny = 1;
            auto [x, y] = SetupTestData(dim, ny);
            for (int k = 0; k < ny; ++k) { // test different vectors
                float distance = faiss::fvec_Linf(x.data(), y[k].data(), dim);
                float ref_distance = 0;

                for (int j = 0; j < dim; ++j) {
                    ref_distance =
                            std::max(ref_distance, std::abs(x[j] - y[k][j]));
                }
                ASSERT_EQ(distance, ref_distance);
            }
        }
    }
}

TEST_P(DistancesSIMDTest, inner_product_batch_4) {
    for (int i = 0; i < ntests; ++i) {
        int dim = 128;
        int ny = 4;
        auto [x, y] = SetupTestData(dim, ny);

        std::vector<float> true_distances(ny, 0.F);
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < dim; ++k) {
                true_distances[j] += x[k] * y[j][k];
            }
        }

        std::vector<float> actual_distances(ny, 0.F);
        faiss::fvec_inner_product_batch_4(
                x.data(),
                y[0].data(),
                y[1].data(),
                y[2].data(),
                y[3].data(),
                dim,
                actual_distances[0],
                actual_distances[1],
                actual_distances[2],
                actual_distances[3]);

        ASSERT_EQ(actual_distances, true_distances)
                << "Mismatching fvec_inner_product_batch4 results for test = "
                << i;
    }
}

TEST_P(DistancesSIMDTest, fvec_L2sqr) {
    for (int i = 0; i < ntests; ++i) {
        int ny = 1;
        for (const auto dim : dims) {
            auto [x, y] = SetupTestData(dim, ny);
            float true_distance = 0.F;
            for (int k = 0; k < dim; ++k) {
                const float tmp = x[k] - y[0][k];
                true_distance += tmp * tmp;
            }

            float actual_distance =
                    faiss::fvec_L2sqr(x.data(), y[0].data(), dim);

            ASSERT_EQ(actual_distance, true_distance)
                    << "Mismatching fvec_L2sqr results for test = " << i;
        }
    }
}

TEST_P(DistancesSIMDTest, L2sqr_batch_4) {
    for (int i = 0; i < ntests; ++i) {
        int dim = 128;
        int ny = 4;
        auto [x, y] = SetupTestData(dim, ny);

        std::vector<float> true_distances(ny, 0.F);
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < dim; ++k) {
                const float tmp = x[k] - y[j][k];
                true_distances[j] += tmp * tmp;
            }
        }

        std::vector<float> actual_distances(ny, 0.F);
        faiss::fvec_L2sqr_batch_4(
                x.data(),
                y[0].data(),
                y[1].data(),
                y[2].data(),
                y[3].data(),
                dim,
                actual_distances[0],
                actual_distances[1],
                actual_distances[2],
                actual_distances[3]);

        ASSERT_EQ(actual_distances, true_distances)
                << "Mismatching fvec_L2sqr_batch_4 results for test = " << i;
    }
}

TEST_P(DistancesSIMDTest, fvec_L2sqr_ny) {
    for (const auto dim : {2, 4, 8, 12}) {
        for (const auto ny : {1, 2, 5, 10, 15, 20, 25}) {
            auto [x, y] = SetupTestData(dim, ny);

            std::vector<float> actual_distances(ny, 0.F);

            std::vector<float> flat_y;
            for (auto y_ : y) {
                flat_y.insert(flat_y.end(), y_.begin(), y_.end());
            }

            std::vector<float> true_distances(ny, 0.F);
            for (int i = 0; i < ny; ++i) {
                for (int k = 0; k < dim; ++k) {
                    const float tmp = x[k] - y[i][k];
                    true_distances[i] += tmp * tmp;
                }
            }

            faiss::fvec_L2sqr_ny(
                    actual_distances.data(), x.data(), flat_y.data(), dim, ny);

            ASSERT_EQ(actual_distances, true_distances)
                    << "Mismatching fvec_L2sqr_ny results for dim = " << dim
                    << ", ny = " << ny;
        }
    }
}

TEST_P(DistancesSIMDTest, fvec_inner_products_ny) {
    for (const auto dim : {2, 4, 8, 12}) {
        for (const auto ny : {1, 2, 5, 10, 15, 20, 25}) {
            auto [x, y] = SetupTestData(dim, ny);
            auto flat_y = flatten_2d_vector(y);

            std::vector<float> actual_distances(ny, 0.F);
            faiss::fvec_inner_products_ny(
                    actual_distances.data(), x.data(), flat_y.data(), dim, ny);

            std::vector<float> true_distances(ny, 0.F);
            for (int i = 0; i < ny; ++i) {
                for (int k = 0; k < dim; ++k) {
                    true_distances[i] += x[k] * y[i][k];
                }
            }

            ASSERT_EQ(actual_distances, true_distances)
                    << "Mismatching fvec_inner_products_ny results for dim = "
                    << dim << ", ny = " << ny;
        }
    }
}

TEST_P(DistancesSIMDTest, L2SqrNYNearest) {
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 32);

    int dim = 128;
    int ny = 11;

    auto [x, y] = SetupTestData(dim, ny);
    auto flat_y = flatten_2d_vector(y);

    std::vector<float> true_tmp_buffer_distances(ny, 0.F);
    for (int i = 0; i < ny; ++i) {
        for (int k = 0; k < dim; ++k) {
            const float tmp = x[k] - y[i][k];
            true_tmp_buffer_distances[i] += tmp * tmp;
        }
    }

    size_t true_nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (true_tmp_buffer_distances[i] < min_dis) {
            min_dis = true_tmp_buffer_distances[i];
            true_nearest_idx = i;
        }
    }

    std::vector<float> actual_distances(ny);
    auto actual_nearest_index = faiss::fvec_L2sqr_ny_nearest(
            actual_distances.data(), x.data(), flat_y.data(), dim, ny);

    EXPECT_EQ(actual_nearest_index, true_nearest_idx);
}

TEST_P(DistancesSIMDTest, multiple_add) {
    // modulo 8 results - 16 is to repeat the while loop in the function
    for (const auto dim : {8, 9, 10, 11, 12, 13, 14, 15, 16}) {
        auto [x, y] = SetupTestData(dim, 1);
        const float bf = uniform(rng);
        std::vector<float> true_distances(dim);
        for (size_t i = 0; i < x.size(); i++) {
            true_distances[i] = x[i] + bf * y[0][i];
        }

        std::vector<float> actual_distances(dim);
        faiss::fvec_madd(
                x.size(), x.data(), bf, y[0].data(), actual_distances.data());

        ASSERT_EQ(actual_distances, true_distances)
                << "Mismatching fvec_madd results for nrows = " << dim;
    }
}

TEST_P(DistancesSIMDTest, manhattan_distance) {
    // modulo 8 results - 16 is to repeat the while loop in the function
    for (const auto dim : {8, 9, 10, 11, 12, 13, 14, 15, 16}) {
        auto [x, y] = SetupTestData(dim, 1);
        float true_distance = 0;
        for (size_t i = 0; i < x.size(); i++) {
            true_distance += std::abs(x[i] - y[0][i]);
        }

        auto actual_distances = faiss::fvec_L1(x.data(), y[0].data(), x.size());

        ASSERT_EQ(actual_distances, true_distance)
                << "Mismatching fvec_Linf results for nrows = " << dim;
    }
}

TEST_P(DistancesSIMDTest, add_value) {
    for (const auto dim : {1, 2, 5, 10, 15, 20, 25}) {
        auto [x, y] = SetupTestData(dim, 1);
        const float b = uniform(rng); // value to add
        std::vector<float> true_distances(dim);
        for (size_t i = 0; i < x.size(); i++) {
            true_distances[i] = x[i] + b;
        }

        std::vector<float> actual_distances(dim);
        faiss::fvec_add(x.size(), x.data(), b, actual_distances.data());

        ASSERT_EQ(actual_distances, true_distances)
                << "Mismatching array-value fvec_add results for nrows = "
                << dim;
    }
}

TEST_P(DistancesSIMDTest, add_array) {
    for (const auto dim : {1, 2, 5, 10, 15, 20, 25}) {
        auto [x, y] = SetupTestData(dim, 1);
        std::vector<float> true_distances(dim);
        for (size_t i = 0; i < x.size(); i++) {
            true_distances[i] = x[i] + y[0][i];
        }

        std::vector<float> actual_distances(dim);
        faiss::fvec_add(
                x.size(), x.data(), y[0].data(), actual_distances.data());

        ASSERT_EQ(actual_distances, true_distances)
                << "Mismatching array-array fvec_add results for nrows = "
                << dim;
    }
}

TEST_P(DistancesSIMDTest, distances_L2_squared_y_transposed) {
    // modulo 8 results - 16 is to repeat the loop in the function
    int ny = 11; // this value will hit all the codepaths
    for (const auto d : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
        auto [x, y] = SetupTestData(d, ny);
        float x_sqlen = 0;
        for (size_t i = 0; i < d; ++i) {
            x_sqlen += x[i] * x[i];
        }
        auto flat_y = flatten_2d_vector(y);
        std::vector<float> y_sqlens(ny, 0);
        for (size_t i = 0; i < ny; ++i) {
            for (size_t j = 0; j < d; ++j) {
                y_sqlens[i] += flat_y[j] * flat_y[j];
            }
        }

        // perform function
        std::vector<float> true_distances(ny, 0);
        for (size_t i = 0; i < ny; ++i) {
            float dp = 0;
            for (size_t j = 0; j < d; ++j) {
                dp += x[j] * flat_y[i + j * ny];
            }
            true_distances[i] = x_sqlen + y_sqlens[i] - 2 * dp;
        }

        std::vector<float> distances(ny);
        faiss::fvec_L2sqr_ny_transposed(
                distances.data(),
                x.data(),
                flat_y.data(),
                y_sqlens.data(),
                d,
                ny, // no need for special offset to test all lines of code
                ny);

        ASSERT_EQ(distances, true_distances)
                << "Mismatching fvec_L2sqr_ny_transposed results for d = " << d;
    }
}

TEST_P(DistancesSIMDTest, nearest_L2_squared_y_transposed) {
    // modulo 8 results - 16 is to repeat the loop in the function
    int ny = 11; // this value will hit all the codepaths
    for (const auto dim : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
        auto [x, y] = SetupTestData(dim, ny);
        float x_sqlen = 0.F;
        for (size_t i = 0; i < dim; i++) {
            x_sqlen += x[i] * x[i];
        }

        auto flat_y = flatten_2d_vector(y);
        std::vector<float> y_sqlens(ny, 0);

        for (size_t i = 0; i < ny; i++) {
            for (size_t j = 0; j < dim; j++) {
                y_sqlens[i] += y[i][j] * y[i][j];
            }
        }

        std::vector<float> distances(ny, 0);
        for (size_t i = 0; i < ny; i++) {
            float dp = 0;
            for (size_t j = 0; j < dim; j++) {
                dp += x[j] * flat_y[i + j * ny];
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
        size_t actual_nearest_idx = faiss::fvec_L2sqr_ny_nearest_y_transposed(
                buffer.data(),
                x.data(),
                flat_y.data(),
                y_sqlens.data(),
                dim,
                ny, // no need for special offset to test all lines of code
                ny);

        ASSERT_EQ(actual_nearest_idx, true_nearest_idx)
                << "Mismatching fvec_L2sqr_ny_nearest_y_transposed results for d = "
                << dim;
    }
}

std::vector<faiss::SIMDLevel> GetSupportedSIMDLevels() {
    std::vector<faiss::SIMDLevel> supported_levels = {faiss::SIMDLevel::NONE};

    for (int level = static_cast<int>(faiss::SIMDLevel::NONE) + 1;
         level < static_cast<int>(faiss::SIMDLevel::COUNT);
         level++) {
        faiss::SIMDLevel simd_level = static_cast<faiss::SIMDLevel>(level);
        if (faiss::SIMDConfig::is_simd_level_available(simd_level)) {
            supported_levels.push_back(simd_level);
        }
    }

    EXPECT_TRUE(supported_levels.size() > 0);

    return std::vector<faiss::SIMDLevel>(
            supported_levels.begin(), supported_levels.end());
}

::testing::internal::ParamGenerator<faiss::SIMDLevel> SupportedSIMDLevels() {
    std::vector<faiss::SIMDLevel> levels = GetSupportedSIMDLevels();
    return ::testing::ValuesIn(levels);
}

INSTANTIATE_TEST_SUITE_P(SIMDLevels, DistancesSIMDTest, SupportedSIMDLevels());
