/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <omp.h>

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/utils/utils.h>

TEST(Threading, openmp) {
    EXPECT_TRUE(faiss::check_openmp());
}

TEST(Threading, flat_search_in_parallel_region) {
    struct RestoreOpenMP {
        int threads = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
#if _OPENMP >= 200805
        int max_active_levels = omp_get_max_active_levels();
#else
        int nested = omp_get_nested();
#endif

        ~RestoreOpenMP() {
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
#if _OPENMP >= 200805
            omp_set_max_active_levels(max_active_levels);
#else
            omp_set_nested(nested);
#endif
        }
    } restore;
    omp_set_num_threads(4);
    omp_set_dynamic(0);
#if _OPENMP >= 200805
    omp_set_max_active_levels(1);
#else
    omp_set_nested(0);
#endif

    constexpr faiss::idx_t nb = 10001;
    for (auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        SCOPED_TRACE(metric);
        const float query = metric == faiss::METRIC_L2 ? 0.0f : 1.0f;
        std::vector<float> database(nb, 10.0f);
        database.back() = metric == faiss::METRIC_L2 ? 0.0f : 100.0f;
        faiss::IndexFlat index(1, metric);
        index.add(nb, database.data());

        float expected_distance;
        faiss::idx_t expected_label;
        index.search(1, &query, 1, &expected_distance, &expected_label);
        ASSERT_EQ(expected_label, nb - 1);

        float distance = -1;
        faiss::idx_t label = -1;
        int outer_threads = 0;
#pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0) {
                outer_threads = omp_get_num_threads();
                omp_set_num_threads(4);
                index.search(1, &query, 1, &distance, &label);
            }
        }
        ASSERT_EQ(outer_threads, 2);
        EXPECT_EQ(label, expected_label);
        EXPECT_EQ(distance, expected_distance);
    }
}
