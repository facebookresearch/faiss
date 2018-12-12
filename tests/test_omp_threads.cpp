/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <gtest/gtest.h>

#include <omp.h>

TEST(Threading, openmp) {

    omp_set_num_threads(10);

    EXPECT_EQ(omp_get_max_threads(), 10);

    std::vector<int> nt_per_thread(10);
    size_t sum = 0;
#pragma omp parallel reduction(+: sum)
    {
        int nt = omp_get_num_threads ();
        int rank = omp_get_thread_num ();

        nt_per_thread[rank] = nt;
#pragma omp for
        for(int i = 0; i < 1000 * 1000 * 10; i++) {
            sum += i;
        }
    }


    EXPECT_EQ (nt_per_thread[0], 10);
    EXPECT_GT (sum, 0);
}
