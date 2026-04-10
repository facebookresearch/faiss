/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/NNDescent.h>
#include <gtest/gtest.h>
#include <random>

using namespace faiss::nndescent;

/// Helper: build a Nhood with known data in all fields.
static Nhood make_populated_nhood() {
    std::mt19937 rng(42);
    int N = 200;
    Nhood nh(/*l=*/50, /*s=*/10, rng, N);

    nh.pool.clear();
    nh.pool.push_back(Neighbor(1, 0.1f, true));
    nh.pool.push_back(Neighbor(2, 0.2f, false));
    nh.pool.push_back(Neighbor(3, 0.3f, true));

    nh.nn_old = {10, 20, 30};
    nh.rnn_new = {40, 50};
    nh.rnn_old = {60, 70, 80};

    return nh;
}

TEST(NhoodCopy, CopyConstructorPreservesAllFields) {
    Nhood original = make_populated_nhood();
    Nhood copy(original);

    EXPECT_EQ(copy.M, original.M);
    EXPECT_EQ(copy.pool.size(), original.pool.size());
    EXPECT_EQ(copy.nn_new, original.nn_new);
    EXPECT_EQ(copy.nn_old, original.nn_old);
    EXPECT_EQ(copy.rnn_new, original.rnn_new);
    EXPECT_EQ(copy.rnn_old, original.rnn_old);
}

TEST(NhoodCopy, CopyAssignmentPreservesAllFields) {
    Nhood original = make_populated_nhood();
    Nhood assigned;
    assigned = original;

    EXPECT_EQ(assigned.M, original.M);
    EXPECT_EQ(assigned.pool.size(), original.pool.size());
    EXPECT_EQ(assigned.nn_new, original.nn_new);
    EXPECT_EQ(assigned.nn_old, original.nn_old);
    EXPECT_EQ(assigned.rnn_new, original.rnn_new);
    EXPECT_EQ(assigned.rnn_old, original.rnn_old);
}

TEST(NhoodCopy, CopyAssignmentSelfAssign) {
    Nhood nh = make_populated_nhood();
    auto expected_pool_size = nh.pool.size();
    auto expected_nn_new = nh.nn_new;

    // Use a reference to avoid -Wself-assign-overloaded.
    Nhood& ref = nh;
    nh = ref;

    EXPECT_EQ(nh.pool.size(), expected_pool_size);
    EXPECT_EQ(nh.nn_new, expected_nn_new);
}

/// Simulates std::vector<Nhood> reallocation during push_back.
TEST(NhoodCopy, VectorReallocationPreservesData) {
    std::vector<Nhood> vec;
    // Do NOT reserve — force reallocation during push_back
    for (int i = 0; i < 20; i++) {
        Nhood nh = make_populated_nhood();
        nh.pool[0].id = i;
        vec.push_back(std::move(nh));
    }

    for (int i = 0; i < 20; i++) {
        EXPECT_EQ(vec[i].pool[0].id, i) << "pool lost at index " << i;
        EXPECT_EQ(vec[i].pool.size(), 3) << "pool truncated at index " << i;
        EXPECT_EQ(vec[i].nn_old.size(), 3) << "nn_old lost at index " << i;
        EXPECT_EQ(vec[i].rnn_new.size(), 2) << "rnn_new lost at index " << i;
        EXPECT_EQ(vec[i].rnn_old.size(), 3) << "rnn_old lost at index " << i;
    }
}
