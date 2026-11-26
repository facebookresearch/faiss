/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>
#include <faiss/impl/FaissException.h>
#include <gtest/gtest.h>

namespace {

TEST(IndexSplitVectors, SearchUsesChildLpExponent) {
    faiss::IndexFlat first(2, faiss::METRIC_Lp);
    faiss::IndexFlat second(1, faiss::METRIC_Lp);
    first.metric_arg = second.metric_arg = 2.0f;
    const float first_vectors[] = {3.0f, 0.0f, 2.0f, 2.0f};
    const float second_vectors[] = {0.0f, 1.0f};
    first.add(2, first_vectors);
    second.add(2, second_vectors);

    faiss::IndexSplitVectors index(3);
    index.add_sub_index(&first);
    index.add_sub_index(&second);

    const float query[] = {0.0f, 0.0f, 1.0f};
    float distance;
    faiss::idx_t label;
    EXPECT_FLOAT_EQ(index.metric_arg, 2.0f);
    index.search(1, query, 1, &distance, &label);
    EXPECT_EQ(label, 3);
    EXPECT_FLOAT_EQ(distance, 8.0f);

    index.metric_arg = 1.0f;
    EXPECT_THROW(
            index.search(1, query, 1, &distance, &label),
            faiss::FaissException);

    first.metric_arg = second.metric_arg = 1.0f;
    index.sync_with_sub_indexes();
    index.search(1, query, 1, &distance, &label);
    EXPECT_EQ(label, 2);
    EXPECT_FLOAT_EQ(distance, 3.0f);
}

TEST(IndexSplitVectors, RejectsMismatchedChildLpExponent) {
    faiss::IndexFlat first(2, faiss::METRIC_Lp);
    faiss::IndexFlat second(1, faiss::METRIC_Lp);
    first.metric_arg = 2.0f;
    second.metric_arg = 1.0f;

    faiss::IndexSplitVectors index(3);
    index.add_sub_index(&first);
    EXPECT_THROW(index.add_sub_index(&second), faiss::FaissException);
    EXPECT_EQ(index.sub_indexes.size(), 1);
    EXPECT_EQ(index.sum_d, 2);

    second.metric_arg = 2.0f;
    index.add_sub_index(&second);
    first.metric_arg = 1.0f;
    EXPECT_THROW(index.sync_with_sub_indexes(), faiss::FaissException);
    EXPECT_FLOAT_EQ(index.metric_arg, 2.0f);
}

} // namespace
