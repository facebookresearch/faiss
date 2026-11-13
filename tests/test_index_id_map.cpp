/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/FaissException.h>
#include <gtest/gtest.h>

namespace faiss {

TEST(IndexIDMap2, MergeFromIDMap) {
    IndexFlatL2 storage(2);
    IndexIDMap2 index(&storage);
    const float old_vector[] = {1, 2};
    const idx_t old_id = 123;
    index.add_with_ids(1, old_vector, &old_id);

    IndexFlatL2 other_storage(2);
    IndexIDMap other(&other_storage);
    const float new_vector[] = {3, 4};
    const idx_t new_id = 456;
    other.add_with_ids(1, new_vector, &new_id);

    EXPECT_NO_THROW(index.check_compatible_for_merge(other));
    index.merge_from(other, 10);
    EXPECT_EQ(2, index.ntotal);
    EXPECT_EQ(0, other.ntotal);
    EXPECT_TRUE(other.id_map.empty());
    EXPECT_NO_THROW(index.check_consistency());

    float reconstructed[2];
    index.reconstruct(new_id + 10, reconstructed);
    EXPECT_FLOAT_EQ(new_vector[0], reconstructed[0]);
    EXPECT_FLOAT_EQ(new_vector[1], reconstructed[1]);
    index.reconstruct(old_id, reconstructed);
    EXPECT_FLOAT_EQ(old_vector[0], reconstructed[0]);
    EXPECT_FLOAT_EQ(old_vector[1], reconstructed[1]);
}

TEST(IndexIDMap2, MergeIntoIDMapClearsReverseMap) {
    IndexFlatL2 storage(2);
    IndexIDMap index(&storage);
    IndexFlatL2 other_storage(2);
    IndexIDMap2 other(&other_storage);
    const float vector[] = {1, 2};
    const idx_t id = 123;
    other.add_with_ids(1, vector, &id);

    index.merge_from(other);
    EXPECT_EQ(1, index.ntotal);
    EXPECT_EQ(0, other.ntotal);
    EXPECT_TRUE(other.id_map.empty());
    EXPECT_TRUE(other.rev_map.empty());
    EXPECT_NO_THROW(other.check_consistency());

    const idx_t new_id = 456;
    other.add_with_ids(1, vector, &new_id);
    float reconstructed[2];
    EXPECT_THROW(other.reconstruct(id, reconstructed), FaissException);
    other.reconstruct(new_id, reconstructed);
    EXPECT_FLOAT_EQ(vector[0], reconstructed[0]);
    EXPECT_FLOAT_EQ(vector[1], reconstructed[1]);
    EXPECT_NO_THROW(other.check_consistency());
}

} // namespace faiss
