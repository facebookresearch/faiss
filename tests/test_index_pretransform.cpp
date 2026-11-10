/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <gtest/gtest.h>

namespace faiss {

TEST(IndexPreTransform, MergeUpdatesSourceCount) {
    for (bool with_transform : {false, true}) {
        SCOPED_TRACE(with_transform);
        IndexFlatL2 storage(2);
        IndexPreTransform index(&storage);
        IndexFlatL2 other_storage(2);
        IndexPreTransform other(&other_storage);
        LinearTransform identity(2, 2);
        identity.A = {1, 0, 0, 1};
        identity.is_trained = true;
        identity.is_orthonormal = true;
        if (with_transform) {
            index.prepend_transform(&identity);
            other.prepend_transform(&identity);
        }

        const float vectors[] = {1, 2, 3, 4, 5, 6};
        index.add(1, vectors);
        other.add(2, vectors + 2);

        index.merge_from(other);
        EXPECT_EQ(3, index.ntotal);
        EXPECT_EQ(index.ntotal, storage.ntotal);
        EXPECT_EQ(0, other.ntotal);
        EXPECT_EQ(other.ntotal, other_storage.ntotal);

        float reconstructed[6];
        index.reconstruct_n(0, index.ntotal, reconstructed);
        for (size_t i = 0; i < 6; i++) {
            EXPECT_FLOAT_EQ(vectors[i], reconstructed[i]);
        }

        // The emptied source remains available for reuse.
        other.add(1, vectors);
        EXPECT_EQ(1, other.ntotal);
        EXPECT_EQ(other.ntotal, other_storage.ntotal);
    }
}

} // namespace faiss
