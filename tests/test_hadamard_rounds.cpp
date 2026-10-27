/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>

#include <memory>

namespace faiss {

TEST(HadamardRounds, RejectsInvalidConstructorCount) {
    for (int rounds : {-1, 0, 4}) {
        EXPECT_THROW(HadamardRotation(8, 42, rounds), FaissException);
    }
}

TEST(HadamardRounds, RejectsInvalidMutableCountBeforeAccessingBuffers) {
    HadamardRotation transform(8);
    HadamardRotation other(8);
    for (int rounds : {-1, 0, 4}) {
        transform.rounds = rounds;
        EXPECT_THROW(transform.init(42), FaissException);
        EXPECT_THROW(transform.train(0, nullptr), FaissException);
        EXPECT_THROW(
                transform.apply_noalloc(0, nullptr, nullptr), FaissException);
        EXPECT_THROW(
                transform.reverse_transform(0, nullptr, nullptr),
                FaissException);
        EXPECT_THROW(transform.check_identical(other), FaissException);
        EXPECT_THROW(other.check_identical(transform), FaissException);
        VectorIOWriter writer;
        EXPECT_THROW(
                write_VectorTransform(&transform, &writer), FaissException);
        EXPECT_TRUE(writer.data.empty());
    }
}

TEST(HadamardRounds, EmptySquareBatchAndRoundTrip) {
    for (int rounds : {1, 2, 3}) {
        HadamardRotation transform(8, 42, rounds);
        EXPECT_NO_THROW(transform.apply_noalloc(0, nullptr, nullptr));
        EXPECT_NO_THROW(transform.reverse_transform(0, nullptr, nullptr));
        float x[8] = {1, 2, 3, 4, -1, -2, -3, -4};
        float y[8], restored[8];
        transform.apply_noalloc(1, x, y);
        transform.reverse_transform(1, y, restored);
        for (int j = 0; j < 8; j++) {
            EXPECT_NEAR(x[j], restored[j], 1e-5);
        }
    }
}

TEST(HadamardRounds, LegacySerializationBytes) {
    HadamardRotation transform(8, 42);
    const std::vector<uint8_t> legacy = {
            'H', 'R', 'o', 't', 42, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 1};
    VectorIOWriter writer;
    write_VectorTransform(&transform, &writer);
    EXPECT_EQ(writer.data, legacy);
    VectorIOReader reader;
    reader.data = legacy;
    std::unique_ptr<VectorTransform> restored(read_VectorTransform(&reader));
    auto* hr = dynamic_cast<HadamardRotation*>(restored.get());
    ASSERT_NE(hr, nullptr);
    EXPECT_EQ(hr->rounds, 3);
    EXPECT_NO_THROW(transform.check_identical(*hr));
}

} // namespace faiss
