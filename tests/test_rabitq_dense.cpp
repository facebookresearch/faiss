/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

TEST(RaBitQDense, RoundTripPreservesLayout) {
    constexpr int d = 128;
    constexpr int nb = 32;
    constexpr int nq = 4;
    constexpr int k = 8;
    std::mt19937 rng(42);
    std::normal_distribution<float> distribution;
    std::vector<float> xb(nb * d), xq(nq * d);
    for (float& value : xb) {
        value = distribution(rng);
    }
    for (float& value : xq) {
        value = distribution(rng);
    }

    faiss::IndexRaBitQ original(d, faiss::METRIC_L2, 4, true);
    original.train(nb, xb.data());
    original.add(nb, xb.data());

    std::vector<float> before_distances(nq * k), after_distances(nq * k);
    std::vector<faiss::idx_t> before_labels(nq * k), after_labels(nq * k);
    original.search(
            nq,
            xq.data(),
            k,
            before_distances.data(),
            before_labels.data());

    faiss::VectorIOWriter writer;
    faiss::write_index(&original, &writer);
    ASSERT_GE(writer.data.size(), 4);
    EXPECT_EQ(
            std::string(writer.data.begin(), writer.data.begin() + 4),
            "Ixrd");

    faiss::VectorIOReader reader;
    reader.data = writer.data;
    std::unique_ptr<faiss::Index> restored_base(faiss::read_index(&reader));
    auto* restored =
            dynamic_cast<faiss::IndexRaBitQ*>(restored_base.get());
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->rabitq.dense_layout);
    EXPECT_EQ(restored->code_size, original.code_size);
    EXPECT_EQ(restored->codes, original.codes);

    restored->search(
            nq,
            xq.data(),
            k,
            after_distances.data(),
            after_labels.data());
    EXPECT_EQ(after_labels, before_labels);
    EXPECT_EQ(after_distances, before_distances);
    EXPECT_THROW(
            {
                faiss::IndexRaBitQFastScan converted(*restored);
            },
            faiss::FaissException);
}
