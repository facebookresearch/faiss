/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <vector>

#include <faiss/impl/ScalarQuantizer.h>

TEST(ScalarQuantizer, RSQuantilesClamping) {
    int d = 8;
    int n = 100;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i % 100);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;

    sq.rangestat_arg = 0.05f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.0f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = -0.1f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.8f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.5f;
    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(ScalarQuantizer, RSQuantilesOddSize) {
    int d = 4;
    int n = 5;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;

    sq.rangestat_arg = 0.4f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.5f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.6f;
    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(ScalarQuantizer, RSQuantilesValidRange) {
    int d = 8;
    int n = 100;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.1f;

    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n);
    ASSERT_NO_THROW(sq.compute_codes(x.data(), codes.data(), n));

    std::vector<float> decoded(d * n);
    ASSERT_NO_THROW(sq.decode(codes.data(), decoded.data(), n));
}

TEST(ScalarQuantizer, RSQuantilesSmallDataset) {
    int d = 2;
    int n = 2;

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.1f;

    ASSERT_NO_THROW(sq.train(n, x.data()));
}
