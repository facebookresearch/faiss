/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/index_factory.h>

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

TEST(TestSQ0bit, CoarseOnlySearch) {
    // Test QT_0bit: centroid-only distance
    int d = 64;
    int nlist = 8;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    std::vector<float> xb(nb * d), xq(nq * d);
    for (int i = 0; i < nb * d; i++) {
        xb[i] = distrib(rng);
    }
    for (int i = 0; i < nq * d; i++) {
        xq[i] = distrib(rng);
    }

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFScalarQuantizer index(
            &quantizer,
            d,
            nlist,
            faiss::ScalarQuantizer::QT_0bit,
            faiss::METRIC_L2,
            false);
    EXPECT_EQ(index.code_size, 0);
    EXPECT_FALSE(index.by_residual);

    index.train(nb, xb.data());
    index.add(nb, xb.data());
    EXPECT_EQ(index.ntotal, nb);

    index.nprobe = nlist;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // Verify we got results
    for (int q = 0; q < nq; q++) {
        EXPECT_GE(labels[q * k], 0);
    }

    // Compare with direct quantizer search - distances should match
    std::vector<float> coarse_dis(nq * nlist);
    std::vector<faiss::idx_t> coarse_ids(nq * nlist);
    quantizer.search(
            nq, xq.data(), nlist, coarse_dis.data(), coarse_ids.data());

    for (int q = 0; q < nq; q++) {
        float ivf_dis = distances[q * k];
        bool found = false;
        for (int j = 0; j < nlist; j++) {
            if (std::abs(ivf_dis - coarse_dis[q * nlist + j]) < 1e-5) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "IVF distance " << ivf_dis
                           << " not found in coarse distances for query " << q;
    }
}

TEST(TestSQ0bit, IndexFactory) {
    int d = 32;
    std::unique_ptr<faiss::Index> index(faiss::index_factory(d, "IVF8,SQ0"));
    EXPECT_NE(index, nullptr);
    auto* ivfsq = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(index.get());
    EXPECT_NE(ivfsq, nullptr);
    EXPECT_EQ(ivfsq->sq.qtype, faiss::ScalarQuantizer::QT_0bit);
    EXPECT_EQ(ivfsq->code_size, 0);
}

TEST(TestSQ0bit, InnerProduct) {
    int d = 64;
    int nlist = 4;
    int nb = 500;
    int nq = 5;
    int k = 3;

    std::mt19937 rng2(43);
    std::uniform_real_distribution<float> distrib2(0.0f, 1.0f);
    std::vector<float> xb(nb * d), xq(nq * d);
    for (int i = 0; i < nb * d; i++) {
        xb[i] = distrib2(rng2);
    }
    for (int i = 0; i < nq * d; i++) {
        xq[i] = distrib2(rng2);
    }

    faiss::IndexFlatIP quantizer(d);
    faiss::IndexIVFScalarQuantizer index(
            &quantizer,
            d,
            nlist,
            faiss::ScalarQuantizer::QT_0bit,
            faiss::METRIC_INNER_PRODUCT,
            false);

    index.train(nb, xb.data());
    index.add(nb, xb.data());

    index.nprobe = nlist;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int q = 0; q < nq; q++) {
        EXPECT_GE(labels[q * k], 0);
    }
}
