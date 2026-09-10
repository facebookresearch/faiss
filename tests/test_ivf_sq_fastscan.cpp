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
#include <faiss/IndexIVFSQFastScan.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

namespace {

std::vector<float> make_data(int n, int d, int seed = 42) {
    std::vector<float> x(n * d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x) {
        v = dist(rng);
    }
    return x;
}

} // namespace

TEST(IndexIVFSQFastScan, Construct4bit) {
    int d = 32, nlist = 8;
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_4bit);
    EXPECT_EQ(idx.d, d);
    EXPECT_EQ(idx.nlist, nlist);
}

TEST(IndexIVFSQFastScan, Construct8bit) {
    int d = 32, nlist = 8;
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_8bit);
    EXPECT_EQ(idx.d, d);
    EXPECT_EQ(idx.nlist, nlist);
}

TEST(IndexIVFSQFastScan, TrainAddSearch4bit) {
    int d = 32, nlist = 8, n = 1000;
    auto xt = make_data(n, d, 1);
    auto xb = make_data(n, d, 2);
    auto xq = make_data(10, d, 3);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_4bit);

    idx.train(n, xt.data());
    EXPECT_TRUE(idx.is_trained);

    idx.add(n, xb.data());
    EXPECT_EQ(idx.ntotal, n);

    idx.nprobe = nlist;
    std::vector<float> D(10 * 10);
    std::vector<faiss::idx_t> I(10 * 10);
    idx.search(10, xq.data(), 10, D.data(), I.data());

    for (int i = 0; i < 10 * 10; i++) {
        EXPECT_GE(I[i], 0);
    }
}

TEST(IndexIVFSQFastScan, TrainAddSearch8bit) {
    int d = 32, nlist = 8, n = 1000;
    auto xt = make_data(n, d, 1);
    auto xb = make_data(n, d, 2);
    auto xq = make_data(10, d, 3);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_8bit);

    idx.train(n, xt.data());
    idx.add(n, xb.data());
    EXPECT_EQ(idx.ntotal, n);

    idx.nprobe = nlist;
    std::vector<float> D(10 * 10);
    std::vector<faiss::idx_t> I(10 * 10);
    idx.search(10, xq.data(), 10, D.data(), I.data());

    for (int i = 0; i < 10 * 10; i++) {
        EXPECT_GE(I[i], 0);
    }
}

TEST(IndexIVFSQFastScan, FallbackFP16) {
    int d = 32, nlist = 8, n = 1000;
    auto xt = make_data(n, d, 1);
    auto xb = make_data(n, d, 2);
    auto xq = make_data(10, d, 3);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_fp16);

    idx.train(n, xt.data());
    idx.add(n, xb.data());
    EXPECT_EQ(idx.ntotal, n);

    idx.nprobe = nlist;
    std::vector<float> D(10 * 10);
    std::vector<faiss::idx_t> I(10 * 10);
    idx.search(10, xq.data(), 10, D.data(), I.data());

    for (int i = 0; i < 10 * 10; i++) {
        EXPECT_GE(I[i], 0);
    }
}

TEST(IndexIVFSQFastScan, FactoryString) {
    int d = 32;
    std::unique_ptr<faiss::Index> idx(faiss::index_factory(d, "IVF16,SQ8fs"));
    EXPECT_NE(dynamic_cast<faiss::IndexIVFSQFastScan*>(idx.get()), nullptr);
}

TEST(IndexIVFSQFastScan, FactoryStringWithBbs) {
    int d = 32;
    std::unique_ptr<faiss::Index> idx(
            faiss::index_factory(d, "IVF16,SQ8fs_64"));
    auto* ivfsqfs = dynamic_cast<faiss::IndexIVFSQFastScan*>(idx.get());
    ASSERT_NE(ivfsqfs, nullptr);
    EXPECT_EQ(ivfsqfs->bbs, 64);
}

TEST(IndexIVFSQFastScan, Reset) {
    int d = 32, nlist = 8, n = 500;
    auto xt = make_data(n, d, 1);
    auto xb = make_data(n, d, 2);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_8bit);
    idx.train(n, xt.data());
    idx.add(n, xb.data());
    EXPECT_EQ(idx.ntotal, n);

    idx.reset();
    EXPECT_EQ(idx.ntotal, 0);

    idx.add(n, xb.data());
    EXPECT_EQ(idx.ntotal, n);
}

TEST(IndexIVFSQFastScan, InnerProduct) {
    int d = 32, nlist = 8, n = 1000;
    auto xt = make_data(n, d, 1);
    auto xb = make_data(n, d, 2);
    auto xq = make_data(10, d, 3);

    faiss::IndexFlatIP quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer,
            d,
            nlist,
            faiss::ScalarQuantizer::QT_4bit,
            faiss::METRIC_INNER_PRODUCT);

    idx.train(n, xt.data());
    idx.add(n, xb.data());
    idx.nprobe = nlist;

    std::vector<float> D(10 * 10);
    std::vector<faiss::idx_t> I(10 * 10);
    idx.search(10, xq.data(), 10, D.data(), I.data());

    for (int i = 0; i < 10 * 10; i++) {
        EXPECT_GE(I[i], 0);
    }
}

TEST(IndexIVFSQFastScan, OddDimension) {
    int d = 33, nlist = 8, n = 500;
    auto xt = make_data(n, d, 1);
    auto xb = make_data(n, d, 2);
    auto xq = make_data(10, d, 3);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_4bit);
    idx.train(n, xt.data());
    idx.add(n, xb.data());
    idx.nprobe = nlist;

    std::vector<float> D(10 * 5);
    std::vector<faiss::idx_t> I(10 * 5);
    idx.search(10, xq.data(), 5, D.data(), I.data());

    for (int i = 0; i < 50; i++) {
        EXPECT_GE(I[i], 0);
    }
}
