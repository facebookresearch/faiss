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
#include <faiss/impl/FaissAssert.h>
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

TEST(IndexIVFSQFastScan, UnsupportedTypeThrows) {
    int d = 32, nlist = 8;
    faiss::IndexFlatL2 quantizer(d);
    // Only native 4-bit types are supported; everything else must throw
    // (use IndexRefine or IndexIVFScalarQuantizer instead).
    EXPECT_THROW(
            faiss::IndexIVFSQFastScan(
                    &quantizer, d, nlist, faiss::ScalarQuantizer::QT_8bit),
            faiss::FaissException);
    EXPECT_THROW(
            faiss::IndexIVFSQFastScan(
                    &quantizer, d, nlist, faiss::ScalarQuantizer::QT_fp16),
            faiss::FaissException);
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

TEST(IndexIVFSQFastScan, FactoryString) {
    int d = 32;
    std::unique_ptr<faiss::Index> idx(faiss::index_factory(d, "IVF16,SQ4fs"));
    EXPECT_NE(dynamic_cast<faiss::IndexIVFSQFastScan*>(idx.get()), nullptr);
}

TEST(IndexIVFSQFastScan, FactoryStringWithBbs) {
    int d = 32;
    std::unique_ptr<faiss::Index> idx(
            faiss::index_factory(d, "IVF16,SQ4fs_64"));
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
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_4bit);
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

TEST(IndexIVFSQFastScan, IOAndReconstruct) {
    int d = 32, nlist = 8, n = 1000;
    auto xt = make_data(n, d, 1);
    auto xb = make_data(n, d, 2);
    auto xq = make_data(10, d, 3);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFSQFastScan idx(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_4bit);
    idx.train(n, xt.data());
    idx.add(n, xb.data());
    idx.nprobe = nlist;

    std::vector<float> D1(10 * 5);
    std::vector<faiss::idx_t> I1(10 * 5);
    idx.search(10, xq.data(), 5, D1.data(), I1.data());

    // Serialize / deserialize and confirm identical results. This exercises
    // the read path that must wire fine_quantizer for the base sa_decode /
    // reconstruct to work.
    faiss::write_index(&idx, "/tmp/ivfsqfs_test.faissindex");
    std::unique_ptr<faiss::Index> loaded(
            faiss::read_index("/tmp/ivfsqfs_test.faissindex"));
    auto* idx2 = dynamic_cast<faiss::IndexIVFSQFastScan*>(loaded.get());
    ASSERT_NE(idx2, nullptr);
    idx2->nprobe = nlist;

    std::vector<float> D2(10 * 5);
    std::vector<faiss::idx_t> I2(10 * 5);
    idx2->search(10, xq.data(), 5, D2.data(), I2.data());
    for (int i = 0; i < 50; i++) {
        EXPECT_EQ(I1[i], I2[i]);
        EXPECT_FLOAT_EQ(D1[i], D2[i]);
    }

    // reconstruct via the inherited base path (fine_quantizer -> sq.decode).
    idx2->make_direct_map();
    std::vector<float> recon(d);
    idx2->reconstruct(0, recon.data());
    float err = 0;
    for (int j = 0; j < d; j++) {
        float e = recon[j] - xb[j];
        err += e * e;
    }
    EXPECT_LT(std::sqrt(err), 5.0f);
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
