/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissException.h>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

namespace {

faiss::gpu::GpuIndexFlatConfig makeConfig(bool useFloat16) {
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = 0;
    config.useFloat16 = useFloat16;
    return config;
}

} // namespace

TEST(TestGpuIndexFlatNonFinite, AddRejectsNonFinite) {
    faiss::gpu::StandardGpuResources res;
    for (bool fp16 : {false, true}) {
        faiss::gpu::GpuIndexFlatL2 index(&res, 2, makeConfig(fp16));
        std::vector<float> xb = {NAN, 1, 0.5f, 1};
        EXPECT_THROW(index.add(2, xb.data()), faiss::FaissException);
        xb = {std::numeric_limits<float>::infinity(), 1, 0.5f, 1};
        EXPECT_THROW(index.add(2, xb.data()), faiss::FaissException);
        EXPECT_EQ(index.ntotal, 0);
    }
}

TEST(TestGpuIndexFlatNonFinite, AddKeepsFiniteData) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatL2 index(&res, 2, makeConfig(false));
    std::vector<float> xb = {1, 0, 0, 1};
    index.add(2, xb.data());
    std::vector<float> q = {0.6f, 0.8f};
    std::vector<float> d(2);
    std::vector<faiss::idx_t> l(2);
    index.search(1, q.data(), 2, d.data(), l.data());
    EXPECT_EQ(l[0], 1);
    EXPECT_EQ(l[1], 0);
}

TEST(TestGpuIndexFlatNonFinite, L2OverflowIsReported) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatL2 index(&res, 2, makeConfig(false));
    float a = 3e19f;
    std::vector<float> xb = {a, 0, 0, a};
    index.add(2, xb.data());
    std::vector<float> q = {0.6f * a, 0.8f * a};
    std::vector<float> d(2);
    std::vector<faiss::idx_t> l(2);
    EXPECT_THROW(
            index.search(1, q.data(), 2, d.data(), l.data()),
            faiss::FaissException);
}

TEST(TestGpuIndexFlatNonFinite, KLargerThanNtotalIsFine) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatL2 index(&res, 2, makeConfig(false));
    std::vector<float> xb = {1, 0, 0, 1};
    index.add(2, xb.data());
    std::vector<float> q = {0.6f, 0.8f};
    std::vector<float> d(5);
    std::vector<faiss::idx_t> l(5);
    index.search(1, q.data(), 5, d.data(), l.data());
    EXPECT_EQ(l[0], 1);
    EXPECT_EQ(l[1], 0);
    EXPECT_EQ(l[2], -1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
