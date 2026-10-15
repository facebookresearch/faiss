/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRowwiseInt8.h>
#include <faiss/gpu/GpuIndexRowwiseInt8.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

std::vector<float> make_data(size_t n, size_t d, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::uniform_real_distribution<float> scale(0.1f, 3.0f);
    std::vector<float> x(n * d);
    for (size_t i = 0; i < n; i++) {
        float s = scale(rng);
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] = normal(rng) * s;
        }
    }
    return x;
}

void compare_with_cpu(faiss::MetricType metric, int tileQ, int tileB) {
    size_t d = 128, nb = 5000, nq = 300; // nb % 64 != 0: padding rows
    int k = 10;
    auto xb = make_data(nb, d, 11);
    auto xq = make_data(nq, d, 12);

    faiss::IndexRowwiseInt8 cpu(d, metric);
    cpu.add(nb, xb.data());

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexRowwiseInt8Config config;
    config.device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    config.tileQueries = tileQ;
    config.tileBank = tileB;
    faiss::gpu::GpuIndexRowwiseInt8 gpu(&res, &cpu, config);
    EXPECT_EQ(gpu.ntotal, cpu.ntotal);

    std::vector<float> Dc(nq * k), Dg(nq * k);
    std::vector<faiss::idx_t> Ic(nq * k), Ig(nq * k);
    cpu.search(nq, xq.data(), k, Dc.data(), Ic.data());
    gpu.search(nq, xq.data(), k, Dg.data(), Ig.data());

    // same neighbour sets (ties aside), distances equal up to rounding
    size_t same = 0;
    float maxAbs = 0;
    for (size_t i = 0; i < nq * k; i++) {
        maxAbs = std::max(maxAbs, std::fabs(Dc[i]));
    }
    for (size_t q = 0; q < nq; q++) {
        std::vector<faiss::idx_t> a(Ic.begin() + q * k, Ic.begin() + q * k + k);
        std::vector<faiss::idx_t> b(Ig.begin() + q * k, Ig.begin() + q * k + k);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        same += (a == b);
        for (int r = 0; r < k; r++) {
            EXPECT_NEAR(Dg[q * k + r], Dc[q * k + r], 1e-5f * maxAbs);
        }
    }
    EXPECT_GE(same, nq - 1);

    // round trip back to the CPU keeps the codes
    faiss::IndexRowwiseInt8 back;
    gpu.copyTo(&back);
    EXPECT_EQ(back.ntotal, cpu.ntotal);
    EXPECT_EQ(
            0,
            std::memcmp(back.codes.data(), cpu.codes.data(), cpu.codes.size()));
}

} // namespace

TEST(TestGpuIndexRowwiseInt8, L2) {
    compare_with_cpu(faiss::METRIC_L2, 4096, 32768);
}

TEST(TestGpuIndexRowwiseInt8, IP) {
    compare_with_cpu(faiss::METRIC_INNER_PRODUCT, 4096, 32768);
}

TEST(TestGpuIndexRowwiseInt8, Tiled) {
    compare_with_cpu(faiss::METRIC_L2, 64, 1024);
}

TEST(TestGpuIndexRowwiseInt8, EmptyAndSmall) {
    size_t d = 64;
    int k = 5;
    auto xq = make_data(4, d, 13);
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexRowwiseInt8 gpu(&res, d);
    std::vector<float> D(4 * k);
    std::vector<faiss::idx_t> I(4 * k);
    gpu.search(4, xq.data(), k, D.data(), I.data());
    for (auto i : I) {
        EXPECT_EQ(i, -1);
    }
    auto xb = make_data(3, d, 14);
    gpu.add(3, xb.data());
    gpu.search(4, xq.data(), k, D.data(), I.data());
    for (int q = 0; q < 4; q++) {
        for (int r = 0; r < k; r++) {
            EXPECT_EQ(I[q * k + r] >= 0, r < 3);
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
