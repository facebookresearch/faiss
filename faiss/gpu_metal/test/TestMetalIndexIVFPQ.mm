// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Tests for MetalIndexIVFPQ: train, add, search, reset, cloner round-trip.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexIVFPQ.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
#import <memory>
#import <set>
#import <vector>
#include <algorithm>
#include <map>

namespace {

void compareSearchLabels(
        int nq,
        int k,
        const faiss::idx_t* refLab,
        const faiss::idx_t* testLab) {
    int matches = 0;
    for (int q = 0; q < nq; ++q) {
        std::set<faiss::idx_t> refSet(refLab + q * k, refLab + (q + 1) * k);
        for (int j = 0; j < k; ++j) {
            if (refSet.count(testLab[q * k + j]))
                ++matches;
        }
    }
    EXPECT_GT(matches, nq * k / 2);
}

} // namespace

class TestMetalIndexIVFPQ : public ::testing::Test {
   protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        if (!resources_->isAvailable()) {
            GTEST_SKIP() << "Metal not available";
        }
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

TEST_F(TestMetalIndexIVFPQ, L2_TrainAddSearch) {
    const int dim = 64;
    const int nb = 2000;
    const int nq = 10;
    const int nlist = 8;
    const int nprobe = 4;
    const int k = 5;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFPQ cpuIndex(&quantizer, dim, nlist, M, nbits);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFPQ metalIndex(resources_, &cpuIndex);
    ASSERT_EQ(metalIndex.ntotal, nb);

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> gpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> gpuLab((size_t)nq * k);
    metalIndex.search(nq, queries.data(), k, gpuDist.data(), gpuLab.data());

    compareSearchLabels(nq, k, refLab.data(), gpuLab.data());
}

TEST_F(TestMetalIndexIVFPQ, IP_TrainAddSearch) {
    const int dim = 64;
    const int nb = 2000;
    const int nq = 10;
    const int nlist = 8;
    const int nprobe = 4;
    const int k = 5;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatIP quantizer(dim);
    faiss::IndexIVFPQ cpuIndex(
            &quantizer, dim, nlist, M, nbits, faiss::METRIC_INNER_PRODUCT);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFPQ metalIndex(resources_, &cpuIndex);
    ASSERT_EQ(metalIndex.ntotal, nb);

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> gpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> gpuLab((size_t)nq * k);
    metalIndex.search(nq, queries.data(), k, gpuDist.data(), gpuLab.data());

    compareSearchLabels(nq, k, refLab.data(), gpuLab.data());
}

TEST_F(TestMetalIndexIVFPQ, Reset) {
    const int dim = 64;
    const int nb = 500;
    const int nlist = 4;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);

    faiss::gpu_metal::MetalIndexIVFPQ metalIndex(
            resources_, dim, nlist, M, nbits, faiss::METRIC_L2);
    metalIndex.train(nb, vecs.data());
    metalIndex.add(nb, vecs.data());
    ASSERT_EQ(metalIndex.ntotal, nb);

    metalIndex.reset();
    ASSERT_EQ(metalIndex.ntotal, 0);
}

TEST_F(TestMetalIndexIVFPQ, CpuToGpuRoundTrip) {
    const int dim = 64;
    const int nb = 1000;
    const int nlist = 8;
    const int M = 8;
    const int nbits = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 99);

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFPQ cpuIndex(&quantizer, dim, nlist, M, nbits);
    cpuIndex.train(nb, vecs.data());
    cpuIndex.add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    auto* metalRaw = dynamic_cast<faiss::gpu_metal::MetalIndexIVFPQ*>(
            faiss::gpu_metal::index_cpu_to_metal_gpu(&stdRes, 0, &cpuIndex));
    ASSERT_NE(metalRaw, nullptr);
    std::unique_ptr<faiss::gpu_metal::MetalIndexIVFPQ> metalIndex(metalRaw);
    ASSERT_EQ(metalIndex->ntotal, nb);

    auto* cpuBack = faiss::gpu_metal::index_metal_gpu_to_cpu(metalIndex.get());
    ASSERT_NE(cpuBack, nullptr);
    std::unique_ptr<faiss::Index> cpuBackPtr(cpuBack);
    ASSERT_EQ(cpuBack->ntotal, nb);
}

TEST_F(TestMetalIndexIVFPQ, LargeMSearchParametersKeepCpuSemantics) {
    constexpr int d = 128, nb = 2000, nq = 4, k = 20;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 123);
    faiss::float_rand(xq.data(), xq.size(), 456);
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFPQ cpu(&quantizer, d, 8, 32, 8);
    cpu.nprobe = 2;
    cpu.train(nb, xb.data());
    cpu.add(nb, xb.data());
    faiss::gpu_metal::MetalIndexIVFPQ metal(resources_, &cpu);
    faiss::IDSelectorRange selector(0, 10);
    for (bool useSelector : {true, false}) {
        faiss::IVFPQSearchParameters params;
        params.nprobe = 8;
        params.sel = useSelector ? &selector : nullptr;
        params.max_codes = useSelector ? 0 : 12;
        std::vector<float> cpuD(nq * k), gpuD(nq * k);
        std::vector<faiss::idx_t> cpuI(nq * k), gpuI(nq * k);
        cpu.search(nq, xq.data(), k, cpuD.data(), cpuI.data(), &params);
        metal.search(nq, xq.data(), k, gpuD.data(), gpuI.data(), &params);
        EXPECT_EQ(cpuI, gpuI);
        EXPECT_EQ(cpuD, gpuD);
        EXPECT_EQ(metal.nprobe(), 2);
    }
}

// Exercise the Metal entry point directly: a CPU fallback cannot satisfy this
// test. The oracle reconstructs PQ vectors and computes their distances,
// without using the GPU's lookup-table decomposition or selection
// implementation.
TEST_F(TestMetalIndexIVFPQ, ScanVariantsAgainstReconstruction) {
    struct Config {
        int M;
        int d;
        bool onTheFly;
    };
    const Config configs[] = {
            {8, 128, false},
            {16, 128, false},
            {8, 128, true},
            {16, 128, true},
            {17, 136, true},
            {32, 128, true},
            {48, 192, true},
            {64, 128, true},
            {32, 512, false},
            {64, 512, false}};
    for (const auto& config : configs) {
        for (int shape : {0, 1, 2, 3, 4}) {
            const bool shortLists = shape != 1 && shape != 3;
            for (bool isL2 : {true, false}) {
                @autoreleasepool {
                    const int M = config.M, d = config.d, dsub = d / M;
                    const int nq = 3;
                    std::vector<uint32_t> lengths = shortLists
                            ? std::vector<uint32_t>{0, 1, 6, 24, 127, 128}
                            : std::vector<uint32_t>{0, 1, 6, 24, 127, 128,
                                  129, 511, 512, 513, 1025, 2051, 3};
                    if (shape == 2)
                        lengths = {0, 0, 0};
                    else if (shape == 3)
                        lengths = {2051};
                    else if (shape == 4)
                        lengths = {128};
                    const int nl = lengths.size(), np = nl;
                    std::vector<uint32_t> offsets(nl);
                    size_t nb = 0;
                    for (int l = 0; l < nl; ++l) {
                        offsets[l] = nb;
                        nb += lengths[l];
                    }
                    std::vector<float> queries(nq * d), coarse(nl * d),
                            pq(d * 256);
                    faiss::float_rand(queries.data(), queries.size(), 43);
                    faiss::float_rand(coarse.data(), coarse.size(), 44);
                    faiss::float_rand(pq.data(), pq.size(), 45);
                    std::vector<uint8_t> codes(nb * M);
                    for (size_t i = 0; i < codes.size(); ++i)
                        codes[i] = (i * 131 + i / 7 + i / 257) % 256;
                    std::vector<int64_t> ids(nb);
                    for (size_t i = 0; i < nb; ++i)
                        ids[i] = (int64_t(1) << 40) + i * 17;
                    std::vector<int32_t> assign(nq * np);
                    std::vector<float> coarseDist(nq * np);
                    std::vector<std::vector<std::pair<float, int64_t>>> ref(nq);
                    std::vector<std::map<int64_t, float>> score(nq);
                    for (int q = 0; q < nq; ++q) {
                        for (int p = 0; p < np; ++p) {
                            int l = (p + q) % nl;
                            assign[q * np + p] = np > 1 && p == np - 1 ? -1 : l;
                            if (assign[q * np + p] < 0)
                                continue;
                            for (int j = 0; j < d; ++j) {
                                float x = queries[q * d + j],
                                      c = coarse[l * d + j];
                                coarseDist[q * np + p] +=
                                        isL2 ? (x - c) * (x - c) : x * c;
                            }
                            for (uint32_t v = offsets[l];
                                 v < offsets[l] + lengths[l];
                                 ++v) {
                                double distance = 0;
                                for (int j = 0; j < d; ++j) {
                                    float reconstructed = coarse[l * d + j] +
                                            pq[((j / dsub) * 256 +
                                                codes[size_t(v) * M +
                                                      j / dsub]) *
                                                       dsub +
                                               j % dsub];
                                    double x = queries[q * d + j];
                                    distance += isL2 ? (x - reconstructed) *
                                                    (x - reconstructed)
                                                     : x * reconstructed;
                                }
                                score[q][ids[v]] = distance;
                                ref[q].emplace_back(distance, ids[v]);
                            }
                        }
                        std::sort(
                                ref[q].begin(),
                                ref[q].end(),
                                [isL2](const auto& a, const auto& b) {
                                    return a.first == b.first
                                            ? a.second < b.second
                                            : (isL2 ? a.first < b.first
                                                    : a.first > b.first);
                                });
                    }
                    auto device = resources_->getDevice();
                    auto buffer = [&](const void* src, size_t bytes) {
                        if (bytes == 0)
                            return [device
                                    newBufferWithLength:1
                                                options:MTLResourceStorageModeShared];
                        return src
                                ? [device newBufferWithBytes:src
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared]
                                : [device newBufferWithLength:bytes
                                                      options:MTLResourceStorageModeShared];
                    };
                    auto qb = buffer(queries.data(), queries.size() * 4);
                    auto cb = buffer(coarse.data(), coarse.size() * 4);
                    auto pb = buffer(pq.data(), pq.size() * 4);
                    auto codesb = buffer(codes.data(), codes.size());
                    auto ib = buffer(ids.data(), ids.size() * 8);
                    auto ob = buffer(offsets.data(), offsets.size() * 4);
                    auto lb = buffer(lengths.data(), lengths.size() * 4);
                    auto ab = buffer(assign.data(), assign.size() * 4);
                    auto db = buffer(coarseDist.data(), coarseDist.size() * 4);
                    auto qt = buffer(nullptr, size_t(nq) * M * 256 * 4);
                    auto t2 = buffer(nullptr, size_t(nl) * M * 256 * 4);
                    if (isL2 && !config.onTheFly) {
                        ASSERT_TRUE(
                                faiss::gpu_metal::runMetalIVFPQPrecomputeTerm2(
                                        device,
                                        resources_->getCommandQueue(),
                                        cb,
                                        pb,
                                        t2,
                                        nl,
                                        d,
                                        M));
                    }
                    for (int k : {1, 20, 128, 512}) {
                        for (bool compact : {false, true}) {
                            SCOPED_TRACE(
                                    ::testing::Message()
                                    << "M=" << M << " d=" << d
                                    << " short=" << shortLists << " L2=" << isL2
                                    << " onTheFly=" << config.onTheFly
                                    << " k=" << k);
                            auto outD = buffer(nullptr, nq * k * 4),
                                 outI = buffer(nullptr, nq * k * 8);
                            auto perD = buffer(nullptr, nq * np * k * 4),
                                 perI = buffer(nullptr, nq * np * k * 8);
                            auto tmpD = buffer(nullptr, nq * np * k * 4),
                                 tmpI = buffer(nullptr, nq * np * k * 8);
                            ASSERT_TRUE(
                                    faiss::gpu_metal::
                                            runMetalIVFPQPrecompSearch(
                                                    device,
                                                    resources_
                                                            ->getCommandQueue(),
                                                    qb,
                                                    ab,
                                                    db,
                                                    t2,
                                                    qt,
                                                    pb,
                                                    codesb,
                                                    ib,
                                                    ob,
                                                    lb,
                                                    nq,
                                                    d,
                                                    M,
                                                    k,
                                                    np,
                                                    isL2,
                                                    true,
                                                    outD,
                                                    outI,
                                                    perD,
                                                    perI,
                                                    tmpD,
                                                    tmpI,
                                                    true,
                                                    cb,
                                                    config.onTheFly,
                                                    shortLists,
                                                    compact));
                            const auto* gd =
                                    static_cast<const float*>(outD.contents);
                            const auto* gi =
                                    static_cast<const int64_t*>(outI.contents);
                            for (int q = 0; q < nq; ++q) {
                                std::set<int64_t> seen;
                                for (int j = 0; j < k; ++j) {
                                    int pos = q * k + j;
                                    if (j >= ref[q].size()) {
                                        EXPECT_EQ(gi[pos], -1);
                                        EXPECT_GE(
                                                isL2 ? gd[pos] : -gd[pos],
                                                1e38f);
                                        continue;
                                    }
                                    ASSERT_TRUE(score[q].count(gi[pos]));
                                    EXPECT_TRUE(seen.insert(gi[pos]).second);
                                    EXPECT_NEAR(
                                            gd[pos], score[q][gi[pos]], 0.002);
                                    EXPECT_NEAR(
                                            gd[pos], ref[q][j].first, 0.002);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
