// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIVFSQScan.h>
#include <faiss/gpu_metal/MetalIndexIVFScalarQuantizer.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/impl/IDSelector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace faiss;
using namespace faiss::gpu_metal;
namespace {
std::vector<float> randomData(size_t n, int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2, 2);
    std::vector<float> x(n);
    for (auto& v : x)
        v = dist(rng);
    return x;
}
id<MTLBuffer> buffer(
        id<MTLDevice> device,
        size_t size,
        const void* data = nullptr) {
    auto b = [device newBufferWithLength:std::max(size, size_t(8))
                                 options:MTLResourceStorageModeShared];
    if (data && size)
        memcpy(b.contents, data, size);
    return b;
}
} // namespace
class TestMetalIndexIVFSQ : public ::testing::Test {
   protected:
    std::shared_ptr<MetalResources> res;
    void SetUp() override {
        res = std::make_shared<MetalResources>();
        if (!res->isAvailable())
            GTEST_SKIP() << "Metal unavailable";
    }
};

// Invoke the shader entry directly; the oracle decodes CPU codes independently
// and sorts all candidates by score and final 64-bit label.
TEST_F(TestMetalIndexIVFSQ, DirectExactSegmentsAndTies) {
    for (auto type : {ScalarQuantizer::QT_8bit, ScalarQuantizer::QT_fp16})
        for (int d : {1, 7, 33, 513})
            for (bool l2 : {false, true})
                for (bool residual : {false, true}) {
                    const int nq = 2, probes = 4, lists = 3;
                    auto training = randomData(128 * d);
                    ScalarQuantizer sq(d, type);
                    sq.train(128, training.data());
                    if (type == ScalarQuantizer::QT_8bit) {
                        std::fill_n(sq.trained.data(), d, 0.f);
                        std::fill_n(sq.trained.data() + d, d, 255.f);
                    }
                    auto q = randomData(nq * d, 88),
                         c = randomData(lists * d, 99);
                    for (int dim = 0; dim < d; ++dim) {
                        for (int qi = 0; qi < nq; ++qi)
                            q[qi * d + dim] = float(dim % 3) - 1;
                        for (int l = 0; l < lists; ++l)
                            c[l * d + dim] = float(l + dim % 2) - 1;
                    }
                    auto dev = res->getDevice();
                    auto qb = buffer(dev, q.size() * 4, q.data());
                    auto cb = buffer(dev, c.size() * 4, c.data());
                    auto rb = buffer(
                            dev, sq.trained.size() * 4, sq.trained.data());
                    int32_t assign[] = {0, -1, 2, 1, 2, 1, 0, -1};
                    auto ab = buffer(dev, sizeof(assign), assign);
                    float scores[nq * probes] = {};
                    for (int qi = 0; qi < nq; ++qi)
                        for (int probe = 0; probe < probes; ++probe) {
                            int l = assign[qi * probes + probe];
                            if (l >= 0)
                                for (int dim = 0; dim < d; ++dim)
                                    scores[qi * probes + probe] +=
                                            q[qi * d + dim] * c[l * d + dim];
                        }
                    auto sb = buffer(dev, sizeof(scores), scores);
                    for (uint32_t size :
                         {0,
                          1,
                          127,
                          128,
                          129,
                          511,
                          512,
                          513,
                          1024,
                          1025,
                          2051}) {
                        // Empty middle list, one extra vector in the last list.
                        uint32_t offsets[] = {0, size, size},
                                 lengths[] = {size, 0, 1};
                        size_t nb = size + 1;
                        std::vector<float> x(nb * d);
                        for (size_t v = 0; v < nb; ++v)
                            for (int dim = 0; dim < d; ++dim)
                                x[v * d + dim] =
                                        float(v % 32) + (dim % 4) * 0.25f;
                        // Exact duplicate decoded codes, deliberately reverse
                        // label order.
                        if (nb > 1)
                            std::copy_n(x.data(), d, x.data() + (nb - 1) * d);
                        std::vector<uint8_t> code(nb * sq.code_size);
                        sq.compute_codes(x.data(), code.data(), nb);
                        std::vector<float> decoded(nb * d);
                        sq.decode(code.data(), decoded.data(), nb);
                        std::vector<idx_t> id(nb);
                        for (size_t j = 0; j < nb; ++j)
                            id[j] = (idx_t(1) << 42) + nb - j;
                        auto codes = buffer(dev, code.size(), code.data());
                        auto ids = buffer(dev, id.size() * 8, id.data());
                        auto off = buffer(dev, sizeof(offsets), offsets);
                        auto len = buffer(dev, sizeof(lengths), lengths);
                        for (uint32_t k : {1, 4, 33, 512}) {
                            auto outD = buffer(dev, nq * k * 4),
                                 outI = buffer(dev, nq * k * 8);
                            uint32_t p[] = {
                                    nq,
                                    uint32_t(d),
                                    k,
                                    probes,
                                    lists,
                                    type == ScalarQuantizer::QT_fp16,
                                    residual,
                                    l2,
                                    uint32_t(sq.code_size)};
                            runMetalIVFSQScan(
                                    dev,
                                    res->getCommandQueue(),
                                    qb,
                                    codes,
                                    ids,
                                    off,
                                    len,
                                    ab,
                                    cb,
                                    rb,
                                    outD,
                                    outI,
                                    sb,
                                    p);
                            for (int qi = 0; qi < nq; ++qi) {
                                std::vector<std::pair<float, idx_t>> expected;
                                for (int probe = 0; probe < probes; ++probe) {
                                    int l = assign[qi * probes + probe];
                                    if (l < 0)
                                        continue;
                                    for (uint32_t j = 0; j < lengths[l]; ++j) {
                                        size_t v = offsets[l] + j;
                                        float score = 0, coarse = 0;
                                        for (int dim = 0; dim < d; ++dim) {
                                            float query = q[qi * d + dim];
                                            if (residual && l2)
                                                query -= c[l * d + dim];
                                            float value = decoded[v * d + dim],
                                                  delta = query - value;
                                            score += l2 ? delta * delta
                                                        : query * value;
                                            if (residual && !l2)
                                                coarse += q[qi * d + dim] *
                                                        c[l * d + dim];
                                        }
                                        expected.emplace_back(
                                                score + coarse, id[v]);
                                    }
                                }
                                std::sort(
                                        expected.begin(),
                                        expected.end(),
                                        [=](auto a, auto b) {
                                            return a.first == b.first
                                                    ? a.second < b.second
                                                    : (l2 ? a.first < b.first
                                                          : a.first > b.first);
                                        });
                                for (uint32_t j = 0; j < k; ++j) {
                                    idx_t actual =
                                            ((idx_t*)outI.contents)[qi * k + j];
                                    float score =
                                            ((float*)outD.contents)[qi * k + j];
                                    if (j >= expected.size()) {
                                        ASSERT_EQ(actual, -1);
                                        ASSERT_EQ(
                                                score,
                                                l2 ? std::numeric_limits<
                                                             float>::max()
                                                   : -std::numeric_limits<
                                                             float>::max());
                                    } else {
                                        ASSERT_EQ(actual, expected[j].second)
                                                << "d=" << d << " size=" << size
                                                << " k=" << k << " j=" << j
                                                << " type=" << type
                                                << " L2=" << l2
                                                << " residual=" << residual
                                                << " expected score="
                                                << expected[j].first
                                                << " actual score=" << score;
                                        ASSERT_NEAR(
                                                score,
                                                expected[j].first,
                                                2e-4f *
                                                        std::max(
                                                                1.f,
                                                                std::abs(
                                                                        expected[j]
                                                                                .first)));
                                    }
                                }
                            }
                        }
                    }
                }
}

TEST_F(TestMetalIndexIVFSQ, CloneAppendResetAndFallback) {
    for (auto type :
         {ScalarQuantizer::QT_8bit,
          ScalarQuantizer::QT_fp16,
          ScalarQuantizer::QT_4bit})
        for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT})
            for (bool residual : {false, true}) {
                const int d = 17, nb = 1025, nq = 3, k = 13;
                auto x = randomData(nb * d), q = randomData(nq * d, 88);
                IndexFlat coarse(d, metric);
                IndexIVFScalarQuantizer cpu(
                        &coarse, d, 4, type, metric, residual);
                cpu.cp.min_points_per_centroid = 1;
                cpu.train(nb, x.data());
                cpu.nprobe = 4;
                cpu.add(513, x.data());
                MetalIndexIVFScalarQuantizer gpu(res, &cpu);
                cpu.add(nb - 513, x.data() + 513 * d);
                gpu.add(nb - 513, x.data() + 513 * d);
                std::vector<float> cd(nq * 513), gd(nq * 513);
                std::vector<idx_t> ci(nq * 513), gi(nq * 513);
                auto check = [&](idx_t count,
                                 const SearchParameters* p,
                                 bool dispatch) {
                    size_t before = gpu.gpuSearchCount();
                    cpu.search(nq, q.data(), count, cd.data(), ci.data(), p);
                    gpu.search(nq, q.data(), count, gd.data(), gi.data(), p);
                    ASSERT_EQ(gpu.gpuSearchCount(), before + size_t(dispatch));
                    for (int j = 0; j < nq * count; ++j) {
                        ASSERT_EQ(ci[j], gi[j]);
                        ASSERT_NEAR(
                                cd[j],
                                gd[j],
                                1e-4f * std::max(1.f, std::abs(cd[j])));
                    }
                    ASSERT_EQ(gpu.nprobe(), 4);
                };
                check(k, nullptr, type != ScalarQuantizer::QT_4bit);
                check(513, nullptr, false);
                SearchParametersIVF p;
                p.nprobe = 2;
                p.max_codes = 17;
                check(k, &p, false);
                IDSelectorRange selector(3, 19);
                p.max_codes = 0;
                p.sel = &selector;
                check(k, &p, false);
                std::unique_ptr<IndexIVFScalarQuantizer> back(gpu.copyTo());
                ASSERT_EQ(back->nprobe, cpu.nprobe);
                ASSERT_EQ(back->sq.trained, cpu.sq.trained);
                ASSERT_EQ(back->by_residual, residual);
                ASSERT_EQ(back->ntotal, nb);
                gpu.reset();
                cpu.reset();
                gpu.add(32, x.data());
                cpu.add(32, x.data());
                check(k, nullptr, type != ScalarQuantizer::QT_4bit);
            }
}

TEST_F(TestMetalIndexIVFSQ, OwningClonesAndConfiguration) {
    auto x = randomData(256 * 7);
    std::unique_ptr<MetalIndexIVFScalarQuantizer> gpu;
    {
        auto quantizer = std::make_unique<IndexFlatL2>(7);
        auto cpu = std::make_unique<IndexIVFScalarQuantizer>(
                quantizer.get(), 7, 2, ScalarQuantizer::QT_fp16);
        cpu->own_fields = true;
        quantizer.release();
        cpu->train(256, x.data());
        cpu->add(256, x.data());
        cpu->nprobe = 2;
        cpu->metric_arg = 3;
        cpu->verbose = true;
        cpu->make_direct_map();
        gpu = std::make_unique<MetalIndexIVFScalarQuantizer>(res, cpu.get());
    }
    std::vector<float> d(8), reconstructed(7);
    std::vector<idx_t> i(8);
    gpu->search(1, x.data(), 8, d.data(), i.data());
    ASSERT_EQ(gpu->gpuSearchCount(), 1);
    gpu->reconstruct(0, reconstructed.data());
    std::unique_ptr<IndexIVFScalarQuantizer> back(gpu->copyTo());
    ASSERT_EQ(back->metric_arg, 3);
    ASSERT_TRUE(back->verbose);
    ASSERT_EQ(back->direct_map.type, DirectMap::Array);
    gpu.reset();
    res.reset();
    back->search(1, x.data(), 8, d.data(), i.data());
    ASSERT_EQ(i[0], 0);
}

namespace {
struct TrackingLists : ArrayInvertedLists {
    mutable size_t acquiredCodes = 0, acquiredIds = 0, releasedCodes = 0,
                   releasedIds = 0;
    TrackingLists(size_t lists, size_t codeSize)
            : ArrayInvertedLists(lists, codeSize) {}
    const uint8_t* get_codes(size_t list) const override {
        ++acquiredCodes;
        return ArrayInvertedLists::get_codes(list);
    }
    const idx_t* get_ids(size_t list) const override {
        ++acquiredIds;
        return ArrayInvertedLists::get_ids(list);
    }
    void release_codes(size_t, const uint8_t*) const override {
        ++releasedCodes;
    }
    void release_ids(size_t, const idx_t*) const override {
        ++releasedIds;
    }
};
} // namespace
TEST_F(TestMetalIndexIVFSQ, CustomListsAndBorrowedCoarseStorage) {
    auto x = randomData(256 * 7);
    IndexFlatL2 coarse(7);
    IndexIVFScalarQuantizer cpu(&coarse, 7, 2, ScalarQuantizer::QT_fp16);
    cpu.train(256, x.data());
    cpu.add(256, x.data());
    cpu.nprobe = 2;
    auto lists = std::make_unique<TrackingLists>(2, cpu.code_size);
    for (size_t l = 0; l < 2; ++l) {
        InvertedLists::ScopedCodes codes(cpu.invlists, l);
        InvertedLists::ScopedIds ids(cpu.invlists, l);
        lists->add_entries(
                l, cpu.invlists->list_size(l), ids.get(), codes.get());
    }
    cpu.replace_invlists(lists.get(), false);
    std::vector<uint8_t> borrowed(
            coarse.codes.data(), coarse.codes.data() + coarse.codes.size());
    coarse.codes = MaybeOwnedVector<uint8_t>::create_view(
            borrowed.data(), borrowed.size(), nullptr);
    MetalIndexIVFScalarQuantizer gpu(res, &cpu);
    ASSERT_EQ(lists->acquiredCodes, lists->releasedCodes);
    ASSERT_EQ(lists->acquiredIds, lists->releasedIds);
    ASSERT_GT(lists->releasedCodes, 0);
    std::unique_ptr<IndexIVFScalarQuantizer> back(gpu.copyTo());
    borrowed.assign(borrowed.size(), 0);
    float cd[5], gd[5];
    idx_t ci[5], gi[5];
    back->search(1, x.data(), 5, cd, ci);
    gpu.search(1, x.data(), 5, gd, gi);
    for (int j = 0; j < 5; ++j) {
        ASSERT_EQ(ci[j], gi[j]);
        ASSERT_NEAR(cd[j], gd[j], 1e-4);
    }
}

TEST_F(TestMetalIndexIVFSQ, EmptyTrainedIndexAndInvalidBuffer) {
    MetalIndexIVFScalarQuantizer gpu(res, 7, 1, ScalarQuantizer::QT_fp16);
    auto x = randomData(128 * 7);
    gpu.train(128, x.data());
    gpu.reset();
    gpu.reserveMemory(128);
    float d[5];
    idx_t i[5];
    gpu.search(1, x.data(), 5, d, i);
    ASSERT_EQ(gpu.gpuSearchCount(), 1);
    for (int j = 0; j < 5; ++j) {
        ASSERT_EQ(i[j], -1);
        ASSERT_EQ(d[j], std::numeric_limits<float>::max());
    }
    auto b = buffer(res->getDevice(), 8);
    uint32_t p[] = {1, 7, 5, 1, 1, 1, 0, 1, 14};
    ASSERT_THROW(
            runMetalIVFSQScan(
                    res->getDevice(),
                    res->getCommandQueue(),
                    b,
                    b,
                    b,
                    b,
                    b,
                    b,
                    b,
                    b,
                    b,
                    b,
                    b,
                    p),
            std::exception);
}

TEST_F(TestMetalIndexIVFSQ, ReserveCompactAndCoarseScoreSemantics) {
    // Deliberately use an L2 coarse quantizer for an IP index. The CPU scanner
    // adds the actual coarse score, rather than recomputing a centroid dot.
    auto x = randomData(256 * 7);
    IndexFlatL2 coarse(7);
    IndexIVFScalarQuantizer cpu(
            &coarse,
            7,
            2,
            ScalarQuantizer::QT_fp16,
            METRIC_INNER_PRODUCT,
            true);
    cpu.train(256, x.data());
    cpu.add(128, x.data());
    cpu.nprobe = 2;
    MetalIndexIVFScalarQuantizer gpu(res, &cpu);
    gpu.reserveMemory(1024);
    gpu.add(128, x.data() + 128 * 7);
    cpu.add(128, x.data() + 128 * 7);
    gpu.reclaimMemory();
    float cd[5], gd[5];
    idx_t ci[5], gi[5];
    cpu.search(1, x.data(), 5, cd, ci);
    gpu.search(1, x.data(), 5, gd, gi);
    ASSERT_EQ(gpu.gpuSearchCount(), 1);
    for (int j = 0; j < 5; ++j) {
        ASSERT_EQ(ci[j], gi[j]);
        ASSERT_NEAR(cd[j], gd[j], 1e-4);
    }
}
