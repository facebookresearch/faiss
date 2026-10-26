// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIVFSQScan.h>
#include <faiss/gpu_metal/MetalKernels.h>
#include <faiss/gpu_metal/MetalIndexIVFScalarQuantizer.h>
#include <faiss/gpu_metal/MetalIndexFlat.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/impl/IDSelector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>
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

// Keep first in this executable to exercise cold generic and specialized
// caches together. Every index is serialized individually, sharing a device.
TEST_F(TestMetalIndexIVFSQ, ConcurrentIndependentIndexes) {
    std::vector<std::unique_ptr<Index>> indexes;
    std::vector<std::vector<float>> queries;
    for (int dim : {128, 512}) {
        queries.emplace_back(dim, 0.f);
        IndexFlatL2 coarse(dim);
        coarse.add(1, queries.back().data());
        IndexIVFScalarQuantizer cpu(
                &coarse, dim, 1, ScalarQuantizer::QT_fp16, METRIC_L2, false);
        cpu.is_trained = true;
        cpu.add(1, queries.back().data());
        indexes.push_back(
                std::make_unique<MetalIndexIVFScalarQuantizer>(res, &cpu));
    }
    for (int dim : {7, 19}) {
        queries.emplace_back(dim, 0.f);
        auto flat = std::make_unique<MetalIndexFlat>(res, dim, METRIC_L2);
        flat->add(1, queries.back().data());
        indexes.push_back(std::move(flat));
    }
    std::atomic<int> ready{0};
    std::vector<std::exception_ptr> errors(indexes.size());
    std::vector<float> distances(indexes.size(), -1);
    std::vector<idx_t> labels(indexes.size(), -1);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < indexes.size(); ++i) {
        workers.emplace_back([&, i] {
            @autoreleasepool {
                ++ready;
                while (ready.load() < int(indexes.size()))
                    std::this_thread::yield();
                try {
                    for (int repeat = 0; repeat < 20; ++repeat)
                        indexes[i]->search(
                                1, queries[i].data(), 1,
                                &distances[i], &labels[i]);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    for (size_t i = 0; i < indexes.size(); ++i) {
        if (errors[i]) std::rethrow_exception(errors[i]);
        ASSERT_EQ(labels[i], 0);
        ASSERT_EQ(distances[i], 0.f);
        if (auto* sq = dynamic_cast<MetalIndexIVFScalarQuantizer*>(indexes[i].get()))
            ASSERT_EQ(sq->gpuSearchCount(), 20);
    }
}

TEST_F(TestMetalIndexIVFSQ, RejectBorrowedTrainingDependency) {
    auto x = randomData(128);
    std::unique_ptr<MetalIndexIVFScalarQuantizer> gpu;
    std::unique_ptr<IndexIVFScalarQuantizer> back;
    {
        IndexFlatL2 coarse(1), trainingAssignment(1);
        IndexIVFScalarQuantizer cpu(&coarse, 1, 2, ScalarQuantizer::QT_fp16);
        cpu.cp.niter = 1;
        cpu.cp.min_points_per_centroid = 1;
        gpu = std::make_unique<MetalIndexIVFScalarQuantizer>(res, &cpu);
        cpu.clustering_index = &trainingAssignment;
        EXPECT_THROW(
                MetalIndexIVFScalarQuantizer rejected(res, &cpu),
                std::exception);
        EXPECT_THROW(gpu->copyFrom(&cpu), std::exception);
        ASSERT_EQ(cpu.clustering_index, &trainingAssignment);
        // A rejected replacement preserves the previous owning index.
        back.reset(gpu->copyTo());
        ASSERT_EQ(back->clustering_index, nullptr);
        ASSERT_FALSE(back->is_trained);
    }
    // All original dependencies have left scope before either clone trains.
    gpu->train(128, x.data());
    gpu->add(128, x.data());
    float distance;
    idx_t label;
    gpu->search(1, x.data(), 1, &distance, &label);
    ASSERT_EQ(gpu->gpuSearchCount(), 1);
    ASSERT_GE(label, 0);
    gpu.reset();
    back->train(128, x.data());
    back->add(128, x.data());
    back->search(1, x.data(), 1, &distance, &label);
    ASSERT_GE(label, 0);
}

TEST_F(TestMetalIndexIVFSQ, SortedPartialRunsMergeExactly) {
    auto device = res->getDevice();
    for (bool l2 : {false, true})
        for (uint32_t k : {1, 10, 64, 128, 512})
            for (uint32_t parts : {1, 2, 3, 8, 17}) {
                SCOPED_TRACE(::testing::Message() << "l2=" << l2
                    << " k=" << k << " parts=" << parts);
                const uint32_t nq = 2, fanIn = std::min(64u, 1024u / k);
                uint32_t groups = (parts - 1) / fanIn + 1;
                float neutral = l2 ? std::numeric_limits<float>::max()
                                   : -std::numeric_limits<float>::max();
                std::vector<float> distances(nq * parts * k, -123.f);
                std::vector<idx_t> ids(nq * parts * k, -1);
                using Candidate = std::pair<float, idx_t>;
                auto before = [=](const Candidate& a, const Candidate& b) {
                    return a.first == b.first ? a.second < b.second
                        : (l2 ? a.first < b.first : a.first > b.first);
                };
                for (uint32_t q = 0; q < nq; ++q)
                    for (uint32_t run = 0; run < parts; ++run) {
                        uint32_t mode = (q + run) % 4;
                        uint32_t count = mode == 0 ? 0 : (mode == 1 ? 1 : k - run % k);
                        std::vector<Candidate> sorted;
                        for (uint32_t i = 0; i < count; ++i) {
                            float score = l2 ? float(i % 11) : -float(i % 11);
                            if (!l2 && i == 0) score = std::numeric_limits<float>::infinity();
                            sorted.emplace_back(score, (idx_t(1) << 42) + i % 5);
                        }
                        std::sort(sorted.begin(), sorted.end(), before);
                        size_t base = (q * parts + run) * k;
                        for (size_t i = 0; i < sorted.size(); ++i) {
                            distances[base+i] = sorted[i].first;
                            ids[base+i] = sorted[i].second;
                        }
                    }
                auto inputD = buffer(device, distances.size() * 4, distances.data());
                auto inputI = buffer(device, ids.size() * 8, ids.data());
                auto outputD = buffer(device, nq * groups * k * 4);
                auto outputI = buffer(device, nq * groups * k * 8);
                // Detect missing writers independently of allocator contents
                // or a previous merge producing coincidentally equal keys.
                std::fill_n((float*)outputD.contents, nq * groups * k,
                            std::numeric_limits<float>::quiet_NaN());
                std::fill_n((idx_t*)outputI.contents, nq * groups * k, idx_t(-777));
                uint32_t params[] = {nq, parts, fanIn, k, uint32_t(l2)};
                auto command = [res->getCommandQueue() commandBuffer];
                auto encoder = [command computeCommandEncoder];
                getMetalKernels(device).encodeIVFSQMerge(
                        encoder, inputD, inputI, outputD, outputI, params);
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                ASSERT_EQ(command.status, MTLCommandBufferStatusCompleted);
                for (uint32_t q = 0; q < nq; ++q)
                    for (uint32_t group = 0; group < groups; ++group) {
                        std::vector<Candidate> expected;
                        for (uint32_t run = group * fanIn;
                             run < std::min(parts, (group + 1) * fanIn); ++run)
                            for (uint32_t i = 0; i < k; ++i) {
                                size_t at = (q * parts + run) * k + i;
                                if (ids[at] >= 0) expected.emplace_back(distances[at], ids[at]);
                            }
                        std::sort(expected.begin(), expected.end(), before);
                        for (uint32_t i = 0; i < k; ++i) {
                            size_t at = (q * groups + group) * k + i;
                            ASSERT_EQ(((idx_t*)outputI.contents)[at],
                                      i < expected.size() ? expected[i].second : -1);
                            ASSERT_EQ(((float*)outputD.contents)[at],
                                      i < expected.size() ? expected[i].first : neutral);
                        }
                    }
            }
}

// Invoke the shader entry directly; the oracle decodes CPU codes independently
// and sorts all candidates by score and final 64-bit label.
TEST_F(TestMetalIndexIVFSQ, DirectExactSegmentsAndTies) {
    for (int nq : {2, 33})
    for (auto type : {ScalarQuantizer::QT_8bit, ScalarQuantizer::QT_fp16})
        for (int d : {1, 7, 33, 64, 128, 512, 513, 768})
            for (bool l2 : {false, true})
                for (bool residual : {false, true}) {
                    if (nq == 33 && d != 128) continue;
                    const int probes = 4, lists = 3;
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
                            q[qi * d + dim] = float(dim % 3) - 1 +
                                (nq == 33 ? float(qi % 4) * 0.25f : 0.f);
                        for (int l = 0; l < lists; ++l)
                            c[l * d + dim] = float(l + dim % 2) - 1;
                    }
                    auto dev = res->getDevice();
                    auto qb = buffer(dev, q.size() * 4, q.data());
                    auto cb = buffer(dev, c.size() * 4, c.data());
                    auto rb = buffer(
                            dev, sq.trained.size() * 4, sq.trained.data());
                    const int32_t pattern[] = {0, -1, 2, 1, 2, 1, 0, -1};
                    const int32_t pairedPattern[] = {0, 0, -1, 2, 2, 0, 0, 1};
                    std::vector<int32_t> assign(nq * probes);
                    for (int i = 0; i < nq * probes; ++i)
                        assign[i] = nq == 33 ? pairedPattern[i % 8] : pattern[i % 8];
                    // Mix short fused queries with long paired/unpaired ones
                    // in the same batch. A long query must not steal a probe
                    // from a neighbor that independently takes the fused path.
                    if (nq == 33) {
                        for (int qi = 3; qi < nq; qi += 4) {
                            assign[qi * probes] = 2;
                            assign[qi * probes + 1] = 1;
                            assign[qi * probes + 2] = -1;
                            assign[qi * probes + 3] = -1;
                        }
                    }
                    auto ab = buffer(dev, assign.size() * 4, assign.data());
                    std::vector<float> scores(nq * probes, 0.f);
                    for (int qi = 0; qi < nq; ++qi)
                        for (int probe = 0; probe < probes; ++probe) {
                            int l = assign[qi * probes + probe];
                            if (l >= 0)
                                for (int dim = 0; dim < d; ++dim)
                                    scores[qi * probes + probe] +=
                                            q[qi * d + dim] * c[l * d + dim];
                        }
                    auto sb = buffer(dev, scores.size() * 4, scores.data());
                    for (uint32_t size :
                         {0,
                          1,
                          127,
                          128,
                          129,
                          255,
                          256,
                          257,
                          511,
                          512,
                          513,
                          1024,
                          1025,
                          2051}) {
                        if (nq == 33 && size != 129 && size != 513 && size != 2051) continue;
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
                        for (uint32_t k : {1, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512}) {
                            if (nq == 33 && k != 1 && k != 16 && k != 32) continue;
                            auto outD = buffer(dev, nq * k * 4),
                                 outI = buffer(dev, nq * k * 8);
                            uint32_t p[] = {
                                    uint32_t(nq),
                                    uint32_t(d),
                                    k,
                                    probes,
                                    lists,
                                    type == ScalarQuantizer::QT_fp16,
                                    residual,
                                    l2,
                                    uint32_t(sq.code_size)};
                            SCOPED_TRACE(::testing::Message() << "nq=" << nq << " d=" << d
                                << " size=" << size << " k=" << k << " type=" << type
                                << " residual=" << residual << " l2=" << l2);
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

TEST_F(TestMetalIndexIVFSQ, NonFiniteScoresFollowCpuAdmission) {
    // Finite fp32 values outside the fp16 range encode to +/- infinity.
    // Exercise NaN (0 * infinity), both infinities, and finite overflow.
    std::vector<float> x = {
            100000, 1, 2, 3, -100000, std::numeric_limits<float>::quiet_NaN()};
    std::vector<float> queries = {0, 1, -1, std::numeric_limits<float>::max()};
    for (uint32_t dim : {1, 128, 512}) {
        std::vector<float> expanded(x.size() * dim, 0),
                expandedQueries(queries.size() * dim, 0);
        for (size_t i = 0; i < x.size(); ++i)
            expanded[i * dim] = x[i];
        for (size_t i = 0; i < queries.size(); ++i)
            expandedQueries[i * dim] = queries[i];
        ScalarQuantizer sq(dim, ScalarQuantizer::QT_fp16);
        std::vector<uint8_t> codes(x.size() * sq.code_size);
        sq.compute_codes(expanded.data(), codes.data(), x.size());
        std::vector<float> decoded(x.size() * dim);
        sq.decode(codes.data(), decoded.data(), x.size());
        auto dev = res->getDevice();
        std::vector<idx_t> ids(x.size());
        for (size_t j = 0; j < ids.size(); ++j)
            ids[j] = (idx_t(1) << 42) + j;
        uint32_t offset = 0, length = x.size();
        std::vector<int32_t> assignment(queries.size(), 0);
        auto qb =
                buffer(dev, expandedQueries.size() * 4, expandedQueries.data());
        auto cb = buffer(dev, codes.size(), codes.data());
        auto ib = buffer(dev, ids.size() * 8, ids.data());
        auto ob = buffer(dev, 4, &offset), lb = buffer(dev, 4, &length);
        auto ab = buffer(dev, assignment.size() * 4, assignment.data());
        std::vector<float> zeros(std::max(size_t(dim), queries.size()), 0);
        auto dummy = buffer(dev, zeros.size() * 4, zeros.data());
        for (bool l2 : {false, true})
            for (bool residual : {false, true}) {
                uint32_t k = x.size() + 1;
                auto distances = buffer(dev, queries.size() * k * 4),
                     labels = buffer(dev, queries.size() * k * 8);
                uint32_t p[] = {
                        uint32_t(queries.size()),
                        dim,
                        k,
                        1,
                        1,
                        1,
                        residual,
                        l2,
                        uint32_t(sq.code_size)};
                runMetalIVFSQScan(
                        dev,
                        res->getCommandQueue(),
                        qb,
                        cb,
                        ib,
                        ob,
                        lb,
                        ab,
                        dummy,
                        dummy,
                        distances,
                        labels,
                        dummy,
                        p);
                for (size_t qi = 0; qi < queries.size(); ++qi) {
                    std::vector<std::pair<float, idx_t>> expected;
                    float worst = l2 ? std::numeric_limits<float>::max()
                                     : -std::numeric_limits<float>::max();
                    for (size_t j = 0; j < x.size(); ++j) {
                        float delta = queries[qi] - decoded[j * dim];
                        float score = l2 ? delta * delta
                                         : queries[qi] * decoded[j * dim];
                        if (l2 ? score < worst : score > worst)
                            expected.emplace_back(score, ids[j]);
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
                    for (size_t j = 0; j < k; ++j) {
                        ASSERT_EQ(
                                ((idx_t*)labels.contents)[qi * k + j],
                                j < expected.size() ? expected[j].second : -1);
                        ASSERT_EQ(
                                ((float*)distances.contents)[qi * k + j],
                                j < expected.size() ? expected[j].first
                                                    : worst);
                    }
                }
            }
    }
    // The public index must take the GPU path for the original reproduction.
    for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT}) {
        IndexFlat coarse(1, metric);
        IndexIVFScalarQuantizer cpu(
                &coarse, 1, 1, ScalarQuantizer::QT_fp16, metric, false);
        float train[] = {1, 2, 3, 4};
        cpu.train(4, train);
        cpu.add(4, x.data());
        MetalIndexIVFScalarQuantizer gpu(res, &cpu);
        for (float query : {0.f, 1.f}) {
            float cd[4], gd[4];
            idx_t ci[4], gi[4];
            cpu.search(1, &query, 4, cd, ci);
            gpu.search(1, &query, 4, gd, gi);
            for (int j = 0; j < 4; ++j)
                ASSERT_EQ(cd[j], gd[j]);
            std::sort(ci, ci + 4);
            std::sort(gi, gi + 4);
            for (int j = 0; j < 4; ++j)
                ASSERT_EQ(ci[j], gi[j]);
        }
        ASSERT_EQ(gpu.gpuSearchCount(), 2);
    }
}

TEST_F(TestMetalIndexIVFSQ, NoHeapInitPreservesCallerHeap) {
    for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT}) {
        IndexFlat coarse(1, metric);
        IndexIVFScalarQuantizer cpu(
                &coarse, 1, 1, ScalarQuantizer::QT_fp16, metric, false);
        float x[] = {1, 2, 3, 4}, query = 0;
        cpu.train(4, x);
        cpu.add(4, x);
        cpu.parallel_mode = cpu.PARALLEL_MODE_NO_HEAP_INIT;
        MetalIndexIVFScalarQuantizer gpu(res, &cpu);
        SearchParametersIVF p;
        p.nprobe = 1;
        for (const SearchParameters* params :
             {static_cast<const SearchParameters*>(nullptr),
              static_cast<const SearchParameters*>(&p)}) {
            float cd = metric == METRIC_L2 ? 0 : 100, gd = cd;
            idx_t ci = 999, gi = ci;
            cpu.search(1, &query, 1, &cd, &ci, params);
            gpu.search(1, &query, 1, &gd, &gi, params);
            ASSERT_EQ(cd, gd);
            ASSERT_EQ(ci, gi);
            ASSERT_EQ(gi, 999);
            ASSERT_EQ(gpu.gpuSearchCount(), 0);
            ASSERT_EQ(gpu.nprobe(), cpu.nprobe);
        }
    }
}

TEST_F(TestMetalIndexIVFSQ, WrappedQuantizersOwnBorrowedLeavesAndGraphs) {
    for (int kind : {0, 1, 2}) {
        std::unique_ptr<MetalIndexIVFScalarQuantizer> gpu;
        std::unique_ptr<IndexIVFScalarQuantizer> back;
        {
            auto leaf = std::make_unique<IndexFlatL2>(1);
            std::unique_ptr<Index> graph;
            float centroids[] = {0, 10};
            if (kind == 1) {
                auto mapped = std::make_unique<IndexIDMap2>(leaf.get());
                idx_t ids[] = {0, 1};
                mapped->add_with_ids(2, centroids, ids);
                graph = std::move(mapped);
            } else if (kind == 2) {
                graph = std::make_unique<IndexHNSW>(leaf.get(), 2);
                graph->add(2, centroids);
            } else
                leaf->add(2, centroids);
            std::vector<uint8_t> borrowed(
                    leaf->codes.data(),
                    leaf->codes.data() + leaf->codes.size());
            leaf->codes = MaybeOwnedVector<uint8_t>::create_view(
                    borrowed.data(), borrowed.size(), nullptr);
            std::vector<IndexHNSW::storage_idx_t> borrowedNeighbors;
            if (kind == 2) {
                auto* hnsw = static_cast<IndexHNSW*>(graph.get());
                auto& neighbors = hnsw->hnsw.neighbors;
                borrowedNeighbors.assign(
                        neighbors.data(), neighbors.data() + neighbors.size());
                neighbors =
                        MaybeOwnedVector<IndexHNSW::storage_idx_t>::create_view(
                                borrowedNeighbors.data(),
                                borrowedNeighbors.size(),
                                nullptr);
            }
            IndexPreTransform inner(graph ? graph.get() : leaf.get()),
                    outer(&inner);
            IndexIVFScalarQuantizer cpu(
                    &outer, 1, 2, ScalarQuantizer::QT_fp16, METRIC_L2, false);
            float x[] = {0, 1, 9, 10};
            cpu.train(4, x);
            cpu.add(4, x);
            gpu = std::make_unique<MetalIndexIVFScalarQuantizer>(res, &cpu);
            back.reset(gpu->copyTo());
            // Corrupt borrowed memory before destroying both it and the source
            // graph. This catches aliases even when freed pages stay readable.
            std::fill(borrowed.begin(), borrowed.end(), 0xff);
            std::fill(borrowedNeighbors.begin(), borrowedNeighbors.end(), -1);
        }
        float query = 0, gd, cd;
        idx_t gi, ci;
        gpu->search(1, &query, 1, &gd, &gi);
        back->search(1, &query, 1, &cd, &ci);
        ASSERT_EQ(gi, 0);
        ASSERT_EQ(ci, 0);
        ASSERT_EQ(gd, 0);
        ASSERT_EQ(cd, 0);
        back.reset(gpu->copyTo());
        gpu.reset();
        back->search(1, &query, 1, &cd, &ci);
        ASSERT_EQ(ci, 0);
        ASSERT_EQ(cd, 0);
    }
}

TEST_F(TestMetalIndexIVFSQ, CutoffKeepsLateLabelTiesAndDuplicateKeys) {
    constexpr uint32_t nb = 2051;
    auto dev = res->getDevice();
    for (bool l2 : {false, true}) {
        float query = l2 ? 0.f : 1.f;
        std::vector<float> x(nb, 1.f);
        x.back() = l2 ? 0.f : 2.f;
        ScalarQuantizer sq(1, ScalarQuantizer::QT_fp16);
        std::vector<uint8_t> codes(nb * sq.code_size);
        sq.compute_codes(x.data(), codes.data(), nb);
        std::vector<idx_t> ids(nb);
        for (uint32_t i = 0; i < nb; ++i)
            ids[i] = (idx_t(1) << 42) + nb - i + 100;
        // The same (score, label) can legitimately occur more than once.
        // Put duplicates in separate segments and lists to exercise merging.
        for (uint32_t i : {10, 600, 1200, 1800})
            ids[i] = (idx_t(1) << 42) + 1;
        uint32_t offsets[] = {0, 1025}, lengths[] = {1025, nb - 1025};
        int32_t assignments[] = {0, 1};
        auto qb = buffer(dev, 4, &query);
        auto cb = buffer(dev, codes.size(), codes.data());
        auto ib = buffer(dev, ids.size() * 8, ids.data());
        auto ob = buffer(dev, sizeof(offsets), offsets);
        auto lb = buffer(dev, sizeof(lengths), lengths);
        auto ab = buffer(dev, sizeof(assignments), assignments);
        auto dummy = buffer(dev, 8);
        std::vector<std::pair<float, idx_t>> expected;
        for (uint32_t i = 0; i < nb; ++i)
            expected.emplace_back(l2 ? x[i] * x[i] : x[i], ids[i]);
        std::sort(expected.begin(), expected.end(), [=](auto a, auto b) {
            return a.first == b.first
                    ? a.second < b.second
                    : (l2 ? a.first < b.first : a.first > b.first);
        });
        for (uint32_t k : {1, 10, 512}) {
            auto db = buffer(dev, k * 4), labels = buffer(dev, k * 8);
            uint32_t p[] = {1, 1, k, 2, 2, 1, 0, l2, 2};
            runMetalIVFSQScan(
                    dev,
                    res->getCommandQueue(),
                    qb,
                    cb,
                    ib,
                    ob,
                    lb,
                    ab,
                    dummy,
                    dummy,
                    db,
                    labels,
                    dummy,
                    p);
            for (uint32_t i = 0; i < k; ++i) {
                ASSERT_EQ(((float*)db.contents)[i], expected[i].first);
                ASSERT_EQ(((idx_t*)labels.contents)[i], expected[i].second);
            }
        }
    }
}

TEST_F(TestMetalIndexIVFSQ, ParallelChunksAndManyProbeMergeRounds) {
    constexpr uint32_t nb = 2051;
    std::vector<float> x(nb);
    std::vector<idx_t> ids(nb);
    for (uint32_t i = 0; i < nb; ++i) {
        x[i] = float(i % 32);
        ids[i] = (idx_t(1) << 42) + nb - i;
    }
    ScalarQuantizer sq(1, ScalarQuantizer::QT_fp16);
    std::vector<uint8_t> code(nb * 2);
    sq.compute_codes(x.data(), code.data(), nb);
    auto dev = res->getDevice();
    float queries[] = {0, 1};
    uint32_t offsets[] = {0, nb}, lengths[] = {nb, 0};
    auto qb = buffer(dev, sizeof(queries), queries),
         cb = buffer(dev, code.size(), code.data());
    auto ib = buffer(dev, ids.size() * 8, ids.data()),
         ob = buffer(dev, sizeof(offsets), offsets);
    auto lb = buffer(dev, sizeof(lengths), lengths), dummy = buffer(dev, 8);
    MetalIVFSQScanScratch scratch;
    for (bool l2 : {false, true})
        for (uint32_t probes : {1, 1025}) {
            std::vector<int32_t> assignments(2 * probes, 0);
            uint32_t valid = 0;
            for (uint32_t pi = 0; pi < probes; ++pi) {
                int32_t list = pi && pi % 17 == 0
                        ? -1
                        : (pi && pi % 19 == 0 ? 99
                                              : (pi && pi % 23 == 0 ? 1 : 0));
                assignments[pi] = assignments[probes + pi] = list;
                valid += list == 0;
            }
            auto ab = buffer(dev, assignments.size() * 4, assignments.data());
            auto coarse = buffer(dev, assignments.size() * 4);
            for (uint32_t k : {1, 512}) {
                auto db = buffer(dev, 2 * k * 4),
                     labels = buffer(dev, 2 * k * 8);
                uint32_t p[] = {2, 1, k, probes, 2, 1, 0, l2, 2};
                runMetalIVFSQScan(
                        dev,
                        res->getCommandQueue(),
                        qb,
                        cb,
                        ib,
                        ob,
                        lb,
                        ab,
                        dummy,
                        dummy,
                        db,
                        labels,
                        coarse,
                        p,
                        &scratch);
                for (uint32_t qi = 0; qi < 2; ++qi) {
                    std::vector<std::pair<float, idx_t>> expected;
                    for (uint32_t i = 0; i < nb; ++i) {
                        float delta = queries[qi] - x[i];
                        expected.emplace_back(
                                l2 ? delta * delta : queries[qi] * x[i],
                                ids[i]);
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
                        ASSERT_EQ(
                                ((float*)db.contents)[qi * k + j],
                                expected[j / valid].first);
                        ASSERT_EQ(
                                ((idx_t*)labels.contents)[qi * k + j],
                                expected[j / valid].second);
                    }
                }
            }
        }
    MetalIVFSQScanPlan plan;
    ASSERT_FALSE(makeMetalIVFSQScanPlan(
            UINT32_MAX, UINT32_MAX, 512, UINT32_MAX, SIZE_MAX, plan));
    ASSERT_FALSE(makeMetalIVFSQScanPlan(1, 2, 512, 4096, 64, plan));
}

TEST_F(TestMetalIndexIVFSQ, FlatCoarseRunsInScanCommand) {
    for (int dim : {7, 128, 512})
        for (bool coarseL2 : {false, true})
            for (bool residual : {false, true}) {
                constexpr int lists = 128, nq = 3, k = 10;
                std::vector<float> centroids(lists * dim, 0.f),
                        q(nq * dim, 0.f);
                for (int i = 0; i < lists; ++i)
                    centroids[i * dim] = float(i + 1);
                q[0] = -1.25f;
                q[dim] = 17.25f;
                q[2 * dim] = 126.75f;
                IndexFlat coarse(
                        dim, coarseL2 ? METRIC_L2 : METRIC_INNER_PRODUCT);
                coarse.add(lists, centroids.data());
                // Exercise a coarse metric different from the IVF metric too.
                IndexIVFScalarQuantizer cpu(
                        &coarse,
                        dim,
                        lists,
                        ScalarQuantizer::QT_fp16,
                        METRIC_INNER_PRODUCT,
                        residual);
                cpu.is_trained = true;
                cpu.add(lists, centroids.data());
                MetalIndexIVFScalarQuantizer gpu(res, &cpu);
                for (int probes : {1, 8, 128}) {
                    cpu.nprobe = probes;
                    gpu.setNumProbes(probes);
                    std::vector<float> expected(nq * k), actual(nq * k);
                    std::vector<idx_t> ei(nq * k), ai(nq * k);
                    cpu.search(nq, q.data(), k, expected.data(), ei.data());
                    size_t before = gpu.gpuCoarseSearchCount();
                    gpu.search(nq, q.data(), k, actual.data(), ai.data());
                    ASSERT_EQ(gpu.gpuCoarseSearchCount(), before + 1);
                    for (int i = 0; i < nq * k; ++i) {
                        ASSERT_EQ(ai[i], ei[i]);
                        ASSERT_FLOAT_EQ(actual[i], expected[i]);
                    }
                }
                // Wrapped quantizers retain their own CPU search semantics.
                IndexPreTransform wrapped(&coarse);
                cpu.quantizer = &wrapped;
                MetalIndexIVFScalarQuantizer wrappedGpu(res, &cpu);
                std::vector<float> scores(nq * k);
                std::vector<idx_t> labels(nq * k);
                wrappedGpu.search(
                        nq, q.data(), k, scores.data(), labels.data());
                ASSERT_EQ(wrappedGpu.gpuSearchCount(), 1);
                ASSERT_EQ(wrappedGpu.gpuCoarseSearchCount(), 0);
                cpu.quantizer = &coarse;
            }
}

TEST_F(TestMetalIndexIVFSQ, DirectFlatCoarseTiesAndNonFiniteAdmission) {
    constexpr uint32_t lists = 1025, nq = 3, k = 10;
    std::vector<float> c(lists);
    std::vector<uint32_t> offsets(lists), lengths(lists, 1);
    std::vector<idx_t> ids(lists);
    for (uint32_t i = 0; i < lists; ++i) {
        c[i] = float(i % 31);
        offsets[i] = i;
        ids[i] = i;
    }
    c[0] = std::numeric_limits<float>::quiet_NaN();
    c[1] = std::numeric_limits<float>::infinity();
    c[2] = -std::numeric_limits<float>::infinity();
    c[3] = 1e30f;
    float q[] = {0, 1e20f, -1e20f};
    std::vector<uint8_t> code(lists * 2, 0);
    auto dev = res->getDevice();
    auto qb = buffer(dev, sizeof(q), q),
         cb = buffer(dev, code.size(), code.data());
    auto ib = buffer(dev, ids.size() * 8, ids.data());
    auto ob = buffer(dev, lists * 4, offsets.data()),
         lb = buffer(dev, lists * 4, lengths.data());
    auto centroids = buffer(dev, lists * 4, c.data()), dummy = buffer(dev, 8);
    auto db = buffer(dev, nq * k * 4), labels = buffer(dev, nq * k * 8);
    for (bool l2 : {false, true}) {
        IndexFlat coarse(1, l2 ? METRIC_L2 : METRIC_INNER_PRODUCT);
        coarse.add(lists, c.data());
        for (uint32_t probes : {1, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512}) {
            SCOPED_TRACE(
                    ::testing::Message()
                    << "l2=" << l2 << " probes=" << probes);
            std::vector<float> expected(nq * probes);
            std::vector<idx_t> expectedIds(nq * probes);
            coarse.search(nq, q, probes, expected.data(), expectedIds.data());
            auto ab = buffer(dev, nq * probes * 4),
                 scores = buffer(dev, nq * probes * 4);
            uint32_t p[] = {nq, 1, k, probes, lists, 1, 0, 1, 2};
            runMetalIVFSQScan(
                    dev,
                    res->getCommandQueue(),
                    qb,
                    cb,
                    ib,
                    ob,
                    lb,
                    ab,
                    centroids,
                    dummy,
                    db,
                    labels,
                    scores,
                    p,
                    nullptr,
                    true,
                    l2);
            // CPU flat selection does not promise a list-ID order for ties.
            // Compare scores with CPU and IDs with an independent ordered
            // oracle.
            for (uint32_t qi = 0; qi < nq; ++qi) {
                std::vector<std::pair<float, idx_t>> ordered;
                for (uint32_t li = 0; li < lists; ++li) {
                    float delta = q[qi] - c[li];
                    float score = l2 ? delta * delta : q[qi] * c[li];
                    if (l2 ? score < std::numeric_limits<float>::max()
                           : score > -std::numeric_limits<float>::max())
                        ordered.emplace_back(score, li);
                }
                std::sort(ordered.begin(), ordered.end(), [=](auto a, auto b) {
                    return a.first == b.first
                            ? a.second < b.second
                            : (l2 ? a.first < b.first : a.first > b.first);
                });
                for (uint32_t j = 0; j < probes; ++j) {
                    size_t i = qi * probes + j;
                    ASSERT_EQ(((float*)scores.contents)[i], expected[i]);
                    ASSERT_EQ(
                            ((int32_t*)ab.contents)[i],
                            j < ordered.size() ? ordered[j].second : -1);
                }
            }
        }
    }
}
