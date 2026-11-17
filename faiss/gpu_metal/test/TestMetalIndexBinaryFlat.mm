// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryIVF.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexBinaryFlat.h>
#include <faiss/gpu_metal/MetalKernels.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/impl/IDSelector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using faiss::idx_t;
using namespace faiss::gpu_metal;

namespace {
std::vector<uint8_t> randomBytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes(n);
    for (auto& b : bytes) {
        b = uint8_t(rng());
    }
    return bytes;
}

// Independent full-sort oracle, including deterministic ties and sentinels.
void checkExact(
        const std::vector<uint8_t>& database,
        const std::vector<uint8_t>& queries,
        int codeSize,
        int k,
        const int32_t* distances,
        const idx_t* labels) {
    size_t nb = database.size() / codeSize;
    size_t nq = queries.size() / codeSize;
    for (size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<int32_t, idx_t>> expected;
        for (size_t v = 0; v < nb; ++v) {
            int32_t distance = 0;
            for (int b = 0; b < codeSize; ++b) {
                distance += __builtin_popcount(
                        unsigned(
                                queries[q * codeSize + b] ^
                                database[v * codeSize + b]));
            }
            expected.emplace_back(distance, v);
        }
        std::sort(expected.begin(), expected.end());
        for (int j = 0; j < k; ++j) {
            auto value = size_t(j) < nb
                    ? expected[j]
                    : std::make_pair(
                              std::numeric_limits<int32_t>::max(), idx_t(-1));
            ASSERT_EQ(distances[q * k + j], value.first)
                    << "q=" << q << " j=" << j;
            ASSERT_EQ(labels[q * k + j], value.second)
                    << "q=" << q << " j=" << j;
        }
    }
}
} // namespace

class TestMetalIndexBinaryFlat : public ::testing::Test {
   protected:
    void SetUp() override {
        resources = std::make_shared<MetalResources>();
        if (!resources->isAvailable()) {
            GTEST_SKIP() << "Metal not available";
        }
    }
    std::shared_ptr<MetalResources> resources;
};

TEST_F(TestMetalIndexBinaryFlat, DimensionsAndKBoundaries) {
    // Odd code strides exercise unaligned vectors. 513/1024-byte codes exceed
    // the prototype's fixed 512-byte threadgroup query cache.
    for (int cs : {1, 3, 5, 32, 513, 1024}) {
        SCOPED_TRACE(cs);
        constexpr int nb = 1537, nq = 7;
        auto database = randomBytes(nb * cs, 1234);
        auto queries = randomBytes(nq * cs, 5678);
        MetalIndexBinaryFlat metal(resources, cs * 8);
        metal.add(511, database.data());
        metal.add(nb - 511, database.data() + 511 * cs);
        faiss::IndexBinaryFlat cpu(cs * 8);
        cpu.add(nb, database.data());
        for (int k : {1, 7, 255, 256, 511, 512}) {
            SCOPED_TRACE(k);
            std::vector<int32_t> distances(nq * k), cpuDist(nq * k);
            std::vector<idx_t> labels(nq * k), cpuLabels(nq * k);
            metal.search(
                    nq, queries.data(), k, distances.data(), labels.data());
            checkExact(
                    database, queries, cs, k, distances.data(), labels.data());
            cpu.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
            EXPECT_EQ(distances, cpuDist);
        }
    }
}

TEST_F(TestMetalIndexBinaryFlat, DirectCompiledKernelAdversarialCandidates) {
    // All nearest neighbors belong to the same strided prototype thread.
    // Keeping only four candidates per thread would lose most of these hits.
    constexpr int nb = 8193, cs = 3, nq = 1, k = 32;
    std::vector<uint8_t> database(nb * cs, 255), queries(cs, 0);
    for (int v = 0; v < nb; v += 256) {
        std::fill_n(database.data() + v * cs, cs, uint8_t(0));
    }
    auto device = resources->getDevice();
    id<MTLBuffer> db = [device newBufferWithBytes:database.data()
                                           length:database.size()
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> q = [device newBufferWithBytes:queries.data()
                                          length:queries.size()
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> dist =
            [device newBufferWithLength:k * sizeof(int32_t)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> ids =
            [device newBufferWithLength:k * sizeof(idx_t)
                                options:MTLResourceStorageModeShared];
    ASSERT_TRUE(db && q && dist && ids);
    auto& kernels = getMetalKernels(device);
    ASSERT_TRUE(kernels.isValid());
    auto command = [resources->getCommandQueue() commandBuffer];
    auto encoder = [command computeCommandEncoder];
    kernels.encodeHammingDistanceTopK(encoder, q, db, dist, ids, nq, nb, cs, k);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    ASSERT_EQ(command.status, MTLCommandBufferStatusCompleted);
    checkExact(
            database,
            queries,
            cs,
            k,
            static_cast<const int32_t*>(dist.contents),
            static_cast<const idx_t*>(ids.contents));
}

TEST_F(TestMetalIndexBinaryFlat, PackedAndOddStridesSelectionBoundaries) {
    // Exercise the CUDA-style word path and its byte-stride fallback. Values
    // around powers of two cover selection capacities and partial segments.
    for (int cs : {3, 4, 8, 12, 16, 32, 64, 128, 129, 132}) {
        SCOPED_TRACE(cs);
        constexpr int nb = 2053, nq = 3;
        auto database = randomBytes(nb * cs, 177);
        auto queries = randomBytes(nq * cs, 991);
        MetalIndexBinaryFlat metal(resources, cs * 8);
        metal.add(nb, database.data());
        for (int k :
             {1,
              2,
              31,
              32,
              33,
              63,
              64,
              65,
              127,
              128,
              129,
              255,
              256,
              257,
              511,
              512}) {
            SCOPED_TRACE(k);
            std::vector<int32_t> distances(nq * k);
            std::vector<idx_t> labels(nq * k);
            metal.search(
                    nq, queries.data(), k, distances.data(), labels.data());
            checkExact(
                    database, queries, cs, k, distances.data(), labels.data());
        }
    }
}

TEST_F(TestMetalIndexBinaryFlat, LateImprovementsAndCutoffTies) {
    constexpr int cs = 8, nb = 8193;
    std::vector<uint8_t> database(nb * cs, 0), query(cs, 0);
    for (int v = 0; v < nb; ++v) {
        // Repeatedly improve the current cutoff, including the last partial
        // segment, with enough equal distances to require deterministic IDs.
        int distance = (nb - v) % (cs * 8 + 1);
        for (int bit = 0; bit < distance; ++bit) {
            database[v * cs + bit / 8] |= uint8_t(1u << (bit % 8));
        }
    }
    MetalIndexBinaryFlat metal(resources, cs * 8);
    metal.add(nb, database.data());
    for (int k : {1, 20, 128, 512}) {
        std::vector<int32_t> distances(k);
        std::vector<idx_t> labels(k);
        metal.search(1, query.data(), k, distances.data(), labels.data());
        checkExact(database, query, cs, k, distances.data(), labels.data());
    }
}

TEST_F(TestMetalIndexBinaryFlat, IntegerDistancesBeyondFloatPrecision) {
    constexpr int cs = (1 << 21) + 1, d = cs * 8;
    std::vector<uint8_t> database(size_t(3) * cs, 255), query(cs, 0);
    database[cs] = 254;
    std::fill(database.begin() + 2 * cs, database.end(), 0);
    MetalIndexBinaryFlat metal(resources, d);
    metal.add(3, database.data());
    int32_t distances[4];
    idx_t labels[4];
    metal.search(1, query.data(), 4, distances, labels);
    EXPECT_EQ(distances[0], 0);
    EXPECT_EQ(distances[1], d - 1);
    EXPECT_EQ(distances[2], d);
    EXPECT_EQ(distances[3], std::numeric_limits<int32_t>::max());
    EXPECT_EQ(labels[0], 2);
    EXPECT_EQ(labels[1], 1);
    EXPECT_EQ(labels[2], 0);
    EXPECT_EQ(labels[3], -1);
}

TEST_F(TestMetalIndexBinaryFlat, EmptyShortAndReset) {
    constexpr int cs = 5, nq = 3, k = 512;
    auto queries = randomBytes(nq * cs, 42);
    for (int nb : {0, 1, 17, 511, 512, 513}) {
        SCOPED_TRACE(nb);
        auto database = randomBytes(nb * cs, 13);
        MetalIndexBinaryFlat metal(resources, cs * 8);
        metal.add(nb, database.data());
        std::vector<int32_t> distances(nq * k, -2);
        std::vector<idx_t> labels(nq * k, -2);
        metal.search(nq, queries.data(), k, distances.data(), labels.data());
        checkExact(database, queries, cs, k, distances.data(), labels.data());
        metal.reset();
        EXPECT_EQ(metal.ntotal, 0);
        metal.search(nq, queries.data(), k, distances.data(), labels.data());
        checkExact({}, queries, cs, k, distances.data(), labels.data());
        metal.add(nq, queries.data());
        metal.search(nq, queries.data(), k, distances.data(), labels.data());
        checkExact(queries, queries, cs, k, distances.data(), labels.data());
    }
}

TEST_F(TestMetalIndexBinaryFlat, RepeatedTiesAcrossSegmentsAndLargeBatch) {
    constexpr int nb = 100003, cs = 1, nq = 65, k = 512;
    std::vector<uint8_t> database(nb * cs, 0), queries(nq * cs, 0);
    MetalIndexBinaryFlat metal(resources, cs * 8);
    metal.add(nb, database.data());
    std::vector<int32_t> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    metal.search(nq, queries.data(), k, distances.data(), labels.data());
    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            EXPECT_EQ(distances[q * k + j], 0);
            EXPECT_EQ(labels[q * k + j], j);
        }
    }
}

TEST_F(TestMetalIndexBinaryFlat, CpuFallbackLargeKAndSelectors) {
    constexpr int nb = 701, cs = 8, nq = 3;
    auto database = randomBytes(nb * cs, 91);
    auto queries = randomBytes(nq * cs, 12);
    faiss::IndexBinaryFlat cpu(cs * 8);
    cpu.add(nb, database.data());
    MetalIndexBinaryFlat metal(resources, &cpu);
    faiss::IDSelectorRange selector(7, 25);
    faiss::SearchParameters params;
    params.sel = &selector;
    for (int k : {7, 513, 709}) {
        for (const auto* p :
             {static_cast<const faiss::SearchParameters*>(nullptr),
              static_cast<const faiss::SearchParameters*>(&params)}) {
            if (k == 7 && !p)
                continue;
            std::vector<int32_t> distances(nq * k), expected(nq * k);
            std::vector<idx_t> labels(nq * k), expectedLabels(nq * k);
            cpu.search(
                    nq,
                    queries.data(),
                    k,
                    expected.data(),
                    expectedLabels.data(),
                    p);
            metal.search(
                    nq, queries.data(), k, distances.data(), labels.data(), p);
            EXPECT_EQ(distances, expected);
            EXPECT_EQ(labels, expectedLabels);
        }
    }
}

TEST_F(TestMetalIndexBinaryFlat, CloneOwnsSourceAndPreservesCpuOptions) {
    constexpr int cs = 3, nb = 19;
    auto bytes = randomBytes(nb * cs, 71);
    faiss::IndexBinaryFlat cpu(cs * 8);
    // A borrowed source must be deep-copied rather than retaining its view.
    cpu.xb = faiss::MaybeOwnedVector<uint8_t>::create_view(
            bytes.data(), bytes.size(), nullptr);
    cpu.ntotal = nb;
    cpu.use_heap = false;
    cpu.query_batch_size = 7;
    cpu.verbose = true;
    StandardMetalResources res;
    std::unique_ptr<faiss::IndexBinary> metal(
            index_binary_cpu_to_metal_gpu(&res, 0, &cpu));
    auto expected = bytes;
    std::fill(bytes.begin(), bytes.end(), 0);
    cpu.xb = faiss::MaybeOwnedVector<uint8_t>();
    cpu.reset();
    auto* flat = dynamic_cast<MetalIndexBinaryFlat*>(metal.get());
    ASSERT_NE(flat, nullptr);
    std::vector<uint8_t> recons(cs);
    for (int v = 0; v < nb; ++v) {
        flat->reconstruct(v, recons.data());
        EXPECT_TRUE(
                std::equal(
                        recons.begin(),
                        recons.end(),
                        expected.data() + v * cs));
    }
    std::unique_ptr<faiss::IndexBinary> back(
            index_binary_metal_gpu_to_cpu(flat));
    auto* cpuBack = dynamic_cast<faiss::IndexBinaryFlat*>(back.get());
    ASSERT_NE(cpuBack, nullptr);
    EXPECT_EQ(cpuBack->ntotal, nb);
    EXPECT_FALSE(cpuBack->use_heap);
    EXPECT_EQ(cpuBack->query_batch_size, 7);
    EXPECT_TRUE(cpuBack->verbose);
    EXPECT_EQ(
            std::memcmp(cpuBack->xb.data(), expected.data(), expected.size()),
            0);
    flat->reset();
    EXPECT_EQ(
            std::memcmp(cpuBack->xb.data(), expected.data(), expected.size()),
            0);
    // copyFrom changes dimensions and replaces existing GPU data.
    flat->copyFrom(cpuBack);
    faiss::IndexBinaryFlat other(8);
    const uint8_t one = 42;
    other.add(1, &one);
    flat->copyFrom(&other);
    EXPECT_EQ(flat->d, 8);
    EXPECT_EQ(flat->ntotal, 1);
    flat->reconstruct(0, recons.data());
    EXPECT_EQ(recons[0], one);
}

TEST_F(TestMetalIndexBinaryFlat, InvalidInputsDoNotChangeState) {
    EXPECT_ANY_THROW(MetalIndexBinaryFlat(resources, 0));
    EXPECT_ANY_THROW(MetalIndexBinaryFlat(resources, 7));
    EXPECT_ANY_THROW(MetalIndexBinaryFlat(nullptr, 8));
    EXPECT_ANY_THROW(MetalIndexBinaryFlat(
            resources, static_cast<const faiss::IndexBinaryFlat*>(nullptr)));
    MetalIndexBinaryFlat metal(resources, 8);
    uint8_t value = 42;
    int32_t distance;
    idx_t label;
    metal.add(1, &value);
    EXPECT_ANY_THROW(metal.add(-1, &value));
    EXPECT_ANY_THROW(metal.add(1, nullptr));
    EXPECT_ANY_THROW(metal.add(std::numeric_limits<idx_t>::max(), &value));
    EXPECT_ANY_THROW(metal.search(-1, &value, 1, &distance, &label));
    EXPECT_ANY_THROW(metal.search(1, &value, 0, &distance, &label));
    EXPECT_ANY_THROW(metal.search(1, nullptr, 1, &distance, &label));
    EXPECT_ANY_THROW(metal.search(1, &value, 1, nullptr, &label));
    EXPECT_ANY_THROW(metal.search(
            1, &value, std::numeric_limits<idx_t>::max(), &distance, &label));
    EXPECT_ANY_THROW(metal.reconstruct(-1, &value));
    EXPECT_ANY_THROW(metal.reconstruct(1, &value));
    EXPECT_ANY_THROW(metal.copyFrom(nullptr));
    EXPECT_ANY_THROW(metal.copyTo(nullptr));
    metal.search(0, nullptr, 1, nullptr, nullptr);
    metal.add(0, nullptr);
    faiss::IndexBinaryFlat invalid(16);
    invalid.ntotal = 1;
    EXPECT_ANY_THROW(metal.copyFrom(&invalid));
    EXPECT_EQ(metal.d, 8);
    EXPECT_EQ(metal.ntotal, 1);
    metal.search(1, &value, 1, &distance, &label);
    EXPECT_EQ(distance, 0);
    EXPECT_EQ(label, 0);
    StandardMetalResources res;
    EXPECT_ANY_THROW(index_binary_cpu_to_metal_gpu(&res, 1, &invalid));
    EXPECT_ANY_THROW(index_binary_cpu_to_metal_gpu(nullptr, 0, &invalid));
    EXPECT_ANY_THROW(index_binary_cpu_to_metal_gpu(&res, 0, nullptr));
    EXPECT_ANY_THROW(index_binary_metal_gpu_to_cpu(&invalid));
    faiss::IndexBinaryFlat quantizer(8);
    faiss::IndexBinaryIVF unsupported(&quantizer, 8, 2);
    EXPECT_ANY_THROW(index_binary_cpu_to_metal_gpu(&res, 0, &unsupported));
}
