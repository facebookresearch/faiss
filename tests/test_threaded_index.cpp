/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexReplicas.h>
#include <faiss/IndexShards.h>
#include <faiss/IndexShardsIVF.h>
#include <faiss/impl/ThreadedIndex.h>
#include <faiss/impl/idselector.h>

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace {

struct TestException : public std::exception {};

using idx_t = faiss::idx_t;

struct MockIndex : public faiss::Index {
    explicit MockIndex(idx_t d_in) : faiss::Index(d_in) {
        resetMock();
    }

    void resetMock() {
        flag = false;
        nCalled = 0;
        xCalled = nullptr;
        kCalled = 0;
        distancesCalled = nullptr;
        labelsCalled = nullptr;
    }

    void add(idx_t n, const float* x) override {
        nCalled = n;
        xCalled = x;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const faiss::SearchParameters* params) const override {
        FAISS_THROW_IF_MSG(params, "search params not supported");
        nCalled = n;
        xCalled = x;
        kCalled = k;
        distancesCalled = distances;
        labelsCalled = labels;
    }

    void reset() override {}

    bool flag;

    mutable idx_t nCalled;
    mutable const float* xCalled;
    mutable idx_t kCalled;
    mutable float* distancesCalled;
    mutable idx_t* labelsCalled;
};

template <typename IndexT>
struct MockThreadedIndex : public faiss::ThreadedIndex<IndexT> {
    using idx_t = faiss::idx_t;

    explicit MockThreadedIndex(bool threaded)
            : faiss::ThreadedIndex<IndexT>(threaded) {}

    void add(idx_t, const float*) override {}
    void search(
            idx_t,
            const float*,
            idx_t,
            float*,
            idx_t*,
            const faiss::SearchParameters*) const override {}
    void reset() override {}
};

} // namespace

TEST(ThreadedIndex, SingleException) {
    std::vector<std::unique_ptr<MockIndex>> idxs;

    for (int i = 0; i < 3; ++i) {
        idxs.emplace_back(new MockIndex(1));
    }

    auto fn = [](int i, MockIndex* index) {
        if (i == 1) {
            throw TestException();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(i * 250));

            index->flag = true;
        }
    };

    for (bool threaded : {true, false}) {
        for (auto& idx : idxs) {
            idx->resetMock();
        }

        MockThreadedIndex<MockIndex> ti(threaded);
        for (auto& idx : idxs) {
            ti.addIndex(idx.get());
        }

        EXPECT_THROW(ti.runOnIndex(fn), TestException);

        EXPECT_TRUE(idxs[0]->flag);
        EXPECT_TRUE(idxs[2]->flag);
    }
}

TEST(ThreadedIndex, MultipleException) {
    std::vector<std::unique_ptr<MockIndex>> idxs;

    for (int i = 0; i < 3; ++i) {
        idxs.emplace_back(new MockIndex(1));
    }

    auto fn = [](int i, MockIndex* index) {
        if (i < 2) {
            throw TestException();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(i * 250));

            index->flag = true;
        }
    };

    for (bool threaded : {true, false}) {
        for (auto& idx : idxs) {
            idx->resetMock();
        }

        MockThreadedIndex<MockIndex> ti(threaded);
        for (auto& idx : idxs) {
            ti.addIndex(idx.get());
        }

        // Multiple indices threw an exception that was aggregated into a
        // FaissException
        EXPECT_THROW(ti.runOnIndex(fn), faiss::FaissException);

        EXPECT_TRUE(idxs[2]->flag);
    }
}

TEST(ThreadedIndex, TestReplica) {
    int numReplicas = 5;
    int n = 10 * numReplicas;
    int d = 3;
    int k = 6;

    for (const bool threaded : {true, false}) {
        std::vector<std::unique_ptr<MockIndex>> idxs;
        faiss::IndexReplicas replica(d, threaded);

        for (int i = 0; i < numReplicas; ++i) {
            idxs.emplace_back(new MockIndex(d));
            replica.addIndex(idxs.back().get());
        }

        std::vector<float> x(n * d);
        std::vector<float> distances(n * k);
        std::vector<faiss::idx_t> labels(n * k);

        replica.add(n, x.data());

        for (size_t i = 0; i < idxs.size(); ++i) {
            EXPECT_EQ(idxs[i]->nCalled, n);
            EXPECT_EQ(idxs[i]->xCalled, x.data());
        }

        for (auto& idx : idxs) {
            idx->resetMock();
        }

        replica.search(n, x.data(), k, distances.data(), labels.data());

        for (size_t i = 0; i < idxs.size(); ++i) {
            auto perReplica = n / idxs.size();

            EXPECT_EQ(idxs[i]->nCalled, perReplica);
            EXPECT_EQ(idxs[i]->xCalled, x.data() + i * perReplica * d);
            EXPECT_EQ(idxs[i]->kCalled, k);
            EXPECT_EQ(
                    idxs[i]->distancesCalled,
                    distances.data() + (i * perReplica) * k);
            EXPECT_EQ(
                    idxs[i]->labelsCalled,
                    labels.data() + (i * perReplica) * k);
        }
    }
}

TEST(ThreadedIndex, TestShards) {
    int numShards = 7;
    int d = 3;
    int n = 10 * numShards;
    int k = 6;

    for (bool threaded : {true, false}) {
        std::vector<std::unique_ptr<MockIndex>> idxs;
        faiss::IndexShards shards(d, threaded);

        for (int i = 0; i < numShards; ++i) {
            idxs.emplace_back(new MockIndex(d));
            shards.addIndex(idxs.back().get());
        }

        std::vector<float> x(n * d);
        std::vector<float> distances(n * k);
        std::vector<faiss::idx_t> labels(n * k);

        shards.add(n, x.data());

        for (size_t i = 0; i < idxs.size(); ++i) {
            auto perShard = n / idxs.size();

            EXPECT_EQ(idxs[i]->nCalled, perShard);
            EXPECT_EQ(idxs[i]->xCalled, x.data() + i * perShard * d);
        }

        for (auto& idx : idxs) {
            idx->resetMock();
        }

        shards.search(n, x.data(), k, distances.data(), labels.data());

        for (size_t i = 0; i < idxs.size(); ++i) {
            EXPECT_EQ(idxs[i]->nCalled, n);
            EXPECT_EQ(idxs[i]->xCalled, x.data());
            EXPECT_EQ(idxs[i]->kCalled, k);
            // There is a temporary buffer used for shards
            EXPECT_EQ(
                    idxs[i]->distancesCalled,
                    idxs[0]->distancesCalled + i * k * n);
            EXPECT_EQ(idxs[i]->labelsCalled, idxs[0]->labelsCalled + i * k * n);
        }
    }
}

// See https://github.com/facebookresearch/faiss/issues/5584

TEST(IndexShardsIVF, SearchForwardsSel) {
    const int d = 1;
    std::vector<float> x = {0.f, 1.f, 10.f, 11.f};
    std::vector<faiss::idx_t> sel_ids = {0, 2};
    faiss::IDSelectorArray sel(sel_ids.size(), sel_ids.data());

    for (bool successive : {false, true}) {
        faiss::IndexFlatL2 quantizer(d);
        auto shard0 = std::make_unique<faiss::IndexIVFFlat>(&quantizer, d, 2);
        auto shard1 = std::make_unique<faiss::IndexIVFFlat>(&quantizer, d, 2);
        for (auto* shard : {shard0.get(), shard1.get()}) {
            shard->nprobe = 2;
            shard->cp.min_points_per_centroid = 1;
            shard->cp.max_points_per_centroid = 4;
            shard->train(4, x.data());
        }

        faiss::IndexShardsIVF shards(&quantizer, 2, false, successive);
        shards.add_shard(shard0.release());
        shards.add_shard(shard1.release());
        if (successive) {
            shards.add(4, x.data()); // global ids 0..3 across the shards
        } else {
            std::vector<faiss::idx_t> ids = {0, 1, 2, 3};
            shards.add_with_ids(4, x.data(), ids.data());
        }
        ASSERT_EQ(shards.ntotal, 4);

        faiss::SearchParametersIVF params;
        params.nprobe = 2;
        params.sel = &sel;
        std::vector<float> dis(4);
        std::vector<faiss::idx_t> lab(4);
        shards.search(1, x.data(), 4, dis.data(), lab.data(), &params);

        // only the selected ids may be reported
        EXPECT_EQ(std::count(lab.begin(), lab.end(), 0), 1) << "successive=" << successive;
        EXPECT_EQ(std::count(lab.begin(), lab.end(), 2), 1) << "successive=" << successive;
        EXPECT_EQ(std::count(lab.begin(), lab.end(), 1), 0) << "successive=" << successive;
        EXPECT_EQ(std::count(lab.begin(), lab.end(), 3), 0) << "successive=" << successive;
    }
}
