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

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
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

    // Try with threading and without
    for (bool threaded : {true, false}) {
        // clear flags
        for (auto& idx : idxs) {
            idx->resetMock();
        }

        MockThreadedIndex<MockIndex> ti(threaded);
        for (auto& idx : idxs) {
            ti.addIndex(idx.get());
        }

        // The second index should throw
        EXPECT_THROW(ti.runOnIndex(fn), TestException);

        // Index 0 and 2 should have processed
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

    // Try with threading and without
    for (bool threaded : {true, false}) {
        // clear flags
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

        // Index 2 should have processed
        EXPECT_TRUE(idxs[2]->flag);
    }
}

TEST(ThreadedIndex, TestReplica) {
    int numReplicas = 5;
    int n = 10 * numReplicas;
    int d = 3;
    int k = 6;

    // Try with threading and without
    for ([[maybe_unused]] const bool threaded : {true, false}) {
        std::vector<std::unique_ptr<MockIndex>> idxs;
        faiss::IndexReplicas replica(d);

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

    // Try with threading and without
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

namespace {

class ShardsIVFTest
        : public ::testing::TestWithParam<std::tuple<bool, faiss::MetricType>> {
   protected:
    faiss::MetricType metric = std::get<1>(GetParam());
    faiss::IndexFlat quantizer{1, metric};
    std::vector<std::unique_ptr<faiss::IndexIVFFlat>> children;
    faiss::IndexShardsIVF shards{&quantizer, 1, std::get<0>(GetParam()), false};

    ShardsIVFTest() {
        const float centroid = 0;
        quantizer.add(1, &centroid);
    }

    void addShard(
            const std::vector<float>& vectors,
            const std::vector<idx_t>& ids = {}) {
        auto child =
                std::make_unique<faiss::IndexIVFFlat>(&quantizer, 1, 1, metric);
        if (ids.empty()) {
            child->add(vectors.size(), vectors.data());
        } else {
            child->add_with_ids(vectors.size(), vectors.data(), ids.data());
        }
        shards.addIndex(child.get());
        children.push_back(std::move(child));
    }

    // Compare against an exact index containing only the expected candidates.
    void expectMatchesFlat(
            const std::vector<float>& vectors,
            const std::vector<idx_t>& ids,
            const faiss::SearchParametersIVF* params = nullptr) {
        faiss::IndexFlat reference(1, metric);
        reference.add(vectors.size(), vectors.data());
        const float queries[] = {1, -1};
        constexpr idx_t nq = 2, k = 4;
        std::vector<float> expected_distances(nq * k), distances(nq * k);
        std::vector<idx_t> expected_labels(nq * k), labels(nq * k);
        reference.search(
                nq,
                queries,
                k,
                expected_distances.data(),
                expected_labels.data());
        for (auto& label : expected_labels) {
            if (label >= 0) {
                label = ids[label];
            }
        }
        shards.search(nq, queries, k, distances.data(), labels.data(), params);
        EXPECT_EQ(labels, expected_labels);
        EXPECT_EQ(distances, expected_distances);
    }
};

} // namespace

TEST_P(ShardsIVFTest, SelectorUsesGlobalIds) {
    addShard({-3, 0.5}, {100, 101});
    addShard({});
    addShard({2, 6}, {200, 201});
    faiss::SearchParametersIVF params;

    const idx_t allowed = 100;
    faiss::IDSelectorArray single(1, &allowed);
    params.sel = &single;
    expectMatchesFlat({-3}, {100}, &params);
    EXPECT_EQ(params.sel, &single);

    const idx_t allowed_ids[] = {101, 201};
    faiss::IDSelectorBatch across_shards(2, allowed_ids);
    params.sel = &across_shards;
    expectMatchesFlat({0.5, 6}, {101, 201}, &params);

    faiss::IDSelectorRange sorted_range(200, 202, true);
    params.sel = &sorted_range;
    expectMatchesFlat({2, 6}, {200, 201}, &params);

    faiss::IDSelectorRange no_matches(500, 600);
    params.sel = &no_matches;
    expectMatchesFlat({}, {}, &params);

    params.sel = nullptr;
    expectMatchesFlat({-3, 0.5, 2, 6}, {100, 101, 200, 201}, &params);
    expectMatchesFlat({-3, 0.5, 2, 6}, {100, 101, 200, 201});
}

TEST_P(ShardsIVFTest, MaxCodesIsAppliedPerShard) {
    addShard({-3, 0.5}, {100, 101});
    addShard({2, 6}, {200, 201});
    faiss::SearchParametersIVF params;
    params.max_codes = 1;
    expectMatchesFlat({-3, 2}, {100, 200}, &params);
    EXPECT_EQ(params.max_codes, 1);

    // A per-call budget must not change subsequent default searches.
    expectMatchesFlat({-3, 0.5, 2, 6}, {100, 101, 200, 201});
}

TEST_P(ShardsIVFTest, NoSelectorWithShiftedIds) {
    shards.successive_ids = true;
    addShard({-3, 0.5});
    addShard({2, 6});
    expectMatchesFlat({-3, 0.5, 2, 6}, {0, 1, 2, 3});

    faiss::SearchParametersIVF params;
    expectMatchesFlat({-3, 0.5, 2, 6}, {0, 1, 2, 3}, &params);
    params.max_codes = 1;
    expectMatchesFlat({-3, 2}, {0, 2}, &params);
}

TEST_P(ShardsIVFTest, SelectorWithSingleShard) {
    shards.successive_ids = true;
    addShard({-3, 0.5});
    faiss::IDSelectorRange selector(1, 2);
    faiss::SearchParametersIVF params;
    params.sel = &selector;
    expectMatchesFlat({0.5}, {1}, &params);
}

TEST_P(ShardsIVFTest, SelectorWithZeroOffsets) {
    shards.successive_ids = true;
    addShard({});
    addShard({});
    faiss::IDSelectorRange selector(1, 2);
    faiss::SearchParametersIVF params;
    params.sel = &selector;
    expectMatchesFlat({}, {}, &params);

    addShard({-3, 0.5});
    expectMatchesFlat({0.5}, {1}, &params);
}

TEST_P(ShardsIVFTest, RejectsSelectorWithShiftedIds) {
    shards.successive_ids = true;
    addShard({-3, 0.5});
    addShard({2, 6});
    faiss::IDSelectorRange selector(2, 3);
    faiss::SearchParametersIVF params;
    params.sel = &selector;
    const float query = 1;

    for (bool empty_last_shard : {false, true}) {
        SCOPED_TRACE(empty_last_shard);
        if (empty_last_shard) {
            shards.runOnIndex([](int i, faiss::Index* index) {
                if (i == 1) {
                    index->reset();
                }
            });
        }
        float distance = -123;
        idx_t label = -456;
        try {
            shards.search(1, &query, 1, &distance, &label, &params);
            FAIL() << "Expected rejection of a selector with shifted IDs";
        } catch (const faiss::FaissException& error) {
            EXPECT_NE(
                    std::string(error.what()).find("nonzero shard ID offsets"),
                    std::string::npos);
        }
        EXPECT_EQ(distance, -123);
        EXPECT_EQ(label, -456);
        EXPECT_EQ(params.sel, &selector);
    }
}

INSTANTIATE_TEST_SUITE_P(
        SerialAndThreaded,
        ShardsIVFTest,
        ::testing::Combine(
                ::testing::Bool(),
                ::testing::Values(
                        faiss::METRIC_L2,
                        faiss::METRIC_INNER_PRODUCT)));
