/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexReplicas.h>
#include <faiss/IndexShards.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ThreadedIndex.h>

#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct TestException : public std::exception {};

using idx_t = faiss::idx_t;

struct RecordingFlatIndex : faiss::IndexFlatL2 {
    using faiss::IndexFlatL2::IndexFlatL2;

    mutable int search_calls = 0;
    mutable const faiss::SearchParameters* last_params = nullptr;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
        ++search_calls;
        last_params = params;
        faiss::IndexFlatL2::search(n, x, k, distances, labels, params);
    }
};

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

TEST(ThreadedIndex, ShardsRejectTranslatedIDSelector) {
    constexpr idx_t d = 1;
    const std::array<float, 4> xb = {0.0F, 10.0F, 20.0F, 30.0F};
    const std::array<float, 1> xq = {20.0F};

    for (bool threaded : {false, true}) {
        for (idx_t selected_id : {2, 0}) {
            RecordingFlatIndex first(d);
            RecordingFlatIndex second(d);
            first.add(2, xb.data());
            second.add(2, xb.data() + 2);

            faiss::IndexShards shards(d, threaded, true);
            shards.add_shard(&first);
            shards.add_shard(&second);

            faiss::IDSelectorBatch selector(1, &selected_id);
            faiss::SearchParametersIVF params;
            params.nprobe = 17;
            params.sel = &selector;

            std::array<float, 2> distances = {-11.0F, -12.0F};
            std::array<idx_t, 2> labels = {-21, -22};
            try {
                shards.search(
                        1,
                        xq.data(),
                        labels.size(),
                        distances.data(),
                        labels.data(),
                        &params);
                FAIL() << "expected translated IDSelector search to fail";
            } catch (const faiss::FaissException& exception) {
                EXPECT_NE(
                        std::string(exception.what())
                                .find("IDSelector search is not supported when "
                                      "successive_ids shifts shard IDs"),
                        std::string::npos);
            }

            EXPECT_EQ(params.sel, &selector);
            EXPECT_EQ(params.nprobe, 17U);
            EXPECT_EQ(first.search_calls, 0);
            EXPECT_EQ(second.search_calls, 0);
            EXPECT_EQ(distances, (std::array<float, 2>{-11.0F, -12.0F}));
            EXPECT_EQ(labels, (std::array<idx_t, 2>{-21, -22}));
        }
    }
}

TEST(ThreadedIndex, ShardsAllowSelectorWithoutIDTranslation) {
    constexpr idx_t d = 1;
    const std::array<float, 1> xb = {20.0F};
    const std::array<float, 1> xq = {20.0F};
    idx_t selected_id = 0;
    faiss::IDSelectorBatch selector(1, &selected_id);
    faiss::SearchParameters params;
    params.sel = &selector;

    for (bool threaded : {false, true}) {
        faiss::IndexFlatL2 single(d);
        single.add(1, xb.data());
        faiss::IndexShards one_shard(d, threaded, true);
        one_shard.add_shard(&single);

        std::array<float, 1> distances{};
        std::array<idx_t, 1> labels{};
        one_shard.search(
                1, xq.data(), 1, distances.data(), labels.data(), &params);
        EXPECT_EQ(labels[0], 0);
        EXPECT_EQ(distances[0], 0.0F);

        faiss::IndexFlatL2 empty(d);
        faiss::IndexFlatL2 populated(d);
        populated.add(1, xb.data());
        faiss::IndexShards zero_offsets(d, threaded, true);
        zero_offsets.add_shard(&empty);
        zero_offsets.add_shard(&populated);

        zero_offsets.search(
                1, xq.data(), 1, distances.data(), labels.data(), &params);
        EXPECT_EQ(labels[0], 0);
        EXPECT_EQ(distances[0], 0.0F);
    }
}

TEST(ThreadedIndex, ShardsControlsRemainUnchanged) {
    constexpr idx_t d = 1;
    const std::array<float, 4> xb = {0.0F, 10.0F, 20.0F, 30.0F};
    const std::array<float, 1> xq = {20.0F};

    for (bool threaded : {false, true}) {
        RecordingFlatIndex first(d);
        RecordingFlatIndex second(d);
        first.add(2, xb.data());
        second.add(2, xb.data() + 2);
        faiss::IndexShards translated(d, threaded, true);
        translated.add_shard(&first);
        translated.add_shard(&second);

        std::array<float, 1> distances{};
        std::array<idx_t, 1> labels{};
        translated.search(1, xq.data(), 1, distances.data(), labels.data());
        EXPECT_EQ(labels[0], 2);
        EXPECT_EQ(distances[0], 0.0F);

        faiss::SearchParametersIVF unfiltered_params;
        unfiltered_params.nprobe = 17;
        translated.search(
                1,
                xq.data(),
                1,
                distances.data(),
                labels.data(),
                &unfiltered_params);
        EXPECT_EQ(labels[0], 2);
        EXPECT_EQ(first.last_params, &unfiltered_params);
        EXPECT_EQ(second.last_params, &unfiltered_params);
        EXPECT_EQ(unfiltered_params.nprobe, 17U);
        EXPECT_EQ(unfiltered_params.sel, nullptr);

        faiss::IndexFlatL2 local_first(d);
        faiss::IndexFlatL2 local_second(d);
        local_first.add(2, xb.data());
        local_second.add(2, xb.data() + 2);
        faiss::IndexShards local_ids(d, threaded, false);
        local_ids.add_shard(&local_first);
        local_ids.add_shard(&local_second);

        idx_t selected_id = 0;
        faiss::IDSelectorBatch selector(1, &selected_id);
        faiss::SearchParameters params;
        params.sel = &selector;
        local_ids.search(
                1, xq.data(), 1, distances.data(), labels.data(), &params);
        EXPECT_EQ(labels[0], 0);
        EXPECT_EQ(distances[0], 0.0F);
    }
}

TEST(ThreadedIndex, BinaryShardsRejectTranslatedIDSelector) {
    const uint8_t xb[] = {0, 1};
    idx_t selected_id = 1;
    faiss::IDSelectorBatch selector(1, &selected_id);
    faiss::SearchParameters params;
    params.sel = &selector;

    for (bool threaded : {false, true}) {
        faiss::IndexBinaryFlat first(8);
        faiss::IndexBinaryFlat second(8);
        first.add(1, xb);
        second.add(1, xb + 1);
        faiss::IndexBinaryShards shards(8, threaded, true);
        shards.add_shard(&first);
        shards.add_shard(&second);

        int32_t distance = -11;
        idx_t label = -21;
        try {
            shards.search(1, xb, 1, &distance, &label, &params);
            FAIL() << "expected translated IDSelector search to fail";
        } catch (const faiss::FaissException& exception) {
            EXPECT_NE(
                    std::string(exception.what())
                            .find("IDSelector search is not supported when "
                                  "successive_ids shifts shard IDs"),
                    std::string::npos);
        }
        EXPECT_EQ(distance, -11);
        EXPECT_EQ(label, -21);
        EXPECT_EQ(params.sel, &selector);
    }
}
