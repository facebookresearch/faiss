/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>

using namespace faiss;

/** A ResultHandler that just collects all results that presented to it. */
struct CollectAllResultHandler : ResultHandler {
    std::vector<float> D;
    std::vector<idx_t> I;

    bool add_result(float distance, idx_t i) override {
        // we never change the default threshold so that all results pass
        D.push_back(distance);
        I.push_back(i);
        return true;
    }

    void get_top_k(
            size_t top_k,
            std::vector<float>& out_D,
            std::vector<idx_t>& out_I,
            bool is_max) const {
        size_t n_results = D.size();

        std::vector<size_t> indices(n_results);
        for (size_t i = 0; i < n_results; i++) {
            indices[i] = i;
        }

        const std::vector<float>& distances = D;

        // Sort by distance (ascending for L2, descending for IP)
        if (is_max) {
            std::sort(
                    indices.begin(),
                    indices.end(),
                    [&distances](size_t a, size_t b) {
                        return distances[a] > distances[b];
                    });
        } else {
            std::sort(
                    indices.begin(),
                    indices.end(),
                    [&distances](size_t a, size_t b) {
                        return distances[a] < distances[b];
                    });
        }

        size_t n_out = std::min(top_k, n_results);
        out_D.resize(n_out);
        out_I.resize(n_out);
        for (size_t i = 0; i < n_out; i++) {
            out_D[i] = D[indices[i]];
            out_I[i] = I[indices[i]];
        }
    }
};

struct ThresholdUpdatingResultHandler : ResultHandler {
    bool add_result(float distance, idx_t /* i */) override {
        threshold = distance;
        return true;
    }
};

namespace {

TEST(TestSingleQueryBlockResultHandler, SynchronizesWrappedThreshold) {
    ThresholdUpdatingResultHandler handler;
    handler.threshold = 10.0f;
    SingleQueryBlockResultHandler<CMax<float, idx_t>, false> block(handler);
    SingleQueryBlockResultHandler<CMax<float, idx_t>, false>::
            SingleResultHandler single(block);

    single.begin(0);
    EXPECT_FLOAT_EQ(single.threshold, 10.0f);
    single.add_result(4.0f, 0);
    EXPECT_FLOAT_EQ(single.threshold, 4.0f);
}

constexpr int d = 64;
constexpr size_t nt = 5000;
constexpr size_t nb = 2000;
constexpr size_t nq = 100;
constexpr int k = 10;

// Intrinsic dimensionality ~10: harder than a subspace, easier than uniform.
std::vector<float> make_smooth_data(size_t n, int64_t seed) {
    std::vector<float> data(n * d);
    rand_smooth_vectors(n, d, data.data(), seed);
    return data;
}

// Compares the top-k of search() with search1() through a collect-all handler.
void test_index(
        const char* index_key,
        MetricType metric,
        double min_match_ratio = 1.0) {
    std::unique_ptr<Index> index(index_factory(d, index_key, metric));

    auto xt = make_smooth_data(nt, 1234);
    auto xb = make_smooth_data(nb, 4567);
    auto xq = make_smooth_data(nq, 7890);

    index->train(nt, xt.data());
    index->add(nb, xb.data());

    if (IndexIVF* index_ivf = dynamic_cast<IndexIVF*>(index.get())) {
        index_ivf->nprobe = 4;
    }

    std::vector<idx_t> Iref(nq * k);
    std::vector<float> Dref(nq * k);
    index->search(nq, xq.data(), k, Dref.data(), Iref.data());

    bool is_max = (metric == METRIC_INNER_PRODUCT);

    for (size_t q = 0; q < nq; q++) {
        CollectAllResultHandler handler;
        // threshold that every result passes
        if (is_max) {
            handler.threshold = -std::numeric_limits<float>::max();
        } else {
            handler.threshold = std::numeric_limits<float>::max();
        }
        index->search1(xq.data() + q * d, handler);

        std::vector<float> D_handler;
        std::vector<idx_t> I_handler;
        handler.get_top_k(k, D_handler, I_handler, is_max);

        const idx_t* Iref_q = Iref.data() + q * k;

        std::unordered_set<idx_t> ref_set(Iref_q, Iref_q + k);
        std::unordered_set<idx_t> handler_set(
                I_handler.begin(), I_handler.end());

        int matches = 0;
        for (idx_t id : handler_set) {
            if (ref_set.count(id) > 0) {
                matches++;
            }
        }

        EXPECT_GE(matches, static_cast<int>(k * min_match_ratio))
                << "Query " << q << ": expected at least "
                << static_cast<int>(k * min_match_ratio) << " matches, got "
                << matches;
    }
}

TEST(TestIndexTypes, IVFFlat_L2) {
    test_index("IVF32,Flat", METRIC_L2);
}

TEST(TestIndexTypes, IVFPQ_L2) {
    test_index("IVF32,PQ8np", METRIC_L2);
}

TEST(TestIndexTypes, IVFRaBitQ_IP) {
    test_index("IVF32,RaBitQ", METRIC_INNER_PRODUCT);
}

TEST(TestIndexTypes, IVFRaBitQ4_L2) {
    // RaBitQ4 does a topk per invlist, so not exactly the same as top over all
    test_index("IVF32,RaBitQ4", METRIC_L2, 0.8);
}

TEST(TestIndexTypes, IVFRaBitQ4_IP) {
    // RaBitQ4 does a topk per invlist, so not exactly the same as top over all
    test_index("IVF32,RaBitQ4", METRIC_INNER_PRODUCT, 0.8);
}

TEST(TestIndexTypes, HNSW) {
    test_index("HNSW32,Flat", METRIC_L2);
}

TEST(TestIndexTypes, SQ8) {
    test_index("SQ8", METRIC_L2);
}

TEST(RaBitQScanner, MatchesDistanceComputer) {
    auto xb = make_smooth_data(nb, 4567);
    auto xq = make_smooth_data(1, 7890);
    for (auto [nb_bits, qb] : {std::pair{1, 4}, std::pair{4, 0}}) {
        IndexFlatL2 quantizer(d);
        IndexIVFRaBitQ index(&quantizer, d, 1, METRIC_L2, true, nb_bits);
        index.qb = qb;
        index.train(nb, xb.data());
        index.add(nb, xb.data());
        InvertedLists::ScopedCodes codes(index.invlists, 0);

        std::unique_ptr<InvertedListScanner> scanner(
                index.get_InvertedListScanner(true, nullptr, nullptr));
        scanner->set_query(xq.data());
        scanner->set_list(0, 0);

        std::vector<float> dis(nb);
        for (size_t j = 0; j < nb; j++) {
            dis[j] = scanner->distance_to_code(
                    codes.get() + j * index.code_size);
        }
        std::vector<float> sorted_dis = dis;
        std::sort(sorted_dis.begin(), sorted_dis.end());
        CollectAllResultHandler handler;
        handler.threshold = sorted_dis[nb / 2];

        std::vector<idx_t> expected;
        for (size_t j = 0; j < nb; j++) {
            if (dis[j] < handler.threshold) {
                expected.push_back(j);
            }
        }

        scanner->scan_codes(nb, codes.get(), nullptr, handler);
        if (nb_bits > 1) {
            // The probabilistic error-bound prune may drop a true hit.
            EXPECT_TRUE(
                    std::includes(
                            expected.begin(),
                            expected.end(),
                            handler.I.begin(),
                            handler.I.end()));
        } else {
            EXPECT_EQ(handler.I, expected);
        }
    }
}

} // namespace
