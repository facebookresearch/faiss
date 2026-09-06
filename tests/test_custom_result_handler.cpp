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
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/index_factory.h>
#include <faiss/invlists/InvertedLists.h>
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

    // Sort results by distance and return the top k
    void get_top_k(
            size_t top_k,
            std::vector<float>& out_D,
            std::vector<idx_t>& out_I,
            bool is_max) const {
        size_t n_results = D.size();

        // Create indices for sorting
        std::vector<size_t> indices(n_results);
        for (size_t i = 0; i < n_results; i++) {
            indices[i] = i;
        }

        // Make local copies for lambda capture
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

        // Take top k
        size_t n_out = std::min(top_k, n_results);
        out_D.resize(n_out);
        out_I.resize(n_out);
        for (size_t i = 0; i < n_out; i++) {
            out_D[i] = D[indices[i]];
            out_I[i] = I[indices[i]];
        }
    }
};

namespace {

// dimension of the vectors to index
constexpr int d = 64;

// nb of training vectors
constexpr size_t nt = 5000;

// size of the database
constexpr size_t nb = 2000;

// nb of queries
constexpr size_t nq = 100;

// k for search
constexpr int k = 10;

/**
 * Generate smooth random data using Faiss's rand_smooth_vectors.
 * This produces data with intrinsic dimensionality ~10 that is harder to
 * index than a subspace but easier than uniform random data.
 */
std::vector<float> make_smooth_data(size_t n, int64_t seed) {
    std::vector<float> data(n * d);
    rand_smooth_vectors(n, d, data.data(), seed);
    return data;
}

/**
 * Test helper: trains an index, adds data, and performs a search using
 * both the standard search method and the custom handler search1 method.
 * Compares the results to ensure they match.
 */
void test_index(
        const char* index_key,
        MetricType metric,
        double min_match_ratio = 1.0) {
    // Create index using factory
    std::unique_ptr<Index> index(index_factory(d, index_key, metric));
    ASSERT_NE(index, nullptr) << "Failed to create IVF index for " << index_key;

    // Generate smooth random data for training and database
    auto xt = make_smooth_data(nt, 1234);
    auto xb = make_smooth_data(nb, 4567);
    auto xq = make_smooth_data(nq, 7890);

    // Train the index
    index->train(nt, xt.data());

    // Add database vectors
    index->add(nb, xb.data());

    // Set nprobe for IVF indexes
    if (IndexIVF* index_ivf = dynamic_cast<IndexIVF*>(index.get())) {
        index_ivf->nprobe = 4;
    }

    // Perform reference search using standard search method
    std::vector<idx_t> Iref(nq * k);
    std::vector<float> Dref(nq * k);
    index->search(nq, xq.data(), k, Dref.data(), Iref.data());

    // For IP metric, we need to sort in descending order
    bool is_max = (metric == METRIC_INNER_PRODUCT);

    // Now test search1 with custom handler for each query
    for (size_t q = 0; q < nq; q++) {
        CollectAllResultHandler handler;
        // Set threshold to collect all results
        // For L2: use max float (condition: threshold > distance)
        // For IP: use min float (condition: threshold < distance)
        if (is_max) {
            handler.threshold = -std::numeric_limits<float>::max();
        } else {
            handler.threshold = std::numeric_limits<float>::max();
        }
        index->search1(xq.data() + q * d, handler);

        // Sort the handler results and get top k
        std::vector<float> D_handler;
        std::vector<idx_t> I_handler;
        handler.get_top_k(k, D_handler, I_handler, is_max);

        // Compare with reference results for this query using set comparison
        const idx_t* Iref_q = Iref.data() + q * k;

        // Create sets of IDs for comparison
        std::unordered_set<idx_t> ref_set(Iref_q, Iref_q + k);
        std::unordered_set<idx_t> handler_set(
                I_handler.begin(), I_handler.end());

        // Count how many results from handler are in the reference set
        int matches = 0;
        for (idx_t id : handler_set) {
            if (ref_set.count(id) > 0) {
                matches++;
            }
        }

        // Check that matches meet the minimum threshold
        EXPECT_GE(matches, static_cast<int>(k * min_match_ratio))
                << "Query " << q << ": expected at least "
                << static_cast<int>(k * min_match_ratio) << " matches, got "
                << matches;
    }
}

void test_ivfflat_batch4_matches_scalar(MetricType metric) {
    constexpr size_t local_d = 37;
    constexpr size_t local_nb = 11;

    std::vector<float> xb(local_nb * local_d);
    std::vector<float> xq(local_d);
    rand_smooth_vectors(local_nb, local_d, xb.data(), 1234);
    rand_smooth_vectors(1, local_d, xq.data(), 5678);
    std::copy_n(xb.data() + 4 * local_d, local_d, xb.data() + 5 * local_d);

    std::vector<idx_t> ids(local_nb);
    for (size_t i = 0; i < local_nb; i++) {
        ids[i] = 1000 + 3 * i;
    }

    IndexFlat quantizer(local_d, metric);
    IndexIVFFlat index(&quantizer, local_d, 1, metric);
    index.train(local_nb, xb.data());
    index.add_with_ids(local_nb, xb.data(), ids.data());

    const size_t list_size = index.invlists->list_size(0);
    ASSERT_EQ(list_size, local_nb);
    InvertedLists::ScopedCodes codes(index.invlists, 0);
    InvertedLists::ScopedIds stored_ids(index.invlists, 0);
    IDSelectorAll select_all;

    std::unique_ptr<InvertedListScanner> distance_scanner(
            index.get_InvertedListScanner(false, &select_all, nullptr));
    distance_scanner->set_query(xq.data());
    distance_scanner->set_list(0, 0.0f);
    const float tied_threshold = distance_scanner->distance_to_code(
            codes.get() + 4 * index.code_size);
    EXPECT_EQ(
            tied_threshold,
            distance_scanner->distance_to_code(
                    codes.get() + 5 * index.code_size));

    const float open_threshold = metric == METRIC_INNER_PRODUCT
            ? -std::numeric_limits<float>::max()
            : std::numeric_limits<float>::max();
    for (float threshold : {open_threshold, tied_threshold}) {
        for (bool store_pairs : {false, true}) {
            SCOPED_TRACE(
                    testing::Message()
                    << "metric=" << metric << ", threshold=" << threshold
                    << ", store_pairs=" << store_pairs);

            std::unique_ptr<InvertedListScanner> batch_scanner(
                    index.get_InvertedListScanner(
                            store_pairs, nullptr, nullptr));
            std::unique_ptr<InvertedListScanner> scalar_scanner(
                    index.get_InvertedListScanner(
                            store_pairs, &select_all, nullptr));
            batch_scanner->set_query(xq.data());
            batch_scanner->set_list(0, 0.0f);
            scalar_scanner->set_query(xq.data());
            scalar_scanner->set_list(0, 0.0f);

            CollectAllResultHandler batch_handler;
            CollectAllResultHandler scalar_handler;
            batch_handler.threshold = threshold;
            scalar_handler.threshold = threshold;
            const idx_t* scan_ids = store_pairs ? nullptr : stored_ids.get();
            const size_t batch_nup = batch_scanner->scan_codes(
                    list_size, codes.get(), scan_ids, batch_handler);
            const size_t scalar_nup = scalar_scanner->scan_codes(
                    list_size, codes.get(), scan_ids, scalar_handler);

            EXPECT_EQ(batch_handler.D, scalar_handler.D);
            EXPECT_EQ(batch_handler.I, scalar_handler.I);
            EXPECT_EQ(batch_handler.stats.scan_cnt, list_size);
            EXPECT_EQ(
                    batch_handler.stats.scan_cnt,
                    scalar_handler.stats.scan_cnt);
            EXPECT_EQ(
                    batch_handler.stats.nheap_updates,
                    scalar_handler.stats.nheap_updates);
            EXPECT_EQ(batch_nup, scalar_nup);
        }
    }
}

/*************************************************************
 * Test cases for different IVF index types
 *************************************************************/

TEST(TestIndexTypes, IVFFlat_L2) {
    test_index("IVF32,Flat", METRIC_L2);
}

TEST(TestIndexTypes, IVFFlat_IP) {
    test_index("IVF32,Flat", METRIC_INNER_PRODUCT);
}

TEST(IVFFlatScanner, Batch4MatchesScalarL2) {
    test_ivfflat_batch4_matches_scalar(METRIC_L2);
}

TEST(IVFFlatScanner, Batch4MatchesScalarIP) {
    test_ivfflat_batch4_matches_scalar(METRIC_INNER_PRODUCT);
}

TEST(TestIndexTypes, IVFPQ_L2) {
    test_index("IVF32,PQ8np", METRIC_L2);
}

TEST(TestIndexTypes, IVFPQ_IP) {
    test_index("IVF32,PQ8np", METRIC_INNER_PRODUCT);
}

TEST(TestIndexTypes, IVFSQ_L2) {
    test_index("IVF32,SQ8", METRIC_L2);
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

} // namespace
