/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

TEST(IVFPQ, onTheFlyDistanceToCodeMatchesPrecomputedTable) {
    constexpr int d = 4;
    constexpr size_t nt = 256;
    faiss::IndexFlatL2 coarse_quantizer(d);
    faiss::IndexIVFPQ index(&coarse_quantizer, d, 2, 2, 4);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);
    std::vector<float> training(nt * d);
    for (float& value : training) {
        value = distrib(rng);
    }
    index.train(nt, training.data());

    const std::vector<float> query = {0.2f, -0.4f, 0.6f, -0.8f};
    const std::vector<float> value = {-0.1f, 0.3f, 0.7f, -0.5f};
    faiss::idx_t list_no;
    float coarse_distance;
    index.quantizer->search(1, query.data(), 1, &coarse_distance, &list_no);
    ASSERT_GE(list_no, 0);

    std::vector<uint8_t> code(index.code_size);
    index.encode(list_no, value.data(), code.data());
    std::unique_ptr<faiss::InvertedListScanner> table_scanner(
            index.get_InvertedListScanner(false, nullptr, nullptr));
    std::unique_ptr<faiss::InvertedListScanner> on_the_fly_scanner(
            index.get_InvertedListScanner(
                    false, nullptr, faiss::IndexIVFPQ::ScannerMode::OnTheFly));

    for (auto* scanner : {table_scanner.get(), on_the_fly_scanner.get()}) {
        scanner->set_query(query.data());
        scanner->set_list(list_no, coarse_distance);
    }

    EXPECT_NEAR(
            on_the_fly_scanner->distance_to_code(code.data()),
            table_scanner->distance_to_code(code.data()),
            1e-4);
}

TEST(IVFPQ, accuracy) {
    // dimension of the vectors to index
    int d = 64;

    // size of the database we plan to index
    size_t nb = 1000;

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt = 1500;

    // make the index object and train it
    faiss::IndexFlatL2 coarse_quantizer(d);

    // a reasonable number of centroids to index nb vectors
    int ncentroids = 25;

    faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids, 16, 8);

    // index that gives the ground-truth
    faiss::IndexFlatL2 index_gt(d);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    { // training

        std::vector<float> trainvecs(nt * d);
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }
        index.verbose = true;
        index.train(nt, trainvecs.data());
    }

    { // populating the database

        std::vector<float> database(nb * d);
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        index.add(nb, database.data());
        index_gt.add(nb, database.data());
    }

    int nq = 200;
    int n_ok;

    { // searching the database

        std::vector<float> queries(nq * d);
        for (int i = 0; i < nq * d; i++) {
            queries[i] = distrib(rng);
        }

        std::vector<faiss::idx_t> gt_nns(nq);
        std::vector<float> gt_dis(nq);

        index_gt.search(nq, queries.data(), 1, gt_dis.data(), gt_nns.data());

        index.nprobe = 5;
        int k = 5;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        index.search(nq, queries.data(), k, dis.data(), nns.data());

        n_ok = 0;
        for (int q = 0; q < nq; q++) {
            for (int i = 0; i < k; i++)
                if (nns[q * k + i] == gt_nns[q])
                    n_ok++;
        }
        EXPECT_GT(n_ok, nq * 0.4);
    }
}
