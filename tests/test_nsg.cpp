/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <gtest/gtest.h>

#include <faiss/IndexNSG.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/NSG.h>

using namespace std::chrono;


TEST(NSG, accuracy) {

    // dimension of the vectors to index
    int d = 16;

    // size of the database we plan to index
    size_t nb = 10000;

    // K of KNN graph
    int GK = 64;

    int R = 16;  // graph degree

    faiss::IndexNSGFlat index (d, R);

    // index that gives the ground-truth
    faiss::IndexFlatL2 index_gt (d);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    { // building NSG
        std::vector <float> database (nb * d);
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        // Build a KNN graph, the point itself is included
        // but it won't affect the result too much
        index_gt.add (nb, database.data());
        std::vector<faiss::Index::idx_t> knng (nb * GK);
        std::vector<float>               tmp (nb * GK);

        index_gt.search (nb, database.data(), GK,
                         tmp.data(), knng.data());

        index.nsg.L = 50;
        index.nsg.C = 100;
        index.build(nb, database.data(), knng.data(), GK);
    }

    int nq = 200;
    int n_ok;

    { // searching the database

        std::vector <float> queries (nq * d);
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = distrib(rng);
        }

        int k = 1;

        std::vector<faiss::Index::idx_t> gt_nns (nq);
        std::vector<float>               gt_dis (nq);

        index_gt.search (nq, queries.data(), 1,
                         gt_dis.data(), gt_nns.data());

        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);

        index.nsg.search_L = 10;
        index.search (nq, queries.data(), k, dis.data(), nns.data());

        n_ok = 0;
        for (int q = 0; q < nq; q++) {
            for (int i = 0; i < k; i++)
                if (nns[q * k + i] == gt_nns[q])
                    n_ok++;
        }
        double recall = 1.0 * n_ok / nq;
        printf("NSG Recall-1@%d: %lf\n", k, recall);

        // 0.93 is a resonable result.
        EXPECT_GT(recall, 0.9);
    }

}

TEST(NSG, connectivity) {

    // dimension of the vectors to index
    int d = 16;

    // size of the database we plan to index
    size_t nb = 10000;

    // K of KNN graph
    int GK = 64;

    faiss::IndexNSGFlat index (d, 16);

    // index that gives the ground-truth
    faiss::IndexFlatL2 index_gt (d);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    { // build NSG
        std::vector <float> database (nb * d);
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        // Build a KNN graph, the point itself is included
        // but it won't affect the result too much
        index_gt.add (nb, database.data());
        std::vector<faiss::Index::idx_t> knng (nb * GK);
        std::vector<float>               tmp (nb * GK);

        index_gt.search (nb, database.data(), GK,
                         tmp.data(), knng.data());

        index.nsg.L = 20;
        index.nsg.C = 50;
        index.build(nb, database.data(), knng.data(), GK);
    }

    { // check the connectivity
        faiss::nsg::VisitedTable vt(nb);
        int root = index.nsg.enterpoint;
        int count;
        index.nsg.dfs(vt, root, count);

        // The NSG should be fully connected.
        EXPECT_EQ(count, nb);
    }

}


TEST(NSG, knng) {
    // Test incase of `GK < R`

    // dimension of the vectors to index
    int d = 16;

    // size of the database we plan to index
    size_t nb = 10000;

    // K of KNN graph
    int GK = 8;

    int R = 16;  // graph degree

    faiss::IndexNSGFlat index (d, R);

    // index that gives the ground-truth
    faiss::IndexFlatL2 index_gt (d);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    { // populating the database
        std::vector <float> database (nb * d);
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        // Build a KNN graph, the point itself is included
        // but it won't affect the result too much
        index_gt.add (nb, database.data());
        std::vector<faiss::Index::idx_t> knng (nb * GK);
        std::vector<float>               tmp (nb * GK);

        index_gt.search (nb, database.data(), GK,
                         tmp.data(), knng.data());

        index.nsg.L = 20;
        index.nsg.C = 50;
        index.build(nb, database.data(), knng.data(), GK);
    }

    int nq = 200;
    int n_ok;

    { // searching the database

        std::vector <float> queries (nq * d);
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = distrib(rng);
        }

        int k = 1;

        std::vector<faiss::Index::idx_t> gt_nns (nq);
        std::vector<float>               gt_dis (nq);

        index_gt.search (nq, queries.data(), 1,
                         gt_dis.data(), gt_nns.data());

        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);

        index.nsg.search_L = 10;
        index.search (nq, queries.data(), k, dis.data(), nns.data());

        n_ok = 0;
        for (int q = 0; q < nq; q++) {
            for (int i = 0; i < k; i++)
                if (nns[q * k + i] == gt_nns[q])
                    n_ok++;
        }
        double nsg_recall = 1.0 * n_ok / nq;
        printf("NSG Recall-1@%d: %lf\n", k, nsg_recall);

        // 0.84 is a resonable result.
        EXPECT_GT(nsg_recall, 0.8);
    }

}