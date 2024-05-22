/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <random>

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/MetricType.h>
#include <faiss/impl/IDGrouper.h>

// 64-bit int
using idx_t = faiss::idx_t;

using namespace faiss;

TEST(IdGrouper, get_group) {
    uint64_t ids1[1] = {0b1000100010001000};
    IDGrouperBitmap bitmap(1, ids1);

    ASSERT_EQ(3, bitmap.get_group(0));
    ASSERT_EQ(3, bitmap.get_group(1));
    ASSERT_EQ(3, bitmap.get_group(2));
    ASSERT_EQ(3, bitmap.get_group(3));
    ASSERT_EQ(7, bitmap.get_group(4));
    ASSERT_EQ(7, bitmap.get_group(5));
    ASSERT_EQ(7, bitmap.get_group(6));
    ASSERT_EQ(7, bitmap.get_group(7));
    ASSERT_EQ(11, bitmap.get_group(8));
    ASSERT_EQ(11, bitmap.get_group(9));
    ASSERT_EQ(11, bitmap.get_group(10));
    ASSERT_EQ(11, bitmap.get_group(11));
    ASSERT_EQ(15, bitmap.get_group(12));
    ASSERT_EQ(15, bitmap.get_group(13));
    ASSERT_EQ(15, bitmap.get_group(14));
    ASSERT_EQ(15, bitmap.get_group(15));
    ASSERT_EQ(bitmap.NO_MORE_DOCS, bitmap.get_group(16));
}

TEST(IdGrouper, set_group) {
    idx_t group_ids[] = {64, 127, 128, 1022};
    uint64_t ids[16] = {}; // 1023 / 64 + 1
    IDGrouperBitmap bitmap(16, ids);

    for (int i = 0; i < 4; i++) {
        bitmap.set_group(group_ids[i]);
    }

    int group_id_index = 0;
    for (int i = 0; i <= group_ids[3]; i++) {
        ASSERT_EQ(group_ids[group_id_index], bitmap.get_group(i));
        if (group_ids[group_id_index] == i) {
            group_id_index++;
        }
    }
    ASSERT_EQ(bitmap.NO_MORE_DOCS, bitmap.get_group(group_ids[3] + 1));
}

TEST(IdGrouper, sanity_test) {
    int d = 1;   // dimension
    int nb = 10; // database size

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    uint64_t bitmap[1] = {};
    faiss::IDGrouperBitmap id_grouper(1, bitmap);
    for (int i = 0; i < nb; i++) {
        id_grouper.set_group(i);
    }

    int k = 5;
    int m = 8;
    faiss::Index* index =
            new faiss::IndexHNSWFlat(d, m, faiss::MetricType::METRIC_L2);
    index->add(nb, xb); // add vectors to the index

    // search
    auto pSearchParameters = new faiss::SearchParametersHNSW();

    idx_t* expectedI = new idx_t[k];
    float* expectedD = new float[k];
    index->search(1, xb, k, expectedD, expectedI, pSearchParameters);

    idx_t* I = new idx_t[k];
    float* D = new float[k];
    pSearchParameters->grp = &id_grouper;
    index->search(1, xb, k, D, I, pSearchParameters);

    // compare
    for (int j = 0; j < k; j++) {
        ASSERT_EQ(expectedI[j], I[j]);
        ASSERT_EQ(expectedD[j], D[j]);
    }

    delete[] expectedI;
    delete[] expectedD;
    delete[] I;
    delete[] D;
    delete[] xb;
}

TEST(IdGrouper, bitmap_with_hnsw) {
    int d = 1;   // dimension
    int nb = 10; // database size

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    uint64_t bitmap[1] = {};
    faiss::IDGrouperBitmap id_grouper(1, bitmap);
    for (int i = 0; i < nb; i++) {
        if (i % 2 == 1) {
            id_grouper.set_group(i);
        }
    }

    int k = 10;
    int m = 8;
    faiss::Index* index =
            new faiss::IndexHNSWFlat(d, m, faiss::MetricType::METRIC_L2);
    index->add(nb, xb); // add vectors to the index

    // search
    idx_t* I = new idx_t[k];
    float* D = new float[k];

    auto pSearchParameters = new faiss::SearchParametersHNSW();
    pSearchParameters->grp = &id_grouper;

    index->search(1, xb, k, D, I, pSearchParameters);

    std::unordered_set<int> group_ids;
    ASSERT_EQ(0, I[0]);
    ASSERT_EQ(0, D[0]);
    group_ids.insert(id_grouper.get_group(I[0]));
    for (int j = 1; j < 5; j++) {
        ASSERT_NE(-1, I[j]);
        ASSERT_NE(std::numeric_limits<float>::max(), D[j]);
        group_ids.insert(id_grouper.get_group(I[j]));
    }
    for (int j = 5; j < k; j++) {
        ASSERT_EQ(-1, I[j]);
        ASSERT_EQ(std::numeric_limits<float>::max(), D[j]);
    }
    ASSERT_EQ(5, group_ids.size());

    delete[] I;
    delete[] D;
    delete[] xb;
}

TEST(IdGrouper, bitmap_with_hnswn_idmap) {
    int d = 1;   // dimension
    int nb = 10; // database size

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    idx_t* xids = new idx_t[d * nb];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    uint64_t bitmap[1] = {};
    faiss::IDGrouperBitmap id_grouper(1, bitmap);
    int num_grp = 0;
    int grp_size = 2;
    int id_in_grp = 0;
    for (int i = 0; i < nb; i++) {
        xids[i] = i + num_grp;
        id_in_grp++;
        if (id_in_grp == grp_size) {
            id_grouper.set_group(i + num_grp + 1);
            num_grp++;
            id_in_grp = 0;
        }
    }

    int k = 10;
    int m = 8;
    faiss::Index* index =
            new faiss::IndexHNSWFlat(d, m, faiss::MetricType::METRIC_L2);
    faiss::IndexIDMap id_map =
            faiss::IndexIDMap(index); // add vectors to the index
    id_map.add_with_ids(nb, xb, xids);

    // search
    idx_t* I = new idx_t[k];
    float* D = new float[k];

    auto pSearchParameters = new faiss::SearchParametersHNSW();
    pSearchParameters->grp = &id_grouper;

    id_map.search(1, xb, k, D, I, pSearchParameters);

    std::unordered_set<int> group_ids;
    ASSERT_EQ(0, I[0]);
    ASSERT_EQ(0, D[0]);
    group_ids.insert(id_grouper.get_group(I[0]));
    for (int j = 1; j < 5; j++) {
        ASSERT_NE(-1, I[j]);
        ASSERT_NE(std::numeric_limits<float>::max(), D[j]);
        group_ids.insert(id_grouper.get_group(I[j]));
    }
    for (int j = 5; j < k; j++) {
        ASSERT_EQ(-1, I[j]);
        ASSERT_EQ(std::numeric_limits<float>::max(), D[j]);
    }
    ASSERT_EQ(5, group_ids.size());

    delete[] I;
    delete[] D;
    delete[] xb;
}
