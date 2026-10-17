/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <omp.h>

#include <gtest/gtest.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/utils/hamming.h>

TEST(BinaryFlat, accuracy) {
    // dimension of the vectors to index
    int d = 64;

    // size of the database we plan to index
    size_t nb = 1000;

    // make the index object and train it
    faiss::IndexBinaryFlat index(d);

    std::vector<uint8_t> database(nb * (d / 8));
    for (size_t i = 0; i < nb * (d / 8); i++) {
        database[i] = rand() % 0x100;
    }

    { // populating the database
        index.add(nb, database.data());
    }

    size_t nq = 200;

    { // searching the database

        std::vector<uint8_t> queries(nq * (d / 8));
        for (size_t i = 0; i < nq * (d / 8); i++) {
            queries[i] = rand() % 0x100;
        }

        int k = 5;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<int> dis(k * nq);

        index.search(nq, queries.data(), k, dis.data(), nns.data());

        for (size_t i = 0; i < nq; ++i) {
            faiss::HammingComputer8 hc(queries.data() + i * (d / 8), d / 8);
            hamdis_t dist_min = hc.hamming(database.data());
            for (size_t j = 1; j < nb; ++j) {
                hamdis_t dist = hc.hamming(database.data() + j * (d / 8));
                if (dist < dist_min) {
                    dist_min = dist;
                }
            }
            EXPECT_EQ(dist_min, dis[k * i]);
        }
    }
}

// The database-parallel path (few queries) must give the same results as the
// sequential scan, including when search is called from inside an OpenMP
// parallel region (where a nested region usually gets a single thread).
TEST(BinaryFlat, db_parallel_matches_sequential) {
    int d = 128;
    size_t nb = 50000;
    faiss::IndexBinaryFlat index(d);
    std::vector<uint8_t> database(nb * (d / 8));
    for (size_t i = 0; i < database.size(); i++) {
        database[i] = rand() % 0x100;
    }
    index.add(nb, database.data());
    std::vector<uint8_t> query(d / 8);
    for (size_t i = 0; i < query.size(); i++) {
        query[i] = rand() % 0x100;
    }
    const int k = 10;
    const size_t saved = faiss::hamming_db_parallel_min_vectors;

    for (bool use_heap : {true, false}) {
        index.use_heap = use_heap;
        std::vector<int> dis_seq(k), dis_par(k), dis_nested(k);
        std::vector<faiss::idx_t> ids_seq(k), ids_par(k), ids_nested(k);

        faiss::hamming_db_parallel_min_vectors =
                std::numeric_limits<size_t>::max();
        index.search(1, query.data(), k, dis_seq.data(), ids_seq.data());

        faiss::hamming_db_parallel_min_vectors = 0;
        index.search(1, query.data(), k, dis_par.data(), ids_par.data());
#pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0) {
                index.search(
                        1,
                        query.data(),
                        k,
                        dis_nested.data(),
                        ids_nested.data());
            }
        }
        faiss::hamming_db_parallel_min_vectors = saved;

        EXPECT_EQ(dis_seq, dis_par);
        EXPECT_EQ(dis_seq, dis_nested);
        if (!use_heap) { // the counting variant is deterministic
            EXPECT_EQ(ids_seq, ids_par);
            EXPECT_EQ(ids_seq, ids_nested);
        }
    }
}
