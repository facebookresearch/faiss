/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <gtest/gtest.h>

using namespace faiss;

TEST(TestMemoryLeak, ivfflat) {
    GTEST_SKIP() << "Skipping flaky memory leak test.";
    size_t num_tfidf_faiss_cells = 20;
    size_t max_tfidf_features = 500;

    IndexFlatIP quantizer(max_tfidf_features);
    IndexIVFFlat tfidf_faiss_index(
            &quantizer, max_tfidf_features, num_tfidf_faiss_cells);

    std::vector<float> dense_matrix(5000 * max_tfidf_features);
    float_rand(dense_matrix.data(), dense_matrix.size(), 123);

    tfidf_faiss_index.train(5000, dense_matrix.data());
    tfidf_faiss_index.add(5000, dense_matrix.data());

    int N1 = 1000;
    int N2 = 10000;

    std::vector<float> ent_substr_tfidfs_list(N1 * max_tfidf_features);
    float_rand(
            ent_substr_tfidfs_list.data(), ent_substr_tfidfs_list.size(), 1234);

    for (int bs : {1, 4, 16}) {
        size_t m0 = get_mem_usage_kb();
        double t0 = getmillisecs();

        for (int i = 0; i < N2; i++) {
            std::vector<idx_t> I(10 * bs);
            std::vector<float> D(10 * bs);

            tfidf_faiss_index.search(
                    bs,
                    ent_substr_tfidfs_list.data() +
                            (i % (N1 - bs + 1)) * max_tfidf_features,
                    10,
                    D.data(),
                    I.data());
            if (i % 100 == 0) {
                printf("[%.2f s] BS %d %d: %ld kB %.2f bytes/it\r",
                       (getmillisecs() - t0) / 1000,
                       bs,
                       i,
                       get_mem_usage_kb(),
                       (get_mem_usage_kb() - m0) * 1024.0 / (i + 1));
                fflush(stdout);
            }
        }
        printf("\n");
        EXPECT_GE(50 * bs, (get_mem_usage_kb() - m0) * 1024.0 / N2);
    }
}
