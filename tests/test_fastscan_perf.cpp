/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/impl/AuxIndexStructures.h>

TEST(TestFastScan, knnVSrange) {
    // small vectors and database
    int d = 64;
    size_t nb = 4000;

    // ivf centroids
    size_t nlist = 4;

    // more than 2 threads to surface
    // problems related to multi-threading
    omp_set_num_threads(8);

    // random database, also used as queries
    std::vector<float> database(nb * d);
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = distrib(rng);
    }

    // build index
    faiss::IndexFlatL2 coarse_quantizer(d);
    faiss::IndexIVFPQFastScan index(
            &coarse_quantizer, d, nlist, d / 2, 4, faiss::METRIC_L2, 32);
    index.pq.cp.niter = 10; // speed up train
    index.nprobe = nlist;
    index.train(nb, database.data());
    index.add(nb, database.data());

    std::vector<float> distances(nb);
    std::vector<faiss::idx_t> labels(nb);
    auto t = std::chrono::high_resolution_clock::now();
    index.search(nb, database.data(), 1, distances.data(), labels.data());
    auto knn_time = std::chrono::high_resolution_clock::now() - t;

    faiss::RangeSearchResult rsr(nb);
    t = std::chrono::high_resolution_clock::now();
    index.range_search(nb, database.data(), 1.0, &rsr);
    auto range_time = std::chrono::high_resolution_clock::now() - t;

    // we expect the perf of knn and range search
    // to be similar, at least within a factor of 4
    ASSERT_LE(range_time, knn_time * 4);
    ASSERT_LE(knn_time, range_time * 4);
}
