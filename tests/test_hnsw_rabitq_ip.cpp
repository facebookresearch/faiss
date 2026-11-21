/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <omp.h>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/io.h>
#include <faiss/impl/rabitq_batch/batch_build.h>
#include <faiss/index_io.h>

namespace {

std::vector<float> random_vectors(size_t n, size_t d, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist;
    std::vector<float> x(n * d);
    for (float& v : x) {
        v = dist(rng);
    }
    return x;
}

std::unique_ptr<faiss::IndexHNSWRaBitQ> make_index(
        size_t d,
        const std::vector<float>& xt,
        bool batch_build) {
    auto index = std::make_unique<faiss::IndexHNSWRaBitQ>(
            int(d), 16, 8, faiss::METRIC_INNER_PRODUCT);
    index->batch_build = batch_build;
    index->hnsw.efConstruction = 40;
    index->train(xt.size() / d, xt.data());
    return index;
}

std::vector<uint8_t> serialize(const faiss::Index& index) {
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.data;
}

} // namespace

// With the switch off, add() must be exactly the incremental insertion.
TEST(HNSWRaBitQInnerProduct, SwitchOffIsIncrementalAdd) {
    const size_t d = 64, nt = 1000, nb = 2000;
    auto xt = random_vectors(nt, d, 1);
    auto xb = random_vectors(nb, d, 2);

    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1); // the incremental path is deterministic
    auto a = make_index(d, xt, false);
    a->add(nb, xb.data());
    auto b = make_index(d, xt, false);
    b->IndexHNSW::add(nb, xb.data());
    omp_set_num_threads(saved_threads);

    EXPECT_EQ(serialize(*a), serialize(*b));
}

// The batch build computes the exact k-NN on the codes, and the HNSW pruning
// always keeps the best candidate: every level-0 list must contain a node
// with the highest symmetric_dis() score.
TEST(HNSWRaBitQInnerProduct, BatchBuildKeepsCodeNearestNeighbor) {
    if (faiss::rabitq_batch::select_backend(true) ==
        faiss::rabitq_batch::Backend::none) {
        GTEST_SKIP() << "batch build not available on this build or CPU";
    }
    for (bool use_amx : {true, false}) {
        // d = 100 exercises the padding of the last 64-dimension block
        const size_t d = 100, nt = 1000, nb = 1500;
        auto xt = random_vectors(nt, d, 3);
        auto xb = random_vectors(nb, d, 4);
        auto index = make_index(d, xt, true);
        index->batch_build_use_amx = use_amx;
        index->add(nb, xb.data());
        ASSERT_EQ(index->ntotal, faiss::idx_t(nb));

        std::unique_ptr<faiss::DistanceComputer> dc(
                index->storage->get_distance_computer());
        const faiss::HNSW& hnsw = index->hnsw;
        for (size_t i = 0; i < nb; ++i) {
            float best = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < nb; ++j) {
                if (j != i) {
                    best = std::max(best, dc->symmetric_dis(i, j));
                }
            }
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);
            float best_kept = -std::numeric_limits<float>::infinity();
            for (size_t k = begin; k < end && hnsw.neighbors[k] >= 0; ++k) {
                best_kept = std::max(
                        best_kept, dc->symmetric_dis(i, hnsw.neighbors[k]));
            }
            ASSERT_EQ(best_kept, best) << "node " << i << " amx " << use_amx;
        }
    }
}
