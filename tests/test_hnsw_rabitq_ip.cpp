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
#include <faiss/impl/rabitq_batch/split.h>
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

// Forced partitioned mode (threshold=1): AMX and VNNI produce byte-identical
// graphs and repeated runs are deterministic. Recall is not asserted here; it
// is checked on real data outside the unit tests.
TEST(HNSWRaBitQInnerProduct, PartitionedModeAmxMatchesVnniAndDeterministic) {
    if (faiss::rabitq_batch::select_backend(true) ==
        faiss::rabitq_batch::Backend::none) {
        GTEST_SKIP() << "batch build not available on this build or CPU";
    }
    const size_t d = 100, nt = 1000, nb = 5000;
    auto xt = random_vectors(nt, d, 11);
    auto xb = random_vectors(nb, d, 12);

    auto run_one = [&](bool use_amx) {
        auto idx = make_index(d, xt, true);
        idx->batch_build_use_amx = use_amx;
        idx->batch_build_partition_threshold = 1;
        idx->add(nb, xb.data());
        return idx;
    };

    auto amx1 = run_one(true);
    auto amx2 = run_one(true);
    auto vnni = run_one(false);
    EXPECT_EQ(serialize(*amx1), serialize(*amx2));
    EXPECT_EQ(serialize(*amx1), serialize(*vnni));
}

namespace {

// Checks the invariants of a rank split: both halves within the size bound,
// union equals the bucket, no duplicates inside a half, original order kept.
void check_rank_split(
        const std::vector<uint32_t>& bucket,
        const std::vector<float>& diff,
        double overlap) {
    std::vector<uint32_t> o0, o1;
    faiss::rabitq_batch::split_bucket_by_rank(bucket, diff, overlap, o0, o1);
    const size_t nb = bucket.size();
    const size_t bound = nb / 2 + size_t(double(nb) * overlap / 2.0) + 1;
    EXPECT_LE(o0.size(), bound);
    EXPECT_LE(o1.size(), bound);
    EXPECT_LT(o0.size(), nb);
    EXPECT_LT(o1.size(), nb);

    std::vector<int> pos(
            *std::max_element(bucket.begin(), bucket.end()) + 1, -1);
    for (size_t i = 0; i < nb; ++i)
        pos[bucket[i]] = int(i);
    for (const auto* half : {&o0, &o1}) {
        for (size_t i = 1; i < half->size(); ++i) {
            EXPECT_LT(pos[(*half)[i - 1]], pos[(*half)[i]]);
        }
    }
    std::vector<uint32_t> u(o0);
    u.insert(u.end(), o1.begin(), o1.end());
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    std::vector<uint32_t> b(bucket);
    std::sort(b.begin(), b.end());
    EXPECT_EQ(u, b);
    EXPECT_EQ(
            o0.size() + o1.size() - nb, 2 * size_t(double(nb) * overlap / 2.0));
}

} // namespace

// Oversized-bucket split: the median cut must shrink the bucket even
// when the 2-means projection is skewed, tied or degenerate.
TEST(HNSWRaBitQInnerProduct, RankSplitShrinksSkewedTiedAndDuplicateBuckets) {
    const size_t nb = 10001;
    std::vector<uint32_t> bucket(nb);
    for (size_t i = 0; i < nb; ++i)
        bucket[i] = uint32_t(3 * i + 7);
    std::mt19937 rng(5);
    std::shuffle(bucket.begin(), bucket.end(), rng);

    // Skewed: 90% of the members on one side of zero.
    std::vector<float> skew(nb);
    std::normal_distribution<float> g;
    for (size_t i = 0; i < nb; ++i)
        skew[i] = (i % 10 ? 1.0f : -1.0f) + 0.01f * g(rng);
    check_rank_split(bucket, skew, 0.15);

    // Fully duplicated vectors: every score identical -> even split by id.
    std::vector<float> same(nb, 0.0f);
    check_rank_split(bucket, same, 0.15);
    std::vector<uint32_t> o0, o1;
    faiss::rabitq_batch::split_bucket_by_rank(bucket, same, 0.0, o0, o1);
    std::vector<uint32_t> sorted(bucket);
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(o1.size(), nb / 2);
    for (uint32_t id : o1)
        EXPECT_LT(id, sorted[nb / 2]);

    // Non-finite scores must not break the ordering.
    std::vector<float> bad(skew);
    for (size_t i = 0; i < nb; i += 7)
        bad[i] = std::numeric_limits<float>::quiet_NaN();
    bad[1] = std::numeric_limits<float>::infinity();
    check_rank_split(bucket, bad, 0.15);

    // Repeated splitting converges: sizes shrink geometrically.
    size_t cur = nb;
    for (int pass = 0; pass < 6; ++pass) {
        std::vector<uint32_t> b(bucket.begin(), bucket.begin() + cur);
        std::vector<float> z(cur, 0.0f);
        faiss::rabitq_batch::split_bucket_by_rank(b, z, 0.15, o0, o1);
        const size_t next = std::max(o0.size(), o1.size());
        EXPECT_LE(double(next), 0.575 * double(cur) + 1.0);
        cur = next;
    }
}
