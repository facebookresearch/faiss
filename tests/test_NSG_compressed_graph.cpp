/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexNSG.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>

using namespace faiss;

using FinalNSGGraph = nsg::Graph<int32_t>;

struct CompressedNSGGraph : FinalNSGGraph {
    int bits;
    size_t stride;
    std::vector<uint8_t> compressed_data;

    CompressedNSGGraph(const FinalNSGGraph& graph, int bits_in)
            : FinalNSGGraph(graph.data, graph.N, graph.K), bits(bits_in) {
        FAISS_THROW_IF_NOT((1 << bits) >= K + 1);
        stride = (K * bits + 7) / 8;
        compressed_data.resize(N * stride);
        for (size_t i = 0; i < size_t(N); i++) {
            BitstringWriter writer(compressed_data.data() + i * stride, stride);
            for (size_t j = 0; j < size_t(K); j++) {
                int32_t v = graph.data[i * K + j];
                if (v == -1) {
                    writer.write(K + 1, bits);
                    break;
                } else {
                    writer.write(v, bits);
                }
            }
        }
        data = nullptr;
    }

    size_t get_neighbors(int i, int32_t* neighbors) const override {
        BitstringReader reader(compressed_data.data() + i * stride, stride);
        for (int j = 0; j < K; j++) {
            int32_t v = reader.read(bits);
            if (v == K + 1) {
                return j;
            }
            neighbors[j] = v;
        }
        return K;
    }
};

TEST(NSGCompressed, test_compressed) {
    size_t nq = 10, nt = 0, nb = 5000, d = 32, k = 10;

    using idx_t = faiss::idx_t;

    std::vector<float> buf((nq + nb + nt) * d);
    faiss::rand_smooth_vectors(nq + nb + nt, d, buf.data(), 1234);
    const float* xt = buf.data();
    const float* xb = xt + nt * d;
    const float* xq = xb + nb * d;

    faiss::IndexNSGFlat index(d, 32);

    index.add(nb, xb);

    std::vector<faiss::idx_t> Iref(nq * k);
    std::vector<float> Dref(nq * k);
    index.search(nq, xq, k, Dref.data(), Iref.data());

    // replace the shared ptr
    index.nsg.final_graph.reset(
            new CompressedNSGGraph(*index.nsg.final_graph, 13));

    std::vector<idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    index.search(nq, xq, k, D.data(), I.data());

    // make sure we find back the original results
    EXPECT_EQ(Iref, I);
    EXPECT_EQ(Dref, D);
}

// Regression test for sync_prune out-of-bounds bug.
//
// With ntotal=1 and L=1, search_on_graph produces pool = [{id:0}].
// sync_prune(q=0): pool[0].id == q → start++ → start == pool.size().
// Old code: pool[start] is out-of-bounds → undefined behavior.
// Fix: guard returns early and fills graph row with EMPTY_ID.
//
// Calls NSG::build() directly (bypassing IndexNSG::check_knn_graph)
// to reach the edge case with ntotal=1.
TEST(NSGBugs, SyncPruneSingleNode) {
    constexpr int d = 4;
    constexpr int R = 1;

    faiss::IndexFlat storage(d);
    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    storage.add(1, vec);

    faiss::idx_t knn_data[] = {-1};
    faiss::nsg::Graph<faiss::idx_t> knn_graph(knn_data, 1, 1);

    faiss::NSG nsg_obj(R);
    nsg_obj.L = 1;

    // Old code crashes here. Fixed code handles it.
    ASSERT_NO_THROW(nsg_obj.build(&storage, 1, knn_graph, false));
    EXPECT_TRUE(nsg_obj.is_built);
    EXPECT_EQ(nsg_obj.enterpoint, 0);

    // Search returns the only node
    nsg_obj.search_L = 1;
    faiss::VisitedTable vt(1);
    auto dis = std::unique_ptr<faiss::DistanceComputer>(
            faiss::nsg::storage_distance_computer(&storage));
    dis->set_query(vec);

    faiss::idx_t label = -1;
    float distance = -1;
    nsg_obj.search(*dis, 1, &label, &distance, vt);
    EXPECT_EQ(label, 0);
}
