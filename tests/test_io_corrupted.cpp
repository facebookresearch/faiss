/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexNSG.h>

#include <faiss/utils/random.h>

using namespace faiss;

TEST(IoCorrupted, NSGNeighborIdEqualsNtotal) {
    int d = 8, nb = 500;
    std::vector<float> xb(nb * d);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 1234);

    IndexNSGFlat index(d, 16);
    index.add(nb, xb.data());

    // Set the last non-sentinel neighbor of every node to ntotal: this keeps
    // the graph connected and NSG::search_on_graph() must skip the entry.
    auto* graph = index.nsg.final_graph.get();
    for (int i = 0; i < nb; i++) {
        int last = -1;
        for (int j = 0; j < graph->K; j++) {
            if (graph->at(i, j) >= 0) {
                last = j;
            } else {
                break;
            }
        }
        if (last >= 0) {
            graph->at(i, last) = nb; // inject ntotal
        }
    }

    std::vector<float> xq(d);
    faiss::rand_smooth_vectors(1, d, xq.data(), 5678);
    std::vector<idx_t> I(1);
    std::vector<float> D(1);

    index.search(1, xq.data(), 1, D.data(), I.data());
    EXPECT_GE(I[0], 0);
    EXPECT_LT(I[0], nb);
}
