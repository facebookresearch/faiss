/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/idselector.h>

// See https://github.com/facebookresearch/faiss/issues/5576

namespace {

std::vector<float> increasing_values(int n) {
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = i;
    }
    return x;
}

} // namespace

// With continuous_update=true (the default), the permutation must stay
// in sync across remove_ids, i.e. search must work right after removals.
TEST(TestIndexFlat1D, RemoveIdsUpdatesPermutation) {
    const int n = 50;
    auto x = increasing_values(n);

    faiss::IndexFlat1D index;
    index.add(n, x.data());

    faiss::IDSelectorRange sel(25, n);
    index.remove_ids(sel);
    EXPECT_EQ(index.ntotal, 25);

    // remaining values: 0..24 at positions 0..24
    std::vector<float> query{10.4f};
    std::vector<faiss::idx_t> labels(3);
    std::vector<float> distances(3);
    index.search(1, query.data(), 3, distances.data(), labels.data());
    EXPECT_EQ(labels[0], 10); // value 10, |10.4 - 10| = 0.4
    EXPECT_EQ(labels[1], 11); // value 11, |10.4 - 11| = 0.6
    EXPECT_EQ(labels[2], 9);  // value 9,  |10.4 - 9| = 1.4
    ASSERT_NEAR(distances[0], 0.4f, 1e-5f);
}

// With continuous_update=false, the permutation is only refreshed by
// update_permutation(), also after remove_ids.
TEST(TestIndexFlat1D, RemoveIdsDeferredPermutation) {
    const int n = 50;
    auto x = increasing_values(n);

    faiss::IndexFlat1D index(/* continuous_update = */ false);
    index.add(n, x.data());

    faiss::IDSelectorRange sel(25, n);
    index.remove_ids(sel);

    std::vector<float> query{10.4f};
    std::vector<faiss::idx_t> labels(3);
    std::vector<float> distances(3);
    EXPECT_THROW(
            index.search(
                    1, query.data(), 3, distances.data(), labels.data()),
            faiss::FaissException);

    index.update_permutation();
    index.search(1, query.data(), 3, distances.data(), labels.data());
    EXPECT_EQ(labels[0], 10);
}
