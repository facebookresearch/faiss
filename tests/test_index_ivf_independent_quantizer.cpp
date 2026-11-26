/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFIndependentQuantizer.h>

// See https://github.com/facebookresearch/faiss/issues/5577

TEST(TestIVFIndependentQuantizer, AddUpdatesNtotal) {
    const int d = 8;
    const int n = 64;
    std::vector<float> x(size_t(n) * d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            x[size_t(i) * d + j] = ((i * 5 + j * 3) % 40) * 0.5f;
        }
    }

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index_ivf(&quantizer, d, 4);
    faiss::IndexIVFIndependentQuantizer wrapper(&quantizer, &index_ivf);

    wrapper.train(n, x.data());
    wrapper.add(n, x.data());

    // the wrapper's public ntotal must track the inner IVF index
    EXPECT_EQ(wrapper.ntotal, n);
    EXPECT_EQ(index_ivf.ntotal, n);

    std::vector<faiss::idx_t> labels(1);
    std::vector<float> distances(1);
    wrapper.search(1, x.data(), 1, distances.data(), labels.data());
    EXPECT_EQ(labels[0], 0);
    EXPECT_LT(distances[0], 1e-3f);

    wrapper.reset();
    EXPECT_EQ(wrapper.ntotal, 0);
    EXPECT_EQ(index_ivf.ntotal, 0);
}
