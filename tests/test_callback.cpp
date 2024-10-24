/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/random.h>

TEST(TestCallback, timeout) {
    int n = 1000;
    int k = 100;
    int d = 128;
    int niter = 1000000000;
    int seed = 42;

    std::vector<float> vecs(n * d);
    faiss::float_rand(vecs.data(), vecs.size(), seed);

    auto index(new faiss::IndexFlat(d));

    faiss::ClusteringParameters cp;
    cp.niter = niter;
    cp.verbose = false;

    faiss::Clustering kmeans(d, k, cp);

    faiss::TimeoutCallback::reset(0.010);
    EXPECT_THROW(kmeans.train(n, vecs.data(), *index), faiss::FaissException);
    delete index;
}
