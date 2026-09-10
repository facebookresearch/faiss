/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/random.h>

namespace {

struct OwnedTransformResult {
    const float* data;
    std::unique_ptr<const float[]> owner;

    OwnedTransformResult(const float* input, const float* result)
            : data(result), owner(result == input ? nullptr : result) {}
};

} // namespace

TEST(IndexPreTransformParallel, BlockTailsAndSearchEquivalence) {
    constexpr int d = 32;
    constexpr faiss::idx_t nb = 200;
    constexpr faiss::idx_t k = 10;
    faiss::RandomRotationMatrix rotation(d, d);
    rotation.init(1234);
    faiss::RandomRotationMatrix second_rotation(d, d);
    second_rotation.init(4321);
    faiss::IndexFlatL2 flat(d);
    faiss::IndexPreTransform index(&rotation, &flat);
    index.prepend_transform(&second_rotation);

    std::vector<float> database(size_t(nb) * d);
    faiss::float_rand(database.data(), database.size(), 8765);
    index.add(nb, database.data());

    for (faiss::idx_t nq : {1, 7, 63, 64, 65, 127, 128, 129, 1000}) {
        std::vector<float> queries(size_t(nq) * d);
        faiss::float_rand(queries.data(), queries.size(), 991 + nq);
        OwnedTransformResult serial(
                queries.data(), index.apply_chain(nq, queries.data()));
        OwnedTransformResult parallel(
                queries.data(),
                index.apply_chain_parallel(nq, queries.data(), 4, 64));
        for (size_t i = 0; i < queries.size(); ++i) {
            EXPECT_NEAR(serial.data[i], parallel.data[i], 4e-6);
        }

        std::vector<float> serial_distances(size_t(nq) * k);
        std::vector<float> parallel_distances(size_t(nq) * k);
        std::vector<faiss::idx_t> serial_labels(size_t(nq) * k);
        std::vector<faiss::idx_t> parallel_labels(size_t(nq) * k);
        index.search(
                nq,
                queries.data(),
                k,
                serial_distances.data(),
                serial_labels.data());
        faiss::SearchParametersPreTransform params;
        params.transform_threads = 4;
        params.transform_block_size = 64;
        index.search(
                nq,
                queries.data(),
                k,
                parallel_distances.data(),
                parallel_labels.data(),
                &params);
        EXPECT_EQ(serial_labels, parallel_labels);
        for (size_t i = 0; i < serial_distances.size(); ++i) {
            EXPECT_NEAR(serial_distances[i], parallel_distances[i], 4e-5);
        }
    }

    // Exercise actual parallel work below the reference block size as well.
    std::vector<float> short_queries(size_t(7) * d);
    faiss::float_rand(short_queries.data(), short_queries.size(), 774);
    OwnedTransformResult serial(
            short_queries.data(), index.apply_chain(7, short_queries.data()));
    OwnedTransformResult parallel(
            short_queries.data(),
            index.apply_chain_parallel(7, short_queries.data(), 4, 3));
    for (size_t i = 0; i < short_queries.size(); ++i) {
        EXPECT_NEAR(serial.data[i], parallel.data[i], 4e-6);
    }
}

TEST(IndexPreTransformParallel, ForwardsSubIndexParameters) {
    constexpr int d = 16;
    constexpr faiss::idx_t nb = 100;
    constexpr faiss::idx_t nq = 65;
    constexpr faiss::idx_t k = 10;
    faiss::RandomRotationMatrix rotation(d, d);
    rotation.init(1234);
    faiss::IndexHNSWFlat hnsw(d, 16);
    faiss::IndexPreTransform index(&rotation, &hnsw);
    std::vector<float> database(size_t(nb) * d);
    std::vector<float> queries(size_t(nq) * d);
    faiss::float_rand(database.data(), database.size(), 444);
    faiss::float_rand(queries.data(), queries.size(), 555);
    index.add(nb, database.data());

    faiss::IDSelectorRange selector(20, 80);
    faiss::SearchParametersHNSW inner;
    inner.efSearch = 100;
    inner.sel = &selector;
    faiss::SearchParametersPreTransform outer;
    outer.index_params = &inner;
    outer.transform_threads = 4;
    outer.transform_block_size = 16;
    std::vector<float> distances(size_t(nq) * k);
    std::vector<faiss::idx_t> labels(size_t(nq) * k);
    index.search(
            nq, queries.data(), k, distances.data(), labels.data(), &outer);
    for (faiss::idx_t label : labels) {
        EXPECT_TRUE(label == -1 || (label >= 20 && label < 80));
    }
}

TEST(IndexPreTransformParallel, ValidatesOptionsAndPropagatesExceptions) {
    constexpr int d = 8;
    faiss::LinearTransform malformed(d, d, false);
    malformed.is_trained = true;
    faiss::IndexFlatL2 flat(d);
    faiss::IndexPreTransform index(&malformed, &flat);
    std::vector<float> queries(size_t(65) * d);

    EXPECT_THROW(
            index.apply_chain_parallel(65, queries.data(), 0, 64),
            faiss::FaissException);
    EXPECT_THROW(
            index.apply_chain_parallel(65, queries.data(), 4, 0),
            faiss::FaissException);
    // The malformed matrix throws inside an OpenMP worker. It must be
    // rethrown to the caller rather than terminating the process.
    EXPECT_THROW(
            index.apply_chain_parallel(65, queries.data(), 4, 64),
            faiss::FaissException);
}
