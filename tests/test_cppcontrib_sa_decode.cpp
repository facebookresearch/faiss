/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <faiss/Index.h>
#include <faiss/Index2Layer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/io.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include <faiss/IndexRowwiseMinMax.h>
#include <faiss/cppcontrib/SaDecodeKernels.h>

using namespace ::testing;
using ::testing::TestWithParam;
using ::testing::Values;

std::tuple<std::shared_ptr<faiss::Index>, std::vector<uint8_t>> trainDataset(
        const std::vector<float>& input,
        const uint64_t n,
        const uint64_t d,
        const std::string& description) {
    // train an index
    auto index = std::shared_ptr<faiss::Index>(
            faiss::index_factory((int)d, description.c_str()));
    index->train((int)n, input.data());

    // encode
    const size_t codeSize = index->sa_code_size();

    std::vector<uint8_t> encodedData(n * codeSize);
    index->sa_encode(n, input.data(), encodedData.data());

    return std::make_tuple(std::move(index), std::move(encodedData));
}

bool testIfIVFPQ(
        const faiss::Index* const index,
        const float** pqCoarseCentroidsQ,
        const float** pqFineCentroidsQ) {
    if (pqFineCentroidsQ == nullptr || pqCoarseCentroidsQ == nullptr) {
        return false;
    }

    const faiss::IndexIVFPQ* const indexQ =
            dynamic_cast<const faiss::IndexIVFPQ*>(index);
    if (indexQ == nullptr) {
        return false;
    }

    const auto coarseIndexQ =
            dynamic_cast<const faiss::IndexFlatCodes*>(indexQ->quantizer);
    if (coarseIndexQ == nullptr) {
        return false;
    }

    *pqFineCentroidsQ = indexQ->pq.centroids.data();
    *pqCoarseCentroidsQ =
            reinterpret_cast<const float*>(coarseIndexQ->codes.data());
    return true;
}

bool testIfResidualPQ(
        const faiss::Index* const index,
        const float** pqCoarseCentroidsQ,
        const float** pqFineCentroidsQ) {
    if (pqFineCentroidsQ == nullptr || pqCoarseCentroidsQ == nullptr) {
        return false;
    }

    const faiss::Index2Layer* const indexQ =
            dynamic_cast<const faiss::Index2Layer*>(index);
    if (indexQ == nullptr) {
        return false;
    }

    const auto coarseIndexQ = dynamic_cast<const faiss::MultiIndexQuantizer*>(
            indexQ->q1.quantizer);
    if (coarseIndexQ == nullptr) {
        return false;
    }

    *pqFineCentroidsQ = indexQ->pq.centroids.data();
    *pqCoarseCentroidsQ = coarseIndexQ->pq.centroids.data();
    return true;
}

template <typename T>
void verifyIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData) {
    //
    const float* pqFineCentroidsQ = nullptr;
    const float* pqCoarseCentroidsQ = nullptr;

    //
    testIfIVFPQ(index.get(), &pqCoarseCentroidsQ, &pqFineCentroidsQ);
    testIfResidualPQ(index.get(), &pqCoarseCentroidsQ, &pqFineCentroidsQ);

    //
    const size_t codeSize = index->sa_code_size();

    //
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // test general purpose version vs contrib::store
    std::vector<float> outputFaiss(d, 0);
    std::vector<float> tmpFaiss(d, 0);
    std::vector<float> tmpContrib(d, 0);
    for (size_t i = 0; i < n; i++) {
        // compute using faiss
        index->sa_decode(1, encodedData.data() + i * codeSize, tmpFaiss.data());

        // compute using contrib
        T::store(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + i * codeSize,
                tmpContrib.data());

        // compare
        for (size_t j = 0; j < d; j++)
            ASSERT_FLOAT_EQ(tmpFaiss[j], tmpContrib[j]);

        // save for the further comparison
        const float weight = u(rng);
        for (size_t j = 0; j < d; j++)
            outputFaiss[j] += weight * tmpFaiss[j];
    }

    // test contrib::accum, 1 sample per iteration
    rng.seed(123);

    std::vector<float> outputContrib1s(d, 0);
    for (size_t i = 0; i < n; i++) {
        const float weight0 = u(rng);

        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data());
    }

    // verify
    for (size_t j = 0; j < d; j++) {
        ASSERT_FLOAT_EQ(outputFaiss[j], outputContrib1s[j]);
    }

    // test contrib::accum, 2 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib2s(d, 0);
    std::vector<float> outputContrib2sSame(d, 0);
    for (size_t i = 0; i < n; i += 2) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib2s[j] = (j + 1) * (j + 1);
            outputContrib2sSame[j] = (j + 1) * (j + 1);
        }

        // do a single step, 2 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);

        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2s.data());

        // do a single step, 2 samples per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2sSame.data());

        // do two steps, 1 sample per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data());
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data());

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib2s[j]);
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib2sSame[j]);
        }
    }

    // test contrib::accum, 3 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib3s(d, 0);
    std::vector<float> outputContrib3sSame(d, 0);
    const size_t n3 = (n / 3) * 3;
    for (size_t i = 0; i < n3; i += 3) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib3s[j] = (j + 1) * (j + 1);
            outputContrib3sSame[j] = (j + 1) * (j + 1);
        }

        // do a single step, 3 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);
        const float weight2 = u(rng);

        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3s.data());

        // do a single step, 3 samples per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3sSame.data());

        // do three steps, 1 sample per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data());
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data());
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib1s.data());

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib3s[j]);
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib3sSame[j]);
        }
    }
}

template <typename T>
void verifyMinMaxIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData) {
    //
    const float* pqFineCentroidsQ = nullptr;
    const float* pqCoarseCentroidsQ = nullptr;

    // extract an index that is wrapped with IndexRowwiseMinMaxBase
    const std::shared_ptr<faiss::IndexRowwiseMinMaxBase> indexMinMax =
            std::dynamic_pointer_cast<faiss::IndexRowwiseMinMaxBase>(index);
    ASSERT_NE(indexMinMax.get(), nullptr);

    auto subIndex = indexMinMax->index;

    //
    testIfIVFPQ(subIndex, &pqCoarseCentroidsQ, &pqFineCentroidsQ);
    testIfResidualPQ(subIndex, &pqCoarseCentroidsQ, &pqFineCentroidsQ);

    //
    const size_t codeSize = index->sa_code_size();

    //
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // test general purpose version vs contrib::store
    std::vector<float> outputFaiss(d, 0);
    std::vector<float> tmpFaiss(d, 0);
    std::vector<float> tmpContrib(d, 0);
    for (size_t i = 0; i < n; i++) {
        // compute using faiss
        index->sa_decode(1, encodedData.data() + i * codeSize, tmpFaiss.data());

        // compute using contrib
        T::store(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + i * codeSize,
                tmpContrib.data());

        // compare
        for (size_t j = 0; j < d; j++)
            ASSERT_FLOAT_EQ(tmpFaiss[j], tmpContrib[j]);

        // save for the further comparison
        const float weight = u(rng);
        for (size_t j = 0; j < d; j++)
            outputFaiss[j] += weight * tmpFaiss[j];
    }

    // test contrib::accum, 1 sample per iteration.
    // This needs a way of handling that is different from just IVFPQ and PQ
    // because of the scaling, but rather similar to how 2 samples per iteration
    // is processed.
    rng.seed(123);

    std::vector<float> outputContrib1s(d, 0);
    float outputMinv1s = 0;
    for (size_t i = 0; i < n; i++) {
        // compute using faiss
        index->sa_decode(1, encodedData.data() + i * codeSize, tmpFaiss.data());

        // populate some initial data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
        }
        outputMinv1s = 0;

        // generate a weight
        const float weight0 = u(rng);

        //
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data(),
                outputMinv1s);

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    tmpFaiss[j] * weight0 + (j + 1) * (j + 1));
        }
    }

    // test contrib::accum, 2 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib2s(d, 0);
    std::vector<float> outputContrib2sSame(d, 0);
    float outputMinv2s = 0;
    float outputMinv2sSame = 0;
    for (size_t i = 0; i < n; i += 2) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib2s[j] = (j + 1) * (j + 1);
            outputContrib2sSame[j] = (j + 1) * (j + 1);
        }
        outputMinv1s = 0;
        outputMinv2s = 0;
        outputMinv2sSame = 0;

        // do a single step, 2 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);

        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2s.data(),
                outputMinv2s);

        // do a single step, 2 samples per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2sSame.data(),
                outputMinv2sSame);

        // do two steps, 1 sample per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data(),
                outputMinv1s);
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data(),
                outputMinv1s);

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib2s[j] + outputMinv2s);
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib2sSame[j] + outputMinv2sSame);
        }
    }

    // test contrib::accum, 3 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib3s(d, 0);
    float outputMinv3s = 0;
    std::vector<float> outputContrib3sSame(d, 0);
    float outputMinv3sSame = 0;
    const size_t n3 = (n / 3) * 3;
    for (size_t i = 0; i < n3; i += 3) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib3s[j] = (j + 1) * (j + 1);
            outputContrib3sSame[j] = (j + 1) * (j + 1);
        }
        outputMinv1s = 0;
        outputMinv3s = 0;
        outputMinv3sSame = 0;

        // do a single step, 3 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);
        const float weight2 = u(rng);

        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3s.data(),
                outputMinv3s);

        // do a single step, 3 samples per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3sSame.data(),
                outputMinv3sSame);

        // do three steps, 1 sample per step
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data(),
                outputMinv1s);
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data(),
                outputMinv1s);
        T::accum(
                pqCoarseCentroidsQ,
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib1s.data(),
                outputMinv1s);

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib3s[j] + outputMinv3s);
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib3sSame[j] + outputMinv3sSame);
        }
    }
}

template <typename T>
void verifyIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData) {
    //
    const faiss::IndexPQ* const indexQ =
            dynamic_cast<const faiss::IndexPQ*>(index.get());
    const float* const pqFineCentroidsQ = indexQ->pq.centroids.data();

    //
    const size_t codeSize = index->sa_code_size();

    //
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // test general purpose version vs contrib::store
    std::vector<float> outputFaiss(d, 0);
    std::vector<float> tmpFaiss(d, 0);
    std::vector<float> tmpContrib(d, 0);
    for (size_t i = 0; i < n; i++) {
        // compute using faiss
        index->sa_decode(1, encodedData.data() + i * codeSize, tmpFaiss.data());

        // compute using contrib
        T::store(
                pqFineCentroidsQ,
                encodedData.data() + i * codeSize,
                tmpContrib.data());

        // compare
        for (size_t j = 0; j < d; j++)
            ASSERT_FLOAT_EQ(tmpFaiss[j], tmpContrib[j]);

        // save for the further comparison
        const float weight = u(rng);
        for (size_t j = 0; j < d; j++)
            outputFaiss[j] += weight * tmpFaiss[j];
    }

    // test contrib::accum, 1 sample per iteration
    rng.seed(123);

    std::vector<float> outputContrib1s(d, 0);
    for (size_t i = 0; i < n; i++) {
        const float weight0 = u(rng);

        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data());
    }

    // verify
    for (size_t j = 0; j < d; j++) {
        ASSERT_FLOAT_EQ(outputFaiss[j], outputContrib1s[j]);
    }

    // test contrib::accum, 2 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib2s(d, 0);
    std::vector<float> outputContrib2sSame(d, 0);
    for (size_t i = 0; i < n; i += 2) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib2s[j] = (j + 1) * (j + 1);
            outputContrib2sSame[j] = (j + 1) * (j + 1);
        }

        // do a single step, 2 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);

        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2s.data());

        // do a single step, 2 samples per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2sSame.data());

        // do two steps, 1 sample per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data());
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data());

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib2s[j]);
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib2sSame[j]);
        }
    }

    // test contrib::accum, 3 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib3s(d, 0);
    std::vector<float> outputContrib3sSame(d, 0);
    const size_t n3 = (n / 3) * 3;
    for (size_t i = 0; i < n3; i += 3) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib3s[j] = (j + 1) * (j + 1);
            outputContrib3sSame[j] = (j + 1) * (j + 1);
        }

        // do a single step, 3 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);
        const float weight2 = u(rng);

        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3s.data());

        // do a single step, 3 samples per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3sSame.data());

        // do three steps, 1 sample per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data());
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data());
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib1s.data());

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib3s[j]);
            ASSERT_FLOAT_EQ(outputContrib1s[j], outputContrib3sSame[j]);
        }
    }
}

template <typename T>
void verifyMinMaxIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData) {
    // extract an index that is wrapped with IndexRowwiseMinMaxBase
    const std::shared_ptr<faiss::IndexRowwiseMinMaxBase> indexMinMax =
            std::dynamic_pointer_cast<faiss::IndexRowwiseMinMaxBase>(index);
    ASSERT_NE(indexMinMax.get(), nullptr);

    auto subIndex = indexMinMax->index;

    //
    const faiss::IndexPQ* const indexQ =
            dynamic_cast<const faiss::IndexPQ*>(subIndex);
    const float* const pqFineCentroidsQ = indexQ->pq.centroids.data();

    //
    const size_t codeSize = index->sa_code_size();

    //
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // test general purpose version vs contrib::store
    std::vector<float> outputFaiss(d, 0);
    std::vector<float> tmpFaiss(d, 0);
    std::vector<float> tmpContrib(d, 0);
    for (size_t i = 0; i < n; i++) {
        // compute using faiss
        index->sa_decode(1, encodedData.data() + i * codeSize, tmpFaiss.data());

        // compute using contrib
        T::store(
                pqFineCentroidsQ,
                encodedData.data() + i * codeSize,
                tmpContrib.data());

        // compare
        for (size_t j = 0; j < d; j++)
            ASSERT_FLOAT_EQ(tmpFaiss[j], tmpContrib[j]);

        // save for the further comparison
        const float weight = u(rng);
        for (size_t j = 0; j < d; j++)
            outputFaiss[j] += weight * tmpFaiss[j];
    }

    // test contrib::accum, 1 sample per iteration.
    // This needs a way of handling that is different from just IVFPQ and PQ
    // because of the scaling, but rather similar to how 2 samples per iteration
    // is processed.
    rng.seed(123);

    std::vector<float> outputContrib1s(d, 0);
    float outputMinv1s = 0;
    for (size_t i = 0; i < n; i++) {
        // compute using faiss
        index->sa_decode(1, encodedData.data() + i * codeSize, tmpFaiss.data());

        // populate some initial data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
        }
        outputMinv1s = 0;

        // generate a weight
        const float weight0 = u(rng);

        //
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data(),
                outputMinv1s);

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    tmpFaiss[j] * weight0 + (j + 1) * (j + 1));
        }
    }

    // test contrib::accum, 2 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib2s(d, 0);
    float outputMinv2s = 0;
    std::vector<float> outputContrib2sSame(d, 0);
    float outputMinv2sSame = 0;
    for (size_t i = 0; i < n; i += 2) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib2s[j] = (j + 1) * (j + 1);
            outputContrib2sSame[j] = (j + 1) * (j + 1);
        }
        outputMinv1s = 0;
        outputMinv2s = 0;
        outputMinv2sSame = 0;

        // do a single step, 2 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);

        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2s.data(),
                outputMinv2s);

        // do a single step, 2 samples per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib2sSame.data(),
                outputMinv2sSame);

        // do two steps, 1 sample per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data(),
                outputMinv1s);
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data(),
                outputMinv1s);

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib2s[j] + outputMinv2s);
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib2sSame[j] + outputMinv2sSame);
        }
    }

    // test contrib::accum, 3 samples per iteration.
    rng.seed(123);

    std::vector<float> outputContrib3s(d, 0);
    float outputMinv3s = 0;
    std::vector<float> outputContrib3sSame(d, 0);
    float outputMinv3sSame = 0;
    const size_t n3 = (n / 3) * 3;
    for (size_t i = 0; i < n3; i += 3) {
        // populate outputContribs with some existing data
        for (size_t j = 0; j < d; j++) {
            outputContrib1s[j] = (j + 1) * (j + 1);
            outputContrib3s[j] = (j + 1) * (j + 1);
            outputContrib3sSame[j] = (j + 1) * (j + 1);
        }
        outputMinv1s = 0;
        outputMinv3s = 0;
        outputMinv3sSame = 0;

        // do a single step, 3 samples per step
        const float weight0 = u(rng);
        const float weight1 = u(rng);
        const float weight2 = u(rng);

        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3s.data(),
                outputMinv3s);

        // do a single step, 3 samples per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib3sSame.data(),
                outputMinv3sSame);

        // do three steps, 1 sample per step
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 0) * codeSize,
                weight0,
                outputContrib1s.data(),
                outputMinv1s);
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 1) * codeSize,
                weight1,
                outputContrib1s.data(),
                outputMinv1s);
        T::accum(
                pqFineCentroidsQ,
                encodedData.data() + (i + 2) * codeSize,
                weight2,
                outputContrib1s.data(),
                outputMinv1s);

        // compare
        for (size_t j = 0; j < d; j++) {
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib3s[j] + outputMinv3s);
            ASSERT_FLOAT_EQ(
                    outputContrib1s[j] + outputMinv1s,
                    outputContrib3sSame[j] + outputMinv3sSame);
        }
    }
}

std::vector<float> generate(const size_t n, const size_t d) {
    std::vector<float> data(n * d);

    std::minstd_rand rng(345);
    std::uniform_real_distribution<float> ux(0, 1);

    //
    for (size_t k = 0; k < n; k++) {
        for (size_t j = 0; j < d; j++) {
            data[k * d + j] = ux(rng);
        }
    }

    return data;
}

template <typename T>
void testIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyIndex2LevelDecoder<T>(n, d, index, encodedData);
}

template <typename T>
void testMinMaxIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyMinMaxIndex2LevelDecoder<T>(n, d, index, encodedData);
}

template <typename T>
void testIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyIndexPQDecoder<T>(n, d, index, encodedData);
}

template <typename T>
void testMinMaxIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyMinMaxIndexPQDecoder<T>(n, d, index, encodedData);
}

constexpr size_t NSAMPLES = 256;

//
TEST(testCppcontribSaDecode, D256_IVF256_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 256, "IVF256,PQ16np");
}

TEST(testCppcontribSaDecode, D256_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 32>;
    testIndex2LevelDecoder<T>(NSAMPLES, 256, "IVF256,PQ8np");
}

//
TEST(testCppcontribSaDecode, D192_IVF256_PQ24) {
    using T = faiss::cppcontrib::Index2LevelDecoder<192, 192, 8>;
    testIndex2LevelDecoder<T>(NSAMPLES, 192, "IVF256,PQ24np");
}

//
TEST(testCppcontribSaDecode, D192_IVF256_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<192, 192, 12>;
    testIndex2LevelDecoder<T>(NSAMPLES, 192, "IVF256,PQ16np");
}

//
TEST(testCppcontribSaDecode, D192_IVF256_PQ12) {
    using T = faiss::cppcontrib::Index2LevelDecoder<192, 192, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 192, "IVF256,PQ12np");
}

//
TEST(testCppcontribSaDecode, D160_IVF256_PQ40) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 4>;
    testIndex2LevelDecoder<T>(NSAMPLES, 160, "IVF256,PQ40np");
}

//
TEST(testCppcontribSaDecode, D160_IVF256_PQ20) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 8>;
    testIndex2LevelDecoder<T>(NSAMPLES, 160, "IVF256,PQ20np");
}

//
TEST(testCppcontribSaDecode, D160_IVF256_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 160, "IVF256,PQ10np");
}

//
TEST(testCppcontribSaDecode, D160_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 20>;
    testIndex2LevelDecoder<T>(NSAMPLES, 160, "IVF256,PQ8np");
}

//
TEST(testCppcontribSaDecode, D128_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 128, "IVF256,PQ8np");
}

TEST(testCppcontribSaDecode, D128_IVF256_PQ4) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 32>;
    testIndex2LevelDecoder<T>(NSAMPLES, 128, "IVF256,PQ4np");
}

//
TEST(testCppcontribSaDecode, D64_IVF256_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 64, 8>;
    testIndex2LevelDecoder<T>(NSAMPLES, 64, "IVF256,PQ8np");
}

TEST(testCppcontribSaDecode, D64_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 64, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 64, "IVF256,PQ4np");
}

#if defined(__AVX2__)
TEST(testCppcontribSaDecode, D40_IVF256_PQ20) {
    using T = faiss::cppcontrib::Index2LevelDecoder<40, 40, 2>;
    testIndex2LevelDecoder<T>(NSAMPLES, 40, "IVF256,PQ20np");
}
#endif

//
TEST(testCppcontribSaDecode, D256_Residual4x8_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 256, "Residual4x8,PQ16");
}

TEST(testCppcontribSaDecode, D256_Residual4x8_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 32>;
    testIndex2LevelDecoder<T>(NSAMPLES, 256, "Residual4x8,PQ8");
}

//
TEST(testCppcontribSaDecode, D160_Residual4x8_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 40, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 160, "Residual4x8,PQ10");
}

//
TEST(testCppcontribSaDecode, D160_Residual2x8_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 80, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 160, "Residual2x8,PQ10");
}

//
TEST(testCppcontribSaDecode, D160_Residual1x8_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 160, "Residual1x8,PQ10");
}

//
TEST(testCppcontribSaDecode, D128_Residual4x8_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 128, "Residual4x8,PQ8");
}

TEST(testCppcontribSaDecode, D128_Residual4x8_PQ4) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 32>;
    testIndex2LevelDecoder<T>(NSAMPLES, 128, "Residual4x8,PQ4");
}

//
TEST(testCppcontribSaDecode, D64_Residual4x8_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 16, 8>;
    testIndex2LevelDecoder<T>(NSAMPLES, 64, "Residual4x8,PQ8");
}

TEST(testCppcontribSaDecode, D64_Residual4x8_PQ4) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 16, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES, 64, "Residual4x8,PQ4");
}

//
TEST(testCppcontribSaDecode, D256_IVF1024_PQ16) {
    // It is acceptable to use COARSE_BITS=16 in this case,
    // because there's only one coarse quantizer element.
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES * 4, 256, "IVF1024,PQ16np");
}

TEST(testCppcontribSaDecode, D64_Residual1x9_PQ8) {
    // It is acceptable to use COARSE_BITS=16 in this case,
    // because there's only one coarse quantizer element.
    // It won't work for "Residual2x9,PQ8".
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 64, 8, 16>;
    testIndex2LevelDecoder<T>(NSAMPLES * 2, 64, "Residual1x9,PQ8");
}

//
TEST(testCppcontribSaDecode, D256_PQ16) {
    using T = faiss::cppcontrib::IndexPQDecoder<256, 16>;
    testIndexPQDecoder<T>(NSAMPLES, 256, "PQ16np");
}

//
TEST(testCppcontribSaDecode, D160_PQ20) {
    using T = faiss::cppcontrib::IndexPQDecoder<160, 8>;
    testIndexPQDecoder<T>(NSAMPLES, 160, "PQ20np");
}

#if defined(__AVX2__)
TEST(testCppcontribSaDecode, D40_PQ20) {
    using T = faiss::cppcontrib::IndexPQDecoder<40, 2>;
    testIndexPQDecoder<T>(NSAMPLES, 40, "PQ20np");
}
#endif

// test IndexRowwiseMinMaxFP16
TEST(testCppcontribSaDecode, D256_MINMAXFP16_IVF256_PQ16) {
    using SubT = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16>;
    using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
    testMinMaxIndex2LevelDecoder<T>(NSAMPLES, 256, "MinMaxFP16,IVF256,PQ16np");
}

TEST(testCppcontribSaDecode, D256_MINMAXFP16_PQ16) {
    using SubT = faiss::cppcontrib::IndexPQDecoder<256, 16>;
    using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
    testMinMaxIndexPQDecoder<T>(NSAMPLES, 256, "MinMaxFP16,PQ16np");
}

// test IndexRowwiseMinMax
TEST(testCppcontribSaDecode, D256_MINMAX_IVF256_PQ16) {
    using SubT = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16>;
    using T = faiss::cppcontrib::IndexMinMaxDecoder<SubT>;
    testMinMaxIndex2LevelDecoder<T>(NSAMPLES, 256, "MinMax,IVF256,PQ16np");
}

TEST(testCppcontribSaDecode, D256_MINMAX_PQ16) {
    using SubT = faiss::cppcontrib::IndexPQDecoder<256, 16>;
    using T = faiss::cppcontrib::IndexMinMaxDecoder<SubT>;
    testMinMaxIndexPQDecoder<T>(NSAMPLES, 256, "MinMax,PQ16np");
}

// implemented for AVX2 and ARM so far
#if defined(__AVX2__) || defined(__ARM_NEON)
TEST(testCppcontribSaDecode, D256_PQ16x10) {
    using T = faiss::cppcontrib::IndexPQDecoder<256, 16, 10>;
    testIndexPQDecoder<T>(NSAMPLES * 4, 256, "PQ16x10np");
}

TEST(testCppcontribSaDecode, D256_PQ16x12) {
    using T = faiss::cppcontrib::IndexPQDecoder<256, 16, 12>;
    testIndexPQDecoder<T>(NSAMPLES * 16, 256, "PQ16x12np");
}

TEST(testCppcontribSaDecode, D160_PQ20x10) {
    using T = faiss::cppcontrib::IndexPQDecoder<160, 8, 10>;
    testIndexPQDecoder<T>(NSAMPLES * 4, 160, "PQ20x10np");
}

TEST(testCppcontribSaDecode, D160_PQ20x12) {
    using T = faiss::cppcontrib::IndexPQDecoder<160, 8, 12>;
    testIndexPQDecoder<T>(NSAMPLES * 16, 160, "PQ20x12np");
}

TEST(testCppcontribSaDecode, D256_IVF256_PQ16x10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16, 8, 10>;
    testIndex2LevelDecoder<T>(NSAMPLES * 4, 256, "IVF256,PQ16x10np");
}

TEST(testCppcontribSaDecode, D256_IVF256_PQ16x12) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16, 8, 12>;
    testIndex2LevelDecoder<T>(NSAMPLES * 16, 256, "IVF256,PQ16x12np");
}

TEST(testCppcontribSaDecode, D256_MINMAXFP16_IVF256_PQ16x10) {
    using SubT = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16, 8, 10>;
    using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
    testMinMaxIndex2LevelDecoder<T>(
            NSAMPLES * 4, 256, "MinMaxFP16,IVF256,PQ16x10np");
}

TEST(testCppcontribSaDecode, D256_MINMAXFP16_IVF1024_PQ16x10) {
    using SubT = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16, 10, 10>;
    using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
    testMinMaxIndex2LevelDecoder<T>(
            NSAMPLES * 4, 256, "MinMaxFP16,IVF1024,PQ16x10np");
}

TEST(testCppcontribSaDecode, D256_MINMAXFP16_IVF1024_PQ16x10_ALTERNATIVE) {
    using SubT = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16, 16, 10>;
    using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
    testMinMaxIndex2LevelDecoder<T>(
            NSAMPLES * 4, 256, "MinMaxFP16,IVF1024,PQ16x10np");
}

TEST(testCppcontribSaDecode, D160_Residual4x8_PQ8x10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 40, 20, 8, 10>;
    testIndex2LevelDecoder<T>(NSAMPLES * 4, 160, "Residual4x8,PQ8x10");
}

TEST(testCppcontribSaDecode, D256_Residual1x9_PQ16x10) {
    // It is acceptable to use COARSE_BITS=16 in this case,
    // because there's only one coarse quantizer element.
    // It won't work for "Residual2x9,PQ16x10".
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16, 16, 10>;
    testIndex2LevelDecoder<T>(NSAMPLES * 4, 256, "Residual1x9,PQ16x10");
}

TEST(testCppcontribSaDecode, D256_Residual4x10_PQ16x10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 16, 10, 10>;
    testIndex2LevelDecoder<T>(NSAMPLES * 4, 256, "Residual4x10,PQ16x10");
}

TEST(testCppcontribSaDecode, D256_Residual4x12_PQ16x12) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 16, 12, 12>;
    testIndex2LevelDecoder<T>(NSAMPLES * 16, 256, "Residual4x12,PQ16x12");
}

#endif
