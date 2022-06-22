// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
        const std::shared_ptr<faiss::Index>& index,
        float** pqCoarseCentroidsQ,
        float** pqFineCentroidsQ) {
    if (pqFineCentroidsQ == nullptr || pqCoarseCentroidsQ == nullptr) {
        return false;
    }

    faiss::IndexIVFPQ* const indexQ =
            dynamic_cast<faiss::IndexIVFPQ*>(index.get());
    if (indexQ == nullptr) {
        return false;
    }

    auto const coarseIndexQ =
            dynamic_cast<faiss::IndexFlatCodes*>(indexQ->quantizer);
    if (coarseIndexQ == nullptr) {
        return false;
    }

    *pqFineCentroidsQ = indexQ->pq.centroids.data();
    *pqCoarseCentroidsQ = reinterpret_cast<float*>(coarseIndexQ->codes.data());
    return true;
}

bool testIfResidualPQ(
        const std::shared_ptr<faiss::Index>& index,
        float** pqCoarseCentroidsQ,
        float** pqFineCentroidsQ) {
    if (pqFineCentroidsQ == nullptr || pqCoarseCentroidsQ == nullptr) {
        return false;
    }

    faiss::Index2Layer* const indexQ =
            dynamic_cast<faiss::Index2Layer*>(index.get());
    if (indexQ == nullptr) {
        return false;
    }

    auto const coarseIndexQ =
            dynamic_cast<faiss::MultiIndexQuantizer*>(indexQ->q1.quantizer);
    if (coarseIndexQ == nullptr) {
        return false;
    }

    *pqFineCentroidsQ = indexQ->pq.centroids.data();
    *pqCoarseCentroidsQ = coarseIndexQ->pq.centroids.data();
    return true;
}

template <typename T>
void verify(
        const uint64_t n,
        const uint64_t d,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData) {
    //
    float* pqFineCentroidsQ = nullptr;
    float* pqCoarseCentroidsQ = nullptr;

    //
    testIfIVFPQ(index, &pqCoarseCentroidsQ, &pqFineCentroidsQ);
    testIfResidualPQ(index, &pqCoarseCentroidsQ, &pqFineCentroidsQ);

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

    // test contrib::accum, 2 samples per iteration
    rng.seed(123);

    std::vector<float> outputContrib2s(d, 0);
    for (size_t i = 0; i < n; i += 2) {
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
    }

    // verify
    for (size_t j = 0; j < d; j++) {
        ASSERT_NEAR(outputFaiss[j], outputContrib2s[j], 1e-2);
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
void test(const uint64_t n, const uint64_t d, const std::string& description) {
    auto data = generate(n, d);
    auto [index, encodedData] = trainDataset(data, n, d, description);

    verify<T>(n, d, index, encodedData);
}

constexpr size_t NSAMPLES = 4096;

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D256_IVF256_PQ64) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 4>;
    test<T>(NSAMPLES, 256, "IVF256,PQ64np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D256_IVF256_PQ32) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 8>;
    test<T>(NSAMPLES, 256, "IVF256,PQ32np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D256_IVF256_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 16>;
    test<T>(NSAMPLES, 256, "IVF256,PQ16np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D256_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 256, 32>;
    test<T>(NSAMPLES, 256, "IVF256,PQ8np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D192_IVF256_PQ48) {
    using T = faiss::cppcontrib::Index2LevelDecoder<192, 192, 4>;
    test<T>(NSAMPLES, 192, "IVF256,PQ48np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D192_IVF256_PQ24) {
    using T = faiss::cppcontrib::Index2LevelDecoder<192, 192, 8>;
    test<T>(NSAMPLES, 192, "IVF256,PQ24np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D192_IVF256_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<192, 192, 12>;
    test<T>(NSAMPLES, 192, "IVF256,PQ16np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D192_IVF256_PQ12) {
    using T = faiss::cppcontrib::Index2LevelDecoder<192, 192, 16>;
    test<T>(NSAMPLES, 192, "IVF256,PQ12np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D160_IVF256_PQ40) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 4>;
    test<T>(NSAMPLES, 160, "IVF256,PQ40np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D160_IVF256_PQ20) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 8>;
    test<T>(NSAMPLES, 160, "IVF256,PQ20np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D160_IVF256_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 16>;
    test<T>(NSAMPLES, 160, "IVF256,PQ10np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D160_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 20>;
    test<T>(NSAMPLES, 160, "IVF256,PQ8np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D128_IVF256_PQ32) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 4>;
    test<T>(NSAMPLES, 128, "IVF256,PQ32np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D128_IVF256_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 8>;
    test<T>(NSAMPLES, 128, "IVF256,PQ16np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D128_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 16>;
    test<T>(NSAMPLES, 128, "IVF256,PQ8np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D128_IVF256_PQ4) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 32>;
    test<T>(NSAMPLES, 128, "IVF256,PQ4np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D64_IVF256_PQ32) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 64, 4>;
    test<T>(NSAMPLES, 64, "IVF256,PQ16np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D64_IVF256_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 64, 8>;
    test<T>(NSAMPLES, 64, "IVF256,PQ8np");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D64_IVF256_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 64, 16>;
    test<T>(NSAMPLES, 64, "IVF256,PQ4np");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D256_Residual4x8_PQ64) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 4>;
    test<T>(NSAMPLES, 256, "Residual4x8,PQ64");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D256_Residual4x8_PQ32) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 8>;
    test<T>(NSAMPLES, 256, "Residual4x8,PQ32");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D256_Residual4x8_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 16>;
    test<T>(NSAMPLES, 256, "Residual4x8,PQ16");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D256_Residual4x8_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<256, 64, 32>;
    test<T>(NSAMPLES, 256, "Residual4x8,PQ8");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D160_Residual4x8_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 40, 16>;
    test<T>(NSAMPLES, 160, "Residual4x8,PQ10");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D160_Residual2x8_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 80, 16>;
    test<T>(NSAMPLES, 160, "Residual2x8,PQ10");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D160_Residual1x8_PQ10) {
    using T = faiss::cppcontrib::Index2LevelDecoder<160, 160, 16>;
    test<T>(NSAMPLES, 160, "Residual1x8,PQ10");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D128_Residual4x8_PQ32) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 4>;
    test<T>(NSAMPLES, 128, "Residual4x8,PQ32");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D128_Residual4x8_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 8>;
    test<T>(NSAMPLES, 128, "Residual4x8,PQ16");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D128_Residual4x8_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 16>;
    test<T>(NSAMPLES, 128, "Residual4x8,PQ8");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D128_Residual4x8_PQ4) {
    using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 32>;
    test<T>(NSAMPLES, 128, "Residual4x8,PQ4");
}

//
TEST(TEST_CPPCONTRIB_SA_DECODE, D64_Residual4x8_PQ16) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 16, 4>;
    test<T>(NSAMPLES, 64, "Residual4x8,PQ16");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D64_Residual4x8_PQ8) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 16, 8>;
    test<T>(NSAMPLES, 64, "Residual4x8,PQ8");
}

TEST(TEST_CPPCONTRIB_SA_DECODE, D64_Residual4x8_PQ4) {
    using T = faiss::cppcontrib::Index2LevelDecoder<64, 16, 16>;
    test<T>(NSAMPLES, 64, "Residual4x8,PQ4");
}
