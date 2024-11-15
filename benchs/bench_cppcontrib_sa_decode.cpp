/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include <faiss/Index.h>
#include <faiss/Index2Layer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_factory.h>

#include <faiss/IndexRowwiseMinMax.h>
#include <faiss/cppcontrib/SaDecodeKernels.h>

// train a dataset
std::tuple<std::shared_ptr<faiss::Index>, std::vector<uint8_t>> trainDataset(
        const std::vector<float>& input,
        const uint64_t n,
        const uint64_t d,
        const std::string& description) {
    //
    omp_set_num_threads(std::thread::hardware_concurrency());

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

// generate a dataset
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

double getError(
        const uint64_t n,
        const uint64_t d,
        const std::vector<float>& v1,
        const std::vector<float>& v2) {
    double error = 0;
    for (uint64_t i = 0; i < n; i++) {
        double localError = 0;
        for (uint64_t j = 0; j < d; j++) {
            double q = v1[i * d + j] - v2[i * d + j];
            localError += q * q;
        }

        error += localError;
    }

    return error;
}

// a timer
struct StopWatch {
    using timepoint_t = std::chrono::time_point<std::chrono::steady_clock>;

    timepoint_t Start;

    //
    StopWatch() {
        Start = std::chrono::steady_clock::now();
    }

    //
    double elapsed() const {
        const auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - Start;
        return elapsed.count();
    }
};

//
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

//
template <typename T>
static void verifyIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData,
        const uint64_t nIterations) {
    //
    const float* pqFineCentroidsQ = nullptr;
    const float* pqCoarseCentroidsQ = nullptr;

    //
    testIfIVFPQ(index.get(), &pqCoarseCentroidsQ, &pqFineCentroidsQ);
    testIfResidualPQ(index.get(), &pqCoarseCentroidsQ, &pqFineCentroidsQ);

    //
    const size_t codeSize = index->sa_code_size();

    // initialize the random engine
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // use 1 thread
    omp_set_num_threads(1);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // sequential order
    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            index->sa_decode(n, encodedData.data(), outputFaiss.data());
        }
        double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (int iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                T::store(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + i * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        double timeKernel = swKernel.elapsed();

        // evaluate the error
        double error = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\tstore_seq\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random order

    // generate a random order of points
    std::uniform_int_distribution<uint64_t> un(0, n - 1);
    std::vector<uint64_t> pointIncidesToDecode(nIterations * n, 0);
    for (uint64_t i = 0; i < nIterations * n; i++) {
        pointIncidesToDecode[i] = un(rng);
    }

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        outputFaiss.data() + i * d);
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                T::store(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        const double timeKernel = swKernel.elapsed();

        // evaluate the error
        const double error = getError(n, d, outputFaiss, outputKernel1);
        std::cout << description << "\t" << n << "\t" << d << "\tstore_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random accumulate

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);
        std::vector<float> outputKernel2(n * d, 0);
        std::vector<float> outputKernel2u(n * d, 0);
        std::vector<float> outputKernel3(n * d, 0);
        std::vector<float> outputKernel3u(n * d, 0);

        // a temporary buffer for faiss
        std::vector<float> tempFaiss(d, 0);

        // random weights
        std::vector<float> weights(nIterations * n, 0);
        for (uint64_t i = 0; i < nIterations * n; i++) {
            weights[i] = u(rng);
        }

        // faiss
        StopWatch swFaiss;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        tempFaiss.data());
                for (uint64_t j = 0; j < d; j++)
                    outputFaiss[i * d + j] += weight * tempFaiss[j];
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels: accum 1 point
        StopWatch swKernel1;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        weight,
                        outputKernel1.data() + i * d);
            }
        }
        const double timeKernel1 = swKernel1.elapsed();

        // evaluate the error
        const double error1 = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\taccum_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel1
                  << "\t" << error1 << std::endl;

        // kernels: accum 2 points, shared centroids
        StopWatch swKernel2;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2.data() + i * d);
            }
        }
        const double timeKernel2 = swKernel2.elapsed();

        // evaluate the error
        const double error2 = getError(n, d, outputFaiss, outputKernel2);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2
                  << "\t" << error2 << std::endl;

        // kernels: accum 2 points, unique centroids
        StopWatch swKernel2u;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2u.data() + i * d);
            }
        }
        const double timeKernel2u = swKernel2u.elapsed();

        // evaluate the error
        const double error2u = getError(n, d, outputFaiss, outputKernel2u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2u
                  << "\t" << error2u << std::endl;

        // kernels: accum 3 points, shared centroids
        StopWatch swKernel3;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3.data() + i * d);
            }
        }
        const double timeKernel3 = swKernel3.elapsed();

        // evaluate the error
        const double error3 = getError(n, d, outputFaiss, outputKernel3);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3
                  << "\t" << error3 << std::endl;

        // kernels: accum 3 points, unique centroids
        StopWatch swKernel3u;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3u.data() + i * d);
            }
        }
        const double timeKernel3u = swKernel3u.elapsed();

        // evaluate the error
        const double error3u = getError(n, d, outputFaiss, outputKernel3u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3u
                  << "\t" << error3u << std::endl;
    }
}

//
template <typename T>
static void verifyMinMaxIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData,
        const uint64_t nIterations) {
    //
    const float* pqFineCentroidsQ = nullptr;
    const float* pqCoarseCentroidsQ = nullptr;

    // extract an index that is wrapped with IndexRowwiseMinMaxBase
    const std::shared_ptr<faiss::IndexRowwiseMinMaxBase> indexMinMax =
            std::dynamic_pointer_cast<faiss::IndexRowwiseMinMaxBase>(index);

    auto subIndex = indexMinMax->index;

    //
    testIfIVFPQ(subIndex, &pqCoarseCentroidsQ, &pqFineCentroidsQ);
    testIfResidualPQ(subIndex, &pqCoarseCentroidsQ, &pqFineCentroidsQ);

    //
    const size_t codeSize = index->sa_code_size();

    // initialize the random engine
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // use 1 thread
    omp_set_num_threads(1);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // sequential order
    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            index->sa_decode(n, encodedData.data(), outputFaiss.data());
        }
        double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (int iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                T::store(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + i * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        double timeKernel = swKernel.elapsed();

        // evaluate the error
        double error = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\tstore_seq\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random order

    // generate a random order of points
    std::uniform_int_distribution<uint64_t> un(0, n - 1);
    std::vector<uint64_t> pointIncidesToDecode(nIterations * n, 0);
    for (uint64_t i = 0; i < nIterations * n; i++) {
        pointIncidesToDecode[i] = un(rng);
    }

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        outputFaiss.data() + i * d);
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                T::store(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        const double timeKernel = swKernel.elapsed();

        // evaluate the error
        const double error = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\tstore_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random accumulate

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);
        std::vector<float> outputKernel2(n * d, 0);
        std::vector<float> outputKernel2u(n * d, 0);
        std::vector<float> outputKernel3(n * d, 0);
        std::vector<float> outputKernel3u(n * d, 0);

        // a temporary buffer for faiss
        std::vector<float> tempFaiss(d, 0);

        // random weights
        std::vector<float> weights(nIterations * n, 0);
        for (uint64_t i = 0; i < nIterations * n; i++) {
            weights[i] = u(rng);
        }

        // faiss
        StopWatch swFaiss;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        tempFaiss.data());
                for (uint64_t j = 0; j < d; j++) {
                    outputFaiss[i * d + j] += weight * tempFaiss[j];
                }
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels: accum 1 point
        StopWatch swKernel1;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        weight,
                        outputKernel1.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel1[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel1 = swKernel1.elapsed();

        // evaluate the error
        const double error1 = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\taccum_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel1
                  << "\t" << error1 << std::endl;

        // kernels: accum 2 points, shared centroids
        StopWatch swKernel2;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel2[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel2 = swKernel2.elapsed();

        // evaluate the error
        const double error2 = getError(n, d, outputFaiss, outputKernel2);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2
                  << "\t" << error2 << std::endl;

        // kernels: accum 2 points, unique centroids
        StopWatch swKernel2u;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2u.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel2u[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel2u = swKernel2u.elapsed();

        // evaluate the error
        const double error2u = getError(n, d, outputFaiss, outputKernel2u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2u
                  << "\t" << error2u << std::endl;

        // kernels: accum 3 points, shared centroids
        StopWatch swKernel3;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel3[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel3 = swKernel3.elapsed();

        // evaluate the error
        const double error3 = getError(n, d, outputFaiss, outputKernel3);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3
                  << "\t" << error3 << std::endl;

        // kernels: accum 3 points, unique centroids
        StopWatch swKernel3u;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        pqCoarseCentroidsQ,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3u.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel3u[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel3u = swKernel3u.elapsed();

        // evaluate the error
        const double error3u = getError(n, d, outputFaiss, outputKernel3u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3u
                  << "\t" << error3u << std::endl;
    }
}

//
template <typename T>
static void verifyIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData,
        const uint64_t nIterations) {
    //
    const faiss::IndexPQ* const indexQ =
            dynamic_cast<const faiss::IndexPQ*>(index.get());
    const float* const pqFineCentroidsQ = indexQ->pq.centroids.data();

    //
    const size_t codeSize = index->sa_code_size();

    // initialize the random engine
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // use 1 thread
    omp_set_num_threads(1);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // sequential order
    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            index->sa_decode(n, encodedData.data(), outputFaiss.data());
        }
        double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (int iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                T::store(
                        pqFineCentroidsQ,
                        encodedData.data() + i * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        double timeKernel = swKernel.elapsed();

        // evaluate the error
        double error = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\tstore_seq\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random order

    // generate a random order of points
    std::uniform_int_distribution<uint64_t> un(0, n - 1);
    std::vector<uint64_t> pointIncidesToDecode(nIterations * n, 0);
    for (uint64_t i = 0; i < nIterations * n; i++) {
        pointIncidesToDecode[i] = un(rng);
    }

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        outputFaiss.data() + i * d);
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                T::store(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        const double timeKernel = swKernel.elapsed();

        // evaluate the error
        const double error = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\tstore_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random accumulate

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);
        std::vector<float> outputKernel2(n * d, 0);
        std::vector<float> outputKernel2u(n * d, 0);
        std::vector<float> outputKernel3(n * d, 0);
        std::vector<float> outputKernel3u(n * d, 0);

        // a temporary buffer for faiss
        std::vector<float> tempFaiss(d, 0);

        // random weights
        std::vector<float> weights(nIterations * n, 0);
        for (uint64_t i = 0; i < nIterations * n; i++) {
            weights[i] = u(rng);
        }

        // faiss
        StopWatch swFaiss;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        tempFaiss.data());
                for (uint64_t j = 0; j < d; j++) {
                    outputFaiss[i * d + j] += weight * tempFaiss[j];
                }
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels: accum 1 point
        StopWatch swKernel1;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        weight,
                        outputKernel1.data() + i * d);
            }
        }
        const double timeKernel1 = swKernel1.elapsed();

        // evaluate the error
        const double error1 = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\taccum_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel1
                  << "\t" << error1 << std::endl;

        // kernels: accum 2 points, shared centroids
        StopWatch swKernel2;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2.data() + i * d);
            }
        }
        const double timeKernel2 = swKernel2.elapsed();

        // evaluate the error
        const double error2 = getError(n, d, outputFaiss, outputKernel2);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2
                  << "\t" << error2 << std::endl;

        // kernels: accum 2 points, unique centroids
        StopWatch swKernel2u;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2u.data() + i * d);
            }
        }
        const double timeKernel2u = swKernel2u.elapsed();

        // evaluate the error
        const double error2u = getError(n, d, outputFaiss, outputKernel2u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2u
                  << "\t" << error2u << std::endl;

        // kernels: accum 3 points, shared centroids
        StopWatch swKernel3;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3.data() + i * d);
            }
        }
        const double timeKernel3 = swKernel3.elapsed();

        // evaluate the error
        const double error3 = getError(n, d, outputFaiss, outputKernel3);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3
                  << "\t" << error3 << std::endl;

        // kernels: accum 3 points, unique centroids
        StopWatch swKernel3u;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3u.data() + i * d);
            }
        }
        const double timeKernel3u = swKernel3u.elapsed();

        // evaluate the error
        const double error3u = getError(n, d, outputFaiss, outputKernel3u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3u
                  << "\t" << error3u << std::endl;
    }
}

//
template <typename T>
static void verifyMinMaxIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const std::shared_ptr<faiss::Index>& index,
        const std::vector<uint8_t>& encodedData,
        const uint64_t nIterations) {
    // extract an index that is wrapped with IndexRowwiseMinMaxBase
    const std::shared_ptr<faiss::IndexRowwiseMinMaxBase> indexMinMax =
            std::dynamic_pointer_cast<faiss::IndexRowwiseMinMaxBase>(index);

    auto subIndex = indexMinMax->index;

    //
    const faiss::IndexPQ* const indexQ =
            dynamic_cast<const faiss::IndexPQ*>(subIndex);
    const float* const pqFineCentroidsQ = indexQ->pq.centroids.data();

    //
    const size_t codeSize = index->sa_code_size();
    // initialize the random engine
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // use 1 thread
    omp_set_num_threads(1);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // sequential order
    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            index->sa_decode(n, encodedData.data(), outputFaiss.data());
        }
        double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (int iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                T::store(
                        pqFineCentroidsQ,
                        encodedData.data() + i * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        double timeKernel = swKernel.elapsed();

        // evaluate the error
        double error = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\tstore_seq\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random order

    // generate a random order of points
    std::uniform_int_distribution<uint64_t> un(0, n - 1);
    std::vector<uint64_t> pointIncidesToDecode(nIterations * n, 0);
    for (uint64_t i = 0; i < nIterations * n; i++) {
        pointIncidesToDecode[i] = un(rng);
    }

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);

        // faiss
        StopWatch swFaiss;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        outputFaiss.data() + i * d);
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels
        StopWatch swKernel;
        for (uint64_t iter = 0; iter < nIterations; iter++) {
            for (uint64_t i = 0; i < n; i++) {
                const auto pointIdx = pointIncidesToDecode[i + iter * n];
                T::store(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        outputKernel1.data() + i * d);
            }
        }
        const double timeKernel = swKernel.elapsed();

        // evaluate the error
        const double error = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\tstore_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel
                  << "\t" << error << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random accumulate

    {
        std::vector<float> outputFaiss(n * d, 0);
        std::vector<float> outputKernel1(n * d, 0);
        std::vector<float> outputKernel2(n * d, 0);
        std::vector<float> outputKernel2u(n * d, 0);
        std::vector<float> outputKernel3(n * d, 0);
        std::vector<float> outputKernel3u(n * d, 0);

        // a temporary buffer for faiss
        std::vector<float> tempFaiss(d, 0);

        // random weights
        std::vector<float> weights(nIterations * n, 0);
        for (uint64_t i = 0; i < nIterations * n; i++) {
            weights[i] = u(rng);
        }

        // faiss
        StopWatch swFaiss;
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                index->sa_decode(
                        1,
                        encodedData.data() + pointIdx * codeSize,
                        tempFaiss.data());
                for (uint64_t j = 0; j < d; j++) {
                    outputFaiss[i * d + j] += weight * tempFaiss[j];
                }
            }
        }
        const double timeFaiss = swFaiss.elapsed();

        // kernels: accum 1 point
        StopWatch swKernel1;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter++) {
                const auto pointIdx =
                        pointIncidesToDecode[i * nIterations + iter];
                const auto weight = weights[i * nIterations + iter];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx * codeSize,
                        weight,
                        outputKernel1.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel1[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel1 = swKernel1.elapsed();

        // evaluate the error
        const double error1 = getError(n, d, outputFaiss, outputKernel1);

        std::cout << description << "\t" << n << "\t" << d << "\taccum_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel1
                  << "\t" << error1 << std::endl;

        // kernels: accum 2 points, shared centroids
        StopWatch swKernel2;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel2[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel2 = swKernel2.elapsed();

        // evaluate the error
        const double error2 = getError(n, d, outputFaiss, outputKernel2);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2
                  << "\t" << error2 << std::endl;

        // kernels: accum 2 points, unique centroids
        StopWatch swKernel2u;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 2) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        outputKernel2u.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel2u[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel2u = swKernel2u.elapsed();

        // evaluate the error
        const double error2u = getError(n, d, outputFaiss, outputKernel2u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum2u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel2u
                  << "\t" << error2u << std::endl;

        // kernels: accum 3 points, shared centroids
        StopWatch swKernel3;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel3[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel3 = swKernel3.elapsed();

        // evaluate the error
        const double error3 = getError(n, d, outputFaiss, outputKernel3);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3
                  << "\t" << error3 << std::endl;

        // kernels: accum 3 points, unique centroids
        StopWatch swKernel3u;
        for (uint64_t i = 0; i < n; i++) {
            float outputAccumMin = 0;
            for (uint64_t iter = 0; iter < nIterations; iter += 3) {
                const auto pointIdx0 =
                        pointIncidesToDecode[i * nIterations + iter + 0];
                const auto weight0 = weights[i * nIterations + iter + 0];
                const auto pointIdx1 =
                        pointIncidesToDecode[i * nIterations + iter + 1];
                const auto weight1 = weights[i * nIterations + iter + 1];
                const auto pointIdx2 =
                        pointIncidesToDecode[i * nIterations + iter + 2];
                const auto weight2 = weights[i * nIterations + iter + 2];
                T::accum(
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx0 * codeSize,
                        weight0,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx1 * codeSize,
                        weight1,
                        pqFineCentroidsQ,
                        encodedData.data() + pointIdx2 * codeSize,
                        weight2,
                        outputKernel3u.data() + i * d,
                        outputAccumMin);
            }
            for (uint64_t j = 0; j < d; j++) {
                outputKernel3u[i * d + j] += outputAccumMin;
            }
        }
        const double timeKernel3u = swKernel3u.elapsed();

        // evaluate the error
        const double error3u = getError(n, d, outputFaiss, outputKernel3u);

        std::cout << description << "\t" << n << "\t" << d << "\taccum3u_rnd\t"
                  << nIterations << "\t" << timeFaiss << "\t" << timeKernel3u
                  << "\t" << error3u << std::endl;
    }
}

template <typename T>
void testIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const uint64_t nIterations) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyIndex2LevelDecoder<T>(
            n, d, description, index, encodedData, nIterations);
}

template <typename T>
void testMinMaxIndex2LevelDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const uint64_t nIterations) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyMinMaxIndex2LevelDecoder<T>(
            n, d, description, index, encodedData, nIterations);
}

template <typename T>
void testIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const uint64_t nIterations) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyIndexPQDecoder<T>(n, d, description, index, encodedData, nIterations);
}

template <typename T>
void testMinMaxIndexPQDecoder(
        const uint64_t n,
        const uint64_t d,
        const std::string& description,
        const uint64_t nIterations) {
    auto data = generate(n, d);
    std::shared_ptr<faiss::Index> index;
    std::vector<uint8_t> encodedData;
    std::tie(index, encodedData) = trainDataset(data, n, d, description);

    verifyMinMaxIndexPQDecoder<T>(
            n, d, description, index, encodedData, nIterations);
}

//
int main(int argc, char** argv) {
    // 1 MB points
    const uint64_t INDEX_SIZE = 65536 * 16;
    const uint64_t N_ITERATIONS = 18;

    static_assert(
            (N_ITERATIONS % 6) == 0, "Number of iterations should be 6*x");

    // print the header
    auto delim = "\t";
    std::cout << "Codec" << delim << "n" << delim << "d" << delim
              << "Experiment" << delim << "Iterations" << delim << "Faiss time"
              << delim << "SADecodeKernel time" << delim << "Error"
              << std::endl;

    // The following experiment types are available:
    // * store_seq - decode a contiguous block of codes into vectors, one by one
    // * store_rnd - decode a contiguous block of codes into vectors in a random
    // order
    // * accum_rnd - create a linear combination from decoded vectors,
    // random order
    // * accum2_rnd - create a linear combination from decoded vectors,
    // random order, decode 2 codes per call, centroid tables are shared
    // * accum2u_rnd - create a linear combination from decoded vectors,
    // random order, decode 2 codes per call, centroid tables are not shared
    // * accum3_rnd - create a linear combination from decoded vectors,
    // random order, decode 3 codes per call, centroid tables are shared
    // * accum3u_rnd - create a linear combination from decoded vectors,
    // random order, decode 3 codes per call, centroid tables are not shared
    //
    // It is expected that:
    // * store_seq is faster than store_rnd
    // * accum2 is faster than accum
    // * accum3 is faster than accum2

    // test plain PQx8
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 2>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ64np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 4>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ32np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 8>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ16np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 16>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ8np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 32>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ4np", N_ITERATIONS);
    }

    // test PQx10
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 2, 10>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ64x10np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 4, 10>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ32x10np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 8, 10>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ16x10np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 16, 10>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ8x10np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::IndexPQDecoder<128, 32, 10>;
        testIndexPQDecoder<T>(INDEX_SIZE, 128, "PQ4x10np", N_ITERATIONS);
    }

    // test MinMaxFP16,PQx8
    {
        using SubT = faiss::cppcontrib::IndexPQDecoder<128, 2>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndexPQDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,PQ64np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::IndexPQDecoder<128, 4>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndexPQDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,PQ32np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::IndexPQDecoder<128, 8>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndexPQDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,PQ16np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::IndexPQDecoder<128, 16>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndexPQDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,PQ8np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::IndexPQDecoder<128, 32>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndexPQDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,PQ4np", N_ITERATIONS);
    }

    // test IVFPQ
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 2>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "IVF256,PQ64np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 4>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "IVF256,PQ32np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 8>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "IVF256,PQ16np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 16>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "IVF256,PQ8np", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 32>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "IVF256,PQ4np", N_ITERATIONS);
    }

    // test Residual,PQ
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 2>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual4x8,PQ64", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 4>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual4x8,PQ32", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 8>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual4x8,PQ16", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 16>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual4x8,PQ8", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 32, 32>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual4x8,PQ4", N_ITERATIONS);
    }

    // test MinMaxFP16,IVFPQ
    {
        using SubT = faiss::cppcontrib::Index2LevelDecoder<128, 128, 2>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,IVF256,PQ64np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::Index2LevelDecoder<128, 128, 4>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,IVF256,PQ32np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::Index2LevelDecoder<128, 128, 8>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,IVF256,PQ16np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::Index2LevelDecoder<128, 128, 16>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,IVF256,PQ8np", N_ITERATIONS);
    }
    {
        using SubT = faiss::cppcontrib::Index2LevelDecoder<128, 128, 32>;
        using T = faiss::cppcontrib::IndexMinMaxFP16Decoder<SubT>;
        testMinMaxIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "MinMaxFP16,IVF256,PQ4np", N_ITERATIONS);
    }

    // test Residual,PQ with unusual bits
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 2, 16, 10>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual1x10,PQ64x10", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 4, 16, 10>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual1x10,PQ32x10", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 8, 16, 10>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual1x10,PQ16x10", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 16, 16, 10>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual1x10,PQ8x10", N_ITERATIONS);
    }
    {
        using T = faiss::cppcontrib::Index2LevelDecoder<128, 128, 32, 16, 10>;
        testIndex2LevelDecoder<T>(
                INDEX_SIZE, 128, "Residual1x10,PQ4x10", N_ITERATIONS);
    }

    return 0;
}
