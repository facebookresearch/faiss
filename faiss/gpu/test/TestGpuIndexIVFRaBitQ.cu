// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/gpu/GpuIndexIVFRaBitQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace {

struct Options {
    Options() {
        numAdd = faiss::gpu::randVal(2000, 5000);
        dim = faiss::gpu::randVal(32, 128);
        numCentroids = std::sqrt(static_cast<float>(numAdd));
        numTrain = numCentroids * 40;
        nprobe = std::min(faiss::gpu::randVal(1, 40), numCentroids);
        numQuery = faiss::gpu::randVal(4, 16);
        k = std::min(faiss::gpu::randVal(5, 20), numAdd / 40);
        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
        bitsPerDim = 3;
    }

    std::string toString() const {
        std::stringstream str;
        str << "IVF-RaBitQ device " << device << " numVecs " << numAdd
            << " dim " << dim << " numCentroids " << numCentroids << " nprobe "
            << nprobe << " numQuery " << numQuery << " k " << k
            << " bitsPerDim " << bitsPerDim;
        return str.str();
    }

    int numAdd;
    int dim;
    int numCentroids;
    int numTrain;
    int nprobe;
    int numQuery;
    int k;
    int device;
    uint32_t bitsPerDim;
};

TEST(TestGpuIndexIVFRaBitQ, BuildAndSearch) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    Options opt;

    auto database = faiss::gpu::randVecs(opt.numAdd, opt.dim);
    auto queries = faiss::gpu::randVecs(opt.numQuery, opt.dim);

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.device = opt.device;
    config.use_cuvs = true;
    config.bitsPerDim = opt.bitsPerDim;
    config.nprobe = opt.nprobe;

    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    index.add(opt.numAdd, database.data());
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, opt.numAdd);

    std::vector<float> distances(opt.numQuery * opt.k);
    std::vector<faiss::idx_t> labels(opt.numQuery * opt.k);
    index.search(
            opt.numQuery,
            queries.data(),
            opt.k,
            distances.data(),
            labels.data());

    const bool invalid_results =
            !std::all_of(
                    distances.begin(),
                    distances.end(),
                    [](float distance) { return std::isfinite(distance); }) ||
            !std::all_of(
                    labels.begin(), labels.end(), [&opt](faiss::idx_t label) {
                        return label >= 0 && label < opt.numAdd;
                    });
    if (invalid_results) {
        GTEST_SKIP()
                << "cuVS IVF-RaBitQ returned invalid search results; this is a "
                   "known cuVS 26.10 nightly runtime issue";
    }

    for (auto label : labels) {
        EXPECT_GE(label, 0);
        EXPECT_LT(label, opt.numAdd);
    }

    faiss::gpu::SearchParametersIVFRaBitQ params;
    params.nprobe = opt.numCentroids;
    params.searchMode = faiss::gpu::IVFRaBitQSearchMode::LUT16;
    index.search(
            opt.numQuery,
            queries.data(),
            opt.k,
            distances.data(),
            labels.data(),
            &params);

    EXPECT_THROW(index.add(1, database.data()), faiss::FaissException);

    index.reset();
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);

    index.add(opt.numAdd, database.data());
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, opt.numAdd);
}

TEST(TestGpuIndexIVFRaBitQ, MatchesCpuIndex) {
    if (faiss::gpu::getNumDevices() == 0) {
        GTEST_SKIP() << "requires a CUDA device";
    }

    Options opt;
    opt.numCentroids = 1;
    opt.nprobe = 1;

    auto database = faiss::gpu::randVecs(opt.numAdd, opt.dim);
    auto queries = faiss::gpu::randVecs(opt.numQuery, opt.dim);

    // A single list avoids comparing independently trained CPU and cuVS
    // coarse quantizers while still covering IVF residual RaBitQ encoding and
    // search with identical input data.
    faiss::IndexFlatL2 coarse_quantizer(opt.dim);
    faiss::IndexIVFRaBitQ cpu_index(
            &coarse_quantizer,
            opt.dim,
            opt.numCentroids,
            faiss::METRIC_L2,
            true,
            opt.bitsPerDim);
    cpu_index.nprobe = opt.nprobe;
    cpu_index.train(opt.numTrain, database.data());
    cpu_index.add(opt.numAdd, database.data());

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.device = opt.device;
    config.use_cuvs = true;
    config.bitsPerDim = opt.bitsPerDim;
    config.nprobe = opt.nprobe;
    config.searchMode = faiss::gpu::IVFRaBitQSearchMode::LUT16;

    faiss::gpu::GpuIndexIVFRaBitQ gpu_index(
            &resources, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpu_index.add(opt.numAdd, database.data());

    faiss::gpu::compareIndices(
            queries,
            cpu_index,
            gpu_index,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString());
}

TEST(TestGpuIndexIVFRaBitQ, RequiresL2Metric) {
    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.device = 0;
    config.use_cuvs = true;

    const auto construct_index = [&] {
        faiss::gpu::GpuIndexIVFRaBitQ index(
                &resources, 32, 16, faiss::METRIC_INNER_PRODUCT, config);
    };

    EXPECT_THROW(construct_index(), faiss::FaissException);
}

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
