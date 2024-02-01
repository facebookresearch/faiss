/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <cstddef>
#include "faiss/MetricType.h"

struct Options {
    Options() {
        numTrain = 2 * faiss::gpu::randVal(2000, 5000);
        dim = faiss::gpu::randVal(64, 200);

        graphDegree = faiss::gpu::randSelect({16, 32});
        intermediateGraphDegree = faiss::gpu::randSelect({32, 64});
        buildAlgo = faiss::gpu::randSelect(
                {faiss::gpu::graph_build_algo::IVF_PQ, faiss::gpu::graph_build_algo::NN_DESCENT});

        numQuery = faiss::gpu::randVal(32, 100);
        k = faiss::gpu::randVal(10, 30);

        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    }

    std::string toString() const {
        std::stringstream str;
        str << "CAGRA device " << device << " numVecs " << numTrain << " dim "
            << dim << " graphDegree " << graphDegree << " intermediateGraphDegree " << intermediateGraphDegree
            << "buildAlgo " << static_cast<int>(buildAlgo)
            << " numQuery " << numQuery << " k " << k;

        return str.str();
    }

    int numTrain;
    int dim;
    size_t graphDegree;
    size_t intermediateGraphDegree;
    faiss::gpu::graph_build_algo buildAlgo;
    int numQuery;
    int k;
    int device;
};

void queryTest() {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);

        faiss::IndexHNSWFlat cpuIndex(
            opt.dim, opt.graphDegree / 2);
        cpuIndex.train(opt.numTrain, trainVecs.data());
        cpuIndex.add(opt.numTrain, trainVecs.data());

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        faiss::gpu::GpuIndexCagra gpuIndex(
                &res, cpuIndex.d, faiss::METRIC_L2, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

        faiss::gpu::compareIndices(
                cpuIndex,
                gpuIndex,
                opt.numQuery,
                opt.dim,
                opt.k,
                opt.toString(),
                0.15f,
                1.0f,
                0.15f);
    }
}

TEST(TestGpuIndexCagra, Float32_Query_L2) {
    queryTest();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
