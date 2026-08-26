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

#include <faiss/gpu/GpuIndexIVFRaBitQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>

#include <gtest/gtest.h>

#include <vector>

namespace {

TEST(TestGpuIndexIVFRaBitQ, BuildAndSearch) {
    constexpr int d = 32;
    constexpr int nlist = 16;
    constexpr int nb = 2048;
    constexpr int nq = 16;
    constexpr int k = 10;

    auto database = faiss::gpu::randVecs(nb, d);
    auto queries = faiss::gpu::randVecs(nq, d);

    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFRaBitQConfig config;
    config.device = 0;
    config.use_cuvs = true;
    config.bitsPerDim = 3;
    config.nprobe = 8;

    faiss::gpu::GpuIndexIVFRaBitQ index(
            &resources, d, nlist, faiss::METRIC_L2, config);
    index.add(nb, database.data());
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, nb);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, queries.data(), k, distances.data(), labels.data());

    for (auto label : labels) {
        EXPECT_GE(label, 0);
        EXPECT_LT(label, nb);
    }

    faiss::gpu::SearchParametersIVFRaBitQ params;
    params.nprobe = nlist;
    params.searchMode = faiss::gpu::IVFRaBitQSearchMode::LUT16;
    index.search(
            nq, queries.data(), k, distances.data(), labels.data(), &params);

    index.reset();
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);
}

} // namespace
