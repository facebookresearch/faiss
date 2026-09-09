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

#pragma once

#include <faiss/gpu/GpuIndex.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace faiss {
namespace gpu {

class CuvsIVFRaBitQ;

/// Search implementation used by cuVS IVF-RaBitQ.
enum class IVFRaBitQSearchMode {
    LUT16 = 0,
    LUT32 = 1,
    QUANT4 = 2,
    QUANT8 = 3,
};

struct GpuIndexIVFRaBitQConfig : public GpuIndexConfig {
    /// Number of bits used to encode each residual dimension. Supported values
    /// are 1 through 9.
    uint32_t bitsPerDim = 3;

    /// Number of k-means iterations to use while building the coarse IVF
    /// quantizer.
    uint32_t kmeansNIterations = 20;

    /// Maximum number of training vectors sampled from each coarse cluster.
    uint32_t maxTrainPointsPerCluster = 256;

    /// Enable cuVS's fast quantization path during index construction.
    bool useFastQuantize = true;

    /// Maximum number of vectors processed in a host-memory streaming build.
    size_t streamingBatchSize = 100000;

    /// Force streaming construction when the input is resident on the host.
    bool forceStreaming = false;

    /// Default number of IVF lists searched for each query.
    uint32_t nprobe = 20;

    /// Default search implementation.
    IVFRaBitQSearchMode searchMode = IVFRaBitQSearchMode::QUANT4;
};

/// Per-search IVF-RaBitQ parameters.
struct SearchParametersIVFRaBitQ : SearchParameters {
    uint32_t nprobe = 20;
    IVFRaBitQSearchMode searchMode = IVFRaBitQSearchMode::QUANT4;
};

/// GPU-native IVF-RaBitQ index backed by cuVS.
///
/// cuVS builds IVF-RaBitQ from a complete dataset. Consequently, the first
/// call to train() or add() constructs the complete index and incremental
/// additions are not supported.
class GpuIndexIVFRaBitQ : public GpuIndex {
   public:
    GpuIndexIVFRaBitQ(
            GpuResourcesProvider* provider,
            int dims,
            idx_t nlist,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexIVFRaBitQConfig config = GpuIndexIVFRaBitQConfig());

    ~GpuIndexIVFRaBitQ() override;

    void train(idx_t n, const float* x) override;

    /// Build the index from x. This is equivalent to train() for this index.
    void add(idx_t n, const float* x) override;

    void reset() override;

   protected:
    bool addImplRequiresIDs_() const override;

    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* search_params) const override;

   private:
    const idx_t nlist_;
    const GpuIndexIVFRaBitQConfig ivfRabitqConfig_;
    std::shared_ptr<CuvsIVFRaBitQ> index_;
};

} // namespace gpu
} // namespace faiss
