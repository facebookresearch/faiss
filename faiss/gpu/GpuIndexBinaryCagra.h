// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/gpu/GpuIndexCagra.h>

namespace faiss {
namespace gpu {

class BinaryCuvsCagra;

struct GpuIndexBinaryCagra : public IndexBinary {
   public:
    GpuIndexBinaryCagra(
            GpuResourcesProvider* provider,
            int dims,
            GpuIndexCagraConfig config = GpuIndexCagraConfig());

    ~GpuIndexBinaryCagra() override;

    int getDevice() const;

    /// Returns a reference to our GpuResources object that manages memory,
    /// stream and handle resources on the GPU
    std::shared_ptr<GpuResources> getResources();

    /// Trains CAGRA based on the given vector data and add them along with ids.
    /// NB: The use of the add function here is to build the CAGRA graph on
    /// the base dataset. Use this function when you want to add vectors with
    /// ids. Ref: https://github.com/facebookresearch/faiss/issues/4107
    void add(idx_t n, const uint8_t* x) override;

    /// Trains CAGRA based on the given vector data.
    /// NB: The use of the train function here is to build the CAGRA graph on
    /// the base dataset and is currently the only function to add the full set
    /// of vectors (without IDs) to the index. There is no external quantizer to
    /// be trained here.
    void train(idx_t n, const uint8_t* x) override;

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexBinaryHNSW* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexBinaryHNSW* index) const;

    void reset() override;

    std::vector<idx_t> get_knngraph() const;

    void search(
            idx_t n,
            const uint8_t* x,
            // faiss::IndexBinary has idx_t for k
            idx_t k,
            int* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;

   protected:
    /// Called from search when the input data is on the CPU;
    /// potentially allows for pinned memory usage
    void searchFromCpuPaged_(
            idx_t n,
            const uint8_t* x,
            int k,
            int* outDistancesData,
            idx_t* outIndicesData,
            const SearchParameters* search_params) const;

    void searchNonPaged_(
            idx_t n,
            const uint8_t* x,
            int k,
            int* outDistancesData,
            idx_t* outIndicesData,
            const SearchParameters* search_params) const;

    void searchImpl_(
            idx_t n,
            const uint8_t* x,
            int k,
            int* distances,
            idx_t* labels,
            const SearchParameters* search_params) const;

   protected:
    /// Manages streans, cuBLAS handles and scratch memory for devices
    std::shared_ptr<GpuResources> resources_;

    /// Configuration options
    const GpuIndexCagraConfig cagraConfig_;

    /// Instance that we own; contains the cuVS index
    std::shared_ptr<BinaryCuvsCagra> index_;
};

} // namespace gpu
} // namespace faiss
