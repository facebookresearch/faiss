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

#include <memory>

namespace faiss {
namespace gpu {

class BinaryCuvsCagra;
#ifndef SWIG
class Dataset;
#endif

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

    /// Pointer-based construction is unsupported. Use a caller-owned Dataset.
    void add(idx_t n, const uint8_t* x) override;
    void train(idx_t n, const uint8_t* x) override;

#ifndef SWIG
    /// Build the full graph, borrowing UInt8 vectors with dim == d / 8.
    /// The backing storage must remain alive and unchanged until all index
    /// operations have completed. Reset before building a second graph.
    void train(const Dataset& dataset);
    void add(const Dataset& dataset);

    /// Rebind search storage without changing the graph. The vectors and their
    /// ordering must match the training data. Requires compatible device data;
    /// any padding must be zero and the caller retains ownership. Previous
    /// work on the index stream completes before rebinding.
    void updateDataset(const Dataset& dataset);
    bool isSearchReady() const;

    /// Copy the CPU graph while borrowing caller-supplied vectors in the same
    /// order as the CPU index. The CPU index can be released after this call
    /// when the supplied Dataset does not borrow its vector storage.
    void copyFrom(
            const faiss::IndexBinaryHNSWCagra* index,
            const Dataset& dataset);
#endif

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexBinaryHNSWCagra* index) const;

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
    /// Manages streams, cuBLAS handles and scratch memory for devices
    std::shared_ptr<GpuResources> resources_;

    /// Configuration options
    const GpuIndexCagraConfig cagraConfig_;

    /// Instance that we own; contains the cuVS index
    std::shared_ptr<BinaryCuvsCagra> index_;
};

} // namespace gpu
} // namespace faiss
