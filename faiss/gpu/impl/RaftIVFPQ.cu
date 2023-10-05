/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/neighbors/ivf_flat_codepacker.hpp>
#include <raft/neighbors/ivf_flat.cuh>

namespace faiss {
namespace gpu {
RaftIVFPQ::RaftIVFPQ(
        GpuResources* res,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space): IVFBase(res,
                  dim,
                  nlist,
                  metric,
                  metricArg,
                  // we use IVF cell residuals for encoding vectors
                  true,
                  interleavedLayout,
                  indicesOptions,
                  space),
          numSubQuantizers_(numSubQuantizers),
          bitsPerSubQuantizer_(bitsPerSubQuantizer),
          numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
          dimPerSubQuantizer_(dim_ / numSubQuantizers),
          useFloat16LookupTables_(useFloat16LookupTables),
          useMMCodeDistance_(useMMCodeDistance),
          precomputedCodes_(false) {
    FAISS_ASSERT(pqCentroidData);

    FAISS_ASSERT(useResidual);

    const raft::device_resources& raft_handle = res->getRaftHandleCurrentDevice();

    raft::neighbors::ivf_pq::index_params pams;
    switch (metric) {
        case faiss::METRIC_L2:
            pams.metric = raft::distance::DistanceType::L2Expanded;
            break;
        case faiss::METRIC_INNER_PRODUCT:
            pams.metric = raft::distance::DistanceType::InnerProduct;
            break;
        default:
            FAISS_THROW_MSG("Metric is not supported.");
    }

    raft::neighbors::ivf_flat::index_params pams;
    pams.codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    pams.n_lists = nlist;
    pams.pq_bits = bitsPerSubQuantizer;
    pams.pq_dim = numSubQuantizers;
    raft_knn_index.emplace(raft::neighbors::ivf_pq::index(raft_handle, pams, static_cast<uint32_t>(dim)));
}

void RaftIVFPQ::reset() {
    raft_knn_index.reset();
}

RaftIVFPQ::~RaftIVFPQ() {}

/// Find the approximate k nearest neighbors for `queries` against
/// our database
void RaftIVFFlat::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    // TODO: We probably don't want to ignore the coarse quantizer here...

    uint32_t numQueries = queries.getSize(0);
    uint32_t cols = queries.getSize(1);
    uint32_t k_ = k;

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    raft::neighbors::ivf_flat::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view = raft::make_device_matrix_view<const float, idx_t>(
            queries.data(), (idx_t)numQueries, (idx_t)cols);
    auto out_inds_view = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), (idx_t)numQueries, (idx_t)k_);
    auto out_dists_view = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(), (idx_t)numQueries, (idx_t)k_);

    raft::neighbors::ivf_pq::search<float, idx_t>(
            raft_handle,
            pams,
            raft_knn_index.value(),
            queries_view,
            out_inds_view,
            out_dists_view);

    /// Identify NaN rows and mask their nearest neighbors
    auto nan_flag = raft::make_device_vector<bool>(raft_handle, numQueries);

    validRowIndices_(queries, nan_flag.data_handle());

    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view(outIndices.data(), numQueries * k_),
            [nan_flag = nan_flag.data_handle(),
             out_inds = outIndices.data(),
             k_] __device__(uint32_t i) {
                uint32_t row = i / k_;
                if (!nan_flag[row])
                    return idx_t(-1);
                return out_inds[i];
            });

    float max_val = std::numeric_limits<float>::max();
    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view(outDistances.data(), numQueries * k_),
            [nan_flag = nan_flag.data_handle(),
             out_dists = outDistances.data(),
             max_val,
             k_] __device__(uint32_t i) {
                uint32_t row = i / k_;
                if (!nan_flag[row])
                    return max_val;
                return out_dists[i];
            });
}

idx_t RaftIVFPQ::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    idx_t n_rows = vecs.getSize(0);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// Remove NaN values
    auto nan_flag = raft::make_device_vector<bool, idx_t>(raft_handle, n_rows);

    validRowIndices_(vecs, nan_flag.data_handle());

    idx_t n_rows_valid = thrust::reduce(
            raft_handle.get_thrust_policy(),
            nan_flag.data_handle(),
            nan_flag.data_handle() + n_rows,
            0);

    if (n_rows_valid < n_rows) {
        auto gather_indices = raft::make_device_vector<idx_t, idx_t>(
                raft_handle, n_rows_valid);

        auto count = thrust::make_counting_iterator(0);

        thrust::copy_if(
                raft_handle.get_thrust_policy(),
                count,
                count + n_rows,
                gather_indices.data_handle(),
                [nan_flag = nan_flag.data_handle()] __device__(auto i) {
                    return nan_flag[i];
                });

        raft::matrix::gather(
                raft_handle,
                raft::make_device_matrix_view<float, idx_t>(
                        vecs.data(), n_rows, dim_),
                raft::make_const_mdspan(gather_indices.view()),
                (idx_t)16);

        auto valid_indices = raft::make_device_vector<idx_t, idx_t>(
                raft_handle, n_rows_valid);

        raft::matrix::gather(
                raft_handle,
                raft::make_device_matrix_view<idx_t>(
                        indices.data(), n_rows, (idx_t)1),
                raft::make_const_mdspan(gather_indices.view()));
    }

    /// TODO: We probably don't want to ignore the coarse quantizer here

    FAISS_ASSERT(raft_knn_index.has_value());
    raft_knn_index.emplace(raft::neighbors::ivf_pq::extend(
            raft_handle,
            raft::make_device_matrix_view<const float, idx_t>(
                    vecs.data(), n_rows_valid, dim_),
            std::make_optional<raft::device_vector_view<const idx_t, idx_t>>(
                    raft::make_device_vector_view<const idx_t, idx_t>(
                            indices.data(), n_rows_valid)),
            raft_knn_index.value()));

    return n_rows_valid;
}

void RaftIVFPQ::build_index() {
        const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
        raft::neighbors::ivf_pq::build_index(raft_handle,
           params,
           x,
           numVecs,
           dim);
}

// Convert the CPU layout to the GPU layout
std::vector<uint8_t> RaftIVFPQ::translateCodesToGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    if (!interleavedLayout_) {
        return codes;
    }

    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);

    auto up = unpackNonInterleaved(
            std::move(codes), numVecs, numSubQuantizers_, bitsPerSubQuantizer_);
    
    RaftIVFPQCodePackerInterleaved packer(
            (size_t)numVecs, (uint32_t)dim_);
    packer.pack_all(
            std::move(up), interleaved_codes.data());
    
    return interleaved_codes;
}

// Convert the GPU layout to the CPU layout
std::vector<uint8_t> RaftIVFPQ::translateCodesFromGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    if (!interleavedLayout_) {
        return codes;
    }

    std::vector<uint8_t> flat_codes(cpuListSizeInBytes);

    RaftIVFPQCodePackerInterleaved packer(
            (size_t)numVecs, (uint32_t)dim_);
    packer.unpack_all(
            codes, interleaved_codes.data());

    return packNonInterleaved(
            std::move(up), numVecs, numSubQuantizers_, bitsPerSubQuantizer_);
}

RaftIVFPQCodePackerInterleaved::RaftIVFPQCodePackerInterleaved(
        size_t list_size,
        int numSubQuantizers,
        int bitsPerSubQuantizer) {
    this->numSubQuantizers_ = numSubQuantizers;
    this->bitsPerSubQuantizer_ = bitsPerSubQuantizer;
    nvec = list_size;
}

void RaftIVFPQCodePackerInterleaved::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    raft::neighbors::ivf_pq::codepacker::pack_1(
            flat_code,
            block,
            static_cast<uint32_t>(numSubQuantizers),
            static_cast<uint32_t>(offset));
}

void RaftIVFPQCodePackerInterleaved::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    raft::neighbors::ivf_pq::codepacker::unpack_1(
            block,
            flat_code,
            static_cast<uint32_t>(numSubQuantizers),
            static_cast<uint32_t>(offset));
}
}
}