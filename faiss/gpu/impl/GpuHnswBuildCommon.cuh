// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, 6sense Insights Inc.
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

// Graph/vector upload helpers shared by the HNSW->GPU build paths. These are
// intentionally free of any dependency on a specific IndexHNSW flavor:
//   - extract_hnsw_layers() is templated on the graph struct and works with any
//     type exposing the faiss::HNSW accessor interface (neighbor_range,
//     nb_neighbors, neighbors, levels, entry_point, max_level).
//   - the upload_* helpers take raw host buffers.
// The flavor-specific entry points (from a vanilla faiss::IndexHNSW, or from
// Knowhere's cppcontrib IndexHNSW) live in separate headers and reuse these.

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/gpu/impl/GpuHnswSearchKernel.cuh>
#include <faiss/gpu/impl/GpuHnswTypes.h>

// GpuHnswUploadFaultInjection (used below) lives in GpuHnswTypes.h so it is
// reachable from host-compiled unit tests without pulling in device kernels.
#define GPU_HNSW_BUILD_CUDA_CHECK(expr)                                  \
    do {                                                                 \
        if (faiss::gpu::GpuHnswUploadFaultInjection::should_fail()) {    \
            throw std::runtime_error(                                    \
                    std::string("CUDA error (injected): simulated ") +   \
                    "upload failure at " + __FILE__ + ":" +              \
                    std::to_string(__LINE__));                           \
        }                                                                \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            throw std::runtime_error(                                    \
                    std::string("CUDA error: ") +                        \
                    cudaGetErrorString(_e) + " at " + __FILE__ + ":" +   \
                    std::to_string(__LINE__));                           \
        }                                                                \
    } while (0)

namespace faiss {
namespace gpu {

/// Extract HNSW graph layers from an HNSW struct.
/// Template parameter HnswT can be faiss::HNSW or any type exposing the same
/// interface:
///   - neighbor_range(node, layer, &begin, &end)
///   - nb_neighbors(layer) -> int
///   - neighbors            : flat neighbor array indexed by neighbor_range
///   - levels : per-node array; levels[i] is the 1-based layer count, so
///              node i lives on layers 0..levels[i]-1 and "i is on layer L"
///              is the test levels[i] > L
///   - entry_point, max_level
/// The levels[] convention above matches faiss::HNSW (HNSW.cpp uses
/// pt_level = levels[i] - 1); a variant with different semantics would need a
/// different membership test at the levels[i] > layer check below.
template <typename HnswT>
inline void extract_hnsw_layers(
        const HnswT& hnsw,
        int64_t n_rows,
        std::vector<GpuHnswDeviceUpperLayer>& h_upper_layers,
        std::vector<uint32_t>& h_layer0_flat,
        uint32_t& entry_point,
        int& M,
        int& max_degree0,
        int& num_layers) {
    const int maxM0 = hnsw.nb_neighbors(0);
    const int max_lv = hnsw.max_level;
    // nb_neighbors(1) indexes cum_nneighbor_per_level[2], which a degenerate
    // layer-0-only index (max_level == 0) need not contain — reading it would
    // be out of bounds. maxM is only used for the upper-layer loop below, which
    // is empty when max_lv == 0, so fall back to maxM0 in that case.
    const int maxM = (max_lv >= 1) ? hnsw.nb_neighbors(1) : maxM0;

    // The kernels dereference d_dataset + entry_point * dim directly, so a
    // malformed CPU index carrying an invalid/sentinel entry_point (e.g. -1
    // cast to UINT32_MAX) would be an out-of-bounds device read. Validate here
    // rather than faulting on the GPU.
    if (hnsw.entry_point < 0 || hnsw.entry_point >= n_rows) {
        throw std::runtime_error(
                std::string("gpu_hnsw: invalid HNSW entry_point ") +
                std::to_string(static_cast<long long>(hnsw.entry_point)) +
                " (n_rows=" + std::to_string(n_rows) + ")");
    }
    entry_point = static_cast<uint32_t>(hnsw.entry_point);
    M = maxM;
    max_degree0 = maxM0;
    num_layers = max_lv + 1;

    // Layer 0: dense [n_rows x maxM0]
    h_layer0_flat.assign(n_rows * maxM0, UINT32_MAX);
    for (int64_t i = 0; i < n_rows; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        uint32_t count = static_cast<uint32_t>(end - begin);
        for (uint32_t j = 0; j < count; j++) {
            auto nb = hnsw.neighbors[begin + j];
            if (nb >= 0)
                h_layer0_flat[i * maxM0 + j] = static_cast<uint32_t>(nb);
        }
    }

    // Upper layers (1..max_level): sparse [num_nodes_at_L x maxM]
    h_upper_layers.resize(max_lv);
    for (int layer = 1; layer <= max_lv; layer++) {
        auto& ul = h_upper_layers[layer - 1];
        ul.max_degree = static_cast<uint32_t>(maxM);

        std::vector<uint32_t> node_ids;
        for (int64_t i = 0; i < n_rows; i++) {
            if (hnsw.levels[i] > layer)
                node_ids.push_back(static_cast<uint32_t>(i));
        }
        ul.num_nodes = static_cast<uint32_t>(node_ids.size());

        std::vector<uint32_t> h_neighbors(ul.num_nodes * maxM, UINT32_MAX);

        for (uint32_t idx = 0; idx < ul.num_nodes; idx++) {
            int64_t i = node_ids[idx];
            size_t begin, end;
            hnsw.neighbor_range(i, layer, &begin, &end);
            uint32_t count = static_cast<uint32_t>(end - begin);
            for (uint32_t j = 0; j < count; j++) {
                auto nb = hnsw.neighbors[begin + j];
                if (nb >= 0)
                    h_neighbors[idx * maxM + j] = static_cast<uint32_t>(nb);
            }
        }

        GPU_HNSW_BUILD_CUDA_CHECK(
                cudaMalloc(&ul.d_node_ids, ul.num_nodes * sizeof(uint32_t)));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                ul.d_node_ids,
                node_ids.data(),
                ul.num_nodes * sizeof(uint32_t),
                cudaMemcpyHostToDevice));

        GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(
                &ul.d_neighbors,
                ul.num_nodes * maxM * sizeof(uint32_t)));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                ul.d_neighbors,
                h_neighbors.data(),
                ul.num_nodes * maxM * sizeof(uint32_t),
                cudaMemcpyHostToDevice));
    }
}

inline void normalize_vectors(
        std::vector<float>& h_vectors,
        int64_t n_rows,
        int64_t dim) {
    for (int64_t i = 0; i < n_rows; i++) {
        float* v = h_vectors.data() + i * dim;
        float sq_norm = 0.0f;
        for (int64_t d = 0; d < dim; d++)
            sq_norm += v[d] * v[d];
        if (sq_norm > 0.0f) {
            float inv = 1.0f / std::sqrt(sq_norm);
            for (int64_t d = 0; d < dim; d++)
                v[d] *= inv;
        }
    }
}

template <typename HnswT>
inline void upload_graph_to_gpu(
        GpuHnswDeviceIndex& idx,
        const HnswT& hnsw,
        int64_t n_rows) {
    std::vector<uint32_t> h_layer0_flat;
    extract_hnsw_layers(
            hnsw,
            n_rows,
            idx.upper_layers,
            h_layer0_flat,
            idx.entry_point,
            idx.M,
            idx.max_degree0,
            idx.num_layers);

    size_t graph0_bytes =
            static_cast<size_t>(n_rows) * idx.max_degree0 * sizeof(uint32_t);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_layer0_graph, graph0_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_layer0_graph,
            h_layer0_flat.data(),
            graph0_bytes,
            cudaMemcpyHostToDevice));

    int num_upper = static_cast<int>(idx.upper_layers.size());
    idx.num_upper_layers_built = num_upper;
    if (num_upper > 0) {
        using kernel_ptrs = hnsw_kernel::upper_layer_ptrs;
        std::vector<kernel_ptrs> h_ptrs(num_upper);
        for (int i = 0; i < num_upper; i++) {
            const auto& ul = idx.upper_layers[i];
            h_ptrs[i] = {
                    ul.d_node_ids, ul.d_neighbors, ul.num_nodes, ul.max_degree};
        }
        size_t ptrs_bytes = num_upper * sizeof(kernel_ptrs);
        GPU_HNSW_BUILD_CUDA_CHECK(
                cudaMalloc(&idx.d_upper_layer_ptrs, ptrs_bytes));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                idx.d_upper_layer_ptrs,
                h_ptrs.data(),
                ptrs_bytes,
                cudaMemcpyHostToDevice));
    }
}

// Upload precomputed per-row inverse L2 norms to the device. Optional: only the
// Knowhere cosine path (which records the *original* input norms in the CPU
// index) uses this; the vanilla faiss path leaves d_inv_norms == nullptr and
// obtains cosine by normalizing the stored vectors (see upload_fp32_dataset).
inline void upload_inv_norms(
        GpuHnswDeviceIndex& idx,
        const float* inv_norms,
        int64_t n_rows) {
    size_t norms_bytes = static_cast<size_t>(n_rows) * sizeof(float);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_inv_norms, norms_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_inv_norms, inv_norms, norms_bytes, cudaMemcpyHostToDevice));
}

inline void upload_fp32_dataset(
        GpuHnswDeviceIndex& idx,
        std::vector<float>& h_vectors,
        int64_t n_rows,
        bool is_cosine,
        const float* stored_inv_norms = nullptr) {
    int64_t dim = idx.dim;
    // Flat cosine (stored_inv_norms == nullptr): the stored vectors are the
    // exact originals, so normalizing them in place yields cosine via plain
    // inner product. Lossy-SQ cosine decoded to fp32 (stored_inv_norms !=
    // null): keep the decoded vectors un-normalized and apply the CPU index's
    // inverse norms at search time, matching CPU semantics for lossy codes.
    if (is_cosine && stored_inv_norms == nullptr)
        normalize_vectors(h_vectors, n_rows, dim);

    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * sizeof(float);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset,
            h_vectors.data(),
            dataset_bytes,
            cudaMemcpyHostToDevice));
    idx.dataset_type = GpuHnswDatasetType::FP32;

    if (is_cosine && stored_inv_norms != nullptr)
        upload_inv_norms(idx, stored_inv_norms, n_rows);
}

// Upload fp16 (QT_fp16) or bf16 (QT_bf16) ScalarQuantizer codes to the GPU in
// their native 2-byte layout. faiss stores these codes row-major as raw IEEE
// half / bfloat16, which are bit-compatible with CUDA half / __nv_bfloat16, so
// the bytes are copied verbatim (no up-conversion to fp32).
inline void upload_halfwidth_dataset(
        GpuHnswDeviceIndex& idx,
        const uint8_t* codes,
        int64_t n_rows,
        bool is_cosine,
        GpuHnswDatasetType dtype,
        const float* stored_inv_norms) {
    int64_t dim = idx.dim;
    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * 2;
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset, codes, dataset_bytes, cudaMemcpyHostToDevice));
    idx.dataset_type = dtype;

    // fp16/bf16 codes are not normalized in place; when original inverse norms
    // are supplied (Knowhere path) apply them at search time, else the vanilla
    // path handles cosine via query/vector normalization upstream.
    if (is_cosine && stored_inv_norms != nullptr)
        upload_inv_norms(idx, stored_inv_norms, n_rows);
}

inline void upload_int8_dataset(
        GpuHnswDeviceIndex& idx,
        const uint8_t* codes,
        int64_t n_rows,
        bool is_cosine,
        const float* stored_inv_norms) {
    int64_t dim = idx.dim;
    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim;

    std::vector<int8_t> signed_codes(dataset_bytes);
    for (size_t i = 0; i < dataset_bytes; i++) {
        signed_codes[i] =
                static_cast<int8_t>(static_cast<int>(codes[i]) - 128);
    }

    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset,
            signed_codes.data(),
            dataset_bytes,
            cudaMemcpyHostToDevice));
    idx.dataset_type = GpuHnswDatasetType::INT8;

    if (is_cosine && stored_inv_norms != nullptr)
        upload_inv_norms(idx, stored_inv_norms, n_rows);
}

} // namespace gpu
} // namespace faiss
