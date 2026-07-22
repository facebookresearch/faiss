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

// Build a GpuHnswDeviceIndex from a *vanilla* faiss::IndexHNSW (Flat or SQ
// storage). This path has no dependency on Knowhere's cppcontrib layer and is
// the one intended for upstream facebookresearch/faiss.
//
// Metric contract (matches the rest of faiss/gpu):
//   - METRIC_L2            -> L2 (use_ip = false)
//   - METRIC_INNER_PRODUCT -> inner product (use_ip = true)
// Cosine is not a distinct metric here. As elsewhere in faiss, cosine is
// obtained by L2-normalizing the vectors and building an inner-product index;
// the GPU index then computes plain IP. Consequently no per-row inverse-norm
// buffer is uploaded (d_inv_norms stays nullptr).
//
// NOTE on quantized storage + cosine: if a user builds an IndexHNSWSQ with
// METRIC_INNER_PRODUCT over pre-normalized inputs, the codes encode normalized
// vectors, so plain IP on the decoded codes reproduces cosine up to
// quantization error. That error is the subject of the recall gate discussed in
// the design doc for SQ8/fp16/bf16/int8; it is a property of quantization, not
// of this upload path.

#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ScalarQuantizer.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include <faiss/gpu/impl/GpuHnswBuildCommon.cuh>
#include <faiss/gpu/impl/GpuHnswTypes.h>

namespace faiss {
namespace gpu {

/// Build from a vanilla faiss::IndexHNSW with Flat storage.
inline std::unique_ptr<GpuHnswDeviceIndex> from_index_hnsw_flat(
        const faiss::IndexHNSW& hnsw_index,
        bool use_ip,
        int device = 0) {
    const auto* flat_storage =
            dynamic_cast<const faiss::IndexFlat*>(hnsw_index.storage);
    if (!flat_storage)
        throw std::runtime_error("gpu_hnsw: storage is not IndexFlat");

    int64_t n_rows = hnsw_index.ntotal;
    int64_t dim = hnsw_index.d;

    // reconstruct_n (not get_xb) so this works even if the storage keeps its
    // vectors somewhere other than a plain host array.
    std::vector<float> h_vectors(n_rows * dim);
    flat_storage->reconstruct_n(0, n_rows, h_vectors.data());

    auto idx = std::make_unique<GpuHnswDeviceIndex>();
    idx->n_rows = n_rows;
    idx->dim = dim;
    idx->use_ip = use_ip;
    idx->device = device;
    idx->scratch_pool = std::make_unique<GpuHnswScratchPool>(4, device);

    // is_cosine = false: no in-index normalization and no inverse-norm buffer.
    upload_fp32_dataset(*idx, h_vectors, n_rows, /*is_cosine=*/false);
    upload_graph_to_gpu(*idx, hnsw_index.hnsw, n_rows);
    return idx;
}

/// Build from a vanilla faiss::IndexHNSW with ScalarQuantizer storage.
/// Supports QT_8bit_direct_signed (native INT8/DP4A), QT_fp16, QT_bf16 (native
/// 2-byte), and other SQ types (decoded to fp32).
inline std::unique_ptr<GpuHnswDeviceIndex> from_index_hnsw_sq(
        const faiss::IndexHNSW& hnsw_index,
        bool use_ip,
        int device = 0) {
    const auto* sq_storage =
            dynamic_cast<const faiss::IndexScalarQuantizer*>(
                    hnsw_index.storage);
    if (!sq_storage)
        throw std::runtime_error(
                "gpu_hnsw: storage is not IndexScalarQuantizer");

    int64_t n_rows = hnsw_index.ntotal;
    int64_t dim = hnsw_index.d;

    auto idx = std::make_unique<GpuHnswDeviceIndex>();
    idx->n_rows = n_rows;
    idx->dim = dim;
    idx->use_ip = use_ip;
    idx->device = device;
    idx->scratch_pool = std::make_unique<GpuHnswScratchPool>(4, device);

    auto qtype = sq_storage->sq.qtype;

    if (qtype == faiss::ScalarQuantizer::QT_8bit_direct_signed) {
        upload_int8_dataset(
                *idx,
                sq_storage->codes.data(),
                n_rows,
                /*is_cosine=*/false,
                /*stored_inv_norms=*/nullptr);
    } else if (
            qtype == faiss::ScalarQuantizer::QT_fp16 ||
            qtype == faiss::ScalarQuantizer::QT_bf16) {
        GpuHnswDatasetType dtype =
                (qtype == faiss::ScalarQuantizer::QT_fp16)
                        ? GpuHnswDatasetType::FP16
                        : GpuHnswDatasetType::BF16;
        upload_halfwidth_dataset(
                *idx,
                sq_storage->codes.data(),
                n_rows,
                /*is_cosine=*/false,
                dtype,
                /*stored_inv_norms=*/nullptr);
    } else {
        // Other SQ types (e.g. QT_8bit / QT_4bit) are decoded to fp32.
        std::vector<float> h_vectors(n_rows * dim);
        sq_storage->sa_decode(
                n_rows, sq_storage->codes.data(), h_vectors.data());
        upload_fp32_dataset(*idx, h_vectors, n_rows, /*is_cosine=*/false);
    }

    upload_graph_to_gpu(*idx, hnsw_index.hnsw, n_rows);
    return idx;
}

} // namespace gpu
} // namespace faiss
