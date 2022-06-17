/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <limits>

namespace faiss {
namespace gpu {

//
// RaftIndexFlatL2
//

RaftIndexFlatL2::RaftIndexFlatL2(
        GpuResourcesProvider* provider,
        faiss::IndexFlatL2* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, index, config) {}

RaftIndexFlatL2::RaftIndexFlatL2(
        std::shared_ptr<GpuResources> resources,
        faiss::IndexFlatL2* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, index, config) {}

RaftIndexFlatL2::RaftIndexFlatL2(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, dims, faiss::METRIC_L2, config) {}

RaftIndexFlatL2::RaftIndexFlatL2(
        std::shared_ptr<GpuResources> resources,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, dims, faiss::METRIC_L2, config) {}

void RaftIndexFlatL2::copyFrom(faiss::IndexFlat* index) {
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a RaftIndexFlatL2 from an index of "
            "different metric_type");

    GpuIndexFlat::copyFrom(index);
}

void RaftIndexFlatL2::copyTo(faiss::IndexFlat* index) {
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a RaftIndexFlatL2 to an index of "
            "different metric_type");

    GpuIndexFlat::copyTo(index);
}

//
// RaftIndexFlatIP
//

RaftIndexFlatIP::RaftIndexFlatIP(
        GpuResourcesProvider* provider,
        faiss::IndexFlatIP* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, index, config) {}

RaftIndexFlatIP::RaftIndexFlatIP(
        std::shared_ptr<GpuResources> resources,
        faiss::IndexFlatIP* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, index, config) {}

RaftIndexFlatIP::RaftIndexFlatIP(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, dims, faiss::METRIC_INNER_PRODUCT, config) {}

RaftIndexFlatIP::RaftIndexFlatIP(
        std::shared_ptr<GpuResources> resources,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, dims, faiss::METRIC_INNER_PRODUCT, config) {}

void RaftIndexFlatIP::copyFrom(faiss::IndexFlat* index) {
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a RaftIndexFlatIP from an index of "
            "different metric_type");

    GpuIndexFlat::copyFrom(index);
}

void RaftIndexFlatIP::copyTo(faiss::IndexFlat* index) {
    // The passed in index must be IP
    FAISS_THROW_IF_NOT_MSG(
            index->metric_type == metric_type,
            "Cannot copy a RaftIndexFlatIP to an index of "
            "different metric_type");

    GpuIndexFlat::copyTo(index);
}

} // namespace gpu
} // namespace faiss
