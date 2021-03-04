/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "GpuAutoTune_c.h"
#include <faiss/Index.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuResources.h>
#include <vector>
#include "GpuClonerOptions_c.h"
#include "macros_impl.h"

using faiss::Index;
using faiss::gpu::GpuClonerOptions;
using faiss::gpu::GpuMultipleClonerOptions;
using faiss::gpu::GpuResourcesProvider;

int faiss_index_gpu_to_cpu(const FaissIndex* gpu_index, FaissIndex** p_out) {
    try {
        auto cpu_index = faiss::gpu::index_gpu_to_cpu(
                reinterpret_cast<const Index*>(gpu_index));
        *p_out = reinterpret_cast<FaissIndex*>(cpu_index);
    }
    CATCH_AND_HANDLE
}

/// converts any CPU index that can be converted to GPU
int faiss_index_cpu_to_gpu(
        FaissGpuResourcesProvider* provider,
        int device,
        const FaissIndex* index,
        FaissGpuIndex** p_out) {
    try {
        auto res = reinterpret_cast<GpuResourcesProvider*>(provider);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(
                res, device, reinterpret_cast<const Index*>(index));
        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
    }
    CATCH_AND_HANDLE
}

int faiss_index_cpu_to_gpu_with_options(
        FaissGpuResourcesProvider* provider,
        int device,
        const FaissIndex* index,
        const FaissGpuClonerOptions* options,
        FaissGpuIndex** p_out) {
    try {
        auto res = reinterpret_cast<GpuResourcesProvider*>(provider);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(
                res,
                device,
                reinterpret_cast<const Index*>(index),
                reinterpret_cast<const GpuClonerOptions*>(options));
        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
    }
    CATCH_AND_HANDLE
}

int faiss_index_cpu_to_gpu_multiple(
        FaissGpuResourcesProvider* const* providers_vec,
        const int* devices,
        size_t devices_size,
        const FaissIndex* index,
        FaissGpuIndex** p_out) {
    try {
        std::vector<GpuResourcesProvider*> res(devices_size);
        for (auto i = 0u; i < devices_size; ++i) {
            res[i] = reinterpret_cast<GpuResourcesProvider*>(providers_vec[i]);
        }

        std::vector<int> dev(devices, devices + devices_size);

        auto gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(
                res, dev, reinterpret_cast<const Index*>(index));
        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
    }
    CATCH_AND_HANDLE
}

int faiss_index_cpu_to_gpu_multiple_with_options(
        FaissGpuResourcesProvider** providers_vec,
        size_t providers_vec_size,
        int* devices,
        size_t devices_size,
        const FaissIndex* index,
        const FaissGpuMultipleClonerOptions* options,
        FaissGpuIndex** p_out) {
    try {
        std::vector<GpuResourcesProvider*> res(providers_vec_size);
        for (auto i = 0u; i < providers_vec_size; ++i) {
            res[i] = reinterpret_cast<GpuResourcesProvider*>(providers_vec[i]);
        }

        std::vector<int> dev(devices, devices + devices_size);

        auto gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(
                res,
                dev,
                reinterpret_cast<const Index*>(index),
                reinterpret_cast<const GpuMultipleClonerOptions*>(options));
        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
    }
    CATCH_AND_HANDLE
}
