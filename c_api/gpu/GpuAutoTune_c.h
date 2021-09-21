/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_GPU_AUTO_TUNE_C_H
#define FAISS_GPU_AUTO_TUNE_C_H

#include <stddef.h>
#include "../Index_c.h"
#include "../faiss_c.h"
#include "GpuClonerOptions_c.h"
#include "GpuIndex_c.h"
#include "GpuResources_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/// converts any GPU index inside gpu_index to a CPU index
int faiss_index_gpu_to_cpu(const FaissIndex* gpu_index, FaissIndex** p_out);

/// converts any CPU index that can be converted to GPU
int faiss_index_cpu_to_gpu(
        FaissGpuResourcesProvider* provider,
        int device,
        const FaissIndex* index,
        FaissGpuIndex** p_out);

/// converts any CPU index that can be converted to GPU
int faiss_index_cpu_to_gpu_with_options(
        FaissGpuResourcesProvider* provider,
        int device,
        const FaissIndex* index,
        const FaissGpuClonerOptions* options,
        FaissGpuIndex** p_out);

/// converts any CPU index that can be converted to GPU
int faiss_index_cpu_to_gpu_multiple(
        FaissGpuResourcesProvider* const* providers_vec,
        const int* devices,
        size_t devices_size,
        const FaissIndex* index,
        FaissGpuIndex** p_out);

/// converts any CPU index that can be converted to GPU
int faiss_index_cpu_to_gpu_multiple_with_options(
        FaissGpuResourcesProvider* const* providers_vec,
        size_t providers_vec_size,
        const int* devices,
        size_t devices_size,
        const FaissIndex* index,
        const FaissGpuMultipleClonerOptions* options,
        FaissGpuIndex** p_out);

/// parameter space and setters for GPU indexes
FAISS_DECLARE_CLASS_INHERITED(GpuParameterSpace, ParameterSpace)

#ifdef __cplusplus
}
#endif
#endif
