/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "GpuAutoTune_c.h"
#include "macros_impl.h"
#include "Index.h"
#include "gpu/GpuAutoTune.h"
#include "gpu/GpuClonerOptions.h"

using faiss::Index;
using faiss::gpu::GpuResources;
using faiss::gpu::GpuClonerOptions;

int faiss_index_gpu_to_cpu(const FaissIndex* gpu_index, FaissIndex** p_out) {
    try {
        auto cpu_index = faiss::gpu::index_gpu_to_cpu(
            reinterpret_cast<const Index*>(gpu_index)
        );
        *p_out = reinterpret_cast<FaissIndex*>(cpu_index);
    } CATCH_AND_HANDLE
}

/// converts any CPU index that can be converted to GPU
int faiss_index_cpu_to_gpu(FaissGpuResources* resources, int device, const FaissIndex *index, FaissGpuIndex** p_out) {
    try {
        auto res = reinterpret_cast<GpuResources*>(resources);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(
            res, device, reinterpret_cast<const Index*>(index)
        );
        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
    } CATCH_AND_HANDLE
}

int faiss_index_cpu_to_gpu_with_options(
    FaissGpuResources* resources, int device,
    const FaissIndex *index, const FaissGpuClonerOptions* options,
    FaissGpuIndex** p_out)
{
    try {
        auto res = reinterpret_cast<GpuResources*>(resources);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(
            res, device, reinterpret_cast<const Index*>(index),
            reinterpret_cast<const GpuClonerOptions*>(options));
        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
    } CATCH_AND_HANDLE
}
