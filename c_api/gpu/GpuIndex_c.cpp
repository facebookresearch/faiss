/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "GpuIndex_c.h"
#include "GpuAutoTune_c.h"
#include "macros_impl.h"
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/FaissAssert.h>
#include <vector>
#include <cmath>

using faiss::gpu::GpuIndexConfig;

DEFINE_GETTER(GpuIndexConfig, int, device)

int faiss_index_gpu_to_cpu_new(const FaissIndex* gpu_index, FaissIndex** p_out) {
    int result = faiss_index_gpu_to_cpu(gpu_index, p_out);
    return result;
}

int faiss_index_cpu_to_gpu_new(
        FaissGpuResourcesProvider* provider,
        int device,
        const FaissIndex* index,
        FaissGpuIndex** p_out) {
    return faiss_index_cpu_to_gpu(provider, device, index, p_out);
}

int faiss_GpuIndexCagra_new(
        FaissIndex** p_index,
        FaissStandardGpuResources* res,
        int d,
        FaissMetricType metric,
        size_t graph_degree) {
    try {
        faiss::gpu::GpuIndexCagraConfig config;
        config.graph_degree = graph_degree;

        auto gpu_res =
                reinterpret_cast<faiss::gpu::StandardGpuResources*>(res);

        auto cagra_index = new faiss::gpu::GpuIndexCagra(
                gpu_res,
                d,
                static_cast<faiss::MetricType>(metric),
                config);

        *p_index = reinterpret_cast<FaissIndex*>(cagra_index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_SearchParametersCagra_new(
        FaissSearchParameters** p_params,
        size_t itopk_size) {
    try {
        auto cagra_params = new faiss::gpu::SearchParametersCagra();
        cagra_params->itopk_size = itopk_size;

        *p_params = reinterpret_cast<FaissSearchParameters*>(cagra_params);
        return 0;
    }
    CATCH_AND_HANDLE
}
