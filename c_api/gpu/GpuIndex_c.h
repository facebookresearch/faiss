/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_GPU_INDEX_C_H
#define FAISS_GPU_INDEX_C_H

#include "../Index_c.h"
#include "StandardGpuResources_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS(GpuIndexConfig)

FAISS_DECLARE_GETTER(GpuIndexConfig, int, device)

FAISS_DECLARE_CLASS_INHERITED(GpuIndex, Index)

FAISS_DECLARE_CLASS(SearchParameters)

int faiss_GpuIndexCagra_new(
        FaissIndex** p_index,
        FaissStandardGpuResources* res,
        int d,
        FaissMetricType metric,
        size_t graph_degree);

int faiss_SearchParametersCagra_new(
        FaissSearchParameters** p_params,
        size_t itopk_size);

int faiss_index_gpu_to_cpu_new(const FaissIndex* gpu_index, FaissIndex** p_out);

int faiss_index_cpu_to_gpu_new(
        FaissGpuResourcesProvider* provider,
        int device,
        const FaissIndex* index,
        FaissGpuIndex** p_out);

#ifdef __cplusplus
}
#endif

#endif
