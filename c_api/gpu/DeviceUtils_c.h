/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_DEVICE_UTILS_C_H
#define FAISS_DEVICE_UTILS_C_H

#include <cublas.h>
#include <cuda_runtime_api.h>
#include "../faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the number of available GPU devices
int faiss_get_num_gpus(int* p_output);

/// Starts the CUDA profiler (exposed via SWIG)
int faiss_gpu_profiler_start();

/// Stops the CUDA profiler (exposed via SWIG)
int faiss_gpu_profiler_stop();

/// Synchronizes the CPU against all devices (equivalent to
/// cudaDeviceSynchronize for each device)
int faiss_gpu_sync_all_devices();

#ifdef __cplusplus
}
#endif
#endif
