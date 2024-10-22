/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "DeviceUtils_c.h"
#include <faiss/gpu/utils/DeviceUtils.h>
#include "macros_impl.h"

/// Returns the number of available GPU devices
int faiss_get_num_gpus(int* p_output) {
    try {
        int output = faiss::gpu::getNumDevices();
        *p_output = output;
    }
    CATCH_AND_HANDLE
}

/// Starts the CUDA profiler (exposed via SWIG)
int faiss_gpu_profiler_start() {
    try {
        faiss::gpu::profilerStart();
    }
    CATCH_AND_HANDLE
}

/// Stops the CUDA profiler (exposed via SWIG)
int faiss_gpu_profiler_stop() {
    try {
        faiss::gpu::profilerStop();
    }
    CATCH_AND_HANDLE
}

/// Synchronizes the CPU against all devices (equivalent to
/// cudaDeviceSynchronize for each device)
int faiss_gpu_sync_all_devices() {
    try {
        faiss::gpu::synchronizeAllDevices();
    }
    CATCH_AND_HANDLE
}
