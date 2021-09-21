/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "GpuResources_c.h"
#include <faiss/gpu/GpuResources.h>
#include "macros_impl.h"

using faiss::gpu::GpuResources;
using faiss::gpu::GpuResourcesProvider;

DEFINE_DESTRUCTOR(GpuResources)

int faiss_GpuResources_initializeForDevice(FaissGpuResources* res, int device) {
    try {
        reinterpret_cast<GpuResources*>(res)->initializeForDevice(device);
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_getBlasHandle(
        FaissGpuResources* res,
        int device,
        cublasHandle_t* out) {
    try {
        auto o = reinterpret_cast<GpuResources*>(res)->getBlasHandle(device);
        *out = o;
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_getDefaultStream(
        FaissGpuResources* res,
        int device,
        cudaStream_t* out) {
    try {
        auto o = reinterpret_cast<GpuResources*>(res)->getDefaultStream(device);
        *out = o;
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_getPinnedMemory(
        FaissGpuResources* res,
        void** p_buffer,
        size_t* p_size) {
    try {
        auto o = reinterpret_cast<GpuResources*>(res)->getPinnedMemory();
        *p_buffer = o.first;
        *p_size = o.second;
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_getAsyncCopyStream(
        FaissGpuResources* res,
        int device,
        cudaStream_t* out) {
    try {
        auto o = reinterpret_cast<GpuResources*>(res)->getAsyncCopyStream(
                device);
        *out = o;
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_getBlasHandleCurrentDevice(
        FaissGpuResources* res,
        cublasHandle_t* out) {
    try {
        auto o = reinterpret_cast<GpuResources*>(res)
                         ->getBlasHandleCurrentDevice();
        *out = o;
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_getDefaultStreamCurrentDevice(
        FaissGpuResources* res,
        cudaStream_t* out) {
    try {
        auto o = reinterpret_cast<GpuResources*>(res)
                         ->getDefaultStreamCurrentDevice();
        *out = o;
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_syncDefaultStream(FaissGpuResources* res, int device) {
    try {
        reinterpret_cast<GpuResources*>(res)->syncDefaultStream(device);
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_syncDefaultStreamCurrentDevice(FaissGpuResources* res) {
    try {
        reinterpret_cast<GpuResources*>(res)->syncDefaultStreamCurrentDevice();
    }
    CATCH_AND_HANDLE
}

int faiss_GpuResources_getAsyncCopyStreamCurrentDevice(
        FaissGpuResources* res,
        cudaStream_t* out) {
    try {
        auto o = reinterpret_cast<GpuResources*>(res)
                         ->getAsyncCopyStreamCurrentDevice();
        *out = o;
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(GpuResourcesProvider)

int faiss_GpuResourcesProvider_getResources(
        FaissGpuResourcesProvider* res,
        FaissGpuResources** out) {
    try {
        auto o = reinterpret_cast<GpuResourcesProvider*>(res)->getResources();
        *out = reinterpret_cast<FaissGpuResources*>(o.get());
    }
    CATCH_AND_HANDLE
}
