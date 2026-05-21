/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file panorama_kernels-inl.h
 * @brief Private header for Panorama kernel SIMD implementations.
 *
 * This is a PRIVATE header — do not include in public APIs or user code.
 * Only faiss internal .cpp files (the per-SIMD implementation files and
 * panorama_kernels-generic.cpp) should include this header.
 *
 * This header re-exports the public API (panorama_kernels.h) plus the
 * simd_dispatch.h machinery needed by the implementation files.
 */

#include <faiss/impl/panorama_kernels/panorama_kernels.h>
#include <faiss/impl/simd_dispatch.h>
