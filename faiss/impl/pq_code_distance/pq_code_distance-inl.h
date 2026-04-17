/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file pq_code_distance-inl.h
 * @brief Private header for PQ code distance SIMD implementations.
 *
 * This is a PRIVATE header — do not include in public APIs or user code.
 * Only faiss internal .cpp files (the per-SIMD implementation files and
 * pq_code_distance-generic.cpp) should include this header.
 *
 * This header re-exports the public API (pq_code_distance.h) plus the
 * simd_dispatch.h machinery needed by the implementation files.
 */

#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/pq_code_distance.h>
