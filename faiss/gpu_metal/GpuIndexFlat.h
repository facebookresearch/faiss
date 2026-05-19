// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Unified name for flat GPU index when Metal backend is built.
 * Include this when using the Metal backend for API parity with
 * faiss::gpu::GpuIndexFlat.
 */

#pragma once

#include <faiss/gpu_metal/MetalIndexFlat.h>

namespace faiss {

/// When FAISS is built with Metal backend, GpuIndexFlat is MetalIndexFlat.
using GpuIndexFlat = gpu_metal::MetalIndexFlat;

} // namespace faiss
