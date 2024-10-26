/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/AutoTune.h>
#include <faiss/Index.h>

namespace faiss {
namespace gpu {

/// parameter space and setters for GPU indexes
struct GpuParameterSpace : faiss::ParameterSpace {
    /// initialize with reasonable parameters for the index
    void initialize(const faiss::Index* index) override;

    /// set a combination of parameters on an index
    void set_index_parameter(
            faiss::Index* index,
            const std::string& name,
            double val) const override;
};

} // namespace gpu
} // namespace faiss
