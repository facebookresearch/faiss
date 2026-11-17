/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/DistanceComputer.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

void DistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    // compute first, assign next
    const float d0 = this->operator()(idx0);
    const float d1 = this->operator()(idx1);
    const float d2 = this->operator()(idx2);
    const float d3 = this->operator()(idx3);
    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

float FlatCodesDistanceComputer::partial_dot_product(
        const idx_t /* i */,
        const uint32_t /* offset */,
        const uint32_t /* num_components */) {
    FAISS_THROW_MSG("partial_dot_product not implemented");
}

void FlatCodesDistanceComputer::partial_dot_product_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dp0,
        float& dp1,
        float& dp2,
        float& dp3,
        const uint32_t offset,
        const uint32_t num_components) {
    // default implementation for correctness
    const float d0 = this->partial_dot_product(idx0, offset, num_components);
    const float d1 = this->partial_dot_product(idx1, offset, num_components);
    const float d2 = this->partial_dot_product(idx2, offset, num_components);
    const float d3 = this->partial_dot_product(idx3, offset, num_components);
    dp0 = d0;
    dp1 = d1;
    dp2 = d2;
    dp3 = d3;
}

} // namespace faiss
