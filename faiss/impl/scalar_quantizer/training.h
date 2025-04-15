/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*******************************************************************
 * Quantizer range training
 */

#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

namespace scalar_quantizer {

using RangeStat = ScalarQuantizer::RangeStat;

void train_Uniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int k,
        const float* x,
        std::vector<float>& trained);

void train_NonUniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int d,
        int k,
        const float* x,
        std::vector<float>& trained);
} // namespace scalar_quantizer

} // namespace faiss
