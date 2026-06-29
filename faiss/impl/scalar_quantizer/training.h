/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/*******************************************************************
 * Quantizer range training for the scalar quantizer. This is independent of the
 * searching code and needs not to be very optimized (scalar quantizer training
 * is very efficient).
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

/** Build the TurboQuant MSE codebook using the beta-distribution-optimal
 *  quantizer from the TurboQuant paper. The codebook is analytical
 *  (depends only on d and nbits, no training data needed).
 *
 *  @param d         vector dimensionality (used for beta-distribution shape)
 *  @param nbits     bits per component (1-8)
 *  @param trained   output: [centroids (k floats), boundaries (k-1 floats)]
 *                   where k = 2^nbits
 */
void train_TurboQuantMSE(size_t d, size_t nbits, std::vector<float>& trained);

} // namespace scalar_quantizer

} // namespace faiss
