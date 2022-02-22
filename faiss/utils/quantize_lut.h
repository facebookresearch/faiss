/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdio>

namespace faiss {

/** Functions to quantize PQ floating-point Look Up Tables (LUT) to uint8, and
 * biases to uint16. The accumulation is supposed to take place in uint16.
 * The quantization coefficients are float (a, b) such that
 *
 *      original_value = quantized_value * a / b
 *
 * The hardest part of the quantization is with multiple LUTs that need to be
 * added up together. In that case, coefficient a has to be chosen so that
 * the sum fits in a uint16 accumulator.
 */

namespace quantize_lut {

/* affine quantizer, a and b are the affine coefficients, marginalize over d
 *
 * @param tab input/output, size (n, d)
 */
void round_uint8_per_column(
        float* tab,
        size_t n,
        size_t d,
        float* a_out = nullptr,
        float* b_out = nullptr);

/* affine quantizer, a and b are the affine coefficients
 *
 * @param tab input/output, size (m, n, d)
 */
void round_uint8_per_column_multi(
        float* tab,
        size_t m,
        size_t n,
        size_t d,
        float* a_out = nullptr,
        float* b_out = nullptr);

/** LUT quantization to uint8 and bias to uint16.
 *
 * (nprobe, M, ksub, lut_is_3d) determine the size of the the LUT
 *
 *  LUT input:
 *  - 2D size (M, ksub): single matrix per probe (lut_is_3d=false)
 *  - 3D size (nprobe, M, ksub): separate LUT per probe (lut_is_3d=true)
 *  bias input:
 *  - nullptr: bias is 0
 *  - size (nprobe): one bias per probe
 *  Output:
 *  - LUTq uint8 version of the LUT (M size is rounded up to M2)
 *  - biasq (or nullptr): uint16 version of the LUT
 *  - a, b: scalars to approximate the true distance
 */

void quantize_LUT_and_bias(
        size_t nprobe,
        size_t M,
        size_t ksub,
        bool lut_is_3d,
        const float* LUT,
        const float* bias,
        uint8_t* LUTq,
        size_t M2,
        uint16_t* biasq,
        float* a_out = nullptr,
        float* b_out = nullptr);

void aq_quantize_LUT_and_bias(
        size_t nprobe,
        size_t M,
        size_t ksub,
        const float* LUT,
        const float* bias,
        size_t M_norm,
        int norm_scale,
        uint8_t* LUTq,
        size_t M2,
        uint16_t* biasq,
        float* a_out,
        float* b_out);

float aq_estimate_norm_scale(
        size_t M,
        size_t ksub,
        size_t M_norm,
        const float* LUT);

} // namespace quantize_lut

} // namespace faiss
