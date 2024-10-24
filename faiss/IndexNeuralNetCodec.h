/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/utils/NeuralNet.h>

namespace faiss {

struct IndexNeuralNetCodec : IndexFlatCodes {
    NeuralNetCodec* net = nullptr;
    size_t M, nbits;

    explicit IndexNeuralNetCodec(
            int d = 0,
            int M = 0,
            int nbits = 0,
            MetricType metric = METRIC_L2);

    void train(idx_t n, const float* x) override;

    void sa_encode(idx_t n, const float* x, uint8_t* codes) const override;
    void sa_decode(idx_t n, const uint8_t* codes, float* x) const override;

    ~IndexNeuralNetCodec() {}
};

struct IndexQINCo : IndexNeuralNetCodec {
    QINCo qinco;

    IndexQINCo(
            int d,
            int M,
            int nbits,
            int L,
            int h,
            MetricType metric = METRIC_L2);

    ~IndexQINCo() {}
};

} // namespace faiss
