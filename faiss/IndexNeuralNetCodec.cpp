/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuralNetCodec.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>

namespace faiss {

/*********************************************************
 * IndexNeuralNetCodec implementation
 *********************************************************/

IndexNeuralNetCodec::IndexNeuralNetCodec(
        int d,
        int M,
        int nbits,
        MetricType metric)
        : IndexFlatCodes((M * nbits + 7) / 8, d, metric), M(M), nbits(nbits) {
    is_trained = false;
}

void IndexNeuralNetCodec::train(idx_t n, const float* x) {
    FAISS_THROW_MSG("Training not implemented in C++, use Pytorch");
}

void IndexNeuralNetCodec::sa_encode(idx_t n, const float* x, uint8_t* codes)
        const {
    nn::Tensor2D x_tensor(n, d, x);
    nn::Int32Tensor2D codes_tensor = net->encode(x_tensor);
    pack_bitstrings(n, M, nbits, codes_tensor.data(), codes, code_size);
}

void IndexNeuralNetCodec::sa_decode(idx_t n, const uint8_t* codes, float* x)
        const {
    nn::Int32Tensor2D codes_tensor(n, M);
    unpack_bitstrings(n, M, nbits, codes, code_size, codes_tensor.data());
    nn::Tensor2D x_tensor = net->decode(codes_tensor);
    memcpy(x, x_tensor.data(), d * n * sizeof(float));
}

/*********************************************************
 * IndexQINeuralNetCodec implementation
 *********************************************************/

IndexQINCo::IndexQINCo(int d, int M, int nbits, int L, int h, MetricType metric)
        : IndexNeuralNetCodec(d, M, nbits, metric),
          qinco(d, 1 << nbits, L, M, h) {
    net = &qinco;
}

} // namespace faiss
