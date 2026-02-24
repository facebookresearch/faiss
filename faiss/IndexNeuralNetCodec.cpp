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
        int d_in,
        int M_in,
        int nbits_in,
        MetricType metric)
        : IndexFlatCodes((M_in * nbits_in + 7) / 8, d_in, metric),
          M(M_in),
          nbits(nbits_in) {
    is_trained = false;
}

void IndexNeuralNetCodec::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_MSG("Training not implemented in C++, use Pytorch");
}

void IndexNeuralNetCodec::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    nn::Tensor2D x_tensor(n, d, x);
    nn::Int32Tensor2D codes_tensor = net->encode(x_tensor);
    pack_bitstrings(n, M, nbits, codes_tensor.data(), bytes, code_size);
}

void IndexNeuralNetCodec::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    nn::Int32Tensor2D codes_tensor(n, M);
    unpack_bitstrings(n, M, nbits, bytes, code_size, codes_tensor.data());
    nn::Tensor2D x_tensor = net->decode(codes_tensor);
    memcpy(x, x_tensor.data(), d * n * sizeof(float));
}

/*********************************************************
 * IndexQINeuralNetCodec implementation
 *********************************************************/

IndexQINCo::IndexQINCo(
        int d_in,
        int M_in,
        int nbits_in,
        int L,
        int h,
        MetricType metric)
        : IndexNeuralNetCodec(d_in, M_in, nbits_in, metric),
          qinco(d_in, 1 << nbits_in, L, M_in, h) {
    net = &qinco;
}

} // namespace faiss
