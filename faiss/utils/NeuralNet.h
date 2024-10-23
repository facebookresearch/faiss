/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** Implements a few neural net layers, mainly to support QINCo */

#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

namespace faiss {

// the names are based on the Pytorch names (more or less)
namespace nn {

// container for intermediate steps of the neural net
template <typename T>
struct Tensor2DTemplate {
    size_t shape[2];
    std::vector<T> v;

    Tensor2DTemplate(size_t n0, size_t n1, const T* data = nullptr);

    Tensor2DTemplate& operator+=(const Tensor2DTemplate&);

    /// get column #j as a 1-column Tensor2D
    Tensor2DTemplate column(size_t j) const;

    size_t numel() const {
        return shape[0] * shape[1];
    }
    T* data() {
        return v.data();
    }
    const T* data() const {
        return v.data();
    }
};

using Tensor2D = Tensor2DTemplate<float>;
using Int32Tensor2D = Tensor2DTemplate<int32_t>;

/// minimal translation of nn.Linear
struct Linear {
    size_t in_features, out_features;
    std::vector<float> weight;
    std::vector<float> bias;

    Linear(size_t in_features, size_t out_features, bool bias = true);

    Tensor2D operator()(const Tensor2D& x) const;
};

/// minimal translation of nn.Embedding
struct Embedding {
    size_t num_embeddings, embedding_dim;
    std::vector<float> weight;

    Embedding(size_t num_embeddings, size_t embedding_dim);

    Tensor2D operator()(const Int32Tensor2D&) const;

    float* data() {
        return weight.data();
    }

    const float* data() const {
        return weight.data();
    }
};

/// Feed forward layer that expands to a hidden dimension, applies a ReLU non
/// linearity and maps back to the orignal dimension
struct FFN {
    Linear linear1, linear2;

    FFN(int d, int h) : linear1(d, h, false), linear2(h, d, false) {}

    Tensor2D operator()(const Tensor2D& x) const;
};

} // namespace nn

// Translation of the QINCo implementation from
// https://github.com/facebookresearch/Qinco/blob/main/model_qinco.py

struct QINCoStep {
    /// d: input dim, K: codebook size, L: # of residual blocks, h: hidden dim
    int d, K, L, h;

    QINCoStep(int d, int K, int L, int h);

    nn::Embedding codebook;
    nn::Linear MLPconcat;
    std::vector<nn::FFN> residual_blocks;

    nn::FFN& get_residual_block(int i) {
        return residual_blocks[i];
    }

    /** encode a set of vectors x with intial estimate xhat. Optionally return
     * the delta to be added to xhat to form the new xhat */
    nn::Int32Tensor2D encode(
            const nn::Tensor2D& xhat,
            const nn::Tensor2D& x,
            nn::Tensor2D* residuals = nullptr) const;

    nn::Tensor2D decode(
            const nn::Tensor2D& xhat,
            const nn::Int32Tensor2D& codes) const;
};

struct NeuralNetCodec {
    int d, M;

    NeuralNetCodec(int d, int M) : d(d), M(M) {}

    virtual nn::Tensor2D decode(const nn::Int32Tensor2D& codes) const = 0;
    virtual nn::Int32Tensor2D encode(const nn::Tensor2D& x) const = 0;

    virtual ~NeuralNetCodec() {}
};

struct QINCo : NeuralNetCodec {
    int K, L, h;
    nn::Embedding codebook0;
    std::vector<QINCoStep> steps;

    QINCo(int d, int K, int L, int M, int h);

    QINCoStep& get_step(int i) {
        return steps[i];
    }

    nn::Tensor2D decode(const nn::Int32Tensor2D& codes) const override;

    nn::Int32Tensor2D encode(const nn::Tensor2D& x) const override;

    virtual ~QINCo() {}
};

} // namespace faiss
