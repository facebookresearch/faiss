/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/NeuralNet.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

extern "C" {

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

namespace nn {

/*************************************************************
 * Tensor2D implementation
 *************************************************************/

template <typename T>
Tensor2DTemplate<T>::Tensor2DTemplate(size_t n0, size_t n1, const T* data_in)
        : shape{n0, n1}, v(n0 * n1) {
    if (data_in) {
        memcpy(data(), data_in, n0 * n1 * sizeof(T));
    }
}

template <typename T>
Tensor2DTemplate<T>& Tensor2DTemplate<T>::operator+=(
        const Tensor2DTemplate<T>& other) {
    FAISS_THROW_IF_NOT(shape[0] == other.shape[0]);
    FAISS_THROW_IF_NOT(shape[1] == other.shape[1]);
    for (size_t i = 0; i < numel(); i++) {
        v[i] += other.v[i];
    }
    return *this;
}

template <typename T>
Tensor2DTemplate<T> Tensor2DTemplate<T>::column(size_t j) const {
    size_t n = shape[0], d = shape[1];
    Tensor2DTemplate<T> out(n, 1);
    for (size_t i = 0; i < n; i++) {
        out.v[i] = v[i * d + j];
    }
    return out;
}

// explicit template instanciation
template struct Tensor2DTemplate<float>;
template struct Tensor2DTemplate<int32_t>;

/*************************************************************
 * Layers implementation
 *************************************************************/

Linear::Linear(size_t in_features, size_t out_features, bool bias)
        : in_features(in_features),
          out_features(out_features),
          weight(in_features * out_features) {
    if (bias) {
        this->bias.resize(out_features);
    }
}

Tensor2D Linear::operator()(const Tensor2D& x) const {
    FAISS_THROW_IF_NOT(x.shape[1] == in_features);
    size_t n = x.shape[0];
    Tensor2D output(n, out_features);

    float one = 1, zero = 0;
    FINTEGER nbiti = out_features, ni = n, di = in_features;

    sgemm_("Transposed",
           "Not transposed",
           &nbiti,
           &ni,
           &di,
           &one,
           weight.data(),
           &di,
           x.data(),
           &di,
           &zero,
           output.data(),
           &nbiti);

    if (bias.size() > 0) {
        FAISS_THROW_IF_NOT(bias.size() == out_features);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < out_features; j++) {
                output.v[i * out_features + j] += bias[j];
            }
        }
    }

    return output;
}

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
        : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    weight.resize(num_embeddings * embedding_dim);
}

Tensor2D Embedding::operator()(const Int32Tensor2D& code) const {
    FAISS_THROW_IF_NOT(code.shape[1] == 1);
    size_t n = code.shape[0];
    Tensor2D output(n, embedding_dim);
    for (size_t i = 0; i < n; ++i) {
        size_t ci = code.v[i];
        FAISS_THROW_IF_NOT(ci < num_embeddings);
        memcpy(output.data() + i * embedding_dim,
               weight.data() + ci * embedding_dim,
               sizeof(float) * embedding_dim);
    }
    return output; // TODO figure out how std::move works
}

namespace {

void inplace_relu(Tensor2D& x) {
    for (size_t i = 0; i < x.numel(); i++) {
        x.v[i] = std::max(0.0f, x.v[i]);
    }
}

Tensor2D concatenate_rows(const Tensor2D& x, const Tensor2D& y) {
    size_t n = x.shape[0], d1 = x.shape[1], d2 = y.shape[1];
    FAISS_THROW_IF_NOT(n == y.shape[0]);
    Tensor2D out(n, d1 + d2);
    for (size_t i = 0; i < n; i++) {
        memcpy(out.data() + i * (d1 + d2),
               x.data() + i * d1,
               sizeof(float) * d1);
        memcpy(out.data() + i * (d1 + d2) + d1,
               y.data() + i * d2,
               sizeof(float) * d2);
    }
    return out;
}

} // anonymous namespace

Tensor2D FFN::operator()(const Tensor2D& x_in) const {
    Tensor2D x = linear1(x_in);
    inplace_relu(x);
    return linear2(x);
}

} // namespace nn

/*************************************************************
 * QINCoStep implementation
 *************************************************************/

using namespace nn;

QINCoStep::QINCoStep(int d, int K, int L, int h)
        : d(d), K(K), L(L), h(h), codebook(K, d), MLPconcat(2 * d, d) {
    for (int i = 0; i < L; i++) {
        residual_blocks.emplace_back(d, h);
    }
}

nn::Tensor2D QINCoStep::decode(
        const nn::Tensor2D& xhat,
        const nn::Int32Tensor2D& codes) const {
    size_t n = xhat.shape[0];
    FAISS_THROW_IF_NOT(n == codes.shape[0]);
    Tensor2D zqs = codebook(codes);
    Tensor2D cc = concatenate_rows(zqs, xhat);
    zqs += MLPconcat(cc);
    for (int i = 0; i < L; i++) {
        zqs += residual_blocks[i](zqs);
    }
    return zqs;
}

nn::Int32Tensor2D QINCoStep::encode(
        const nn::Tensor2D& xhat,
        const nn::Tensor2D& x,
        nn::Tensor2D* residuals) const {
    size_t n = xhat.shape[0];
    FAISS_THROW_IF_NOT(
            n == x.shape[0] && xhat.shape[1] == d && x.shape[1] == d);

    // repeated codebook
    Tensor2D zqs_r(n * K, d);  // size n, K, d
    Tensor2D cc(n * K, d * 2); // size n, K, d * 2
    size_t d_2 = this->d;

    auto copy_row = [d_2](Tensor2D& t, size_t i, size_t j, const float* data) {
        assert(i <= t.shape[0] && j <= t.shape[1]);
        memcpy(t.data() + i * t.shape[1] + j, data, sizeof(float) * d_2);
    };

    // manual broadcasting
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < K; j++) {
            copy_row(zqs_r, i * K + j, 0, codebook.data() + j * d_2);
            copy_row(cc, i * K + j, 0, codebook.data() + j * d_2);
            copy_row(cc, i * K + j, d_2, xhat.data() + i * d_2);
        }
    }

    zqs_r += MLPconcat(cc);

    // residual blocks
    for (int i = 0; i < L; i++) {
        zqs_r += residual_blocks[i](zqs_r);
    }

    // add the xhat
    for (size_t i = 0; i < n; i++) {
        float* zqs_r_row = zqs_r.data() + i * K * d_2;
        const float* xhat_row = xhat.data() + i * d_2;
        for (size_t l = 0; l < K; l++) {
            for (size_t j = 0; j < d_2; j++) {
                zqs_r_row[j] += xhat_row[j];
            }
            zqs_r_row += d_2;
        }
    }

    // perform assignment, finding the nearest
    nn::Int32Tensor2D codes(n, 1);
    float* res = nullptr;
    if (residuals) {
        FAISS_THROW_IF_NOT(
                residuals->shape[0] == n && residuals->shape[1] == d_2);
        res = residuals->data();
    }

    for (size_t i = 0; i < n; i++) {
        const float* q = x.data() + i * d_2;
        const float* db = zqs_r.data() + i * K * d_2;
        float dis_min = HUGE_VALF;
        int64_t idx = -1;
        for (size_t j = 0; j < K; j++) {
            float dis = fvec_L2sqr(q, db, d_2);
            if (dis < dis_min) {
                dis_min = dis;
                idx = j;
            }
            db += d_2;
        }
        codes.v[i] = idx;
        if (res) {
            const float* xhat_row = xhat.data() + i * d_2;
            const float* xhat_next_row = zqs_r.data() + (i * K + idx) * d_2;
            for (size_t j = 0; j < d_2; j++) {
                res[j] = xhat_next_row[j] - xhat_row[j];
            }
            res += d_2;
        }
    }
    return codes;
}

/*************************************************************
 * QINCo implementation
 *************************************************************/

QINCo::QINCo(int d, int K, int L, int M, int h)
        : NeuralNetCodec(d, M), K(K), L(L), h(h), codebook0(K, d) {
    for (int i = 1; i < M; i++) {
        steps.emplace_back(d, K, L, h);
    }
}

nn::Tensor2D QINCo::decode(const nn::Int32Tensor2D& codes) const {
    FAISS_THROW_IF_NOT(codes.shape[1] == M);
    Tensor2D xhat = codebook0(codes.column(0));
    for (int i = 1; i < M; i++) {
        xhat += steps[i - 1].decode(xhat, codes.column(i));
    }
    return xhat;
}

nn::Int32Tensor2D QINCo::encode(const nn::Tensor2D& x) const {
    FAISS_THROW_IF_NOT(x.shape[1] == d);
    size_t n = x.shape[0];
    Int32Tensor2D codes(n, M);
    Tensor2D xhat(n, d);
    {
        // assign to first codebook as a batch
        std::vector<float> dis(n);
        std::vector<int64_t> codes64(n);
        knn_L2sqr(
                x.data(),
                codebook0.data(),
                d,
                n,
                K,
                1,
                dis.data(),
                codes64.data());
        for (size_t i = 0; i < n; i++) {
            codes.v[i * M] = codes64[i];
            memcpy(xhat.data() + i * d,
                   codebook0.data() + codes64[i] * d,
                   sizeof(float) * d);
        }
    }

    Tensor2D toadd(n, d);
    for (int i = 1; i < M; i++) {
        Int32Tensor2D ci = steps[i - 1].encode(xhat, x, &toadd);
        for (size_t j = 0; j < n; j++) {
            codes.v[j * M + i] = ci.v[j];
        }
        xhat += toadd;
    }
    return codes;
}

} // namespace faiss
