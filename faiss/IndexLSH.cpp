/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexLSH.h>

#include <cstdio>
#include <cstring>

#include <algorithm>
#include <memory>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

/***************************************************************
 * IndexLSH
 ***************************************************************/

IndexLSH::IndexLSH(idx_t d, int nbits, bool rotate_data, bool train_thresholds)
        : IndexFlatCodes((nbits + 7) / 8, d),
          nbits(nbits),
          rotate_data(rotate_data),
          train_thresholds(train_thresholds),
          rrot(d, nbits) {
    is_trained = !train_thresholds;

    if (rotate_data) {
        rrot.init(5);
    } else {
        FAISS_THROW_IF_NOT(d >= nbits);
    }
}

IndexLSH::IndexLSH() : nbits(0), rotate_data(false), train_thresholds(false) {}

const float* IndexLSH::apply_preprocess(idx_t n, const float* x) const {
    float* xt = nullptr;
    if (rotate_data) {
        // also applies bias if exists
        xt = rrot.apply(n, x);
    } else if (d != nbits) {
        assert(nbits < d);
        xt = new float[nbits * n];
        float* xp = xt;
        for (idx_t i = 0; i < n; i++) {
            const float* xl = x + i * d;
            for (int j = 0; j < nbits; j++)
                *xp++ = xl[j];
        }
    }

    if (train_thresholds) {
        if (xt == nullptr) {
            xt = new float[nbits * n];
            memcpy(xt, x, sizeof(*x) * n * nbits);
        }

        float* xp = xt;
        for (idx_t i = 0; i < n; i++)
            for (int j = 0; j < nbits; j++)
                *xp++ -= thresholds[j];
    }

    return xt ? xt : x;
}

void IndexLSH::train(idx_t n, const float* x) {
    if (train_thresholds) {
        thresholds.resize(nbits);
        train_thresholds = false;
        const float* xt = apply_preprocess(n, x);
        std::unique_ptr<const float[]> del(xt == x ? nullptr : xt);
        train_thresholds = true;

        std::unique_ptr<float[]> transposed_x(new float[n * nbits]);

        for (idx_t i = 0; i < n; i++)
            for (idx_t j = 0; j < nbits; j++)
                transposed_x[j * n + i] = xt[i * nbits + j];

        for (idx_t i = 0; i < nbits; i++) {
            float* xi = transposed_x.get() + i * n;
            // std::nth_element
            std::sort(xi, xi + n);
            if (n % 2 == 1)
                thresholds[i] = xi[n / 2];
            else
                thresholds[i] = (xi[n / 2 - 1] + xi[n / 2]) / 2;
        }
    }
    is_trained = true;
}

void IndexLSH::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    const float* xt = apply_preprocess(n, x);
    std::unique_ptr<const float[]> del(xt == x ? nullptr : xt);

    std::unique_ptr<uint8_t[]> qcodes(new uint8_t[n * code_size]);

    fvecs2bitvecs(xt, qcodes.get(), nbits, n);

    std::unique_ptr<int[]> idistances(new int[n * k]);

    int_maxheap_array_t res = {size_t(n), size_t(k), labels, idistances.get()};

    hammings_knn_hc(&res, qcodes.get(), codes.data(), ntotal, code_size, true);

    // convert distances to floats
    for (int i = 0; i < k * n; i++)
        distances[i] = idistances[i];
}

void IndexLSH::transfer_thresholds(LinearTransform* vt) {
    if (!train_thresholds)
        return;
    FAISS_THROW_IF_NOT(nbits == vt->d_out);
    if (!vt->have_bias) {
        vt->b.resize(nbits, 0);
        vt->have_bias = true;
    }
    for (int i = 0; i < nbits; i++)
        vt->b[i] -= thresholds[i];
    train_thresholds = false;
    thresholds.clear();
}

void IndexLSH::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    const float* xt = apply_preprocess(n, x);
    std::unique_ptr<const float[]> del(xt == x ? nullptr : xt);
    fvecs2bitvecs(xt, bytes, nbits, n);
}

void IndexLSH::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    float* xt = x;
    std::unique_ptr<float[]> del;
    if (rotate_data || nbits != d) {
        xt = new float[n * nbits];
        del.reset(xt);
    }
    bitvecs2fvecs(bytes, xt, nbits, n);

    if (train_thresholds) {
        float* xp = xt;
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < nbits; j++) {
                *xp++ += thresholds[j];
            }
        }
    }

    if (rotate_data) {
        rrot.reverse_transform(n, xt, x);
    } else if (nbits != d) {
        for (idx_t i = 0; i < n; i++) {
            memcpy(x + i * d, xt + i * nbits, nbits * sizeof(xt[0]));
        }
    }
}

} // namespace faiss
