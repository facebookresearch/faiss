/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/Index2Layer.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include <algorithm>

#include <faiss/IndexIVFPQ.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*************************************
 * Index2Layer implementation
 *************************************/

Index2Layer::Index2Layer(
        Index* quantizer,
        size_t nlist,
        int M,
        int nbit,
        MetricType metric)
        : IndexFlatCodes(0, quantizer->d, metric),
          q1(quantizer, nlist),
          pq(quantizer->d, M, nbit) {
    is_trained = false;
    for (int nbyte = 0; nbyte < 7; nbyte++) {
        if (((size_t)1 << (8 * nbyte)) >= nlist) {
            code_size_1 = nbyte;
            break;
        }
    }
    code_size_2 = pq.code_size;
    code_size = code_size_1 + code_size_2;
}

Index2Layer::Index2Layer() {
    code_size = code_size_1 = code_size_2 = 0;
}

Index2Layer::~Index2Layer() = default;

void Index2Layer::train(idx_t n, const float* x) {
    if (verbose) {
        printf("training level-1 quantizer %" PRId64 " vectors in %dD\n", n, d);
    }

    q1.train_q1(n, x, verbose, metric_type);

    if (verbose) {
        printf("computing residuals\n");
    }

    const float* x_in = x;

    x = fvecs_maybe_subsample(
            d,
            (size_t*)&n,
            pq.cp.max_points_per_centroid * pq.ksub,
            x,
            verbose,
            pq.cp.seed);

    std::unique_ptr<const float[]> del_x(x_in == x ? nullptr : x);

    std::vector<idx_t> assign(n); // assignment to coarse centroids
    q1.quantizer->assign(n, x, assign.data());
    std::vector<float> residuals(n * d);
    for (idx_t i = 0; i < n; i++) {
        q1.quantizer->compute_residual(
                x + i * d, residuals.data() + i * d, assign[i]);
    }

    if (verbose) {
        printf("training %zdx%zd product quantizer on %" PRId64
               " vectors in %dD\n",
               pq.M,
               pq.ksub,
               n,
               d);
    }
    pq.verbose = verbose;
    pq.train(n, residuals.data());

    is_trained = true;
}

void Index2Layer::search(
        idx_t /*n*/,
        const float* /*x*/,
        idx_t /*k*/,
        float* /*distances*/,
        idx_t* /*labels*/,
        const SearchParameters* /* params */) const {
    FAISS_THROW_MSG("not implemented");
}

void Index2Layer::transfer_to_IVFPQ(IndexIVFPQ& other) const {
    FAISS_THROW_IF_NOT(other.nlist == q1.nlist);
    FAISS_THROW_IF_NOT(other.code_size == code_size_2);
    FAISS_THROW_IF_NOT(other.ntotal == 0);

    const uint8_t* rp = codes.data();

    for (idx_t i = 0; i < ntotal; i++) {
        idx_t key = 0;
        memcpy(&key, rp, code_size_1);
        rp += code_size_1;
        other.invlists->add_entry(key, i, rp);
        rp += code_size_2;
    }

    other.ntotal = ntotal;
}

namespace {

struct Distance2Level : DistanceComputer {
    size_t d;
    const Index2Layer& storage;
    std::vector<float> buf;
    const float* q = nullptr;

    const float* pq_l1_tab = nullptr;
    const float* pq_l2_tab = nullptr;

    explicit Distance2Level(const Index2Layer& storage_) : storage(storage_) {
        d = storage_.d;
        FAISS_ASSERT(storage.pq.dsub == 4);
        pq_l2_tab = storage.pq.centroids.data();
        buf.resize(2 * d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        storage.reconstruct(i, buf.data());
        storage.reconstruct(j, buf.data() + d);
        return fvec_L2sqr(buf.data() + d, buf.data(), d);
    }

    void set_query(const float* x) override {
        q = x;
    }
};

// well optimized for xNN+PQNN
struct DistanceXPQ4 : Distance2Level {
    int M = 0;
    int k = 0;

    explicit DistanceXPQ4(const Index2Layer& storage_)
            : Distance2Level(storage_) {
        const IndexFlat* quantizer =
                dynamic_cast<IndexFlat*>(storage.q1.quantizer);

        FAISS_ASSERT(quantizer);
        M = static_cast<int>(storage.pq.M);
        pq_l1_tab = quantizer->get_xb();
    }

    float operator()(idx_t i) override {
        const uint8_t* code = storage.codes.data() + i * storage.code_size;
        idx_t key = 0;
        memcpy(&key, code, storage.code_size_1);
        code += storage.code_size_1;

        const float* qa = q;
        const float* l1 = pq_l1_tab + d * key;
        const float* l2 = pq_l2_tab;
        float accu = 0;

        for (int m = 0; m < M; m++) {
            for (int j = 0; j < 4; j++) {
                float diff = qa[j] - (l1[m * 4 + j] + l2[*code * 4 + j]);
                accu += diff * diff;
            }
            code++;
            l2 += 256 * 4;
            qa += 4;
        }
        return accu;
    }
};

// well optimized for 2xNN+PQNN
struct Distance2xXPQ4 : Distance2Level {
    int M_2, mi_nbits;

    explicit Distance2xXPQ4(const Index2Layer& storage_)
            : Distance2Level(storage_) {
        const MultiIndexQuantizer* mi =
                dynamic_cast<MultiIndexQuantizer*>(storage.q1.quantizer);

        FAISS_ASSERT(mi);
        FAISS_ASSERT(storage.pq.M % 2 == 0);
        M_2 = static_cast<int>(storage.pq.M / 2);
        mi_nbits = static_cast<int>(mi->pq.nbits);
        pq_l1_tab = mi->pq.centroids.data();
    }

    float operator()(idx_t i) override {
        const uint8_t* code = storage.codes.data() + i * storage.code_size;
        int64_t key01 = 0;
        memcpy(&key01, code, storage.code_size_1);
        code += storage.code_size_1;

        const float* qa = q;
        const float* l1 = pq_l1_tab;
        const float* l2 = pq_l2_tab;
        float accu = 0;

        for (int mi_m = 0; mi_m < 2; mi_m++) {
            int64_t l1_idx = key01 & (((int64_t)1 << mi_nbits) - 1);
            const float* l1_sub = l1 + M_2 * l1_idx * 4;

            for (int m = 0; m < M_2; m++) {
                for (int j = 0; j < 4; j++) {
                    float diff =
                            qa[j] - (l1_sub[m * 4 + j] + l2[*code * 4 + j]);
                    accu += diff * diff;
                }
                code++;
                l2 += 256 * 4;
                qa += 4;
            }
            l1 += (M_2 << mi_nbits) * 4;
            key01 >>= mi_nbits;
        }
        return accu;
    }
};

} // namespace

DistanceComputer* Index2Layer::get_distance_computer() const {
    const MultiIndexQuantizer* mi =
            dynamic_cast<MultiIndexQuantizer*>(q1.quantizer);

    if (mi && pq.M % 2 == 0 && pq.dsub == 4) {
        return new Distance2xXPQ4(*this);
    }

    const IndexFlat* fl = dynamic_cast<IndexFlat*>(q1.quantizer);

    if (fl && pq.dsub == 4) {
        return new DistanceXPQ4(*this);
    }

    return Index::get_distance_computer();
}

/* The standalone codec interface */

// block size used in Index2Layer::sa_encode
int index2layer_sa_encode_bs = 32768;

void Index2Layer::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);

    idx_t bs = index2layer_sa_encode_bs;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(i0 + bs, n);
            if (verbose) {
                printf("Index2Layer::add: adding %" PRId64 ":%" PRId64
                       " / %" PRId64 "\n",
                       i0,
                       i1,
                       n);
            }
            sa_encode(i1 - i0, x + i0 * d, bytes + i0 * code_size);
        }
        return;
    }

    std::unique_ptr<int64_t[]> list_nos(new int64_t[n]);
    q1.quantizer->assign(n, x, list_nos.get());
    std::vector<float> residuals(n * d);
    for (idx_t i = 0; i < n; i++) {
        q1.quantizer->compute_residual(
                x + i * d, residuals.data() + i * d, list_nos[i]);
    }
    pq.compute_codes(residuals.data(), bytes, n);

    for (idx_t i = n - 1; i >= 0; i--) {
        uint8_t* code = bytes + i * code_size;
        memmove(code + code_size_1, bytes + i * code_size_2, code_size_2);
        q1.encode_listno(list_nos[i], code);
    }
}

void Index2Layer::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
#pragma omp parallel
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = bytes + i * code_size;
            int64_t list_no = q1.decode_listno(code);
            float* xi = x + i * d;
            pq.decode(code + code_size_1, xi);
            q1.quantizer->reconstruct(list_no, residual.data());
            for (int j = 0; j < d; j++) {
                xi[j] += residual[j];
            }
        }
    }
}

} // namespace faiss
