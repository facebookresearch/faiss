/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/extra_distances.h>

#include <omp.h>
#include <algorithm>
#include <cmath>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/utils.h>

namespace faiss {

/***************************************************************************
 * Distance functions (other than L2 and IP)
 ***************************************************************************/

namespace {

struct Run_pairwise_extra_distances {
    using T = void;

    template <class VD>
    void f(VD vd,
           int64_t nq,
           const float* xq,
           int64_t nb,
           const float* xb,
           float* dis,
           int64_t ldq,
           int64_t ldb,
           int64_t ldd) {
#pragma omp parallel for if (nq > 10)
        for (int64_t i = 0; i < nq; i++) {
            const float* xqi = xq + i * ldq;
            const float* xbj = xb;
            float* disi = dis + ldd * i;

            for (int64_t j = 0; j < nb; j++) {
                disi[j] = vd(xqi, xbj);
                xbj += ldb;
            }
        }
    }
};

struct Run_knn_extra_metrics {
    using T = void;
    template <class VD>
    void f(VD vd,
           const float* x,
           const float* y,
           size_t nx,
           size_t ny,
           size_t k,
           float* distances,
           int64_t* labels) {
        size_t d = vd.d;
        using C = typename VD::C;
        size_t check_period = InterruptCallback::get_period_hint(ny * d);
        check_period *= omp_get_max_threads();

        for (size_t i0 = 0; i0 < nx; i0 += check_period) {
            size_t i1 = std::min(i0 + check_period, nx);

#pragma omp parallel for
            for (int64_t i = i0; i < i1; i++) {
                const float* x_i = x + i * d;
                const float* y_j = y;
                size_t j;
                float* simi = distances + k * i;
                int64_t* idxi = labels + k * i;

                // maxheap_heapify(k, simi, idxi);
                heap_heapify<C>(k, simi, idxi);
                for (j = 0; j < ny; j++) {
                    float disij = vd(x_i, y_j);

                    if (C::cmp(simi[0], disij)) {
                        heap_replace_top<C>(k, simi, idxi, disij, j);
                    }
                    y_j += d;
                }
                // maxheap_reorder(k, simi, idxi);
                heap_reorder<C>(k, simi, idxi);
            }
            InterruptCallback::check();
        }
    }
};

template <class VD>
struct ExtraDistanceComputer : FlatCodesDistanceComputer {
    VD vd;
    idx_t nb;
    const float* q;
    const float* b;

    float symmetric_dis(idx_t i, idx_t j) final {
        return vd(b + j * vd.d, b + i * vd.d);
    }

    float distance_to_code(const uint8_t* code) final {
        return vd(q, (float*)code);
    }

    ExtraDistanceComputer(
            const VD& vd,
            const float* xb,
            size_t nb,
            const float* q = nullptr)
            : FlatCodesDistanceComputer((uint8_t*)xb, vd.d * sizeof(float)),
              vd(vd),
              nb(nb),
              q(q),
              b(xb) {}

    void set_query(const float* x) override {
        q = x;
    }
};

struct Run_get_distance_computer {
    using T = FlatCodesDistanceComputer*;

    template <class VD>
    FlatCodesDistanceComputer* f(
            VD vd,
            const float* xb,
            size_t nb,
            const float* q = nullptr) {
        return new ExtraDistanceComputer<VD>(vd, xb, nb, q);
    }
};

} // anonymous namespace

void pairwise_extra_distances(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        MetricType mt,
        float metric_arg,
        float* dis,
        int64_t ldq,
        int64_t ldb,
        int64_t ldd) {
    if (nq == 0 || nb == 0)
        return;
    if (ldq == -1)
        ldq = d;
    if (ldb == -1)
        ldb = d;
    if (ldd == -1)
        ldd = nb;

    Run_pairwise_extra_distances run;
    dispatch_VectorDistance(
            d, mt, metric_arg, run, nq, xq, nb, xb, dis, ldq, ldb, ldd);
}

void knn_extra_metrics(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        MetricType mt,
        float metric_arg,
        size_t k,
        float* distances,
        int64_t* indexes) {
    Run_knn_extra_metrics run;
    dispatch_VectorDistance(
            d, mt, metric_arg, run, x, y, nx, ny, k, distances, indexes);
}

FlatCodesDistanceComputer* get_extra_distance_computer(
        size_t d,
        MetricType mt,
        float metric_arg,
        size_t nb,
        const float* xb) {
    Run_get_distance_computer run;
    return dispatch_VectorDistance(d, mt, metric_arg, run, xb, nb);
}

} // namespace faiss
