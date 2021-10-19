/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexNNDescent.h>

#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include <queue>
#include <unordered_set>

#ifdef __SSE__
#endif

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

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

using idx_t = Index::idx_t;
using storage_idx_t = NNDescent::storage_idx_t;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    ~NegativeDistanceComputer() override {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

} // namespace

/**************************************************************
 * IndexNNDescent implementation
 **************************************************************/

IndexNNDescent::IndexNNDescent(int d, int K, MetricType metric)
        : Index(d, metric),
          nndescent(d, K),
          own_fields(false),
          storage(nullptr) {}

IndexNNDescent::IndexNNDescent(Index* storage, int K)
        : Index(storage->d, storage->metric_type),
          nndescent(storage->d, K),
          own_fields(false),
          storage(storage) {}

IndexNNDescent::~IndexNNDescent() {
    if (own_fields) {
        delete storage;
    }
}

void IndexNNDescent::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNNDescentFlat (or variants) "
            "instead of IndexNNDescent directly");
    // nndescent structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexNNDescent::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const

{
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNNDescentFlat (or variants) "
            "instead of IndexNNDescent directly");
    if (verbose) {
        printf("Parameters: k=%" PRId64 ", search_L=%d\n",
               k,
               nndescent.search_L);
    }

    idx_t check_period =
            InterruptCallback::get_period_hint(d * nndescent.search_L);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                nndescent.search(*dis, k, idxi, simi, vt);
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexNNDescent::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNNDescentFlat (or variants) "
            "instead of IndexNNDescent directly");
    FAISS_THROW_IF_NOT(is_trained);

    if (ntotal != 0) {
        fprintf(stderr,
                "WARNING NNDescent doest not support dynamic insertions,"
                "multiple insertions would lead to re-building the index");
    }

    storage->add(n, x);
    ntotal = storage->ntotal;

    DistanceComputer* dis = storage_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del(dis);
    nndescent.build(*dis, ntotal, verbose);
}

void IndexNNDescent::reset() {
    nndescent.reset();
    storage->reset();
    ntotal = 0;
}

void IndexNNDescent::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

/**************************************************************
 * IndexNNDescentFlat implementation
 **************************************************************/

IndexNNDescentFlat::IndexNNDescentFlat() {
    is_trained = true;
}

IndexNNDescentFlat::IndexNNDescentFlat(int d, int M, MetricType metric)
        : IndexNNDescent(new IndexFlat(d, metric), M) {
    own_fields = true;
    is_trained = true;
}

} // namespace faiss
