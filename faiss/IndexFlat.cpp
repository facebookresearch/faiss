/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>
#include <cstring>

namespace faiss {

IndexFlat::IndexFlat(idx_t d, MetricType metric) : Index(d, metric) {}

void IndexFlat::add(idx_t n, const float* x) {
    xb.insert(xb.end(), x, x + n * d);
    ntotal += n;
}

void IndexFlat::reset() {
    xb.clear();
    ntotal = 0;
}

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps

    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, xb.data(), d, n, ntotal, &res);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, xb.data(), d, n, ntotal, &res);
    } else {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, xb.data(), d, n, ntotal, metric_type, metric_arg, &res);
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, xb.data(), d, n, ntotal, radius, result);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, xb.data(), d, n, ntotal, radius, result);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(
                    distances, x, xb.data(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, xb.data(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

size_t IndexFlat::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove(&xb[d * j], &xb[d * i], sizeof(xb[0]) * d);
            }
            j++;
        }
    }
    size_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        xb.resize(ntotal * d);
    }
    return nremove;
}

namespace {

struct FlatL2Dis : DistanceComputer {
    size_t d;
    Index::idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float operator()(idx_t i) override {
        ndis++;
        return fvec_L2sqr(q, b + i * d, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.xb.data()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }
};

struct FlatIPDis : DistanceComputer {
    size_t d;
    Index::idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float operator()(idx_t i) override {
        ndis++;
        return fvec_inner_product(q, b + i * d, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
            : d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.xb.data()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }
};

} // namespace

DistanceComputer* IndexFlat::get_distance_computer() const {
    if (metric_type == METRIC_L2) {
        return new FlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new FlatIPDis(*this);
    } else {
        return get_extra_distance_computer(
                d, metric_type, metric_arg, ntotal, xb.data());
    }
}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    memcpy(recons, &(xb[key * d]), sizeof(*recons) * d);
}

/* The standalone codec interface */
size_t IndexFlat::sa_code_size() const {
    return sizeof(float) * d;
}

void IndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    memcpy(bytes, x, sizeof(float) * d * n);
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    memcpy(x, bytes, sizeof(float) * d * n);
}

/***************************************************
 * IndexFlat1D
 ***************************************************/

IndexFlat1D::IndexFlat1D(bool continuous_update)
        : IndexFlatL2(1), continuous_update(continuous_update) {}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation() {
    perm.resize(ntotal);
    if (ntotal < 1000000) {
        fvec_argsort(ntotal, xb.data(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel(ntotal, xb.data(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset() {
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(
            perm.size() == ntotal, "Call update_permutation before search");

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        float q = x[i]; // query
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q)
                i0 = imed;
            else
                i1 = imed;
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) {
                    goto finish_right;
                }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) {
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}

} // namespace faiss
