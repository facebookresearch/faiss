/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/sorting.h>
#include <omp.h>
#include <cstring>
#include <numeric>

namespace faiss {

IndexFlat::IndexFlat(idx_t d, MetricType metric)
        : IndexFlatCodes(sizeof(float) * d, d, metric) {}

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, sel);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);
    } else {
        FAISS_THROW_IF_NOT(!sel); // TODO implement with selector
        knn_extra_metrics(
                x,
                get_xb(),
                d,
                n,
                ntotal,
                metric_type,
                metric_arg,
                k,
                distances,
                labels);
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result, sel);
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
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

namespace {

struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }
};

struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product(q, (const float*)code, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }
};

} // namespace

FlatCodesDistanceComputer* IndexFlat::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        return new FlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new FlatIPDis(*this);
    } else {
        return get_extra_distance_computer(
                d, metric_type, metric_arg, ntotal, get_xb());
    }
}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key < ntotal);
    memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, x, sizeof(float) * d * n);
    }
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
        memcpy(x, bytes, sizeof(float) * d * n);
    }
}

/***************************************************
 * IndexFlatL2
 ***************************************************/

namespace {
struct FlatL2WithNormsDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    const float* l2norms;
    float query_l2norm;

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float operator()(const idx_t i) final override {
        const float* __restrict y =
                reinterpret_cast<const float*>(codes + i * code_size);

        prefetch_L2(l2norms + i);
        const float dp0 = fvec_inner_product(q, y, d);
        return query_l2norm + l2norms[i] - 2 * dp0;
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        const float* __restrict yi =
                reinterpret_cast<const float*>(codes + i * code_size);
        const float* __restrict yj =
                reinterpret_cast<const float*>(codes + j * code_size);

        prefetch_L2(l2norms + i);
        prefetch_L2(l2norms + j);
        const float dp0 = fvec_inner_product(yi, yj, d);
        return l2norms[i] + l2norms[j] - 2 * dp0;
    }

    explicit FlatL2WithNormsDis(
            const IndexFlatL2& storage,
            const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0),
              l2norms(storage.cached_l2norms.data()),
              query_l2norm(0) {}

    void set_query(const float* x) override {
        q = x;
        query_l2norm = fvec_norm_L2sqr(q, d);
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        prefetch_L2(l2norms + idx0);
        prefetch_L2(l2norms + idx1);
        prefetch_L2(l2norms + idx2);
        prefetch_L2(l2norms + idx3);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
        dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
        dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
        dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
    }
};

} // namespace

void IndexFlatL2::sync_l2norms() {
    cached_l2norms.resize(ntotal);
    fvec_norms_L2sqr(
            cached_l2norms.data(),
            reinterpret_cast<const float*>(codes.data()),
            d,
            ntotal);
}

void IndexFlatL2::clear_l2norms() {
    cached_l2norms.clear();
    cached_l2norms.shrink_to_fit();
}

FlatCodesDistanceComputer* IndexFlatL2::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        if (!cached_l2norms.empty()) {
            return new FlatL2WithNormsDis(*this);
        }
    }

    return IndexFlat::get_FlatCodesDistanceComputer();
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
        fvec_argsort(ntotal, get_xb(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel(ntotal, get_xb(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);
    if (continuous_update) {
        update_permutation();
    }
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
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            perm.size() == ntotal, "Call update_permutation before search");
    const float* xb = get_xb();

#pragma omp parallel for if (n > 10000)
    for (idx_t i = 0; i < n; i++) {
        float q = x[i]; // query
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (ntotal == 0) {
            for (idx_t j = 0; j < k; j++) {
                I[j] = -1;
                D[j] = HUGE_VAL;
            }
            goto done;
        }

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
            if (xb[perm[imed]] <= q) {
                i0 = imed;
            } else {
                i1 = imed;
            }
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

/***************************************************
 * IndexFlatL2Panorama
 ***************************************************/

void IndexFlatL2Panorama::add(idx_t n, const float* x) {
    size_t offset = ntotal;
    ntotal += n;
    size_t num_batches = (ntotal + batch_size - 1) / batch_size;

    codes.resize(num_batches * batch_size * code_size);
    cum_sums.resize(num_batches * batch_size * (n_levels + 1));

    const uint8_t* code = reinterpret_cast<const uint8_t*>(x);
    pano.copy_codes_to_level_layout(codes.data(), offset, n, code);
    pano.compute_cumulative_sums(cum_sums.data(), offset, n, x);

    printf("codes size: %zu\n", codes.size());
    printf("cum_sums size: %zu\n", cum_sums.size());
}

void IndexFlatL2Panorama::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    using SingleResultHandler =
            typename HeapBlockResultHandler<CMax<float, int64_t>, false>::
                    SingleResultHandler;

    IDSelector* sel = params ? params->sel : nullptr;
    FAISS_ASSERT(sel == nullptr);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(batch_size >= k);

    HeapBlockResultHandler<CMax<float, int64_t>, false> handler(
            size_t(n), distances, labels, size_t(k), nullptr);
    [[maybe_unused]] int nt = std::min(int(n), omp_get_max_threads());

    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "d: " << d << std::endl;
    std::cout << "n_levels: " << n_levels << std::endl;
    std::cout << "pano batch_size: " << pano.batch_size << std::endl;
    std::cout << "pano level_width: " << pano.level_width << std::endl;
    std::cout << "pano n_levels: " << pano.n_levels << std::endl;
    std::cout << "pano level_width_floats: " << pano.level_width_floats
              << std::endl;
    std::cout << "pano code_size: " << pano.code_size << std::endl;

    std::cout << "a" << std::endl;

    // print the first point in codes() and its cumsums
    for (size_t i = 0; i < n_levels; i++) {
        for (size_t j = 0; j < d / n_levels; j++) {
            std::cout << reinterpret_cast<const float*>(codes.data())
                                 [i * batch_size * (d / n_levels) + j]
                      << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "b" << std::endl;

    size_t n_batches = (ntotal + batch_size - 1) / batch_size;

    std::cout << "n_batches: " << n_batches << std::endl;

    // #pragma omp parallel num_threads(nt)
    {
        SingleResultHandler res(handler);

        std::vector<float> query_cum_norms(n_levels + 1);
        std::vector<float> suffix_sums(d + 1);
        std::vector<float> query(d);
        std::vector<float> exact_distances(batch_size);
        std::vector<uint32_t> active_indices(batch_size);
        suffix_sums[d] = 0.0f;

        PanoramaStats local_stats;
        local_stats.reset();

        // #pragma omp for
        for (int64_t i = 0; i < n; i++) {
            const float* xi = x + i * d;

            for (int i = d - 1; i >= 0; i--) {
                float squared_val = xi[i] * xi[i];
                suffix_sums[i] = suffix_sums[i + 1] + squared_val;
            }

            // std::cout << "suffix_sums: ";
            // for (size_t i = 0; i < d + 1; i++) {
            //     std::cout << suffix_sums[i] << " ";
            // }
            // std::cout << std::endl;

            for (int level_idx = 0; level_idx < n_levels; level_idx++) {
                int startIdx = level_idx * pano.level_width_floats;
                // std::cout << "startIdx: " << startIdx << std::endl;
                if (startIdx < d) {
                    query_cum_norms[level_idx] = sqrt(suffix_sums[startIdx]);
                } else {
                    query_cum_norms[level_idx] = 0.0f;
                }
            }
            query_cum_norms[n_levels] = 0.0f;

            // // print xi and query_cum_norms
            // for (size_t i = 0; i < d; i++) {
            //     std::cout << xi[i] << " ";
            // }
            // std::cout << std::endl;
            // for (size_t i = 0; i < n_levels + 1; i++) {
            //     std::cout << query_cum_norms[i] << " ";
            // }
            // std::cout << std::endl;

            res.begin(i);

            for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
                // std::cout << "-------" << std::endl;
                size_t curr_batch_size =
                        std::min(ntotal - batch_no * batch_size, batch_size);

                std::cout << "curr_batch_size: " << curr_batch_size << std::endl;

                std::iota(
                        active_indices.begin(),
                        active_indices.begin() + curr_batch_size,
                        0);

                size_t cumsum_batch_offset =
                        batch_no * batch_size * (n_levels + 1);
                const float* batch_cum_sums =
                        cum_sums.data() + cumsum_batch_offset;
                const float* level_cum_sums = batch_cum_sums + batch_size;

                // Initialize with the first cum sums of each point.
                for (size_t idx = 0; idx < curr_batch_size; idx++) {
                    float squared_root = batch_cum_sums[idx];
                    exact_distances[idx] =
                            squared_root * squared_root + suffix_sums[0];
                }

                size_t batch_offset = batch_no * batch_size * code_size;
                // std::cout << "batch_no: " << batch_no << std::endl;
                // std::cout << "batch_offset: " << batch_offset << std::endl;
                // std::cout << "cumsum_batch_offset: " << cumsum_batch_offset << std::endl;
                // std::cout << "level_cum_sums[0]: " << level_cum_sums[0] << std::endl;
                const uint8_t* storage_base = codes.data() + batch_offset;

                printf("res.heap_dis[0]: %f\n", res.heap_dis[0]);

                size_t active_num =
                        pano.progressive_filter_batch<CMax<float, int64_t>>(
                                storage_base,
                                level_cum_sums,
                                xi,
                                query_cum_norms.data(),
                                active_indices,
                                exact_distances,
                                curr_batch_size,
                                res.heap_dis[0],
                                local_stats);

                idx_t idx_offset = batch_no * batch_size;
                for (size_t j = 0; j < active_num; j++) {
                    res.add_result(
                            exact_distances[active_indices[j]],
                            active_indices[j] + idx_offset);
                }
            }

            res.end();
            indexPanorama_stats.add(local_stats);
        }
    }
}
} // namespace faiss
