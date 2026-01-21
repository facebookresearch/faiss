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
                labels,
                sel);
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
    const float* b;
    size_t ndis;
    size_t npartial_dot_products;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products++;
        return fvec_inner_product(
                q + offset, b + i * d + offset, num_components);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size,
                      q),
              d(storage.d),
              nb(storage.ntotal),
              b(storage.get_xb()),
              ndis(0),
              npartial_dot_products(0) {}

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

    void partial_dot_product_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dp0,
            float& dp1,
            float& dp2,
            float& dp3,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0_ = 0;
        float dp1_ = 0;
        float dp2_ = 0;
        float dp3_ = 0;
        fvec_inner_product_batch_4(
                q + offset,
                y0 + offset,
                y1 + offset,
                y2 + offset,
                y3 + offset,
                num_components,
                dp0_,
                dp1_,
                dp2_,
                dp3_);
        dp0 = dp0_;
        dp1 = dp1_;
        dp2 = dp2_;
        dp3 = dp3_;
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

/**************************************************************
 * shared flat Panorama search code
 **************************************************************/

namespace {

template <bool use_radius, typename BlockHandler>
inline void flat_pano_search_core(
        const IndexFlatPanorama& index,
        BlockHandler& handler,
        idx_t n,
        const float* x,
        float radius,
        const SearchParameters* params) {
    using SingleResultHandler = typename BlockHandler::SingleResultHandler;

    IDSelector* sel = params ? params->sel : nullptr;
    bool use_sel = sel != nullptr;

    [[maybe_unused]] int nt = std::min(int(n), omp_get_max_threads());
    size_t n_batches = (index.ntotal + index.batch_size - 1) / index.batch_size;

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler res(handler);

        std::vector<float> query_cum_norms(index.n_levels + 1);
        std::vector<float> exact_distances(index.batch_size);
        std::vector<uint32_t> active_indices(index.batch_size);

#pragma omp for
        for (int64_t i = 0; i < n; i++) {
            const float* xi = x + i * index.d;
            index.pano.compute_query_cum_sums(xi, query_cum_norms.data());

            PanoramaStats local_stats;
            local_stats.reset();

            res.begin(i);

            for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
                size_t batch_start = batch_no * index.batch_size;

                float threshold;
                if constexpr (use_radius) {
                    threshold = radius;
                } else {
                    threshold = res.heap_dis[0];
                }

                size_t num_active = with_metric_type(
                        index.metric_type, [&]<MetricType M>() {
                            return index.pano.progressive_filter_batch<
                                    CMax<float, int64_t>,
                                    M>(
                                    index.codes.data(),
                                    index.cum_sums.data(),
                                    xi,
                                    query_cum_norms.data(),
                                    batch_no,
                                    index.ntotal,
                                    sel,
                                    nullptr,
                                    use_sel,
                                    active_indices,
                                    exact_distances,
                                    threshold,
                                    local_stats);
                        });

                for (size_t j = 0; j < num_active; j++) {
                    res.add_result(
                            exact_distances[active_indices[j]],
                            batch_start + active_indices[j]);
                }
            }

            res.end();
            indexPanorama_stats.add(local_stats);
        }
    }
}

} // anonymous namespace

/***************************************************
 * IndexFlatPanorama
 ***************************************************/

void IndexFlatPanorama::add(idx_t n, const float* x) {
    size_t offset = ntotal;
    ntotal += n;
    size_t num_batches = (ntotal + batch_size - 1) / batch_size;

    codes.resize(num_batches * batch_size * code_size);
    cum_sums.resize(num_batches * batch_size * (n_levels + 1));

    const uint8_t* code = reinterpret_cast<const uint8_t*>(x);
    pano.copy_codes_to_level_layout(codes.data(), offset, n, code);
    pano.compute_cumulative_sums(cum_sums.data(), offset, n, x);
}

void IndexFlatPanorama::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(batch_size >= k);

    HeapBlockResultHandler<CMax<float, int64_t>, false> handler(
            size_t(n), distances, labels, size_t(k), nullptr);

    flat_pano_search_core<false>(*this, handler, n, x, 0.0f, params);
}

void IndexFlatPanorama::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    RangeSearchBlockResultHandler<CMax<float, int64_t>, false> handler(
            result, radius, nullptr);

    flat_pano_search_core<true>(*this, handler, n, x, radius, params);
}

void IndexFlatPanorama::reset() {
    IndexFlat::reset();
    cum_sums.clear();
}

void IndexFlatPanorama::reconstruct(idx_t key, float* recons) const {
    pano.reconstruct(key, recons, codes.data());
}

void IndexFlatPanorama::reconstruct_n(idx_t i, idx_t n, float* recons) const {
    Index::reconstruct_n(i, n, recons);
}

size_t IndexFlatPanorama::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                pano.copy_entry(
                        codes.data(),
                        codes.data(),
                        cum_sums.data(),
                        cum_sums.data(),
                        j,
                        i);
            }
            j++;
        }
    }
    size_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        size_t num_batches = (ntotal + batch_size - 1) / batch_size;
        codes.resize(num_batches * batch_size * code_size);
        cum_sums.resize(num_batches * batch_size * (n_levels + 1));
    }
    return nremove;
}

void IndexFlatPanorama::merge_from(Index& otherIndex, idx_t add_id) {
    FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatPanorama index");
    check_compatible_for_merge(otherIndex);
    IndexFlatPanorama* other = static_cast<IndexFlatPanorama*>(&otherIndex);

    std::vector<float> buffer(other->ntotal * code_size);
    otherIndex.reconstruct_n(0, other->ntotal, buffer.data());

    add(other->ntotal, buffer.data());
    other->reset();
}

void IndexFlatPanorama::add_sa_codes(
        idx_t /* n */,
        const uint8_t* /* codes_in */,
        const idx_t* /* xids */) {
    FAISS_THROW_MSG("add_sa_codes not implemented for IndexFlatPanorama");
}

void IndexFlatPanorama::permute_entries(const idx_t* perm) {
    MaybeOwnedVector<uint8_t> new_codes(codes.size());
    std::vector<float> new_cum_sums(cum_sums.size());

    for (idx_t i = 0; i < ntotal; i++) {
        pano.copy_entry(
                new_codes.data(),
                codes.data(),
                new_cum_sums.data(),
                cum_sums.data(),
                i,
                perm[i]);
    }

    std::swap(codes, new_codes);
    std::swap(cum_sums, new_cum_sums);
}

void IndexFlatPanorama::search_subset(
        idx_t n,
        const float* x,
        idx_t k_base,
        const idx_t* base_labels,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    using SingleResultHandler =
            HeapBlockResultHandler<CMax<float, int64_t>, false>::
                    SingleResultHandler;
    HeapBlockResultHandler<CMax<float, int64_t>, false> handler(
            size_t(n), distances, labels, size_t(k), nullptr);

    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(batch_size == 1);

    [[maybe_unused]] int nt = std::min(int(n), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler res(handler);

        std::vector<float> query_cum_norms(n_levels + 1);

        // Panorama's optimized point-wise refinement (Algorithm 2):
        // Batch-wise Panorama, as implemented in Panorama.h, incurs overhead
        // from maintaining active_indices and exact_distances. This optimized
        // implementation has minimal overhead and is thus preferred for
        // IndexRefine's use case.
        // 1. Initialize exact distance as ||y||^2 + ||x||^2.
        // 2. For each level, refine distance incrementally:
        //    - Compute dot product for current level: exact_dist -= 2*<x,y>.
        //    - Use Cauchy-Schwarz bound on remaining levels to get lower bound.
        //    - If there are less than k points in the heap, add the point to
        //    the heap.
        //    - Else, prune if lower bound exceeds k-th best distance.
        // 3. After all levels, update heap if the point survived.
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const idx_t* __restrict idsi = base_labels + i * k_base;
            const float* xi = x + i * d;

            PanoramaStats local_stats;
            local_stats.reset();

            pano.compute_query_cum_sums(xi, query_cum_norms.data());
            float query_cum_norm = query_cum_norms[0] * query_cum_norms[0];

            res.begin(i);

            for (size_t j = 0; j < k_base; j++) {
                idx_t idx = idsi[j];

                if (idx < 0) {
                    continue;
                }

                size_t cum_sum_offset = (n_levels + 1) * idx;
                float cum_sum = cum_sums[cum_sum_offset];
                float exact_distance = cum_sum * cum_sum + query_cum_norm;
                cum_sum_offset++;

                const float* x_ptr = xi;
                const float* p_ptr =
                        reinterpret_cast<const float*>(codes.data()) + d * idx;

                local_stats.total_dims += d;

                bool pruned = false;
                for (size_t level = 0; level < n_levels; level++) {
                    local_stats.total_dims_scanned += pano.level_width_floats;

                    // Refine distance
                    size_t actual_level_width = std::min(
                            pano.level_width_floats,
                            d - level * pano.level_width_floats);
                    float dot_product = fvec_inner_product(
                            x_ptr, p_ptr, actual_level_width);
                    exact_distance -= 2 * dot_product;

                    float cum_sum = cum_sums[cum_sum_offset];
                    float cauchy_schwarz_bound =
                            2.0f * cum_sum * query_cum_norms[level + 1];
                    float lower_bound = exact_distance - cauchy_schwarz_bound;

                    // Prune using Cauchy-Schwarz bound
                    if (lower_bound > res.heap_dis[0]) {
                        pruned = true;
                        break;
                    }

                    cum_sum_offset++;
                    x_ptr += pano.level_width_floats;
                    p_ptr += pano.level_width_floats;
                }

                if (!pruned) {
                    res.add_result(exact_distance, idx);
                }
            }

            res.end();
            indexPanorama_stats.add(local_stats);
        }
    }
}
} // namespace faiss
