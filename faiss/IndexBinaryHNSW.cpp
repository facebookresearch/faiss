/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryHNSW.h>

#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <cstdint>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

#include <faiss/impl/simd_dispatch.h>

#include <random>

// Scalar (NONE) fallback for dynamic dispatch
#define THE_SIMD_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/binary_hamming/IndexBinaryHNSW_impl.h>
#undef THE_SIMD_LEVEL

namespace faiss {

/**************************************************************
 * IndexBinaryHNSW implementation
 **************************************************************/

IndexBinaryHNSW::IndexBinaryHNSW() {
    is_trained = true;
}

IndexBinaryHNSW::IndexBinaryHNSW(int d_, int M)
        : IndexBinary(d_),
          hnsw(M),
          own_fields(true),
          storage(new IndexBinaryFlat(d_)) {
    is_trained = true;
}

IndexBinaryHNSW::IndexBinaryHNSW(IndexBinary* storage_, int M)
        : IndexBinary(storage_->d),
          hnsw(M),
          own_fields(false),
          storage(storage_) {
    is_trained = true;
}

IndexBinaryHNSW::~IndexBinaryHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexBinaryHNSW::train(idx_t n, const uint8_t* x) {
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexBinaryHNSW::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    const SearchParametersHNSW* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(
                params, "IndexBinaryHNSW params have incorrect type");
    }

    // we use the buffer for distances as float but convert them back
    // to int in the end
    float* distances_f = (float*)distances;

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances_f, labels, k);

    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

#pragma omp parallel
    {
        std::unique_ptr<VisitedTable> vt = VisitedTable::create(ntotal);
        std::unique_ptr<DistanceComputer> dis(get_distance_computer());
        RH::SingleResultHandler res(bres);

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            dis->set_query((float*)(x + i * code_size));
            // Given that IndexBinaryHNSW is not an IndexHNSW, we pass nullptr
            // as the index parameter. This state does not get used in the
            // search function, as it is merely there to enable Panorama
            // execution for IndexHNSWFlatPanorama.
            HNSWStats stats = hnsw.search(*dis, nullptr, res, *vt, params_in);
            n1 += stats.n1;
            n2 += stats.n2;
            ndis += stats.ndis;
            nhops += stats.nhops;
            res.end();
        }
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});

#pragma omp parallel for
    for (idx_t i = 0; i < n * k; ++i) {
        distances[i] = std::round(distances_f[i]);
    }
}

void IndexBinaryHNSW::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_NOT(is_trained);
    size_t n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    bool preset_levels = hnsw.levels.size() == static_cast<size_t>(ntotal);

    hnsw_add_vertices_deterministic(
            hnsw,
            n0,
            n,
            d,
            init_level0,
            keep_max_size_level0,
            preset_levels,
            verbose,
            [this] { return get_distance_computer(); },
            [this, x, n0](DistanceComputer& dc, HNSW::storage_idx_t pt_id) {
                dc.set_query((const float*)(x + (pt_id - n0) * code_size));
            });
}

void IndexBinaryHNSW::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexBinaryHNSW::reconstruct(idx_t key, uint8_t* recons) const {
    storage->reconstruct(key, recons);
}

DistanceComputer* IndexBinaryHNSW::get_distance_computer() const {
    IndexBinaryFlat* flat_storage = dynamic_cast<IndexBinaryFlat*>(storage);
    FAISS_THROW_IF_MSG(
            flat_storage == nullptr,
            "IndexBinaryHNSW requires IndexBinaryFlat storage");
    return with_simd_level([&]<SIMDLevel SL>() {
        return make_binary_hnsw_distance_computer_fixSL<SL>(
                code_size, flat_storage);
    });
}

/**************************************************************
 * IndexBinaryHNSWCagra implementation
 **************************************************************/

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra() : IndexBinaryHNSW() {
    storage = nullptr;
}

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra(int d_, int M)
        : IndexBinaryHNSW(d_, M) {
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexBinaryHNSWCagra::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_MSG(
            base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexBinaryHNSW::add(n, x);
}

void IndexBinaryHNSWCagra::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexBinaryHNSW::search(n, x, k, distances, labels, params);
    } else {
        FAISS_THROW_IF_NOT_MSG(
                ntotal > 0, "IndexBinaryHNSWCagra: cannot search empty index");
        FAISS_THROW_IF_NOT_MSG(
                num_base_level_search_entrypoints > 0,
                "IndexBinaryHNSWCagra: "
                "num_base_level_search_entrypoints must be > 0");

        float* distances_f = (float*)distances;

        using RH = HeapBlockResultHandler<HNSW::C>;
        RH bres(n, distances_f, labels, k);

        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(get_distance_computer());
            dis->set_query((float*)(x + i * code_size));

            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                float distance = (*dis)(idx);

                if (distance < nearest_d[i]) {
                    nearest[i] = static_cast<storage_idx_t>(idx);
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

#pragma omp parallel
        {
            std::unique_ptr<VisitedTable> vt = VisitedTable::create(ntotal);
            std::unique_ptr<DistanceComputer> dis(get_distance_computer());
            HNSWStats search_stats;
            RH::SingleResultHandler res(bres);

#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                res.begin(i);
                dis->set_query((float*)(x + i * code_size));

                hnsw.search_level_0(
                        *dis,
                        res,
                        1,
                        &nearest[i],
                        &nearest_d[i],
                        1, // search_type
                        search_stats,
                        *vt,
                        params);

                res.end();
            }
#pragma omp critical
            {
                hnsw_stats.combine(search_stats);
            }
        }

#pragma omp parallel for
        for (idx_t i = 0; i < n * k; ++i) {
            distances[i] = std::round(distances_f[i]);
        }
    }
}

} // namespace faiss
