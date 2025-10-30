/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFPQFastScan.h>

#include <array>
#include <cassert>
#include <cstdio>

#include <memory>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/simdlib.h>

#include <faiss/invlists/BlockInvertedLists.h>

#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexIVFPQFastScan::IndexIVFPQFastScan(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits,
        MetricType metric,
        int bbs,
        bool own_invlists)
        : IndexIVFFastScan(quantizer, d, nlist, 0, metric, own_invlists),
          pq(d, M, nbits) {
    by_residual = false; // set to false by default because it's faster

    init_fastscan(&pq, M, nbits, nlist, metric, bbs, own_invlists);
}

IndexIVFPQFastScan::IndexIVFPQFastScan() {
    by_residual = false;
    bbs = 0;
    M2 = 0;
}

IndexIVFPQFastScan::IndexIVFPQFastScan(const IndexIVFPQ& orig, int bbs)
        : IndexIVFFastScan(
                  orig.quantizer,
                  orig.d,
                  orig.nlist,
                  orig.pq.code_size,
                  orig.metric_type,
                  orig.own_invlists),
          pq(orig.pq) {
    FAISS_THROW_IF_NOT(orig.pq.nbits == 4);

    init_fastscan(
            &pq,
            orig.pq.M,
            orig.pq.nbits,
            orig.nlist,
            orig.metric_type,
            bbs,
            orig.own_invlists);

    by_residual = orig.by_residual;
    ntotal = orig.ntotal;
    is_trained = orig.is_trained;
    nprobe = orig.nprobe;

    precomputed_table.resize(orig.precomputed_table.size());

    if (precomputed_table.nbytes() > 0) {
        memcpy(precomputed_table.get(),
               orig.precomputed_table.data(),
               precomputed_table.nbytes());
    }

#pragma omp parallel for if (nlist > 100)
    for (idx_t i = 0; i < nlist; i++) {
        size_t nb = orig.invlists->list_size(i);
        size_t nb2 = roundup(nb, bbs);
        AlignedTable<uint8_t> tmp(nb2 * M2 / 2);
        pq4_pack_codes(
                InvertedLists::ScopedCodes(orig.invlists, i).get(),
                nb,
                M,
                nb2,
                bbs,
                M2,
                tmp.get());
        invlists->add_entries(
                i,
                nb,
                InvertedLists::ScopedIds(orig.invlists, i).get(),
                tmp.get());
    }

    orig_invlists = orig.invlists;
}

/*********************************************************
 * Training
 *********************************************************/

void IndexIVFPQFastScan::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    pq.verbose = verbose;
    pq.train(n, x);

    if (by_residual && metric_type == METRIC_L2) {
        precompute_table();
    }
}

idx_t IndexIVFPQFastScan::train_encoder_num_vectors() const {
    return pq.cp.max_points_per_centroid * pq.ksub;
}

void IndexIVFPQFastScan::precompute_table() {
    initialize_IVFPQ_precomputed_table(
            use_precomputed_table,
            quantizer,
            pq,
            precomputed_table,
            by_residual,
            verbose);
}

/*********************************************************
 * Code management functions
 *********************************************************/

void IndexIVFPQFastScan::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    if (by_residual) {
        AlignedTable<float> residuals(n * d);
        for (size_t i = 0; i < n; i++) {
            if (list_nos[i] < 0) {
                memset(residuals.data() + i * d, 0, sizeof(residuals[0]) * d);
            } else {
                quantizer->compute_residual(
                        x + i * d, residuals.data() + i * d, list_nos[i]);
            }
        }
        pq.compute_codes(residuals.data(), codes, n);
    } else {
        pq.compute_codes(x, codes, n);
    }

    if (include_listnos) {
        size_t coarse_size = coarse_code_size();
        for (idx_t i = n - 1; i >= 0; i--) {
            uint8_t* code = codes + i * (coarse_size + code_size);
            memmove(code + coarse_size, codes + i * code_size, code_size);
            encode_listno(list_nos[i], code);
        }
    }
}

/*********************************************************
 * Look-Up Table functions
 *********************************************************/

void fvec_madd_simd(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    assert(is_aligned_pointer(a));
    assert(is_aligned_pointer(b));
    assert(is_aligned_pointer(c));
    assert(n % 8 == 0);
    simd8float32 bf8(bf);
    n /= 8;
    for (size_t i = 0; i < n; i++) {
        simd8float32 ai(a);
        simd8float32 bi(b);

        simd8float32 ci = fmadd(bf8, bi, ai);
        ci.store(c);
        c += 8;
        a += 8;
        b += 8;
    }
}

bool IndexIVFPQFastScan::lookup_table_is_3d() const {
    return by_residual && metric_type == METRIC_L2;
}

void IndexIVFPQFastScan::compute_LUT(
        size_t n,
        const float* x,
        const CoarseQuantized& cq,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& biases,
        const FastScanDistancePostProcessing&) const {
    size_t dim12 = pq.ksub * pq.M;
    size_t d = pq.d;
    size_t nprobe = cq.nprobe;

    if (by_residual) {
        if (metric_type == METRIC_L2) {
            dis_tables.resize(n * nprobe * dim12);

            if (use_precomputed_table == 1) {
                biases.resize(n * nprobe);
                memcpy(biases.get(), cq.dis, sizeof(float) * n * nprobe);

                AlignedTable<float> ip_table(n * dim12);
                pq.compute_inner_prod_tables(n, x, ip_table.get());

#pragma omp parallel for if (n * nprobe > 8000)
                for (idx_t ij = 0; ij < n * nprobe; ij++) {
                    idx_t i = ij / nprobe;
                    float* tab = dis_tables.get() + ij * dim12;
                    idx_t cij = cq.ids[ij];

                    if (cij >= 0) {
                        fvec_madd_simd(
                                dim12,
                                precomputed_table.get() + cij * dim12,
                                -2,
                                ip_table.get() + i * dim12,
                                tab);
                    } else {
                        // fill with NaNs so that they are ignored during
                        // LUT quantization
                        memset(tab, -1, sizeof(float) * dim12);
                    }
                }

            } else {
                std::unique_ptr<float[]> xrel(new float[n * nprobe * d]);
                biases.resize(n * nprobe);
                memset(biases.get(), 0, sizeof(float) * n * nprobe);

#pragma omp parallel for if (n * nprobe > 8000)
                for (idx_t ij = 0; ij < n * nprobe; ij++) {
                    idx_t i = ij / nprobe;
                    float* xij = &xrel[ij * d];
                    idx_t cij = cq.ids[ij];

                    if (cij >= 0) {
                        quantizer->compute_residual(x + i * d, xij, cij);
                    } else {
                        // will fill with NaNs
                        memset(xij, -1, sizeof(float) * d);
                    }
                }

                pq.compute_distance_tables(
                        n * nprobe, xrel.get(), dis_tables.get());
            }

        } else if (metric_type == METRIC_INNER_PRODUCT) {
            dis_tables.resize(n * dim12);
            pq.compute_inner_prod_tables(n, x, dis_tables.get());
            // compute_inner_prod_tables(pq, n, x, dis_tables.get());

            biases.resize(n * nprobe);
            memcpy(biases.get(), cq.dis, sizeof(float) * n * nprobe);
        } else {
            FAISS_THROW_FMT("metric %d not supported", metric_type);
        }

    } else {
        dis_tables.resize(n * dim12);
        if (metric_type == METRIC_L2) {
            pq.compute_distance_tables(n, x, dis_tables.get());
        } else if (metric_type == METRIC_INNER_PRODUCT) {
            pq.compute_inner_prod_tables(n, x, dis_tables.get());
        } else {
            FAISS_THROW_FMT("metric %d not supported", metric_type);
        }
    }
}

/*********************************************************
 * InvertedListScanner for IVFPQFS
 *********************************************************/

namespace {

struct IVFPQFastScanScanner : InvertedListScanner {
    static constexpr int impl = 10; // based on search_implem_10
    static constexpr size_t nq = 1; // 1 query at a time.
    const IndexIVFPQFastScan& index;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::array<float, 2> normalizers{};
    const float* xi = nullptr;

    IVFPQFastScanScanner(
            const IndexIVFPQFastScan& index,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), index(index) {
        this->keep_max = is_similarity_metric(index.metric_type);
    }

    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        IndexIVFFastScan::CoarseQuantized cq{
                .nprobe = 1,        // 1 due to explicitly passing in list_no
                .dis = &coarse_dis, // dis from query to list_no centroid.
                .ids = &list_no,    // id of the current list we are scanning
        };
        FastScanDistancePostProcessing empty_context{};
        index.compute_LUT_uint8(
                1, xi, cq, dis_tables, biases, &normalizers[0], empty_context);
    }

    float distance_to_code(const uint8_t* /* code */) const override {
        // It's not really possible to implement a distance_to_code since codes
        // for 32 database vectors are intermixed.
        FAISS_THROW_MSG("not implemented");
    }

    // Based on IVFFastScan search_implem_10, since it also deals with 1 query
    // at a time.
    size_t scan_codes(
            size_t ntotal,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const override {
        // initialize the current iteration heap to the worst possible value of
        // the prior loop
        std::vector<float> curr_dists(k, distances[0]);
        std::vector<idx_t> curr_labels(k, labels[0]);
        FastScanDistancePostProcessing empty_context{};
        std::unique_ptr<SIMDResultHandlerToFloat> handler(
                index.make_knn_handler(
                        !keep_max,
                        impl,
                        nq,
                        k,
                        curr_dists.data(),
                        curr_labels.data(),
                        sel,
                        empty_context,
                        &normalizers[0]));

        // This does not quite match search_implem_10, but it is fine because
        // the scanner operates on a single query at a time, and this value is
        // used as the query index. For a single query, the value is always 0.
        int qmap1[1] = {0};

        handler->q_map = qmap1;
        handler->begin(&normalizers[0]);

        const uint8_t* LUT = dis_tables.get();
        handler->dbias = biases.get();

        handler->ntotal = ntotal;
        handler->id_map = ids;

        pq4_accumulate_loop(
                1,
                roundup(ntotal, index.bbs),
                index.bbs,
                static_cast<int>(index.M2),
                codes,
                LUT,
                *handler,
                nullptr);

        // The handler is for the results of this iteration.
        // Then we need a second heap to combine across iterations.
        handler->end();
        if (keep_max) {
            minheap_addn(
                    k,
                    distances,
                    labels,
                    curr_dists.data(),
                    curr_labels.data(),
                    k);
        } else {
            maxheap_addn(
                    k,
                    distances,
                    labels,
                    curr_dists.data(),
                    curr_labels.data(),
                    k);
        }

        return handler->num_updates();
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFPQFastScan::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    return new IVFPQFastScanScanner(*this, store_pairs, sel);
}

} // namespace faiss
