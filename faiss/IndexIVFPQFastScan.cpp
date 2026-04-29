/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFPQFastScan.h>

#include <array>
#include <cstdio>

#include <memory>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/distances_dispatch.h>
#include <faiss/utils/extra_distances.h>

#include <faiss/invlists/BlockInvertedLists.h>

#include <faiss/impl/fast_scan/FastScanDistancePostProcessing.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexIVFPQFastScan::IndexIVFPQFastScan(
        Index* quantizer_in,
        size_t d_in,
        size_t nlist_in,
        size_t M_in,
        size_t nbits_in,
        MetricType metric,
        int bbs_in,
        bool own_invlists_in)
        : IndexIVFFastScan(
                  quantizer_in,
                  d_in,
                  nlist_in,
                  0,
                  metric,
                  own_invlists_in),
          pq(d_in, M_in, nbits_in) {
    by_residual = false; // set to false by default because it's faster

    init_fastscan(
            &pq, M_in, nbits_in, nlist_in, metric, bbs_in, own_invlists_in);
}

IndexIVFPQFastScan::IndexIVFPQFastScan() {
    by_residual = false;
    bbs = 0;
    M2 = 0;
}

IndexIVFPQFastScan::IndexIVFPQFastScan(const IndexIVFPQ& orig, int bbs_in)
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
            bbs_in,
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
    for (idx_t i = 0; i < static_cast<idx_t>(nlist); i++) {
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

size_t IndexIVFPQFastScan::fast_scan_code_size() const {
    return M2 / 2;
}

/*********************************************************
 * Training
 *********************************************************/

void IndexIVFPQFastScan::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* /*assign*/) {
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
        for (idx_t i = 0; i < n; i++) {
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

// Explicit SIMD-level alias (no global bare aliases).
using simd8float32 = simd8float32_tpl<SINGLE_SIMD_LEVEL_256>;

void fvec_madd_simd(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    FAISS_THROW_IF_NOT_MSG(is_aligned_pointer(a), "pointer a is not aligned");
    FAISS_THROW_IF_NOT_MSG(is_aligned_pointer(b), "pointer b is not aligned");
    FAISS_THROW_IF_NOT_MSG(is_aligned_pointer(c), "pointer c is not aligned");
    FAISS_THROW_IF_NOT_MSG(n % 8 == 0, "n must be a multiple of 8");
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
    size_t pq_d = pq.d;
    size_t cq_nprobe = cq.nprobe;

    if (by_residual) {
        if (metric_type == METRIC_L2) {
            dis_tables.resize(n * cq_nprobe * dim12);

            if (use_precomputed_table == 1) {
                biases.resize(n * cq_nprobe);
                memcpy(biases.get(), cq.dis, sizeof(float) * n * cq_nprobe);

                AlignedTable<float> ip_table(n * dim12);
                pq.compute_inner_prod_tables(n, x, ip_table.get());

#pragma omp parallel for if (n * cq_nprobe > 8000)
                for (idx_t ij = 0; ij < static_cast<idx_t>(n * cq_nprobe);
                     ij++) {
                    idx_t i = ij / cq_nprobe;
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
                std::unique_ptr<float[]> xrel(new float[n * cq_nprobe * pq_d]);
                biases.resize(n * cq_nprobe);
                memset(biases.get(), 0, sizeof(float) * n * cq_nprobe);

#pragma omp parallel for if (n * cq_nprobe > 8000)
                for (idx_t ij = 0; ij < static_cast<idx_t>(n * cq_nprobe);
                     ij++) {
                    idx_t i = ij / cq_nprobe;
                    float* xij = &xrel[ij * pq_d];
                    idx_t cij = cq.ids[ij];

                    if (cij >= 0) {
                        quantizer->compute_residual(x + i * pq_d, xij, cij);
                    } else {
                        // will fill with NaNs
                        memset(xij, -1, sizeof(float) * pq_d);
                    }
                }

                pq.compute_distance_tables(
                        n * cq_nprobe, xrel.get(), dis_tables.get());
            }

        } else if (metric_type == METRIC_INNER_PRODUCT) {
            dis_tables.resize(n * dim12);
            pq.compute_inner_prod_tables(n, x, dis_tables.get());
            // compute_inner_prod_tables(pq, n, x, dis_tables.get());

            biases.resize(n * cq_nprobe);
            memcpy(biases.get(), cq.dis, sizeof(float) * n * cq_nprobe);
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
    using InvertedListScanner::scan_codes;
    [[maybe_unused]] static constexpr int impl =
            10;                     // based on search_implem_10
    static constexpr size_t nq = 1; // 1 query at a time.
    const IndexIVFPQFastScan& index;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::vector<float> residual;
    std::array<float, 2> normalizers{};
    const float* xi = nullptr;

    IVFPQFastScanScanner(
            const IndexIVFPQFastScan& index_in,
            bool store_pairs_in,
            const IDSelector* sel_in)
            : InvertedListScanner(store_pairs_in, sel_in), index(index_in) {
        this->keep_max = is_similarity_metric(index_in.metric_type);
        residual.resize(index_in.d);
    }

    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no_in, float coarse_dis_in) override {
        this->list_no = list_no_in;
        IndexIVFFastScan::CoarseQuantized cq{
                .nprobe = 1,           // 1 due to explicitly passing in list_no
                .dis = &coarse_dis_in, // dis from query to list_no centroid.
                .ids = &list_no_in,    // id of the current list we are scanning
        };
        FastScanDistancePostProcessing empty_context{};
        index.compute_LUT_uint8(
                1, xi, cq, dis_tables, biases, &normalizers[0], empty_context);
        // used in distance_to_code
        index.quantizer->compute_residual(
                this->xi, residual.data(), this->list_no);
    }

    float distance_to_code(const uint8_t* code) const override {
        // directly use the PQ tables to compute the distance
        const ProductQuantizer& pq = index.pq;
        // when by_residual, codes are residuals so compare against query
        // residual; otherwise codes are raw vectors so compare against raw
        // query
        const float* x = index.by_residual ? residual.data() : this->xi;
        float accu = 0;
        // implemented for all vector distances, although only L2 and IP are
        // supported by FastScan
        with_VectorDistance(pq.dsub, index.metric_type, 0.0, [&](auto vd) {
            int m;
            for (m = 0; m + 1 < pq.M; m += 2) {
                const float* cent;
                uint8_t c = *code++;
                cent = pq.get_centroids(m, c & 15);
                accu += vd(cent, x);
                x += pq.dsub;
                cent = pq.get_centroids(m + 1, c >> 4);
                accu += vd(cent, x);
                x += pq.dsub;
            }
            if (m < pq.M) { // leftover
                uint8_t c = *code++;
                const float* cent = pq.get_centroids(m, c & 15);
                accu += vd(cent, x);
            }
        });
        return accu;
    }

    // Based on IVFFastScan search_implem_10, since it also deals with 1 query
    // at a time.
    size_t scan_codes(
            size_t ntotal,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        auto scan_with_heap = [&](auto* heap_handler) -> size_t {
            const size_t k = heap_handler->k;
            if (k == 0) {
                return 0;
            }

            // initialize the current iteration heap to the worst possible value
            // of the caller-owned result handler.
            std::vector<float> curr_dists(k, handler.threshold);
            std::vector<idx_t> curr_labels(k, -1);

            auto scanner = index.make_knn_scanner(
                    !keep_max,
                    nq,
                    k,
                    curr_dists.data(),
                    curr_labels.data(),
                    sel);

            SIMDResultHandlerToFloat* rh = scanner->handler();

            // This does not quite match search_implem_10, but it is fine
            // because the scanner operates on a single query at a time, and
            // this value is used as the query index. For a single query, the
            // value is always 0.
            int qmap1[1] = {0};

            rh->q_map = qmap1;
            rh->begin(&normalizers[0]);

            rh->dbias = biases.get();
            rh->ntotal = ntotal;
            rh->id_map = ids;

            scanner->accumulate_loop(
                    1,
                    roundup(ntotal, index.bbs),
                    index.bbs,
                    static_cast<int>(index.M2),
                    codes,
                    dis_tables.get(),
                    0,
                    index.get_block_stride());

            const size_t scan_cnt = rh->count_scanned_rows();
            rh->end();

            handler.stats.scan_cnt += scan_cnt;
            size_t nup = 0;
            for (size_t j = 0; j < k; j++) {
                if (curr_labels[j] < 0) {
                    continue;
                }
                if (handler.add_result(curr_dists[j], curr_labels[j])) {
                    handler.stats.nheap_updates++;
                    nup++;
                }
            }
            return nup;
        };

        if (!keep_max) {
            using C = CMax<float, idx_t>;
            if (auto* heap_handler =
                        dynamic_cast<HeapResultHandler<C, false>*>(&handler)) {
                return scan_with_heap(heap_handler);
            }
        } else {
            using C = CMin<float, idx_t>;
            if (auto* heap_handler =
                        dynamic_cast<HeapResultHandler<C, false>*>(&handler)) {
                return scan_with_heap(heap_handler);
            }
        }

        FAISS_THROW_MSG(
                "IVFPQFastScanScanner::scan_codes requires HeapResultHandler; "
                "custom ResultHandler scan is not supported by this optimized "
                "scanner");
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
