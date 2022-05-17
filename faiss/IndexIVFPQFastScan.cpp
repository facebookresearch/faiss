/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFPQFastScan.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>

#include <omp.h>

#include <memory>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/simdlib.h>
#include <faiss/utils/utils.h>

#include <faiss/invlists/BlockInvertedLists.h>

#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/quantize_lut.h>

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
        int bbs)
        : IndexIVFFastScan(quantizer, d, nlist, 0, metric), pq(d, M, nbits) {
    by_residual = false; // set to false by default because it's much faster

    init_fastscan(M, nbits, nlist, metric, bbs);
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
                  orig.metric_type),
          pq(orig.pq) {
    FAISS_THROW_IF_NOT(orig.pq.nbits == 4);

    init_fastscan(orig.pq.M, orig.pq.nbits, orig.nlist, orig.metric_type, bbs);

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

    for (size_t i = 0; i < nlist; i++) {
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

void IndexIVFPQFastScan::train_residual(idx_t n, const float* x_in) {
    const float* x = fvecs_maybe_subsample(
            d,
            (size_t*)&n,
            pq.cp.max_points_per_centroid * pq.ksub,
            x_in,
            verbose,
            pq.cp.seed);

    std::unique_ptr<float[]> del_x;
    if (x != x_in) {
        del_x.reset((float*)x);
    }

    const float* trainset;
    AlignedTable<float> residuals;

    if (by_residual) {
        if (verbose)
            printf("computing residuals\n");
        std::vector<idx_t> assign(n);
        quantizer->assign(n, x, assign.data());
        residuals.resize(n * d);
        for (idx_t i = 0; i < n; i++) {
            quantizer->compute_residual(
                    x + i * d, residuals.data() + i * d, assign[i]);
        }
        trainset = residuals.data();
    } else {
        trainset = x;
    }

    if (verbose) {
        printf("training %zdx%zd product quantizer on "
               "%" PRId64 " vectors in %dD\n",
               pq.M,
               pq.ksub,
               n,
               d);
    }
    pq.verbose = verbose;
    pq.train(n, trainset);

    if (by_residual && metric_type == METRIC_L2) {
        precompute_table();
    }
}

void IndexIVFPQFastScan::precompute_table() {
    initialize_IVFPQ_precomputed_table(
            use_precomputed_table, quantizer, pq, precomputed_table, verbose);
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

void fvec_madd_avx(
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
        const idx_t* coarse_ids,
        const float* coarse_dis,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& biases) const {
    size_t dim12 = pq.ksub * pq.M;
    size_t d = pq.d;

    if (by_residual) {
        if (metric_type == METRIC_L2) {
            dis_tables.resize(n * nprobe * dim12);

            if (use_precomputed_table == 1) {
                biases.resize(n * nprobe);
                memcpy(biases.get(), coarse_dis, sizeof(float) * n * nprobe);

                AlignedTable<float> ip_table(n * dim12);
                pq.compute_inner_prod_tables(n, x, ip_table.get());

#pragma omp parallel for if (n * nprobe > 8000)
                for (idx_t ij = 0; ij < n * nprobe; ij++) {
                    idx_t i = ij / nprobe;
                    float* tab = dis_tables.get() + ij * dim12;
                    idx_t cij = coarse_ids[ij];

                    if (cij >= 0) {
                        fvec_madd_avx(
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
                    idx_t cij = coarse_ids[ij];

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
            memcpy(biases.get(), coarse_dis, sizeof(float) * n * nprobe);
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

void IndexIVFPQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    pq.decode(bytes, x, n);
}

} // namespace faiss
