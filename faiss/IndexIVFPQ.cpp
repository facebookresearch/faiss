/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFPQ.h>

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>

#include <faiss/utils/distances_dispatch.h>
#include <faiss/utils/utils.h>

#include <faiss/Clustering.h>

#include <faiss/utils/hamming.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/pq_code_distance/pq_code_distance-generic.h>
#include <faiss/impl/simd_dispatch.h>

// Scalar (NONE) fallback for dynamic dispatch
#define THE_SIMD_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/IVFPQScanner_impl.h>
#undef THE_SIMD_LEVEL

namespace faiss {

/*****************************************
 * IndexIVFPQ implementation
 ******************************************/

IndexIVFPQ::IndexIVFPQ(
        Index* quantizer_in,
        size_t d_in,
        size_t nlist_in,
        size_t M,
        size_t nbits_per_idx,
        MetricType metric,
        bool own_invlists_in)
        : IndexIVF(quantizer_in, d_in, nlist_in, 0, metric, own_invlists_in),
          pq(d_in, M, nbits_per_idx) {
    code_size = pq.code_size;
    if (own_invlists_in) {
        invlists->code_size = code_size;
    }
    is_trained = false;
    by_residual = true;
    use_precomputed_table = 0;
    scan_table_threshold = 0;

    polysemous_training = nullptr;
    do_polysemous_training = false;
    polysemous_ht = 0;
}

/****************************************************************
 * training                                                     */

void IndexIVFPQ::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* /*assign*/) {
    pq.train(n, x);

    if (do_polysemous_training) {
        if (verbose) {
            printf("doing polysemous training for PQ\n");
        }
        PolysemousTraining default_pt;
        PolysemousTraining* pt =
                polysemous_training ? polysemous_training : &default_pt;
        pt->optimize_pq_for_hamming(pq, n, x);
    }

    if (by_residual) {
        precompute_table();
    }
}

idx_t IndexIVFPQ::train_encoder_num_vectors() const {
    return pq.cp.max_points_per_centroid * pq.ksub;
}

/****************************************************************
 * IVFPQ as codec                                               */

/* produce a binary signature based on the residual vector */
void IndexIVFPQ::encode(idx_t key, const float* x, uint8_t* code) const {
    if (by_residual) {
        std::vector<float> residual_vec(d);
        quantizer->compute_residual(x, residual_vec.data(), key);
        pq.compute_code(residual_vec.data(), code);
    } else {
        pq.compute_code(x, code);
    }
}

void IndexIVFPQ::encode_multiple(
        size_t n,
        idx_t* keys,
        const float* x,
        uint8_t* xcodes,
        bool compute_keys) const {
    if (compute_keys) {
        quantizer->assign(n, x, keys);
    }

    encode_vectors(n, x, keys, xcodes);
}

void IndexIVFPQ::decode_multiple(
        size_t n,
        const idx_t* keys,
        const uint8_t* xcodes,
        float* x) const {
    pq.decode(xcodes, x, n);
    if (by_residual) {
        std::vector<float> centroid(d);
        for (size_t i = 0; i < n; i++) {
            quantizer->reconstruct(keys[i], centroid.data());
            float* xi = x + i * d;
            for (int j = 0; j < d; j++) {
                xi[j] += centroid[j];
            }
        }
    }
}

/****************************************************************
 * add                                                          */

void IndexIVFPQ::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    add_core_o(n, x, xids, nullptr, coarse_idx, inverted_list_context);
}

static std::unique_ptr<float[]> compute_residuals(
        const Index* quantizer,
        idx_t n,
        const float* x,
        const idx_t* list_nos) {
    size_t d = quantizer->d;
    std::unique_ptr<float[]> residuals(new float[n * d]);
    // Parallelize with OpenMP (each iteration is independent)
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        if (list_nos[i] < 0) {
            memset(residuals.get() + i * d, 0, sizeof(float) * d);
        } else {
            quantizer->compute_residual(
                    x + i * d, residuals.get() + i * d, list_nos[i]);
        }
    }
    return residuals;
}

void IndexIVFPQ::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    if (by_residual) {
        std::unique_ptr<float[]> to_encode =
                compute_residuals(quantizer, n, x, list_nos);
        pq.compute_codes(to_encode.get(), codes, n);
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

void IndexIVFPQ::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* listnos,
        float* x) const {
    return decode_multiple(n, listnos, codes, x);
}

void IndexIVFPQ::sa_decode(idx_t n, const uint8_t* codes, float* x) const {
    size_t coarse_size = coarse_code_size();

#pragma omp parallel
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;
            pq.decode(code + coarse_size, xi);
            if (by_residual) {
                quantizer->reconstruct(list_no, residual.data());
                for (int j = 0; j < d; j++) {
                    xi[j] += residual[j];
                }
            }
        }
    }
}

// block size used in IndexIVFPQ::add_core_o
int index_ivfpq_add_core_o_bs = 32768;

void IndexIVFPQ::add_core_o(
        idx_t n,
        const float* x,
        const idx_t* xids,
        float* residuals_2,
        const idx_t* precomputed_idx,
        void* inverted_list_context) {
    idx_t bs = index_ivfpq_add_core_o_bs;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(i0 + bs, n);
            if (verbose) {
                printf("IndexIVFPQ::add_core_o: adding %" PRId64 ":%" PRId64
                       " / %" PRId64 "\n",
                       i0,
                       i1,
                       n);
            }
            add_core_o(
                    i1 - i0,
                    x + i0 * d,
                    xids ? xids + i0 : nullptr,
                    residuals_2 ? residuals_2 + i0 * d : nullptr,
                    precomputed_idx ? precomputed_idx + i0 : nullptr,
                    inverted_list_context);
        }
        return;
    }

    InterruptCallback::check();

    direct_map.check_can_add(xids);

    FAISS_THROW_IF_NOT(is_trained);
    double t0 = getmillisecs();
    const idx_t* idx;
    std::unique_ptr<idx_t[]> del_idx;

    if (precomputed_idx) {
        idx = precomputed_idx;
    } else {
        idx_t* idx0 = new idx_t[n];
        del_idx.reset(idx0);
        quantizer->assign(n, x, idx0);
        idx = idx0;
    }

    double t1 = getmillisecs();
    std::unique_ptr<uint8_t[]> xcodes(new uint8_t[n * code_size]);

    const float* to_encode = nullptr;
    std::unique_ptr<const float[]> del_to_encode;

    if (by_residual) {
        del_to_encode = compute_residuals(quantizer, n, x, idx);
        to_encode = del_to_encode.get();
    } else {
        to_encode = x;
    }
    pq.compute_codes(to_encode, xcodes.get(), n);

    double t2 = getmillisecs();
    // TODO: parallelize?
    size_t n_ignore = 0;
    for (idx_t i = 0; i < n; i++) {
        idx_t key = idx[i];
        idx_t id = xids ? xids[i] : ntotal + i;
        if (key < 0) {
            direct_map.add_single_id(id, -1, 0);
            n_ignore++;
            if (residuals_2) {
                memset(residuals_2, 0, sizeof(*residuals_2) * d);
            }
            continue;
        }

        uint8_t* code = xcodes.get() + i * code_size;
        size_t offset =
                invlists->add_entry(key, id, code, inverted_list_context);

        if (residuals_2) {
            float* res2 = residuals_2 + i * d;
            const float* xi = to_encode + i * d;
            pq.decode(code, res2);
            for (int j = 0; j < d; j++) {
                res2[j] = xi[j] - res2[j];
            }
        }

        direct_map.add_single_id(id, key, offset);
    }

    double t3 = getmillisecs();
    if (verbose) {
        char comment[100] = {0};
        if (n_ignore > 0) {
            snprintf(comment, 100, "(%zd vectors ignored)", n_ignore);
        }
        printf(" add_core times: %.3f %.3f %.3f %s\n",
               t1 - t0,
               t2 - t1,
               t3 - t2,
               comment);
    }
    ntotal += n;
}

void IndexIVFPQ::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);

    pq.decode(code, recons);
    if (by_residual) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        for (int i = 0; i < d; ++i) {
            recons[i] += centroid[i];
        }
    }
}

/// 2G by default, accommodates tables up to PQ32 w/ 65536 centroids
size_t precomputed_table_max_bytes = ((size_t)1) << 31;

/** Precomputed tables for residuals
 *
 * During IVFPQ search with by_residual, we compute
 *
 *     d = || x - y_C - y_R ||^2
 *
 * where x is the query vector, y_C the coarse centroid, y_R the
 * refined PQ centroid. The expression can be decomposed as:
 *
 *    d = || x - y_C ||^2 + || y_R ||^2 + 2 * (y_C|y_R) - 2 * (x|y_R)
 *        ---------------   ---------------------------       -------
 *             term 1                 term 2                   term 3
 *
 * When using multiprobe, we use the following decomposition:
 * - term 1 is the distance to the coarse centroid, that is computed
 *   during the 1st stage search.
 * - term 2 can be precomputed, as it does not involve x. However,
 *   because of the PQ, it needs nlist * M * ksub storage. This is why
 *   use_precomputed_table is off by default
 * - term 3 is the classical non-residual distance table.
 *
 * Since y_R defined by a product quantizer, it is split across
 * subvectors and stored separately for each subvector. If the coarse
 * quantizer is a MultiIndexQuantizer then the table can be stored
 * more compactly.
 *
 * At search time, the tables for term 2 and term 3 are added up. This
 * is faster when the length of the lists is > ksub * M.
 */

void initialize_IVFPQ_precomputed_table(
        int& use_precomputed_table,
        const Index* quantizer,
        const ProductQuantizer& pq,
        AlignedTable<float>& precomputed_table,
        bool by_residual,
        bool verbose) {
    FAISS_THROW_IF_NOT_MSG(quantizer, "IVF quantizer must not be null");
    size_t nlist = quantizer->ntotal;
    size_t d = quantizer->d;
    FAISS_THROW_IF_NOT(d == pq.d);

    if (use_precomputed_table == -1) {
        precomputed_table.resize(0);
        return;
    }

    if (use_precomputed_table == 0) { // then choose the type of table
        if (!(quantizer->metric_type == METRIC_L2 && by_residual)) {
            if (verbose) {
                printf("IndexIVFPQ::precompute_table: precomputed "
                       "tables needed only for L2 metric and by_residual is enabled\n");
            }
            precomputed_table.resize(0);
            return;
        }
        const MultiIndexQuantizer* miq =
                dynamic_cast<const MultiIndexQuantizer*>(quantizer);
        if (miq && pq.M % miq->pq.M == 0) {
            use_precomputed_table = 2;
        } else {
            size_t table_size = pq.M * pq.ksub * nlist * sizeof(float);
            if (table_size > precomputed_table_max_bytes) {
                if (verbose) {
                    printf("IndexIVFPQ::precompute_table: not precomputing table, "
                           "it would be too big: %zd bytes (max %zd)\n",
                           table_size,
                           precomputed_table_max_bytes);
                    use_precomputed_table = 0;
                }
                return;
            }
            use_precomputed_table = 1;
        }
    } // otherwise assume user has set appropriate flag on input

    if (verbose) {
        printf("precomputing IVFPQ tables type %d\n", use_precomputed_table);
    }

    // squared norms of the PQ centroids
    std::vector<float> r_norms(pq.M * pq.ksub, NAN);
    for (size_t m = 0; m < pq.M; m++) {
        for (size_t j = 0; j < pq.ksub; j++) {
            r_norms[m * pq.ksub + j] =
                    fvec_norm_L2sqr_dispatch(pq.get_centroids(m, j), pq.dsub);
        }
    }

    if (use_precomputed_table == 1) {
        precomputed_table.resize(nlist * pq.M * pq.ksub);
        std::vector<float> centroid(d);

        for (size_t i = 0; i < nlist; i++) {
            quantizer->reconstruct(i, centroid.data());

            float* tab = &precomputed_table[i * pq.M * pq.ksub];
            pq.compute_inner_prod_table(centroid.data(), tab);
            fvec_madd_dispatch(pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
        }
    } else if (use_precomputed_table == 2) {
        const MultiIndexQuantizer* miq =
                dynamic_cast<const MultiIndexQuantizer*>(quantizer);
        FAISS_THROW_IF_NOT(miq);
        const ProductQuantizer& cpq = miq->pq;
        FAISS_THROW_IF_NOT(pq.M % cpq.M == 0);

        precomputed_table.resize(cpq.ksub * pq.M * pq.ksub);

        // reorder PQ centroid table
        std::vector<float> centroids(d * cpq.ksub, NAN);

        for (size_t m = 0; m < cpq.M; m++) {
            for (size_t i = 0; i < cpq.ksub; i++) {
                memcpy(centroids.data() + i * d + m * cpq.dsub,
                       cpq.get_centroids(m, i),
                       sizeof(*centroids.data()) * cpq.dsub);
            }
        }

        pq.compute_inner_prod_tables(
                cpq.ksub, centroids.data(), precomputed_table.data());

        for (size_t i = 0; i < cpq.ksub; i++) {
            float* tab = &precomputed_table[i * pq.M * pq.ksub];
            fvec_madd_dispatch(pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
        }
    }
}

void IndexIVFPQ::precompute_table() {
    initialize_IVFPQ_precomputed_table(
            use_precomputed_table,
            quantizer,
            pq,
            precomputed_table,
            by_residual,
            verbose);
}

InvertedListScanner* IndexIVFPQ::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    return with_simd_level([&]<SIMDLevel SL>() -> InvertedListScanner* {
        return pq_code_distance::make_IVFPQInvertedListScanner<SL>(
                *this, store_pairs, sel);
    });
}

IndexIVFPQStats indexIVFPQ_stats;

void IndexIVFPQStats::reset() {
    memset(this, 0, sizeof(*this));
}

IndexIVFPQ::IndexIVFPQ() {
    // initialize some runtime values
    use_precomputed_table = 0;
    scan_table_threshold = 0;
    do_polysemous_training = false;
    polysemous_ht = 0;
    polysemous_training = nullptr;
}

struct CodeCmp {
    const uint8_t* tab;
    size_t code_size;
    bool operator()(int a, int b) const {
        return cmp(a, b) > 0;
    }
    int cmp(int a, int b) const {
        return memcmp(tab + a * code_size, tab + b * code_size, code_size);
    }
};

size_t IndexIVFPQ::find_duplicates(idx_t* dup_ids, size_t* lims) const {
    size_t ngroup = 0;
    lims[0] = 0;
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t n = invlists->list_size(list_no);
        std::vector<int> ord(n);
        for (size_t i = 0; i < n; i++) {
            ord[i] = static_cast<int>(i);
        }
        InvertedLists::ScopedCodes codes(invlists, list_no);
        CodeCmp cs = {codes.get(), code_size};
        std::sort(ord.begin(), ord.end(), cs);

        InvertedLists::ScopedIds list_ids(invlists, list_no);
        int prev = -1; // all elements from prev to i-1 are equal
        for (size_t i = 0; i < n; i++) {
            if (prev >= 0 && cs.cmp(ord[prev], ord[i]) == 0) {
                // same as previous => remember
                if (static_cast<size_t>(prev + 1) == i) { // start new group
                    ngroup++;
                    lims[ngroup] = lims[ngroup - 1];
                    dup_ids[lims[ngroup]++] = list_ids[ord[prev]];
                }
                dup_ids[lims[ngroup]++] = list_ids[ord[i]];
            } else { // not same as previous.
                prev = static_cast<int>(i);
            }
        }
    }
    return ngroup;
}

} // namespace faiss
