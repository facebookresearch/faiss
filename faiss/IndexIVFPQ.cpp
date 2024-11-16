/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFPQ.h>

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>

#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <faiss/Clustering.h>

#include <faiss/utils/hamming.h>

#include <faiss/impl/FaissAssert.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/ProductQuantizer.h>

#include <faiss/impl/code_distance/code_distance.h>

namespace faiss {

/*****************************************
 * IndexIVFPQ implementation
 ******************************************/

IndexIVFPQ::IndexIVFPQ(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        MetricType metric)
        : IndexIVF(quantizer, d, nlist, 0, metric), pq(d, M, nbits_per_idx) {
    code_size = pq.code_size;
    invlists->code_size = code_size;
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

void IndexIVFPQ::train_encoder(idx_t n, const float* x, const idx_t* assign) {
    pq.train(n, x);

    if (do_polysemous_training) {
        if (verbose)
            printf("doing polysemous training for PQ\n");
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
    } else
        pq.compute_code(x, code);
}

void IndexIVFPQ::encode_multiple(
        size_t n,
        idx_t* keys,
        const float* x,
        uint8_t* xcodes,
        bool compute_keys) const {
    if (compute_keys)
        quantizer->assign(n, x, keys);

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
            for (size_t j = 0; j < d; j++) {
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
    // TODO: parallelize?
    for (size_t i = 0; i < n; i++) {
        if (list_nos[i] < 0)
            memset(residuals.get() + i * d, 0, sizeof(float) * d);
        else
            quantizer->compute_residual(
                    x + i * d, residuals.get() + i * d, list_nos[i]);
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
                for (size_t j = 0; j < d; j++) {
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
    for (size_t i = 0; i < n; i++) {
        idx_t key = idx[i];
        idx_t id = xids ? xids[i] : ntotal + i;
        if (key < 0) {
            direct_map.add_single_id(id, -1, 0);
            n_ignore++;
            if (residuals_2)
                memset(residuals_2, 0, sizeof(*residuals_2) * d);
            continue;
        }

        uint8_t* code = xcodes.get() + i * code_size;
        size_t offset =
                invlists->add_entry(key, id, code, inverted_list_context);

        if (residuals_2) {
            float* res2 = residuals_2 + i * d;
            const float* xi = to_encode + i * d;
            pq.decode(code, res2);
            for (int j = 0; j < d; j++)
                res2[j] = xi[j] - res2[j];
        }

        direct_map.add_single_id(id, key, offset);
    }

    double t3 = getmillisecs();
    if (verbose) {
        char comment[100] = {0};
        if (n_ignore > 0)
            snprintf(comment, 100, "(%zd vectors ignored)", n_ignore);
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
        if (miq && pq.M % miq->pq.M == 0)
            use_precomputed_table = 2;
        else {
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
    for (int m = 0; m < pq.M; m++)
        for (int j = 0; j < pq.ksub; j++)
            r_norms[m * pq.ksub + j] =
                    fvec_norm_L2sqr(pq.get_centroids(m, j), pq.dsub);

    if (use_precomputed_table == 1) {
        precomputed_table.resize(nlist * pq.M * pq.ksub);
        std::vector<float> centroid(d);

        for (size_t i = 0; i < nlist; i++) {
            quantizer->reconstruct(i, centroid.data());

            float* tab = &precomputed_table[i * pq.M * pq.ksub];
            pq.compute_inner_prod_table(centroid.data(), tab);
            fvec_madd(pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
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

        for (int m = 0; m < cpq.M; m++) {
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
            fvec_madd(pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
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

namespace {

#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0

/** QueryTables manages the various ways of searching an
 * IndexIVFPQ. The code contains a lot of branches, depending on:
 * - metric_type: are we computing L2 or Inner product similarity?
 * - by_residual: do we encode raw vectors or residuals?
 * - use_precomputed_table: are x_R|x_C tables precomputed?
 * - polysemous_ht: are we filtering with polysemous codes?
 */
struct QueryTables {
    /*****************************************************
     * General data from the IVFPQ
     *****************************************************/

    const IndexIVFPQ& ivfpq;
    const IVFSearchParameters* params;

    // copied from IndexIVFPQ for easier access
    int d;
    const ProductQuantizer& pq;
    MetricType metric_type;
    bool by_residual;
    int use_precomputed_table;
    int polysemous_ht;

    // pre-allocated data buffers
    float *sim_table, *sim_table_2;
    float *residual_vec, *decoded_vec;

    // single data buffer
    std::vector<float> mem;

    // for table pointers
    std::vector<const float*> sim_table_ptrs;

    explicit QueryTables(
            const IndexIVFPQ& ivfpq,
            const IVFSearchParameters* params)
            : ivfpq(ivfpq),
              d(ivfpq.d),
              pq(ivfpq.pq),
              metric_type(ivfpq.metric_type),
              by_residual(ivfpq.by_residual),
              use_precomputed_table(ivfpq.use_precomputed_table) {
        mem.resize(pq.ksub * pq.M * 2 + d * 2);
        sim_table = mem.data();
        sim_table_2 = sim_table + pq.ksub * pq.M;
        residual_vec = sim_table_2 + pq.ksub * pq.M;
        decoded_vec = residual_vec + d;

        // for polysemous
        polysemous_ht = ivfpq.polysemous_ht;
        if (auto ivfpq_params =
                    dynamic_cast<const IVFPQSearchParameters*>(params)) {
            polysemous_ht = ivfpq_params->polysemous_ht;
        }
        if (polysemous_ht != 0) {
            q_code.resize(pq.code_size);
        }
        init_list_cycles = 0;
        sim_table_ptrs.resize(pq.M);
    }

    /*****************************************************
     * What we do when query is known
     *****************************************************/

    // field specific to query
    const float* qi;

    // query-specific initialization
    void init_query(const float* qi) {
        this->qi = qi;
        if (metric_type == METRIC_INNER_PRODUCT)
            init_query_IP();
        else
            init_query_L2();
        if (!by_residual && polysemous_ht != 0)
            pq.compute_code(qi, q_code.data());
    }

    void init_query_IP() {
        // precompute some tables specific to the query qi
        pq.compute_inner_prod_table(qi, sim_table);
    }

    void init_query_L2() {
        if (!by_residual) {
            pq.compute_distance_table(qi, sim_table);
        } else if (use_precomputed_table) {
            pq.compute_inner_prod_table(qi, sim_table_2);
        }
    }

    /*****************************************************
     * When inverted list is known: prepare computations
     *****************************************************/

    // fields specific to list
    idx_t key;
    float coarse_dis;
    std::vector<uint8_t> q_code;

    uint64_t init_list_cycles;

    /// once we know the query and the centroid, we can prepare the
    /// sim_table that will be used for accumulation
    /// and dis0, the initial value
    float precompute_list_tables() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                dis0 = precompute_list_tables_IP();
            else
                dis0 = precompute_list_tables_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    float precompute_list_table_pointers() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                FAISS_THROW_MSG("not implemented");
            else
                dis0 = precompute_list_table_pointers_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    /*****************************************************
     * compute tables for inner prod
     *****************************************************/

    float precompute_list_tables_IP() {
        // prepare the sim_table that will be used for accumulation
        // and dis0, the initial value
        ivfpq.quantizer->reconstruct(key, decoded_vec);
        // decoded_vec = centroid
        float dis0 = fvec_inner_product(qi, decoded_vec, d);

        if (polysemous_ht) {
            for (int i = 0; i < d; i++) {
                residual_vec[i] = qi[i] - decoded_vec[i];
            }
            pq.compute_code(residual_vec, q_code.data());
        }
        return dis0;
    }

    /*****************************************************
     * compute tables for L2 distance
     *****************************************************/

    float precompute_list_tables_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 0 || use_precomputed_table == -1) {
            ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            pq.compute_distance_table(residual_vec, sim_table);

            if (polysemous_ht != 0) {
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            fvec_madd(
                    pq.M * pq.ksub,
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M,
                    -2.0,
                    sim_table_2,
                    sim_table);

            if (polysemous_ht != 0) {
                ivfpq.quantizer->compute_residual(qi, residual_vec, key);
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            const float* qtab = sim_table_2; // query-specific table
            float* ltab = sim_table;         // (output) list-specific table

            long k = key;
            for (int cm = 0; cm < cpq.M; cm++) {
                // compute PQ index
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                // get corresponding table
                const float* pc = ivfpq.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                if (polysemous_ht == 0) {
                    // sum up with query-specific table
                    fvec_madd(Mf * pq.ksub, pc, -2.0, qtab, ltab);
                    ltab += Mf * pq.ksub;
                    qtab += Mf * pq.ksub;
                } else {
                    for (int m = cm * Mf; m < (cm + 1) * Mf; m++) {
                        q_code[m] = fvec_madd_and_argmin(
                                pq.ksub, pc, -2, qtab, ltab);
                        pc += pq.ksub;
                        ltab += pq.ksub;
                        qtab += pq.ksub;
                    }
                }
            }
        }

        return dis0;
    }

    float precompute_list_table_pointers_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            const float* s =
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M;
            for (int m = 0; m < pq.M; m++) {
                sim_table_ptrs[m] = s;
                s += pq.ksub;
            }
        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            long k = key;
            int m0 = 0;
            for (int cm = 0; cm < cpq.M; cm++) {
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                const float* pc = ivfpq.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                for (int m = m0; m < m0 + Mf; m++) {
                    sim_table_ptrs[m] = pc;
                    pc += pq.ksub;
                }
                m0 += Mf;
            }
        } else {
            FAISS_THROW_MSG("need precomputed tables");
        }

        if (polysemous_ht) {
            FAISS_THROW_MSG("not implemented");
            // Not clear that it makes sense to implemente this,
            // because it costs M * ksub, which is what we wanted to
            // avoid with the tables pointers.
        }

        return dis0;
    }
};

// This way of handling the selector is not optimal since all distances
// are computed even if the id would filter it out.
template <class C, bool use_sel>
struct KnnSearchResults {
    idx_t key;
    const idx_t* ids;
    const IDSelector* sel;

    // heap params
    size_t k;
    float* heap_sim;
    idx_t* heap_ids;

    size_t nup;

    inline bool skip_entry(idx_t j) {
        return use_sel && !sel->is_member(ids[j]);
    }

    inline void add(idx_t j, float dis) {
        if (C::cmp(heap_sim[0], dis)) {
            idx_t id = ids ? ids[j] : lo_build(key, j);
            heap_replace_top<C>(k, heap_sim, heap_ids, dis, id);
            nup++;
        }
    }
};

template <class C, bool use_sel>
struct RangeSearchResults {
    idx_t key;
    const idx_t* ids;
    const IDSelector* sel;

    // wrapped result structure
    float radius;
    RangeQueryResult& rres;

    inline bool skip_entry(idx_t j) {
        return use_sel && !sel->is_member(ids[j]);
    }

    inline void add(idx_t j, float dis) {
        if (C::cmp(radius, dis)) {
            idx_t id = ids ? ids[j] : lo_build(key, j);
            rres.add(dis, id);
        }
    }
};

/*****************************************************
 * Scaning the codes.
 * The scanning functions call their favorite precompute_*
 * function to precompute the tables they need.
 *****************************************************/
template <typename IDType, MetricType METRIC_TYPE, class PQDecoder>
struct IVFPQScannerT : QueryTables {
    const uint8_t* list_codes;
    const IDType* list_ids;
    size_t list_size;

    IVFPQScannerT(const IndexIVFPQ& ivfpq, const IVFSearchParameters* params)
            : QueryTables(ivfpq, params) {
        assert(METRIC_TYPE == metric_type);
    }

    float dis0;

    void init_list(idx_t list_no, float coarse_dis, int mode) {
        this->key = list_no;
        this->coarse_dis = coarse_dis;

        if (mode == 2) {
            dis0 = precompute_list_tables();
        } else if (mode == 1) {
            dis0 = precompute_list_table_pointers();
        }
    }

    /*****************************************************
     * Scaning the codes: simple PQ scan.
     *****************************************************/

    // This is the baseline version of scan_list_with_tables().
    // It demonstrates what this function actually does.
    //
    // /// version of the scan where we use precomputed tables.
    // template <class SearchResultType>
    // void scan_list_with_table(
    //         size_t ncode,
    //         const uint8_t* codes,
    //         SearchResultType& res) const {
    //
    //     for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
    //         if (res.skip_entry(j)) {
    //             continue;
    //         }
    //         float dis = dis0 + distance_single_code<PQDecoder>(
    //             pq, sim_table, codes);
    //         res.add(j, dis);
    //     }
    // }

    // This is the modified version of scan_list_with_tables().
    // It was observed that doing manual unrolling of the loop that
    //    utilizes distance_single_code() speeds up the computations.

    /// version of the scan where we use precomputed tables.
    template <class SearchResultType>
    void scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int counter = 0;

        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }

            saved_j[0] = (counter == 0) ? j : saved_j[0];
            saved_j[1] = (counter == 1) ? j : saved_j[1];
            saved_j[2] = (counter == 2) ? j : saved_j[2];
            saved_j[3] = (counter == 3) ? j : saved_j[3];

            counter += 1;
            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);
                counter = 0;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            res.add(saved_j[0], dis);
        }
        if (counter >= 2) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            res.add(saved_j[1], dis);
        }
        if (counter >= 3) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            res.add(saved_j[2], dis);
        }
    }

    /// tables are not precomputed, but pointers are provided to the
    /// relevant X_c|x_r tables
    template <class SearchResultType>
    void scan_list_with_pointer(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            PQDecoder decoder(codes, pq.nbits);
            float dis = dis0;
            const float* tab = sim_table_2;

            for (size_t m = 0; m < pq.M; m++) {
                int ci = decoder.decode();
                dis += sim_table_ptrs[m][ci] - 2 * tab[ci];
                tab += pq.ksub;
            }
            res.add(j, dis);
        }
    }

    /// nothing is precomputed: access residuals on-the-fly
    template <class SearchResultType>
    void scan_on_the_fly_dist(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        const float* dvec;
        float dis0 = 0;
        if (by_residual) {
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                ivfpq.quantizer->reconstruct(key, residual_vec);
                dis0 = fvec_inner_product(residual_vec, qi, d);
            } else {
                ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            }
            dvec = residual_vec;
        } else {
            dvec = qi;
            dis0 = 0;
        }

        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            pq.decode(codes, decoded_vec);

            float dis;
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                dis = dis0 + fvec_inner_product(decoded_vec, qi, d);
            } else {
                dis = fvec_L2sqr(decoded_vec, dvec, d);
            }
            res.add(j, dis);
        }
    }

    /*****************************************************
     * Scanning codes with polysemous filtering
     *****************************************************/

    // This is the baseline version of scan_list_polysemous_hc().
    // It demonstrates what this function actually does.

    //     template <class HammingComputer, class SearchResultType>
    //     void scan_list_polysemous_hc(
    //             size_t ncode,
    //             const uint8_t* codes,
    //             SearchResultType& res) const {
    //         int ht = ivfpq.polysemous_ht;
    //         size_t n_hamming_pass = 0, nup = 0;
    //
    //         int code_size = pq.code_size;
    //
    //         HammingComputer hc(q_code.data(), code_size);
    //
    //         for (size_t j = 0; j < ncode; j++, codes += code_size) {
    //             if (res.skip_entry(j)) {
    //                 continue;
    //             }
    //             const uint8_t* b_code = codes;
    //             int hd = hc.hamming(b_code);
    //             if (hd < ht) {
    //                 n_hamming_pass++;
    //
    //                 float dis =
    //                         dis0 +
    //                         distance_single_code<PQDecoder>(
    //                             pq, sim_table, codes);
    //
    //                 res.add(j, dis);
    //             }
    //         }
    // #pragma omp critical
    //         { indexIVFPQ_stats.n_hamming_pass += n_hamming_pass; }
    //     }

    // This is the modified version of scan_list_with_tables().
    // It was observed that doing manual unrolling of the loop that
    //    utilizes distance_single_code() speeds up the computations.

    template <class HammingComputer, class SearchResultType>
    void scan_list_polysemous_hc(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int ht = ivfpq.polysemous_ht;
        size_t n_hamming_pass = 0;

        int code_size = pq.code_size;

        size_t saved_j[8];
        int counter = 0;

        HammingComputer hc(q_code.data(), code_size);

        for (size_t j = 0; j < (ncode / 4) * 4; j += 4) {
            const uint8_t* b_code = codes + j * code_size;

            // Unrolling is a key. Basically, doing multiple popcount
            // operations one after another speeds things up.

            // 9999999 is just an arbitrary large number
            int hd0 = (res.skip_entry(j + 0))
                    ? 99999999
                    : hc.hamming(b_code + 0 * code_size);
            int hd1 = (res.skip_entry(j + 1))
                    ? 99999999
                    : hc.hamming(b_code + 1 * code_size);
            int hd2 = (res.skip_entry(j + 2))
                    ? 99999999
                    : hc.hamming(b_code + 2 * code_size);
            int hd3 = (res.skip_entry(j + 3))
                    ? 99999999
                    : hc.hamming(b_code + 3 * code_size);

            saved_j[counter] = j + 0;
            counter = (hd0 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 1;
            counter = (hd1 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 2;
            counter = (hd2 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 3;
            counter = (hd3 < ht) ? (counter + 1) : counter;

            if (counter >= 4) {
                // process four codes at the same time
                n_hamming_pass += 4;

                float distance_0 = dis0;
                float distance_1 = dis0;
                float distance_2 = dis0;
                float distance_3 = dis0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);

                //
                counter -= 4;
                saved_j[0] = saved_j[4];
                saved_j[1] = saved_j[5];
                saved_j[2] = saved_j[6];
                saved_j[3] = saved_j[7];
            }
        }

        for (size_t kk = 0; kk < counter; kk++) {
            n_hamming_pass++;

            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[kk] * pq.code_size);

            res.add(saved_j[kk], dis);
        }

        // process leftovers
        for (size_t j = (ncode / 4) * 4; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }
            const uint8_t* b_code = codes + j * code_size;
            int hd = hc.hamming(b_code);
            if (hd < ht) {
                n_hamming_pass++;

                float dis = dis0 +
                        distance_single_code<PQDecoder>(
                                    pq.M,
                                    pq.nbits,
                                    sim_table,
                                    codes + j * code_size);

                res.add(j, dis);
            }
        }

#pragma omp critical
        { indexIVFPQ_stats.n_hamming_pass += n_hamming_pass; }
    }

    template <class SearchResultType>
    struct Run_scan_list_polysemous_hc {
        using T = void;
        template <class HammingComputer, class... Types>
        void f(const IVFPQScannerT* scanner, Types... args) {
            scanner->scan_list_polysemous_hc<HammingComputer, SearchResultType>(
                    args...);
        }
    };

    template <class SearchResultType>
    void scan_list_polysemous(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        Run_scan_list_polysemous_hc<SearchResultType> r;
        dispatch_HammingComputer(pq.code_size, r, this, ncode, codes, res);
    }
};

/* We put as many parameters as possible in template. Hopefully the
 * gain in runtime is worth the code bloat.
 *
 * C is the comparator < or >, it is directly related to METRIC_TYPE.
 *
 * precompute_mode is how much we precompute (2 = precompute distance tables,
 * 1 = precompute pointers to distances, 0 = compute distances one by one).
 * Currently only 2 is supported
 *
 * use_sel: store or ignore the IDSelector
 */
template <MetricType METRIC_TYPE, class C, class PQDecoder, bool use_sel>
struct IVFPQScanner : IVFPQScannerT<idx_t, METRIC_TYPE, PQDecoder>,
                      InvertedListScanner {
    int precompute_mode;
    const IDSelector* sel;

    IVFPQScanner(
            const IndexIVFPQ& ivfpq,
            bool store_pairs,
            int precompute_mode,
            const IDSelector* sel)
            : IVFPQScannerT<idx_t, METRIC_TYPE, PQDecoder>(ivfpq, nullptr),
              precompute_mode(precompute_mode),
              sel(sel) {
        this->store_pairs = store_pairs;
        this->keep_max = is_similarity_metric(METRIC_TYPE);
    }

    void set_query(const float* query) override {
        this->init_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        this->init_list(list_no, coarse_dis, precompute_mode);
    }

    float distance_to_code(const uint8_t* code) const override {
        assert(precompute_mode == 2);
        float dis = this->dis0 +
                distance_single_code<PQDecoder>(
                            this->pq.M, this->pq.nbits, this->sim_table, code);
        return dis;
    }

    size_t scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float* heap_sim,
            idx_t* heap_ids,
            size_t k) const override {
        KnnSearchResults<C, use_sel> res = {
                /* key */ this->key,
                /* ids */ this->store_pairs ? nullptr : ids,
                /* sel */ this->sel,
                /* k */ k,
                /* heap_sim */ heap_sim,
                /* heap_ids */ heap_ids,
                /* nup */ 0};

        if (this->polysemous_ht > 0) {
            assert(precompute_mode == 2);
            this->scan_list_polysemous(ncode, codes, res);
        } else if (precompute_mode == 2) {
            this->scan_list_with_table(ncode, codes, res);
        } else if (precompute_mode == 1) {
            this->scan_list_with_pointer(ncode, codes, res);
        } else if (precompute_mode == 0) {
            this->scan_on_the_fly_dist(ncode, codes, res);
        } else {
            FAISS_THROW_MSG("bad precomp mode");
        }
        return res.nup;
    }

    void scan_codes_range(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& rres) const override {
        RangeSearchResults<C, use_sel> res = {
                /* key */ this->key,
                /* ids */ this->store_pairs ? nullptr : ids,
                /* sel */ this->sel,
                /* radius */ radius,
                /* rres */ rres};

        if (this->polysemous_ht > 0) {
            assert(precompute_mode == 2);
            this->scan_list_polysemous(ncode, codes, res);
        } else if (precompute_mode == 2) {
            this->scan_list_with_table(ncode, codes, res);
        } else if (precompute_mode == 1) {
            this->scan_list_with_pointer(ncode, codes, res);
        } else if (precompute_mode == 0) {
            this->scan_on_the_fly_dist(ncode, codes, res);
        } else {
            FAISS_THROW_MSG("bad precomp mode");
        }
    }
};

template <class PQDecoder, bool use_sel>
InvertedListScanner* get_InvertedListScanner1(
        const IndexIVFPQ& index,
        bool store_pairs,
        const IDSelector* sel) {
    if (index.metric_type == METRIC_INNER_PRODUCT) {
        return new IVFPQScanner<
                METRIC_INNER_PRODUCT,
                CMin<float, idx_t>,
                PQDecoder,
                use_sel>(index, store_pairs, 2, sel);
    } else if (index.metric_type == METRIC_L2) {
        return new IVFPQScanner<
                METRIC_L2,
                CMax<float, idx_t>,
                PQDecoder,
                use_sel>(index, store_pairs, 2, sel);
    }
    return nullptr;
}

template <bool use_sel>
InvertedListScanner* get_InvertedListScanner2(
        const IndexIVFPQ& index,
        bool store_pairs,
        const IDSelector* sel) {
    if (index.pq.nbits == 8) {
        return get_InvertedListScanner1<PQDecoder8, use_sel>(
                index, store_pairs, sel);
    } else if (index.pq.nbits == 16) {
        return get_InvertedListScanner1<PQDecoder16, use_sel>(
                index, store_pairs, sel);
    } else {
        return get_InvertedListScanner1<PQDecoderGeneric, use_sel>(
                index, store_pairs, sel);
    }
}

} // anonymous namespace

InvertedListScanner* IndexIVFPQ::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel) const {
    if (sel) {
        return get_InvertedListScanner2<true>(*this, store_pairs, sel);
    } else {
        return get_InvertedListScanner2<false>(*this, store_pairs, sel);
    }
    return nullptr;
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
        for (int i = 0; i < n; i++)
            ord[i] = i;
        InvertedLists::ScopedCodes codes(invlists, list_no);
        CodeCmp cs = {codes.get(), code_size};
        std::sort(ord.begin(), ord.end(), cs);

        InvertedLists::ScopedIds list_ids(invlists, list_no);
        int prev = -1; // all elements from prev to i-1 are equal
        for (int i = 0; i < n; i++) {
            if (prev >= 0 && cs.cmp(ord[prev], ord[i]) == 0) {
                // same as previous => remember
                if (prev + 1 == i) { // start new group
                    ngroup++;
                    lims[ngroup] = lims[ngroup - 1];
                    dup_ids[lims[ngroup]++] = list_ids[ord[prev]];
                }
                dup_ids[lims[ngroup]++] = list_ids[ord[i]];
            } else { // not same as previous.
                prev = i;
            }
        }
    }
    return ngroup;
}

} // namespace faiss
