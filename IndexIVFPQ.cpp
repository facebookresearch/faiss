/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright 2004-present Facebook. All Rights Reserved.
   Inverted list structure.
*/

#include "IndexIVFPQ.h"

#include <cmath>
#include <cstdio>
#include <cassert>

#include <sys/mman.h>

#include <algorithm>

#include "Heap.h"
#include "utils.h"

#include "Clustering.h"
#include "IndexFlat.h"

#include "hamming.h"

#include "FaissAssert.h"

#include "AuxIndexStructures.h"

namespace faiss {





/*****************************************
 * IndexIVFPQ implementation
 ******************************************/

IndexIVFPQ::IndexIVFPQ (Index * quantizer, size_t d, size_t nlist,
                        size_t M, size_t nbits_per_idx):
    IndexIVF (quantizer, d, nlist, METRIC_L2),
    pq (d, M, nbits_per_idx)
{
    FAISS_THROW_IF_NOT (nbits_per_idx <= 8);
    code_size = pq.code_size;
    is_trained = false;
    by_residual = true;
    use_precomputed_table = 0;
    scan_table_threshold = 0;
    max_codes = 0; // means unlimited

    polysemous_training = nullptr;
    do_polysemous_training = false;
    polysemous_ht = 0;

}


void IndexIVFPQ::train_residual (idx_t n, const float *x)
{
    train_residual_o (n, x, nullptr);
}


void IndexIVFPQ::train_residual_o (idx_t n, const float *x, float *residuals_2)
{
    const float * x_in = x;

    x = fvecs_maybe_subsample (
         d, (size_t*)&n, pq.cp.max_points_per_centroid * pq.ksub,
         x, verbose, pq.cp.seed);

    ScopeDeleter<float> del_x (x_in == x ? nullptr : x);

    const float *trainset;
    ScopeDeleter<float> del_residuals;
    if (by_residual) {
        if(verbose) printf("computing residuals\n");
        idx_t * assign = new idx_t [n]; // assignement to coarse centroids
        ScopeDeleter<idx_t> del (assign);
        quantizer->assign (n, x, assign);
        float *residuals = new float [n * d];
        del_residuals.set (residuals);
        for (idx_t i = 0; i < n; i++)
           quantizer->compute_residual (x + i * d, residuals+i*d, assign[i]);

        trainset = residuals;
    } else {
        trainset = x;
    }
    if (verbose)
        printf ("training %zdx%zd product quantizer on %ld vectors in %dD\n",
                pq.M, pq.ksub, n, d);
    pq.verbose = verbose;
    pq.train (n, trainset);

    if (do_polysemous_training) {
        if (verbose)
            printf("doing polysemous training for PQ\n");
        PolysemousTraining default_pt;
        PolysemousTraining *pt = polysemous_training;
        if (!pt) pt = &default_pt;
        pt->optimize_pq_for_hamming (pq, n, trainset);
    }

    // prepare second-level residuals for refine PQ
    if (residuals_2) {
        uint8_t *train_codes = new uint8_t [pq.code_size * n];
        ScopeDeleter<uint8_t> del (train_codes);
        pq.compute_codes (trainset, train_codes, n);

        for (idx_t i = 0; i < n; i++) {
            const float *xx = trainset + i * d;
            float * res = residuals_2 + i * d;
            pq.decode (train_codes + i * pq.code_size, res);
            for (int j = 0; j < d; j++)
                res[j] = xx[j] - res[j];
        }

    }

    if (by_residual) {
        precompute_table ();
    }

}


/* produce a binary signature based on the residual vector */
void IndexIVFPQ::encode (long key, const float * x, uint8_t * code) const
{
    if (by_residual) {
        float residual_vec[d];
        quantizer->compute_residual (x, residual_vec, key);
        pq.compute_code (residual_vec, code);
    }
    else pq.compute_code (x, code);
}





void IndexIVFPQ::encode_multiple (size_t n, long *keys,
                                  const float * x, uint8_t * xcodes,
                                  bool compute_keys) const
{
    if (compute_keys)
        quantizer->assign (n, x, keys);

    if (by_residual) {
        float *residuals = new float [n * d];
        ScopeDeleter<float> del (residuals);
        // TODO: parallelize?
        for (size_t i = 0; i < n; i++)
            quantizer->compute_residual (x + i * d, residuals + i * d, keys[i]);
        pq.compute_codes (residuals, xcodes, n);
    } else {
        pq.compute_codes (x, xcodes, n);
    }
}

void IndexIVFPQ::decode_multiple (size_t n, const long *keys,
                                  const uint8_t * xcodes, float * x) const
{
    pq.decode (xcodes, x, n);
    if (by_residual) {
        std::vector<float> centroid (d);
        for (size_t i = 0; i < n; i++) {
            quantizer->reconstruct (keys[i], centroid.data());
            float *xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                xi [j] += centroid [j];
            }
        }
    }
}


void IndexIVFPQ::add_with_ids (idx_t n, const float * x, const long *xids)
{
    add_core_o (n, x, xids, nullptr);
}


void IndexIVFPQ::add_core_o (idx_t n, const float * x, const long *xids,
                             float *residuals_2, const long *precomputed_idx)
{
    FAISS_THROW_IF_NOT (is_trained);
    double t0 = getmillisecs ();
    const long * idx;
    ScopeDeleter<long> del_idx;

    if (precomputed_idx) {
        idx = precomputed_idx;
    } else {
        long * idx0 = new long [n];
        del_idx.set (idx0);
        quantizer->assign (n, x, idx0);
        idx = idx0;
    }

    double t1 = getmillisecs ();
    uint8_t * xcodes = new uint8_t [n * code_size];
    ScopeDeleter<uint8_t> del_xcodes (xcodes);

    const float *to_encode = nullptr;
    ScopeDeleter<float> del_to_encode;

    if (by_residual) {
        float *residuals = new float [n * d];
        // TODO: parallelize?
        for (size_t i = 0; i < n; i++) {
            if (idx[i] < 0)
                memset (residuals + i * d, 0, sizeof(*residuals) * d);
            else
                quantizer->compute_residual (
                    x + i * d, residuals + i * d, idx[i]);
        }
        to_encode = residuals;
        del_to_encode.set (to_encode);
    } else {
        to_encode = x;
    }
    pq.compute_codes (to_encode, xcodes, n);

    double t2 = getmillisecs ();
    // TODO: parallelize?
    size_t n_ignore = 0;
    for (size_t i = 0; i < n; i++) {
        idx_t key = idx[i];
        if (key < 0) {
            n_ignore ++;
            if (residuals_2)
                memset (residuals_2, 0, sizeof(*residuals_2) * d);
            continue;
        }
        idx_t id = xids ? xids[i] : ntotal + i;
        ids[key].push_back (id);
        uint8_t *code = xcodes + i * code_size;
        for (size_t j = 0; j < code_size; j++)
            codes[key].push_back (code[j]);

        if (residuals_2) {
            float *res2 = residuals_2 + i * d;
            const float *xi = to_encode + i * d;
            pq.decode (code, res2);
            for (int j = 0; j < d; j++)
                res2[j] = xi[j] - res2[j];
        }

        if (maintain_direct_map)
            direct_map.push_back (key << 32 | (ids[key].size() - 1));
    }


    double t3 = getmillisecs ();
    if(verbose) {
        char comment[100] = {0};
        if (n_ignore > 0)
            snprintf (comment, 100, "(%ld vectors ignored)", n_ignore);
        printf(" add_core times: %.3f %.3f %.3f %s\n",
               t1 - t0, t2 - t1, t3 - t2, comment);
    }
    ntotal += n;
}

void IndexIVFPQ::reconstruct_n (idx_t i0, idx_t ni, float *recons) const
{
    FAISS_THROW_IF_NOT (ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    std::vector<float> centroid (d);

    for (int key = 0; key < nlist; key++) {
        const std::vector<long> & idlist = ids[key];
        const uint8_t * code_line = codes[key].data();

        for (long ofs = 0; ofs < idlist.size(); ofs++) {
            long id = idlist[ofs];
            if (!(id >= i0 && id < i0 + ni)) continue;
            float *r = recons + d * (id - i0);
            if (by_residual) {
                quantizer->reconstruct (key, centroid.data());
                pq.decode (code_line + ofs * pq.code_size, r);
                for (int j = 0; j < d; j++) {
                    r[j] += centroid[j];
                }
            } else {
                pq.decode (code_line + ofs * pq.code_size, r);
            }
        }
    }
}


void IndexIVFPQ::reconstruct (idx_t key, float * recons) const
{
    FAISS_THROW_IF_NOT (direct_map.size() == ntotal);

    int list_no = direct_map[key] >> 32;
    int ofs = direct_map[key] & 0xffffffff;

    quantizer->reconstruct (list_no, recons);
    const uint8_t * code = &(codes[list_no][ofs * pq.code_size]);

    for (size_t m = 0; m < pq.M; m++) {
        float * out = recons + m * pq.dsub;
        const float * cent = pq.get_centroids (m, code[m]);
        for (size_t i = 0; i < pq.dsub; i++) {
            out[i] += cent[i];
        }
    }
}






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

void IndexIVFPQ::precompute_table ()
{
    if (use_precomputed_table == 0) { // then choose the type of table
        if (quantizer->metric_type == METRIC_INNER_PRODUCT) {
            fprintf(stderr, "IndexIVFPQ::precompute_table: WARN precomputed "
                    "tables not needed for inner product quantizers\n");
            return;
        }
        const MultiIndexQuantizer *miq =
            dynamic_cast<const MultiIndexQuantizer *> (quantizer);
        if (miq && pq.M % miq->pq.M == 0)
            use_precomputed_table = 2;
        else
            use_precomputed_table = 1;
    } // otherwise assume user has set appropriate flag on input

    if (verbose) {
        printf ("precomputing IVFPQ tables type %d\n",
                use_precomputed_table);
    }

    // squared norms of the PQ centroids
    std::vector<float> r_norms (pq.M * pq.ksub, NAN);
    for (int m = 0; m < pq.M; m++)
        for (int j = 0; j < pq.ksub; j++)
            r_norms [m * pq.ksub + j] =
                fvec_norm_L2sqr (pq.get_centroids (m, j), pq.dsub);

    if (use_precomputed_table == 1) {

        precomputed_table.resize (nlist * pq.M * pq.ksub);
        std::vector<float> centroid (d);

        for (size_t i = 0; i < nlist; i++) {
            quantizer->reconstruct (i, centroid.data());

            float *tab = &precomputed_table[i * pq.M * pq.ksub];
            pq.compute_inner_prod_table (centroid.data(), tab);
            fvec_madd (pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
        }
    } else if (use_precomputed_table == 2) {
        const MultiIndexQuantizer *miq =
           dynamic_cast<const MultiIndexQuantizer *> (quantizer);
        FAISS_THROW_IF_NOT (miq);
        const ProductQuantizer &cpq = miq->pq;
        FAISS_THROW_IF_NOT (pq.M % cpq.M == 0);

        precomputed_table.resize(cpq.ksub * pq.M * pq.ksub);

        // reorder PQ centroid table
        std::vector<float> centroids (d * cpq.ksub, NAN);

        for (int m = 0; m < cpq.M; m++) {
            for (size_t i = 0; i < cpq.ksub; i++) {
                memcpy (centroids.data() + i * d + m * cpq.dsub,
                        cpq.get_centroids (m, i),
                        sizeof (*centroids.data()) * cpq.dsub);
            }
        }

        pq.compute_inner_prod_tables (cpq.ksub, centroids.data (),
                                      precomputed_table.data ());

        for (size_t i = 0; i < cpq.ksub; i++) {
            float *tab = &precomputed_table[i * pq.M * pq.ksub];
            fvec_madd (pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
        }

    }
}

namespace {

static uint64_t get_cycles () {
    uint32_t high, low;
    asm volatile("rdtsc \n\t"
                 : "=a" (low),
                   "=d" (high));
    return ((uint64_t)high << 32) | (low);
}

#define TIC t0 = get_cycles()
#define TOC get_cycles () - t0



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

    const IndexIVFPQ & ivfpq;

    // copied from IndexIVFPQ for easier access
    int d;
    const ProductQuantizer & pq;
    MetricType metric_type;
    bool by_residual;
    int use_precomputed_table;

    // pre-allocated data buffers
    float * sim_table, * sim_table_2;
    float * residual_vec, *decoded_vec;

    // single data buffer
    std::vector<float> mem;

    // for table pointers
    std::vector<const float *> sim_table_ptrs;

    explicit QueryTables (const IndexIVFPQ & ivfpq):
        ivfpq(ivfpq),
        d(ivfpq.d),
        pq (ivfpq.pq),
        metric_type (ivfpq.metric_type),
        by_residual (ivfpq.by_residual),
        use_precomputed_table (ivfpq.use_precomputed_table)
    {
        mem.resize (pq.ksub * pq.M * 2 + d *2);
        sim_table = mem.data();
        sim_table_2 = sim_table + pq.ksub * pq.M;
        residual_vec = sim_table_2 + pq.ksub * pq.M;
        decoded_vec = residual_vec + d;

        // for polysemous
        if (ivfpq.polysemous_ht != 0)  {
            q_code.resize (pq.code_size);
        }
        init_list_cycles = 0;
        sim_table_ptrs.resize (pq.M);
    }

    /*****************************************************
     * What we do when query is known
     *****************************************************/

    // field specific to query
    const float * qi;

    // query-specific intialization
    void init_query (const float * qi) {
        this->qi = qi;
        if (metric_type == METRIC_INNER_PRODUCT)
            init_query_IP ();
        else
            init_query_L2 ();
        if (!by_residual && ivfpq.polysemous_ht != 0)
            pq.compute_code (qi, q_code.data());
    }

    void init_query_IP () {
        // precompute some tables specific to the query qi
        pq.compute_inner_prod_table (qi, sim_table);
        // we compute negated inner products for use with the maxheap
        for (int i = 0; i < pq.ksub * pq.M; i++) {
            sim_table[i] = - sim_table[i];
        }
    }

    void init_query_L2 () {
        if (!by_residual) {
            pq.compute_distance_table (qi, sim_table);
        } else if (use_precomputed_table) {
            pq.compute_inner_prod_table (qi, sim_table_2);
        }
    }

    /*****************************************************
     * When inverted list is known: prepare computations
     *****************************************************/

    // fields specific to list
    Index::idx_t key;
    float coarse_dis;
    std::vector<uint8_t> q_code;

    uint64_t init_list_cycles;

    /// once we know the query and the centroid, we can prepare the
    /// sim_table that will be used for accumulation
    /// and dis0, the initial value
    float precompute_list_tables () {
        float dis0 = 0;
        uint64_t t0; TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                dis0 = precompute_list_tables_IP ();
            else
                dis0 = precompute_list_tables_L2 ();
        }
        init_list_cycles += TOC;
        return dis0;
     }

    float precompute_list_table_pointers () {
        float dis0 = 0;
        uint64_t t0; TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
              FAISS_THROW_MSG ("not implemented");
            else
              dis0 = precompute_list_table_pointers_L2 ();
        }
        init_list_cycles += TOC;
        return dis0;
     }

    /*****************************************************
     * compute tables for inner prod
     *****************************************************/

    float precompute_list_tables_IP ()
    {
        // prepare the sim_table that will be used for accumulation
        // and dis0, the initial value
        ivfpq.quantizer->reconstruct (key, decoded_vec);
        // decoded_vec = centroid
        float dis0 = -fvec_inner_product (qi, decoded_vec, d);

        if (ivfpq.polysemous_ht) {
            for (int i = 0; i < d; i++) {
                residual_vec [i] = qi[i] - decoded_vec[i];
            }
            pq.compute_code (residual_vec, q_code.data());
        }
        return dis0;
    }


    /*****************************************************
     * compute tables for L2 distance
     *****************************************************/

    float precompute_list_tables_L2 ()
    {
        float dis0 = 0;

        if (use_precomputed_table == 0) {
            ivfpq.quantizer->compute_residual (qi, residual_vec, key);
            pq.compute_distance_table (residual_vec, sim_table);
        } else if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            fvec_madd (pq.M * pq.ksub,
                       &ivfpq.precomputed_table [key * pq.ksub * pq.M],
                       -2.0, sim_table_2,
                       sim_table);
        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer *miq =
                dynamic_cast<const MultiIndexQuantizer *> (ivfpq.quantizer);
            FAISS_THROW_IF_NOT (miq);
            const ProductQuantizer &cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            const float *qtab = sim_table_2; // query-specific table
            float *ltab = sim_table; // (output) list-specific table

            long k = key;
            for (int cm = 0; cm < cpq.M; cm++) {
                // compute PQ index
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                // get corresponding table
                const float *pc = &ivfpq.precomputed_table
                    [(ki * pq.M + cm * Mf) * pq.ksub];

                if (ivfpq.polysemous_ht == 0) {

                    // sum up with query-specific table
                    fvec_madd (Mf * pq.ksub,
                               pc,
                               -2.0, qtab,
                               ltab);
                    ltab += Mf * pq.ksub;
                    qtab += Mf * pq.ksub;
                } else {
                    for (int m = cm * Mf; m < (cm + 1) * Mf; m++) {
                        q_code[m] = fvec_madd_and_argmin
                            (pq.ksub, pc, -2, qtab, ltab);
                        pc += pq.ksub;
                        ltab += pq.ksub;
                        qtab += pq.ksub;
                    }
                }

            }
        }

        return dis0;
    }

    float precompute_list_table_pointers_L2 ()
    {
        float dis0 = 0;

        if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            const float * s = &ivfpq.precomputed_table [key * pq.ksub * pq.M];
            for (int m = 0; m < pq.M; m++) {
                sim_table_ptrs [m] = s;
                s += pq.ksub;
            }
        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer *miq =
                dynamic_cast<const MultiIndexQuantizer *> (ivfpq.quantizer);
            FAISS_THROW_IF_NOT (miq);
            const ProductQuantizer &cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            long k = key;
            int m0 = 0;
            for (int cm = 0; cm < cpq.M; cm++) {
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                const float *pc = &ivfpq.precomputed_table
                    [(ki * pq.M + cm * Mf) * pq.ksub];

                for (int m = m0; m < m0 + Mf; m++) {
                    sim_table_ptrs [m] = pc;
                    pc += pq.ksub;
                }
                m0 += Mf;
            }
        } else {
          FAISS_THROW_MSG ("need precomputed tables");
        }

        if (ivfpq.polysemous_ht) {
            FAISS_THROW_MSG ("not implemented");
            // Not clear that it makes sense to implemente this,
            // because it costs M * ksub, which is what we wanted to
            // avoid with the tables pointers.
        }

        return dis0;
    }


};


/*****************************************************
 * Scaning the codes.
 * The scanning functions call their favorite precompute_*
 * function to precompute the tables they need.
 *****************************************************/
template <typename IDType>
struct InvertedListScanner: QueryTables {

    const uint8_t * __restrict list_codes;
    const IDType * list_ids;
    size_t list_size;

    explicit InvertedListScanner (const IndexIVFPQ & ivfpq):
        QueryTables (ivfpq)
    {
        FAISS_THROW_IF_NOT (pq.byte_per_idx == 1);
        n_hamming_pass = 0;
    }

    /// list_specific intialization
    void init_list (Index::idx_t key, float coarse_dis,
                    size_t list_size_in, const IDType *list_ids_in,
                    const uint8_t *list_codes_in) {
        this->key = key;
        this->coarse_dis = coarse_dis;
        list_size = list_size_in;
        list_codes = list_codes_in;
        list_ids = list_ids_in;
    }

    /*****************************************************
     * Scaning the codes: simple PQ scan.
     *****************************************************/

    /// version of the scan where we use precomputed tables
    void scan_list_with_table (
             size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
    {
        float dis0 = precompute_list_tables ();

        for (size_t j = 0; j < list_size; j++) {

            float dis = dis0;
            const float *tab = sim_table;

            for (size_t m = 0; m < pq.M; m++) {
                dis += tab[*list_codes++];
                tab += pq.ksub;
            }

            if (dis < heap_sim[0]) {
                maxheap_pop (k, heap_sim, heap_ids);
                long id = store_pairs ? (key << 32 | j) : list_ids[j];
                maxheap_push (k, heap_sim, heap_ids, dis, id);
            }
        }
    }


    /// tables are not precomputed, but pointers are provided to the
    /// relevant X_c|x_r tables
    void scan_list_with_pointer (
             size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
    {

        float dis0 = precompute_list_table_pointers ();

        for (size_t j = 0; j < list_size; j++) {

            float dis = dis0;
            const float *tab = sim_table_2;

            for (size_t m = 0; m < pq.M; m++) {
                int ci = *list_codes++;
                dis += sim_table_ptrs [m][ci] - 2 * tab [ci];
                tab += pq.ksub;
            }

            if (dis < heap_sim[0]) {
                maxheap_pop (k, heap_sim, heap_ids);
                long id = store_pairs ? (key << 32 | j) : list_ids[j];
                maxheap_push (k, heap_sim, heap_ids, dis, id);
            }
        }

    }

    /// nothing is precomputed: access residuals on-the-fly
    void scan_on_the_fly_dist (
             size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
    {

        if (by_residual && use_precomputed_table) {
            scan_list_with_pointer (k, heap_sim, heap_ids, store_pairs);
            return;
        }

        const float *dvec;
        float dis0 = 0;

        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                ivfpq.quantizer->reconstruct (key, residual_vec);
                dis0 = fvec_inner_product (residual_vec, qi, d);
            } else {
                ivfpq.quantizer->compute_residual (qi, residual_vec, key);
            }
            dvec = residual_vec;
        } else {
            dvec = qi;
            dis0 = 0;
        }

        for (size_t j = 0; j < list_size; j++) {

            pq.decode (list_codes, decoded_vec);
            list_codes += pq.code_size;

            float dis;
            if (metric_type == METRIC_INNER_PRODUCT) {
                dis = -dis0 - fvec_inner_product (decoded_vec, qi, d);
            } else {
                dis = fvec_L2sqr (decoded_vec, dvec, d);
            }

            if (dis < heap_sim[0]) {
                maxheap_pop (k, heap_sim, heap_ids);
                long id = store_pairs ? (key << 32 | j) : list_ids[j];
                maxheap_push (k, heap_sim, heap_ids, dis, id);
            }
        }
    }

    /*****************************************************
     * Scanning codes with polysemous filtering
     *****************************************************/

    // code for the query
    size_t n_hamming_pass;


    template <class HammingComputer>
    void scan_list_polysemous_hc (
             size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
    {
        float dis0 = precompute_list_tables ();
        int ht = ivfpq.polysemous_ht;

        int code_size = pq.code_size;

        HammingComputer hc (q_code.data(), code_size);

        for (size_t j = 0; j < list_size; j++) {
            const uint8_t *b_code = list_codes;
            int hd = hc.hamming (b_code);
            if (hd < ht) {
                n_hamming_pass ++;

                float dis = dis0;
                const float *tab = sim_table;

                for (size_t m = 0; m < pq.M; m++) {
                    dis += tab[*b_code++];
                    tab += pq.ksub;
                }

                if (dis < heap_sim[0]) {
                    maxheap_pop (k, heap_sim, heap_ids);
                    long id = store_pairs ? (key << 32 | j) : list_ids[j];
                    maxheap_push (k, heap_sim, heap_ids, dis, id);
                }
            }
            list_codes += code_size;
        }
    }

    void scan_list_polysemous (
             size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
    {
        switch (pq.code_size) {
#define HANDLE_CODE_SIZE(cs)  \
        case cs:  \
            scan_list_polysemous_hc <HammingComputer ## cs> \
                (k, heap_sim, heap_ids, store_pairs); \
            break
        HANDLE_CODE_SIZE(4);
        HANDLE_CODE_SIZE(8);
        HANDLE_CODE_SIZE(16);
        HANDLE_CODE_SIZE(20);
        HANDLE_CODE_SIZE(32);
        HANDLE_CODE_SIZE(64);
#undef HANDLE_CODE_SIZE
        default:
            if (pq.code_size % 8 == 0)
                scan_list_polysemous_hc <HammingComputerM8>
                    (k, heap_sim, heap_ids, store_pairs);
            else
                scan_list_polysemous_hc <HammingComputerM4>
                    (k, heap_sim, heap_ids, store_pairs);
            break;
        }
    }

};




} // anonymous namespace


IndexIVFPQStats indexIVFPQ_stats;

void IndexIVFPQStats::reset () {
    memset (this, 0, sizeof (*this));
}



void IndexIVFPQ::search_preassigned (idx_t nx, const float *qx, idx_t k,
                                     const idx_t *keys,
                                     const float *coarse_dis,
                                     float *distances, idx_t *labels,
                                     bool store_pairs) const
{
    float_maxheap_array_t res = {
        size_t(nx), size_t(k),
        labels, distances
    };

#pragma omp parallel
    {
        InvertedListScanner<long> qt (*this);
        size_t stats_nlist = 0;
        size_t stats_ncode = 0;
        uint64_t init_query_cycles = 0;
        uint64_t scan_cycles = 0;
        uint64_t heap_cycles = 0;

#pragma omp  for
        for (size_t i = 0; i < nx; i++) {
            const float *qi = qx + i * d;
            const long * keysi = keys + i * nprobe;
            const float *coarse_dis_i = coarse_dis + i * nprobe;
            float * heap_sim = res.get_val (i);
            long * heap_ids = res.get_ids (i);

            uint64_t t0;
            TIC;
            maxheap_heapify (k, heap_sim, heap_ids);
            heap_cycles += TOC;

            TIC;
            qt.init_query (qi);
            init_query_cycles += TOC;

            size_t nscan = 0;

            for (size_t ik = 0; ik < nprobe; ik++) {
                long key = keysi[ik];  /* select the list  */
                if (key < 0) {
                    // not enough centroids for multiprobe
                    continue;
                }
                FAISS_THROW_IF_NOT_FMT (
                    key < (long) nlist,
                    "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                    key, ik, nlist);

                size_t list_size = ids[key].size();
                stats_nlist ++;
                nscan += list_size;

                if (list_size == 0) continue;

                qt.init_list (key, coarse_dis_i[ik],
                              list_size, ids[key].data(),
                              codes[key].data());

                TIC;
                if (polysemous_ht > 0) {
                    qt.scan_list_polysemous
                        (k, heap_sim, heap_ids, store_pairs);
                } else if (list_size > scan_table_threshold) {
                   qt.scan_list_with_table (k, heap_sim, heap_ids, store_pairs);
                } else {
                   qt.scan_on_the_fly_dist (k, heap_sim, heap_ids, store_pairs);
                }
                scan_cycles += TOC;

                if (max_codes && nscan >= max_codes) break;
            }
            stats_ncode += nscan;
            TIC;
            maxheap_reorder (k, heap_sim, heap_ids);

            if (metric_type == METRIC_INNER_PRODUCT) {
                for (size_t j = 0; j < k; j++)
                    heap_sim[j] = -heap_sim[j];
            }
            heap_cycles += TOC;
        }

#pragma omp critical
        {
            indexIVFPQ_stats.n_hamming_pass += qt.n_hamming_pass;
            indexIVFPQ_stats.nlist += stats_nlist;
            indexIVFPQ_stats.ncode += stats_ncode;

            indexIVFPQ_stats.init_query_cycles += init_query_cycles;
            indexIVFPQ_stats.init_list_cycles += qt.init_list_cycles;
            indexIVFPQ_stats.scan_cycles += scan_cycles - qt.init_list_cycles;
            indexIVFPQ_stats.heap_cycles += heap_cycles;
        }

    }
    indexIVFPQ_stats.nq += nx;
}


void IndexIVFPQ::search_and_reconstruct (idx_t n, const float *x, idx_t k,
                                         float *distances, idx_t *labels,
                                         float *reconstructed)
{
    long * idx = new long [n * nprobe];
    ScopeDeleter<long> del (idx);
    float * coarse_dis = new float [n * nprobe];
    ScopeDeleter<float> del2 (coarse_dis);

    quantizer->search (n, x, nprobe, coarse_dis, idx);

    search_preassigned (n, x, k, idx, coarse_dis,
                        distances, labels, true);

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < k; j++) {
            long ij = i * k + j;
            idx_t res = labels[ij];
            float *recons = reconstructed + d * (ij);
            if (res < 0) {
                // fill with NaNs
                memset(recons, -1, sizeof(*recons) * d);
            } else {
                int list_no = res >> 32;
                int ofs = res & 0xffffffff;
                labels[ij] = ids[list_no][ofs];

                quantizer->reconstruct (list_no, recons);
                const uint8_t * code = &(codes[list_no][ofs * pq.code_size]);

                for (size_t m = 0; m < pq.M; m++) {
                    float * out = recons + m * pq.dsub;
                    const float * cent = pq.get_centroids (m, code[m]);
                    for (size_t l = 0; l < pq.dsub; l++) {
                        out[l] += cent[l];
                    }
                }
            }
        }
    }

}





IndexIVFPQ::IndexIVFPQ ()
{
    // initialize some runtime values
    use_precomputed_table = 0;
    scan_table_threshold = 0;
    do_polysemous_training = false;
    polysemous_ht = 0;
    max_codes = 0;
    polysemous_training = nullptr;
}


struct CodeCmp {
    const uint8_t *tab;
    size_t code_size;
    bool operator () (int a, int b) const {
        return cmp (a, b) > 0;
    }
    int cmp (int a, int b) const {
        return memcmp (tab + a * code_size, tab + b * code_size,
                       code_size);
    }
};


size_t IndexIVFPQ::find_duplicates (idx_t *dup_ids, size_t *lims) const
{
    size_t ngroup = 0;
    lims[0] = 0;
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t n = ids[list_no].size();
        std::vector<int> ord (n);
        for (int i = 0; i < n; i++) ord[i] = i;
        CodeCmp cs = { codes[list_no].data(), code_size };
        std::sort (ord.begin(), ord.end(), cs);

        const idx_t *list_ids = ids[list_no].data();
        int prev = -1;  // all elements from prev to i-1 are equal
        for (int i = 0; i < n; i++) {
            if (prev >= 0 && cs.cmp (ord [prev], ord [i]) == 0) {
                // same as previous => remember
                if (prev + 1 == i) { // start new group
                    ngroup++;
                    lims[ngroup] = lims[ngroup - 1];
                    dup_ids [lims [ngroup]++] = list_ids [ord [prev]];
                }
                dup_ids [lims [ngroup]++] = list_ids [ord [i]];
            } else { // not same as previous.
                prev = i;
            }
        }
    }
    return ngroup;
}




/*****************************************
 * IndexIVFPQR implementation
 ******************************************/

IndexIVFPQR::IndexIVFPQR (
            Index * quantizer, size_t d, size_t nlist,
            size_t M, size_t nbits_per_idx,
            size_t M_refine, size_t nbits_per_idx_refine):
    IndexIVFPQ (quantizer, d, nlist, M, nbits_per_idx),
    refine_pq (d, M_refine, nbits_per_idx_refine),
    k_factor (4)
{
    by_residual = true;
}

IndexIVFPQR::IndexIVFPQR ():
    k_factor (1)
{
    by_residual = true;
}



void IndexIVFPQR::reset()
{
    IndexIVFPQ::reset();
    refine_codes.clear();
}




void IndexIVFPQR::train_residual (idx_t n, const float *x)
{

    float * residual_2 = new float [n * d];
    ScopeDeleter <float> del(residual_2);

    train_residual_o (n, x, residual_2);

    if (verbose)
        printf ("training %zdx%zd 2nd level PQ quantizer on %ld %dD-vectors\n",
                refine_pq.M, refine_pq.ksub, n, d);

    refine_pq.cp.max_points_per_centroid = 1000;
    refine_pq.cp.verbose = verbose;

    refine_pq.train (n, residual_2);

}


void IndexIVFPQR::add_with_ids (idx_t n, const float *x, const long *xids) {
    add_core (n, x, xids, nullptr);
}

void IndexIVFPQR::add_core (idx_t n, const float *x, const long *xids,
                                const long *precomputed_idx) {

    float * residual_2 = new float [n * d];
    ScopeDeleter <float> del(residual_2);

    idx_t n0 = ntotal;

    add_core_o (n, x, xids, residual_2, precomputed_idx);

    refine_codes.resize (ntotal * refine_pq.code_size);

    refine_pq.compute_codes (
        residual_2, &refine_codes[n0 * refine_pq.code_size], n);


}


void IndexIVFPQR::search (
            idx_t n, const float *x, idx_t k,
            float *distances, idx_t *labels) const
{
    FAISS_THROW_IF_NOT (is_trained);
    long * idx = new long [n * nprobe];
    ScopeDeleter<long> del (idx);
    float * L1_dis = new float [n * nprobe];
    ScopeDeleter<float> del2 (L1_dis);
    uint64_t t0;
    TIC;
    quantizer->search (n, x, nprobe, L1_dis, idx);
    indexIVFPQ_stats.assign_cycles += TOC;

    TIC;
    size_t k_coarse = long(k * k_factor);
    idx_t *coarse_labels = new idx_t [k_coarse * n];
    ScopeDeleter<idx_t> del3 (coarse_labels);
    { // query with quantizer levels 1 and 2.
        float *coarse_distances = new float [k_coarse * n];
        ScopeDeleter<float> del(coarse_distances);

        search_preassigned (n, x, k_coarse,
                            idx, L1_dis, coarse_distances, coarse_labels,
                            true);
    }


    indexIVFPQ_stats.search_cycles += TOC;

    TIC;

    // 3rd level refinement
    size_t n_refine = 0;
#pragma omp parallel reduction(+ : n_refine)
    {
        // tmp buffers
        float *residual_1 = new float [2 * d];
        ScopeDeleter<float> del (residual_1);
        float *residual_2 = residual_1 + d;
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const float *xq = x + i * d;
            const long * shortlist = coarse_labels + k_coarse * i;
            float * heap_sim = distances + k * i;
            long * heap_ids = labels + k * i;
            maxheap_heapify (k, heap_sim, heap_ids);

            for (int j = 0; j < k_coarse; j++) {
                long sl = shortlist[j];

                if (sl == -1) continue;

                int list_no = sl >> 32;
                int ofs = sl & 0xffffffff;

                assert (list_no >= 0 && list_no < nlist);
                assert (ofs >= 0 && ofs < ids[list_no].size());

                // 1st level residual
                quantizer->compute_residual (xq, residual_1, list_no);

                // 2nd level residual
                const uint8_t * l2code = &codes[list_no][ofs * pq.code_size];
                pq.decode (l2code, residual_2);
                for (int l = 0; l < d; l++)
                    residual_2[l] = residual_1[l] - residual_2[l];

                // 3rd level residual's approximation
                idx_t id = ids[list_no][ofs];
                assert (0 <= id && id < ntotal);
                refine_pq.decode (&refine_codes [id * refine_pq.code_size],
                                  residual_1);

                float dis = fvec_L2sqr (residual_1, residual_2, d);

                if (dis < heap_sim[0]) {
                    maxheap_pop (k, heap_sim, heap_ids);
                    maxheap_push (k, heap_sim, heap_ids, dis, id);
                }
                n_refine ++;
            }
            maxheap_reorder (k, heap_sim, heap_ids);
        }
    }
    indexIVFPQ_stats.nrefine += n_refine;
    indexIVFPQ_stats.refine_cycles += TOC;
}

void IndexIVFPQR::reconstruct_n (idx_t i0, idx_t ni, float *recons) const
{
    std::vector<float> r3 (d);

    IndexIVFPQ::reconstruct_n (i0, ni, recons);

    for (idx_t i = i0; i < i0 + ni; i++) {
        float *r = recons + i * d;
        refine_pq.decode (&refine_codes [i * refine_pq.code_size], r3.data());

        for (int j = 0; j < d; j++)
            r[j] += r3[j];

    }

}



void IndexIVFPQR::merge_from (IndexIVF &other_in, idx_t add_id)
{
    IndexIVFPQR *other = dynamic_cast<IndexIVFPQR *> (&other_in);
    FAISS_THROW_IF_NOT(other);

    IndexIVF::merge_from (other_in, add_id);

    refine_codes.insert (refine_codes.end(),
                         other->refine_codes.begin(),
                         other->refine_codes.end());
    other->refine_codes.clear();
}

long IndexIVFPQR::remove_ids(const IDSelector& /*sel*/) {
  FAISS_THROW_MSG("not implemented");
  return 0;
}

/*****************************************
 * IndexIVFPQCompact implementation
 ******************************************/

IndexIVFPQCompact::IndexIVFPQCompact ()
{
    alloc_type = Alloc_type_none;
    limits = nullptr;
    compact_ids = nullptr;
    compact_codes = nullptr;
}


IndexIVFPQCompact::IndexIVFPQCompact (const IndexIVFPQ &other)
{
    FAISS_THROW_IF_NOT_MSG (other.ntotal < (1UL << 31),
                      "IndexIVFPQCompact cannot store more than 2G images");

    // here it would be more convenient to just use the
    // copy-constructor, but it would copy the lists as well: too much
    // overhead...

    // copy fields from Index
    d = other.d;
    ntotal = other.ntotal;
    verbose = other.verbose;
    is_trained = other.is_trained;
    metric_type = other.metric_type;

    // copy fields from IndexIVF (except ids)
    nlist = other.nlist;
    nprobe = other.nprobe;
    quantizer = other.quantizer;
    quantizer_trains_alone = other.quantizer_trains_alone;
    own_fields = false;
    direct_map = other.direct_map;

    // copy fields from IndexIVFPQ (except codes)
    by_residual = other.by_residual;
    use_precomputed_table = other.use_precomputed_table;
    precomputed_table = other.precomputed_table;
    code_size = other.code_size;
    pq = other.pq;
    do_polysemous_training = other.do_polysemous_training;
    polysemous_training = nullptr;

    scan_table_threshold = other.scan_table_threshold;
    max_codes = other.max_codes;
    polysemous_ht = other.polysemous_ht;

    //allocate
    alloc_type = Alloc_type_new;
    limits = new uint32_t [nlist + 1];
    compact_ids = new uint32_t [ntotal];
    compact_codes = new uint8_t [ntotal * code_size];


    // copy content from other
    size_t ofs = 0;
    for (size_t i = 0; i < nlist; i++) {
        limits [i] = ofs;
        const std::vector<long> &other_ids = other.ids[i];
        for (size_t j = 0; j < other_ids.size(); j++) {
            long id = other_ids[j];
            FAISS_THROW_IF_NOT_MSG (id < (1UL << 31),
                              "IndexIVFPQCompact cannot store ids > 2G");
            compact_ids[ofs + j] = id;
        }
        memcpy (compact_codes + ofs * code_size,
                other.codes[i].data(),
                other.codes[i].size());
        ofs += other_ids.size();
    }
    FAISS_THROW_IF_NOT (ofs == ntotal);
    limits [nlist] = ofs;

}

void IndexIVFPQCompact::add (idx_t, const float *) {
    FAISS_THROW_MSG ("cannot add to an IndexIVFPQCompact");
}

void IndexIVFPQCompact::reset () {
    FAISS_THROW_MSG ("cannot reset an IndexIVFPQCompact");
}

void IndexIVFPQCompact::train (idx_t, const float *) {
    FAISS_THROW_MSG ("cannot train an IndexIVFPQCompact");
}




IndexIVFPQCompact::~IndexIVFPQCompact ()
{
    if (alloc_type == Alloc_type_new) {
        delete [] limits;
        delete [] compact_codes;
        delete [] compact_ids;
    } else if (alloc_type == Alloc_type_mmap) {
        munmap (mmap_buffer, mmap_length);

    }

}



void IndexIVFPQCompact::search_preassigned (idx_t nx, const float *qx, idx_t k,
                                     const idx_t *keys,
                                     const float *coarse_dis,
                                     float *distances, idx_t *labels,
                                     bool store_pairs) const
{
    float_maxheap_array_t res = {
        size_t(nx), size_t(k),
        labels, distances
    };

#pragma omp parallel
    {
        InvertedListScanner<uint32_t> qt (*this);
        size_t stats_nlist = 0;
        size_t stats_ncode = 0;
        uint64_t init_query_cycles = 0;
        uint64_t scan_cycles = 0;
        uint64_t heap_cycles = 0;

#pragma omp  for
        for (size_t i = 0; i < nx; i++) {
            const float *qi = qx + i * d;
            const long * keysi = keys + i * nprobe;
            const float *coarse_dis_i = coarse_dis + i * nprobe;
            float * heap_sim = res.get_val (i);
            long * heap_ids = res.get_ids (i);

            uint64_t t0;
            TIC;
            maxheap_heapify (k, heap_sim, heap_ids);
            heap_cycles += TOC;

            TIC;
            qt.init_query (qi);
            init_query_cycles += TOC;

            size_t nscan = 0;

            for (size_t ik = 0; ik < nprobe; ik++) {
                long key = keysi[ik];  /* select the list  */
                if (key < 0) {
                    // not enough centroids for multiprobe
                    continue;
                }
                if (key >= (long) nlist) {
                    fprintf (stderr, "Invalid key=%ld nlist=%ld\n", key, nlist);
                    throw;
                }
                size_t list_size = limits[key + 1] - limits[key];
                stats_nlist ++;
                nscan += list_size;

                if (list_size == 0) continue;

                qt.init_list (key, coarse_dis_i[ik],
                              list_size, compact_ids + limits[key],
                              compact_codes + limits[key] * code_size);

                TIC;
                if (polysemous_ht > 0) {
                    qt.scan_list_polysemous
                        (k, heap_sim, heap_ids, store_pairs);
                } else if (list_size > scan_table_threshold) {
                   qt.scan_list_with_table (k, heap_sim, heap_ids, store_pairs);
                } else {
                   qt.scan_on_the_fly_dist (k, heap_sim, heap_ids, store_pairs);
                }
                scan_cycles += TOC;

                if (max_codes && nscan >= max_codes) break;
            }
            stats_ncode += nscan;
            TIC;
            maxheap_reorder (k, heap_sim, heap_ids);

            if (metric_type == METRIC_INNER_PRODUCT) {
                for (size_t j = 0; j < k; j++) {
                    heap_sim[i] = -heap_sim[i];
                }
            }
            heap_cycles += TOC;
        }

#pragma omp critical
        {
            indexIVFPQ_stats.n_hamming_pass += qt.n_hamming_pass;
            indexIVFPQ_stats.nlist += stats_nlist;
            indexIVFPQ_stats.ncode += stats_ncode;

            indexIVFPQ_stats.init_query_cycles += init_query_cycles;
            indexIVFPQ_stats.init_list_cycles += qt.init_list_cycles;
            indexIVFPQ_stats.scan_cycles += scan_cycles - qt.init_list_cycles;
            indexIVFPQ_stats.heap_cycles += heap_cycles;
        }

    }
    indexIVFPQ_stats.nq += nx;
}



} // namespace faiss
